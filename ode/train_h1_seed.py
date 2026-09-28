"""Train one clean H1 ODE seed on Maize 70/15/15.

Hyperparameters match checkpoints/h1_final_best/best.ckpt (not run_100.sh).
Does not write under checkpoints/h1_final_best/.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from ode._pyc_bootstrap import bootstrap

bootstrap()

import lightning as L
import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EMAWeightAveraging, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from ode.data.datamodule import (
    PlantTrackDataModule,
    _collate_variable_length_tracks,
    audit_z_coverage,
)
from ode.data.dataset import PlantTrackStateDataset
from ode.overfit_test import WandbArtifactCheckpoint
from ode.repro import assert_clean_or_allowed
from ode.train_baselines import clip_tracks_to_time_frac, select_discrete_tracks
from ode.callbacks_h1 import BestEpochGateCallback, FixedHorizonValCallback
from ode.train_multi import EpochGatedModelCheckpoint, MultiTrackLightning
from ode.training_loop import LINEAR_SIZE_WEIGHT, LOG_SIZE_EPS

REPO = Path(__file__).resolve().parent.parent

# From h1_final_best/best.ckpt hyper_parameters (epoch field is 20; train 400).
H1_HP = dict(
    kinetic_reg_weight=0.0,
    z0_reg_weight=0.01,
    geometry_size_loss="log",
    log_size_eps=LOG_SIZE_EPS,
    linear_size_weight=LINEAR_SIZE_WEIGHT,
    xy_loss_weight=0.0,
    size_loss_weight=1.0,
    phys_loss_weight=0.2,
    phys_dropout=0.2,
    latent_dim=64,
    n_context_frames=3,
    encoder="mlp",
    ode_hidden=256,
    normalize_z0=True,
    ode_growth_bias=True,
    ode_solver="dopri5",
    ode_rtol=1e-6,
    ode_atol=1e-6,
    lr=5e-4,
    horizon_start_frac=0.3,
    horizon_ramp_start=100,
    horizon_ramp_end=250,
    affine_decoder=True,
    affine_scale_init=6.0,
    freeze_affine_epochs=50,
    hybrid_decoder=True,
    late_head_lr_scale=0.2,
    diversity_weight=0.0,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--project", default="latent-ode-maize-100")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--max-tracks", type=int, default=100)
    p.add_argument("--species", type=str, default="Maize")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--train-time-frac", type=float, default=1.0,
                    help="If <1, keep only the first fraction of each TRAIN track timeline.")
    p.add_argument("--run-tag", type=str, default="h1_seed",
                    help="Checkpoint directory prefix. Stab jobs use h1_stab_seed.")
    p.add_argument("--ema-decay", type=float, default=0.0,
                    help="If >0, evaluate and checkpoint EMA weights (Lightning EMAWeightAveraging).")
    p.add_argument("--fixed-horizon-val-every", type=int, default=0,
                    help="If >0, log val_full_horizon_mse every N epochs and select best on it.")
    args = p.parse_args()

    git = assert_clean_or_allowed(REPO, args.allow_dirty)
    L.seed_everything(args.seed, workers=True)

    parquet = args.parquet.expanduser()
    if not parquet.is_absolute():
        parquet = REPO / parquet
    if not parquet.is_file():
        raise FileNotFoundError(parquet)

    name = args.name or f"{args.run_tag}{args.seed}"
    train_ids, val_ids, test_ids = select_discrete_tracks(
        parquet, max_tracks=args.max_tracks, species=args.species,
        min_observations=15, val_frac=args.val_frac, test_frac=args.test_frac,
    )
    print(
        f"[h1_seed] seed={args.seed} name={name} train={len(train_ids)} "
        f"val={len(val_ids)} test={len(test_ids)} time_frac={args.train_time_frac} "
        f"git={git['git_hash'][:12] if git['git_hash']!='NO_REPO' else 'NO_REPO'}"
    )

    df = pd.read_parquet(parquet)
    if "valid_track" in df.columns:
        df = df[df["valid_track"]].copy()
    if args.species and "species" in df.columns:
        df = df[df["species"] == args.species].copy()
    train_df = df[df["track_id"].isin(train_ids)].copy()
    val_df = df[df["track_id"].isin(val_ids)].copy()
    if args.train_time_frac < 1.0:
        train_df = clip_tracks_to_time_frac(train_df, train_ids, args.train_time_frac)
        val_df = clip_tracks_to_time_frac(val_df, val_ids, args.train_time_frac)

    dm = PlantTrackDataModule(
        parquet_path=parquet, batch_size=min(args.batch_size, len(train_ids)),
        num_workers=0, sigma_mode="z_score",
    )
    dm.setup()
    dm._train_ds = PlantTrackStateDataset(
        dm.parquet_path, dm.transformer, track_ids=train_ids,
        dataframe=train_df, valid_track_only=True, min_observations=8,
    )
    dm._val_ds = PlantTrackStateDataset(
        dm.parquet_path, dm.transformer, track_ids=val_ids,
        dataframe=val_df, valid_track_only=True, min_observations=8,
    )
    dm.lock_setup()
    audit_z_coverage(dm, label=f"h1_seed {name}", n_tracks=len(train_ids))
    dm.val_dataloader = lambda: DataLoader(
        dm._val_ds, batch_size=len(val_ids), shuffle=False, num_workers=0,
        collate_fn=_collate_variable_length_tracks,
    )

    module = MultiTrackLightning(**H1_HP)

    if "h1_final_best" in args.run_tag:
        raise SystemExit("REFUSE: will not write under h1_final_best")
    out_dir = REPO / "checkpoints" / f"{args.run_tag}{args.seed}"
    if args.train_time_frac < 1.0:
        out_dir = REPO / "checkpoints" / f"{args.run_tag}{args.seed}_extrap60"
    out_dir.mkdir(parents=True, exist_ok=True)

    monitor = "val_track_mse_mean"
    if args.fixed_horizon_val_every > 0:
        monitor = "val_full_horizon_mse"

    if args.wandb:
        logger = WandbLogger(project=args.project, name=name)
        try:
            logger.experiment.config.update({
                "git_hash": git["git_hash"], "git_dirty": git["git_dirty"],
                "seed": args.seed, "train_time_frac": args.train_time_frac,
                "lock_excluded": True,
                "ema_decay": args.ema_decay,
                "fixed_horizon_val_every": args.fixed_horizon_val_every,
                "ckpt_monitor": monitor,
            }, allow_val_change=True)
        except Exception as exc:
            print(f"[h1_seed] wandb config skipped: {exc}")
        art_cb = WandbArtifactCheckpoint(
            monitor=monitor, mode="min", artifact_name=name,
        )
    else:
        logger = True
        art_cb = ModelCheckpoint(
            dirpath=str(out_dir), filename="geo",
            monitor="val_geo_loss", mode="min", save_top_k=1,
        )
    best_cb = EpochGatedModelCheckpoint(
        min_epoch=0, dirpath=str(out_dir), filename="best",
        monitor=monitor, mode="min", save_top_k=1,
    )
    callbacks = []
    if args.fixed_horizon_val_every > 0:
        # Must run before EMAWeightAveraging.on_validation_epoch_end swaps weights back.
        callbacks.append(FixedHorizonValCallback(every_n_epochs=args.fixed_horizon_val_every))
    if args.ema_decay > 0:
        callbacks.append(EMAWeightAveraging(decay=float(args.ema_decay)))
    callbacks.extend([art_cb, best_cb, BestEpochGateCallback(best_cb, min_epoch=20)])

    trainer = Trainer(
        max_epochs=args.epochs, logger=logger, callbacks=callbacks,
        enable_progress_bar=True, accelerator=args.accelerator, devices=1,
        gradient_clip_val=1.0, log_every_n_steps=1,
    )
    trainer.fit(module, datamodule=dm)
    print(f"[h1_seed] done {name} val={trainer.callback_metrics.get(monitor)}")
    print(f"[h1_seed] ckpt dir {out_dir} monitor={monitor}")
    if args.wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.summary["git_hash"] = git["git_hash"]
                wandb.summary["ckpt_monitor"] = monitor
                wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
