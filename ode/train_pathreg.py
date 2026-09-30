"""Train the pathreg H1 arm. Does not write h1_final_best or h1_stab."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from ode.callbacks_h1 import BestEpochGateCallback, FixedHorizonValCallback
from ode.data.datamodule import (
    PlantTrackDataModule,
    _collate_variable_length_tracks,
    audit_z_coverage,
)
from ode.data.dataset import PlantTrackStateDataset
from ode.pathreg import (
    PathRegDiagnosticsCallback,
    PathRegLightning,
    lambda_tag,
    write_provenance,
)
from ode.repro import assert_clean_or_allowed
from ode.train_baselines import clip_tracks_to_time_frac, select_discrete_tracks
from ode.train_h1_seed import H1_HP
from ode.train_multi import EpochGatedModelCheckpoint

REPO = Path(__file__).resolve().parent.parent
LOCKED_TAGS = ("h1_final_best", "h1_stab", "h1_seed")


def _refuse_lock_paths(run_tag: str, out_dir: Path) -> None:
    text = f"{run_tag} {out_dir}"
    if "h1_final_best" in text:
        raise SystemExit("REFUSE: will not write under h1_final_best")
    if "h1_stab_seed" in text:
        raise SystemExit("REFUSE: will not write under h1_stab checkpoints")
    for tag in LOCKED_TAGS:
        if run_tag == tag or run_tag.startswith(f"{tag}_s"):
            raise SystemExit(f"REFUSE: run-tag {run_tag!r} collides with the locked arm")


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
    p.add_argument("--train-time-frac", type=float, default=1.0)
    p.add_argument("--pathreg-lambda", type=float, required=True)
    p.add_argument("--run-tag", type=str, default=None)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--fixed-horizon-val-every", type=int, default=1)
    args = p.parse_args()

    if float(args.pathreg_lambda) <= 0:
        raise SystemExit("REFUSE: pathreg lambda must be > 0; lambda=0 is the locked arm")

    git = assert_clean_or_allowed(REPO, args.allow_dirty)
    L.seed_everything(args.seed, workers=True)

    parquet = args.parquet.expanduser()
    if not parquet.is_absolute():
        parquet = REPO / parquet
    if not parquet.is_file():
        raise FileNotFoundError(parquet)

    tag = args.run_tag or f"h1_pathreg_l{lambda_tag(args.pathreg_lambda)}_seed"
    name = args.name or f"{tag}{args.seed}"
    out_dir = REPO / "checkpoints" / f"{tag}{args.seed}"
    if args.train_time_frac < 1.0:
        out_dir = REPO / "checkpoints" / f"{tag}{args.seed}_extrap60"
    _refuse_lock_paths(tag, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids, test_ids = select_discrete_tracks(
        parquet, max_tracks=args.max_tracks, species=args.species,
        min_observations=15, val_frac=args.val_frac, test_frac=args.test_frac,
    )
    print(
        f"[pathreg] seed={args.seed} name={name} lambda={args.pathreg_lambda} "
        f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} "
        f"time_frac={args.train_time_frac} git={git['git_hash'][:12]}"
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
    audit_z_coverage(dm, label=f"pathreg {name}", n_tracks=len(train_ids))
    dm.val_dataloader = lambda: DataLoader(
        dm._val_ds, batch_size=len(val_ids), shuffle=False, num_workers=0,
        collate_fn=_collate_variable_length_tracks,
    )

    hp = dict(H1_HP)
    hp["pathreg_lambda"] = float(args.pathreg_lambda)
    cfg_hash = hashlib.sha256(
        json.dumps({"H1_HP": H1_HP, "pathreg_lambda": args.pathreg_lambda,
                    "solver": "dopri5", "rtol": 1e-6, "atol": 1e-6,
                    "ema": args.ema_decay, "monitor": "val_full_horizon_mse"},
                   sort_keys=True, default=str).encode()
    ).hexdigest()
    write_provenance(out_dir, {
        "arm": "pathreg",
        "pathreg_lambda": float(args.pathreg_lambda),
        "seed": int(args.seed),
        "run_tag": tag,
        "config_hash": cfg_hash,
        "git_hash": git["git_hash"],
        "git_dirty": git["git_dirty"],
        "solver": "dopri5",
        "ode_rtol": 1e-6,
        "ode_atol": 1e-6,
        "ema_decay": float(args.ema_decay),
        "checkpoint_selection": "min val_full_horizon_mse, EMA state_dict, fail if best epoch < 20",
        "epochs": int(args.epochs),
        "train_time_frac": float(args.train_time_frac),
        "normalize_z0": True,
        "locked_reference": "h1_stab_seed{0..4} / h1_final_best not written",
    })

    module = PathRegLightning(**hp)
    monitor = "val_full_horizon_mse" if args.fixed_horizon_val_every > 0 else "val_track_mse_mean"

    loggers: list = [CSVLogger(save_dir=str(out_dir), name="csv")]
    if args.wandb:
        wb = WandbLogger(project=args.project, name=name)
        try:
            wb.experiment.config.update({
                "git_hash": git["git_hash"], "git_dirty": git["git_dirty"],
                "seed": args.seed, "train_time_frac": args.train_time_frac,
                "pathreg_lambda": args.pathreg_lambda,
                "config_hash": cfg_hash,
                "lock_excluded": True,
                "ema_decay": args.ema_decay,
                "ckpt_monitor": monitor,
            }, allow_val_change=True)
        except Exception as exc:
            print(f"[pathreg] wandb config skipped: {exc}")
        loggers.append(wb)

    art_cb = ModelCheckpoint(
        dirpath=str(out_dir), filename="geo",
        monitor="val_geo_loss", mode="min", save_top_k=1,
    )
    best_cb = EpochGatedModelCheckpoint(
        min_epoch=0, dirpath=str(out_dir), filename="best",
        monitor=monitor, mode="min", save_top_k=1,
    )
    callbacks = [
        FixedHorizonValCallback(every_n_epochs=max(int(args.fixed_horizon_val_every), 1)),
        EMAWeightAveraging(decay=float(args.ema_decay)),
        art_cb, best_cb, BestEpochGateCallback(best_cb, min_epoch=20),
        PathRegDiagnosticsCallback(out_dir),
    ]
    trainer = Trainer(
        max_epochs=args.epochs, logger=loggers, callbacks=callbacks,
        enable_progress_bar=True, accelerator=args.accelerator, devices=1,
        gradient_clip_val=1.0, log_every_n_steps=1,
    )
    trainer.fit(module, datamodule=dm)
    print(f"[pathreg] done {name} val={trainer.callback_metrics.get(monitor)}")
    print(f"[pathreg] ckpt dir {out_dir} monitor={monitor}")
    if args.wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.summary["git_hash"] = git["git_hash"]
                wandb.summary["pathreg_lambda"] = float(args.pathreg_lambda)
                wandb.summary["config_hash"] = cfg_hash
                wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
