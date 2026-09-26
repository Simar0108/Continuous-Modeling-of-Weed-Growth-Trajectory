"""Step 7 training: RHS-family ablation on the 100-track 70/15/15 split."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from ode._pyc_bootstrap import bootstrap

bootstrap()

import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from ode.data.datamodule import PlantTrackDataModule, audit_z_coverage, _collate_variable_length_tracks
from ode.data.dataset import PlantTrackStateDataset
from ode.model import StiffODEMixin
from ode.overfit_test import (
    OVERFIT_GRADIENT_CLIP,
    OVERFIT_KINETIC_REG_WEIGHT,
    OVERFIT_LR,
    OVERFIT_Z0_REG_WEIGHT,
    PLOT_EVERY_N_EPOCHS,
    OverfitPlotCallback,
    WandbArtifactCheckpoint,
)
from ode.rhs_families import (
    ExpRateODEFunc,
    RichardsODEFunc,
    attach_constant_z,
    disable_lag_gate,
)
from ode.train_baselines import select_discrete_tracks
from ode.train_multi import EpochGatedModelCheckpoint, MultiTrackLightning
from ode.training_loop import LINEAR_SIZE_WEIGHT, LOG_SIZE_EPS


def _install_rhs(module: MultiTrackLightning, rhs: str, z_constant: bool) -> None:
    model = module.model
    latent = model.ode_func.latent_dim
    hidden = model.ode_func.shared[0].out_features
    n_layers = sum(1 for m in model.ode_func.shared if m.__class__.__name__ == "Linear")
    if StiffODEMixin not in RichardsODEFunc.__bases__:
        RichardsODEFunc.__bases__ = (StiffODEMixin,) + RichardsODEFunc.__bases__
        ExpRateODEFunc.__bases__ = (StiffODEMixin,) + ExpRateODEFunc.__bases__
    if rhs == "richards":
        model.ode_func = RichardsODEFunc(latent_dim=latent, hidden_dim=hidden, n_layers=n_layers)
    elif rhs == "exprate":
        model.ode_func = ExpRateODEFunc(latent_dim=latent, hidden_dim=hidden, n_layers=n_layers)
    elif rhs == "stable":
        disable_lag_gate(model.ode_func)
    else:
        raise ValueError(rhs)
    if rhs != "stable":
        disable_lag_gate(model.ode_func)
    if z_constant:
        attach_constant_z(model)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--project", default="latent-ode-maize-100")
    p.add_argument("--name", type=str, default="h1-s7")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rhs", choices=("stable", "richards", "exprate"), default="stable")
    p.add_argument("--z-constant", action="store_true")
    p.add_argument("--max-tracks", type=int, default=100)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--head-lr-mult", type=float, default=10.0)
    p.add_argument("--track-embed", action="store_true", default=True)
    p.add_argument("--no-track-embed", action="store_true")
    p.add_argument("--accelerator", default="auto")
    args = p.parse_args()

    L.seed_everything(args.seed, workers=True)
    use_embed = not args.no_track_embed

    parquet = args.parquet.expanduser()
    if not parquet.is_absolute():
        parquet = Path(__file__).parent.parent / parquet
    if not parquet.is_file():
        raise FileNotFoundError(parquet)

    train_ids, val_ids, test_ids = select_discrete_tracks(
        parquet,
        max_tracks=args.max_tracks,
        species=None,
        min_observations=15,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    print(
        f"[step7] rhs={args.rhs} z_const={args.z_constant} seed={args.seed} "
        f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} embed={use_embed}"
    )

    dm = PlantTrackDataModule(
        parquet_path=parquet, batch_size=min(args.batch_size, len(train_ids)),
        num_workers=0, sigma_mode="z_score",
    )
    dm.setup()
    dm._train_ds = PlantTrackStateDataset(
        dm.parquet_path, dm.transformer, track_ids=train_ids, valid_track_only=True, min_observations=15,
    )
    dm._val_ds = PlantTrackStateDataset(
        dm.parquet_path, dm.transformer, track_ids=val_ids, valid_track_only=True, min_observations=15,
    )
    dm.lock_setup()
    audit_z_coverage(dm, label=f"step7 {args.name}", n_tracks=len(train_ids))

    def _val_dl():
        return DataLoader(
            dm._val_ds, batch_size=len(val_ids), shuffle=False, num_workers=0,
            collate_fn=_collate_variable_length_tracks,
        )

    dm.val_dataloader = _val_dl

    module = MultiTrackLightning(
        kinetic_reg_weight=OVERFIT_KINETIC_REG_WEIGHT,
        z0_reg_weight=OVERFIT_Z0_REG_WEIGHT,
        head_lr_mult=args.head_lr_mult,
        diversity_weight=0.0,
        geometry_size_loss="log",
        log_size_eps=LOG_SIZE_EPS,
        linear_size_weight=LINEAR_SIZE_WEIGHT,
        xy_loss_weight=0.0,
        size_loss_weight=1.0,
        phys_loss_weight=0.5,
        phys_dropout=0.2,
        latent_dim=args.latent_dim,
        n_context_frames=3,
        encoder="mlp",
        ode_hidden=256,
        normalize_z0=True,
        ode_growth_bias=True,
        ode_solver="dopri5",
        ode_rtol=1e-6,
        ode_atol=1e-6,
        lr=OVERFIT_LR,
        horizon_start_frac=1.0,
        horizon_ramp_start=0,
        horizon_ramp_end=0,
        affine_decoder=True,
        affine_scale_init=6.0,
        freeze_affine_epochs=50,
        hybrid_decoder=True,
        late_head_lr_scale=0.2,
        use_track_embed=use_embed,
    )
    _install_rhs(module, args.rhs, args.z_constant)

    ckpt_dir = Path(__file__).parent.parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.wandb:
        logger = WandbLogger(project=args.project, name=args.name)
        ckpt_cb: Callback = WandbArtifactCheckpoint(
            monitor="val_geo_loss", mode="min", artifact_name=f"step7_{args.name}",
        )
    else:
        logger = True
        ckpt_cb = ModelCheckpoint(
            dirpath=str(ckpt_dir), filename=f"step7_{args.name}",
            monitor="val_geo_loss", mode="min", save_top_k=1,
        )
    best_val_cb = EpochGatedModelCheckpoint(
        min_epoch=0,
        dirpath=str(ckpt_dir),
        filename=f"step7_valmse_{args.name}",
        monitor="val_track_mse_mean",
        mode="min",
        save_top_k=1,
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[ckpt_cb, best_val_cb, OverfitPlotCallback(PLOT_EVERY_N_EPOCHS, dm.transformer)],
        enable_progress_bar=True,
        accelerator=args.accelerator,
        devices=1,
        gradient_clip_val=OVERFIT_GRADIENT_CLIP,
        log_every_n_steps=1,
    )
    trainer.fit(module, datamodule=dm)
    print(
        f"[step7] done {args.name} val_track_mse_mean="
        f"{trainer.callback_metrics.get('val_track_mse_mean')}"
    )
    if args.wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
