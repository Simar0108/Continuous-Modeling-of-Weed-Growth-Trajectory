"""Step 8: Zwietering–Richards + alignment on the 100-track 70/15/15 split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from ode._pyc_bootstrap import bootstrap

bootstrap()

import lightning as L
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from ode.data.datamodule import (
    PlantTrackDataModule,
    _collate_variable_length_tracks,
    audit_z_coverage,
)
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
from ode.repro import assert_clean_or_allowed
from ode.rhs_families import (
    ExpRateODEFunc,
    ZwieteringRichardsODEFunc,
    attach_constant_z,
    attach_zwietering_initial_condition,
    disable_lag_gate,
    param_near_bound_frac,
)
from ode.train_baselines import select_discrete_tracks
from ode.train_multi import EpochGatedModelCheckpoint, MultiTrackLightning
from ode.training_loop import LINEAR_SIZE_WEIGHT, LOG_SIZE_EPS

REPO = Path(__file__).resolve().parent.parent


class ZwieteringParamCallback(Callback):
    """Log μ/λ/ν/t0 and bound-utilization on every train/val batch end."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log(pl_module, "train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._log(pl_module, "val")

    def _log(self, pl_module, prefix: str) -> None:
        ode = pl_module.model.ode_func
        cache = getattr(ode, "_cache", None)
        if not cache:
            return
        for name, val in cache.items():
            pl_module.log(f"zwiet/{name}_mean", val.mean(), on_step=True, on_epoch=True)
            if val.numel() > 1:
                pl_module.log(f"zwiet/{name}_std", val.std(unbiased=False), on_step=True, on_epoch=True)
        near = param_near_bound_frac(cache)
        for name, frac in near.items():
            pl_module.log(f"zwiet/near_bound_{name}", frac, on_step=True, on_epoch=True)


def _install(module: MultiTrackLightning, rhs: str, use_t0: bool, z_constant: bool) -> None:
    model = module.model
    latent = model.ode_func.latent_dim
    hidden = model.ode_func.shared[0].out_features
    n_layers = sum(1 for m in model.ode_func.shared if m.__class__.__name__ == "Linear")
    if rhs == "exprate":
        if StiffODEMixin not in ExpRateODEFunc.__bases__:
            ExpRateODEFunc.__bases__ = (StiffODEMixin,) + ExpRateODEFunc.__bases__
        model.ode_func = ExpRateODEFunc(latent_dim=latent, hidden_dim=hidden, n_layers=n_layers)
        disable_lag_gate(model.ode_func)
    elif rhs == "zwietering":
        if StiffODEMixin not in ZwieteringRichardsODEFunc.__bases__:
            ZwieteringRichardsODEFunc.__bases__ = (StiffODEMixin,) + ZwieteringRichardsODEFunc.__bases__
        ode = ZwieteringRichardsODEFunc(
            latent_dim=latent, hidden_dim=hidden, n_layers=n_layers, use_t0=use_t0,
        )
        model.ode_func = ode
        attach_zwietering_initial_condition(model, ode)
        if z_constant:
            attach_constant_z(model)
    else:
        raise ValueError(rhs)


def _per_track_size_mse(module, batch) -> torch.Tensor:
    states = batch["states_5d"]
    t_abs = batch["t_absolute"]
    lengths = batch.get("track_lengths")
    if lengths is not None and t_abs.dim() > 1:
        t_vec = t_abs[int(lengths.argmax().item())]
    else:
        t_vec = t_abs[0] if t_abs.dim() > 1 else t_abs
    with torch.no_grad():
        pred = module.model(states, t_vec, return_aux=False, track_ids=batch.get("track_ids"))
        pred = pred.permute(1, 0, 2)
        err = (pred[:, :, 2:4] - states[:, :, 2:4]).pow(2)
        if lengths is not None:
            idx = torch.arange(states.shape[1], device=states.device)
            w = (idx.unsqueeze(0) < lengths.unsqueeze(1)).float()
            return (err * w.unsqueeze(-1)).sum(dim=(1, 2)) / (w.sum(dim=1).clamp(min=1.0) * 2.0)
        return err.mean(dim=(1, 2))


def _run_wilcoxon(module, batch, args, ckpt_dir: Path) -> None:
    import numpy as np
    from scipy.stats import wilcoxon

    ours = _per_track_size_mse(module, batch).cpu().numpy()
    report: dict = {"n": int(ours.size), "ours_mean": float(ours.mean())}

    h1 = REPO / "checkpoints" / "h1_final_best" / "best.ckpt"
    if h1.is_file():
        try:
            ref = MultiTrackLightning.load_from_checkpoint(str(h1), map_location="cpu", strict=False)
            ref.eval()
            ref.horizon_start_frac = 1.0
            ref.horizon_ramp_start = 0
            theirs = _per_track_size_mse(ref, {k: v.cpu() if torch.is_tensor(v) else v for k, v in batch.items()})
            theirs = theirs.cpu().numpy()
            n = min(len(ours), len(theirs))
            if n >= 6:
                stat, p = wilcoxon(ours[:n], theirs[:n], alternative="less")
                report["vs_h1_final_best"] = {
                    "wilcoxon_stat": float(stat), "p": float(p),
                    "ref_mean": float(theirs[:n].mean()),
                    "ours_mean": float(ours[:n].mean()),
                }
        except Exception as exc:
            report["vs_h1_final_best_error"] = str(exc)

    t1 = ckpt_dir / f"step8_valmse_h1-s8-T1-s3-s{args.seed}.ckpt"
    if t1.is_file():
        try:
            ref = MultiTrackLightning.load_from_checkpoint(str(t1), map_location="cpu", strict=False)
            ref.eval()
            theirs = _per_track_size_mse(ref, batch).cpu().numpy()
            n = min(len(ours), len(theirs))
            if n >= 6:
                stat, p = wilcoxon(ours[:n], theirs[:n], alternative="less")
                report["vs_T1"] = {
                    "wilcoxon_stat": float(stat), "p": float(p),
                    "ref_mean": float(theirs[:n].mean()),
                    "ours_mean": float(ours[:n].mean()),
                }
        except Exception as exc:
            report["vs_T1_error"] = str(exc)
    else:
        report["vs_T1"] = "T1 checkpoint not on disk yet"

    out = REPO / "logs" / "step8_diagnostics" / f"wilcoxon_{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[step8] Wilcoxon → {out}\n{json.dumps(report, indent=2)}")
    try:
        import wandb
        if wandb.run is not None:
            wandb.summary["wilcoxon"] = report
    except Exception:
        pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--project", default="latent-ode-maize-100")
    p.add_argument("--name", type=str, default="h1-s8")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rhs", choices=("exprate", "zwietering"), default="zwietering")
    p.add_argument("--t0", action="store_true")
    p.add_argument("--z-constant", action="store_true")
    p.add_argument("--wilcoxon", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--max-tracks", type=int, default=100)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--head-lr-mult", type=float, default=10.0)
    p.add_argument("--accelerator", default="auto")
    args = p.parse_args()

    git = assert_clean_or_allowed(REPO, args.allow_dirty)

    L.seed_everything(args.seed, workers=True)
    parquet = args.parquet.expanduser()
    if not parquet.is_absolute():
        parquet = REPO / parquet
    if not parquet.is_file():
        raise FileNotFoundError(parquet)

    train_ids, val_ids, test_ids = select_discrete_tracks(
        parquet, max_tracks=args.max_tracks, species=None,
        min_observations=15, val_frac=args.val_frac, test_frac=args.test_frac,
    )
    print(
        f"[step8] rhs={args.rhs} t0={args.t0} z_const={args.z_constant} "
        f"seed={args.seed} train={len(train_ids)} val={len(val_ids)} "
        f"git={git['git_hash'][:12] if git['git_hash']!='NO_REPO' else 'NO_REPO'}"
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
    audit_z_coverage(dm, label=f"step8 {args.name}", n_tracks=len(train_ids))
    dm.val_dataloader = lambda: DataLoader(
        dm._val_ds, batch_size=len(val_ids), shuffle=False, num_workers=0,
        collate_fn=_collate_variable_length_tracks,
    )

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
        use_track_embed=True,
    )
    _install(module, args.rhs, args.t0, args.z_constant)

    ckpt_dir = REPO / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.wandb:
        logger = WandbLogger(project=args.project, name=args.name)
        try:
            logger.experiment.config.update({
                "git_hash": git["git_hash"],
                "git_dirty": git["git_dirty"],
                "rhs": args.rhs, "t0": args.t0, "z_constant": args.z_constant,
                "seed": args.seed,
            }, allow_val_change=True)
        except Exception as exc:
            print(f"[step8] wandb config update skipped: {exc}")
        ckpt_cb: Callback = WandbArtifactCheckpoint(
            monitor="val_geo_loss", mode="min", artifact_name=f"step8_{args.name}",
        )
    else:
        logger = True
        ckpt_cb = ModelCheckpoint(
            dirpath=str(ckpt_dir), filename=f"step8_{args.name}",
            monitor="val_geo_loss", mode="min", save_top_k=1,
        )
    best_val_cb = EpochGatedModelCheckpoint(
        min_epoch=0, dirpath=str(ckpt_dir),
        filename=f"step8_valmse_{args.name}",
        monitor="val_track_mse_mean", mode="min", save_top_k=1,
    )
    callbacks = [ckpt_cb, best_val_cb, OverfitPlotCallback(PLOT_EVERY_N_EPOCHS, dm.transformer)]
    if args.rhs == "zwietering":
        callbacks.append(ZwieteringParamCallback())

    trainer = Trainer(
        max_epochs=args.epochs, logger=logger, callbacks=callbacks,
        enable_progress_bar=True, accelerator=args.accelerator, devices=1,
        gradient_clip_val=OVERFIT_GRADIENT_CLIP, log_every_n_steps=1,
    )
    trainer.fit(module, datamodule=dm)
    print(f"[step8] done {args.name} val={trainer.callback_metrics.get('val_track_mse_mean')}")

    if args.wilcoxon:
        val_batch = next(iter(dm.val_dataloader()))
        _run_wilcoxon(module, val_batch, args, ckpt_dir)

    if args.wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.summary["git_hash"] = git["git_hash"]
                wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
