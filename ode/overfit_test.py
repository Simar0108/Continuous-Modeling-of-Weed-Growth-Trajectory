"""Overfit Lightning module restored from Step-6 3.10 bytecode.

CLI ``main()`` is not reconstructed. ``OverfitLightning`` is the H1/NCDE base.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from pathlib import Path

from ode.training_loop import LINEAR_SIZE_WEIGHT, LOG_SIZE_EPS, LatentODELightning

MIN_OBSERVATIONS = 15
TARGET_TRACK_ID = 5208
QUICK_OVERFIT_EPOCHS = 500
FULL_OVERFIT_EPOCHS = 1000
OVERFIT_LR = 0.0005
OVERFIT_GRADIENT_CLIP = 5
SCHEDULER_PATIENCE = 10
PLOT_EVERY_N_EPOCHS = 10
GEO_LOSS_SUCCESS_THRESHOLD = 0.05
OVERFIT_KINETIC_REG_WEIGHT = 0.0
OVERFIT_Z0_REG_WEIGHT = 0.01
LATENT_DIM_SWEEP = [8, 16, 32, 64]


def _split_channel_logs(logs: dict, prefix: str) -> tuple[dict, dict]:
    """Separate aggregate losses from per-channel MSE keys ``{prefix}_mse_*``."""
    main: dict = {}
    ch: dict = {}
    p = f"{prefix}_mse_"
    for k, v in logs.items():
        (ch if k.startswith(p) else main)[k] = v
    return main, ch


def select_overfit_track(parquet_path: Path) -> int:
    """Return track_id 5208 if present with >= MIN_OBSERVATIONS, else longest valid track."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if "valid_track" in df.columns:
        df = df[df["valid_track"]].copy()
    counts = df.groupby("track_id").size()
    long_enough = counts[counts >= MIN_OBSERVATIONS].index.tolist()
    if not long_enough:
        raise ValueError(f"No track has >={MIN_OBSERVATIONS} observations in {parquet_path}")
    if TARGET_TRACK_ID in long_enough:
        return int(TARGET_TRACK_ID)
    return int(counts.loc[counts.index.isin(long_enough)].idxmax())


class OverfitLightning(LatentODELightning):
    """Overfit-specific Lightning module.

    Differences from the base LatentODELightning:
    - ReduceLROnPlateau scheduler with separate ODE / encoder-decoder param groups
    - training_step splits logs into aggregate + per-channel for clean W&B panels
    - Extra W&B logs: NFE counter, z0 L2 norm
    - validation_step stashes predictions for the OverfitPlotCallback

    The three-group Adam split reads ``ode.late_head`` (D-017 crash on NCDE).
    NCDELightning overrides ``configure_optimizers`` (D-017a).
    """

    def __init__(
        self,
        kinetic_reg_weight: float = 0.0,
        z0_reg_weight: float = 0.01,
        geometry_size_loss: Literal["linear", "log"] = "log",
        log_size_eps: float = 0.0001,
        lr: float = 0.0005,
        **kwargs,
    ) -> None:
        super().__init__(
            lr=lr,
            kinetic_reg_weight=kinetic_reg_weight,
            z0_reg_weight=z0_reg_weight,
            geometry_size_loss=geometry_size_loss,
            log_size_eps=log_size_eps,
            **kwargs,
        )

    def configure_optimizers(self):
        """Three-group Adam + ReduceLROnPlateau.

        Groups:
          1. encoder / decoder / affine  — standard lr, no weight decay
          2. shared backbone + early_head + gate params — standard lr
          3. late_head — lr × late_head_lr_scale (default 0.2×)
        """
        ode = self.model.ode_func
        boost_mods = []
        if getattr(self, "head_lr_mult", 1.0) != 1.0:
            if hasattr(ode, "rate_net"):
                boost_mods.append(ode.rate_net)
            if hasattr(ode, "saturation_net"):
                boost_mods.append(ode.saturation_net)
            if getattr(self.model, "track_embedding", None) is not None:
                boost_mods.append(self.model.track_embedding)
        boost_params = [p for m in boost_mods for p in m.parameters()]
        boost_ids = {id(p) for p in boost_params}
        late_ids = {id(p) for p in ode.late_head.parameters()} - boost_ids
        ode_ids = {id(p) for p in ode.parameters()}
        other_params = [
            p for p in self.model.parameters()
            if id(p) not in ode_ids and id(p) not in boost_ids
        ]
        ode_rest = [
            p for p in ode.parameters()
            if id(p) not in late_ids and id(p) not in boost_ids
        ]
        late_params = [
            p for p in ode.late_head.parameters()
            if id(p) not in boost_ids
        ]
        groups = [
            {"params": other_params, "lr": self.lr, "weight_decay": 0.0},
            {"params": ode_rest, "lr": self.lr, "weight_decay": self.ode_weight_decay},
        ]
        if late_params:
            groups.append({
                "params": late_params,
                "lr": self.lr * self.late_head_lr_scale,
                "weight_decay": self.ode_weight_decay,
            })
        if boost_params:
            groups.append({
                "params": boost_params,
                "lr": self.lr * self.head_lr_mult,
                "weight_decay": self.ode_weight_decay,
            })
        groups = [g for g in groups if len(g["params"]) > 0]
        optimizer = torch.optim.Adam(groups, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE, min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        loss, logs, batch_size = self._step(batch, "train")
        main_logs, ch_logs = _split_channel_logs(logs, "train")
        self.log_dict(main_logs, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log_dict(ch_logs, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log(
            "ode_nfe", float(self.model.ode_func.nfe),
            on_step=True, on_epoch=True, batch_size=batch_size,
        )
        if self._last_aux is not None:
            self.log(
                "z0_l2_norm",
                self._last_aux.z0.norm(dim=-1).mean().item(),
                on_step=True, on_epoch=True, batch_size=batch_size,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs, batch_size = self._step(batch, "val")
        main_logs, ch_logs = _split_channel_logs(logs, "val")
        self.log_dict(main_logs, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log_dict(ch_logs, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        if batch_idx == 0:
            states_5d = batch["states_5d"]
            t_absolute = batch["t_absolute"]
            color_mask = batch["color_mask"]
            if states_5d.dim() == 2:
                states_5d = states_5d.unsqueeze(0)
                t_absolute = t_absolute.unsqueeze(0)
                color_mask = color_mask.unsqueeze(0)
            track_lengths = batch.get("track_lengths", None)
            if track_lengths is not None and t_absolute.dim() > 1:
                t_abs = t_absolute[int(track_lengths.argmax().item())]
            elif t_absolute.dim() > 1:
                t_abs = t_absolute[0]
            else:
                t_abs = t_absolute
            pred = self.model(
                states_5d, t_abs, return_aux=False, track_ids=batch.get("track_ids"),
            )
            self._val_pred = pred.detach()
            self._val_states_5d = states_5d.detach()
            self._val_t_absolute = t_absolute.detach()
            self._val_color_mask = color_mask.detach()
            self._val_track_lengths = (
                track_lengths.detach() if track_lengths is not None else None
            )
        return loss


class OverfitPlotCallback:
    """Placeholder — plotting CLI is not reconstructed."""

    def __init__(self, plot_every: int = 10, transformer=None) -> None:
        self.plot_every = plot_every
        self.transformer = transformer


class WandbArtifactCheckpoint:
    """Placeholder — artifact CLI is not reconstructed."""

    def __init__(self, monitor: str = "val_loss", mode: str = "min", artifact_name: str = "ckpt") -> None:
        self.monitor = monitor
        self.mode = mode
        self.artifact_name = artifact_name
        self.best_model_path = ""


def main() -> None:
    raise SystemExit(
        "REFUSE: ode.overfit_test CLI was restored as OverfitLightning only. "
        "Use python -m ode.train_h1_seed or python -m ode.train_ncde."
    )


if __name__ == "__main__":
    main()
