"""Multi-track Lightning wrappers restored from Step-6 3.10 bytecode.

CLI ``main()`` is not reconstructed (use ``ode.train_h1_seed``). Classes
used by H1 / NCDE / pathreg are source-complete.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path

from ode.overfit_test import (
    OVERFIT_GRADIENT_CLIP,
    OVERFIT_KINETIC_REG_WEIGHT,
    OVERFIT_LR,
    OVERFIT_Z0_REG_WEIGHT,
    OverfitLightning,
    _split_channel_logs,
)

MIN_OBSERVATIONS = 15
DEFAULT_MAX_TRACKS = 100
MULTI_TRACK_EPOCHS = 400


class EpochGatedModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that ignores validation results before `min_epoch`.

    Gates BOTH hooks: Lightning routes the top-k save through
    ``on_train_epoch_end`` (not ``on_validation_end``) when validation runs
    every epoch, so gating only one hook is silently bypassed.
    """

    def __init__(self, min_epoch: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._min_epoch = min_epoch

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if trainer.current_epoch < self._min_epoch:
            return None
        return super().on_train_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.current_epoch < self._min_epoch:
            return None
        return super().on_validation_end(trainer, pl_module)


class EpochGatedEarlyStopping(EarlyStopping):
    """EarlyStopping that ignores validation results before `min_epoch` (both hooks)."""

    def __init__(self, min_epoch: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._min_epoch = min_epoch

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if trainer.current_epoch < self._min_epoch:
            return None
        return super().on_train_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.current_epoch < self._min_epoch:
            return None
        return super().on_validation_end(trainer, pl_module)


def select_multi_tracks(
    parquet_path: Path,
    max_tracks: int,
    species: Optional[str] = None,
    min_observations: int = MIN_OBSERVATIONS,
    val_frac: float = 0.2,
) -> tuple[list[int], list[int]]:
    """Select the richest tracks and return (train_ids, val_ids)."""
    df = pd.read_parquet(parquet_path)
    if "valid_track" in df.columns:
        df = df[df["valid_track"]].copy()
    if species is not None and "species" in df.columns:
        df = df[df["species"] == species].copy()
        if df.empty:
            avail = pd.read_parquet(parquet_path).get("species", pd.Series()).unique().tolist()
            raise ValueError(f"No tracks found for species={species!r}. Available: {avail}")
    counts = df.groupby("track_id").size()
    counts = counts[counts >= min_observations]
    if counts.empty:
        raise ValueError(f"No tracks with >= {min_observations} observations after filtering.")
    top_ids = counts.nlargest(max_tracks).index.tolist()
    n_val = max(1, int(len(top_ids) * val_frac))
    sorted_ids = counts.loc[top_ids].sort_values(ascending=True).index.tolist()
    val_ids = sorted_ids[:n_val]
    train_ids = sorted_ids[n_val:]
    extra = f" — species={species!r}" if species else ""
    print(
        f"\n[train_multi] Track selection{extra}\n"
        f"  Total selected : {len(top_ids)}\n"
        f"  Train ({len(train_ids)})  : {train_ids}\n"
        f"  Val   ({len(val_ids)})  : {val_ids}"
    )
    return train_ids, val_ids


class MultiTrackLightning(OverfitLightning):
    """Multi-track Lightning module.

    Extends OverfitLightning with multi-track-aware step logging.
    ``configure_optimizers`` is inherited (three-group Adam + late_head).
    NCDELightning overrides that hook (D-017a).
    """

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


def main() -> None:
    raise SystemExit(
        "REFUSE: ode.train_multi CLI was restored as classes only. "
        "Use python -m ode.train_h1_seed (locked H1) or python -m ode.train_ncde."
    )


if __name__ == "__main__":
    main()
