"""
Training entry point for the discrete baseline suite.

Baselines:
  - LSTM
  - GRU
  - causal Transformer

Training uses teacher forcing only.
Validation / testing use fully autoregressive rollout.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLBACKEND", "Agg")

from ode._pyc_bootstrap import bootstrap

bootstrap()

import lightning as L
import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from .baselines import TARGET_SLICE, build_baseline_model
from .data.datamodule import (
    _collate_single_track,
    _collate_variable_length_tracks,
    audit_z_coverage,
)
from .data.dataset import PlantTrackStateDataset
from .data.transforms import GaussianStateTransformer
from .overfit_test import GEO_LOSS_SUCCESS_THRESHOLD, WandbArtifactCheckpoint
from .repro import assert_clean_or_allowed
from .training_loop import (
    LINEAR_SIZE_WEIGHT,
    LOG_SIZE_EPS,
    hybrid_loss,
    per_channel_mse_logs,
)

MIN_OBSERVATIONS = 15
DEFAULT_MAX_TRACKS = 100
DEFAULT_EPOCHS = 300


def clip_tracks_to_time_frac(
    df: pd.DataFrame,
    track_ids: list,
    frac: float,
    time_col: str = "time_since_germination_hours",
) -> pd.DataFrame:
    """Keep the first ``frac`` of each listed track's observation timeline."""
    if frac >= 1.0:
        return df
    if frac <= 0.0:
        raise ValueError("train_time_frac must be in (0, 1]")
    idset = set(track_ids)
    parts: list[pd.DataFrame] = []
    for tid, group in df.groupby("track_id"):
        group = group.sort_values(time_col)
        if tid not in idset:
            parts.append(group)
            continue
        times = group[time_col].to_numpy(dtype=float)
        if times.size < 2:
            parts.append(group)
            continue
        cut = float(times[0] + frac * (times[-1] - times[0]))
        clipped = group[group[time_col] <= cut + 1e-9]
        if len(clipped) < 3:
            clipped = group.iloc[: max(3, len(group) // 2)]
        parts.append(clipped)
    return pd.concat(parts, ignore_index=True) if parts else df


def _resolve_split_sizes(
    n_tracks: int,
    val_frac: float,
    test_frac: float,
) -> tuple[int, int, int]:
    if n_tracks < 3:
        raise ValueError("Need at least 3 tracks to create train/val/test splits.")

    n_val = max(1, int(round(n_tracks * val_frac)))
    n_test = max(1, int(round(n_tracks * test_frac)))
    n_train = n_tracks - n_val - n_test

    while n_train < 1:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            raise ValueError("Split fractions leave no training tracks.")
        n_train = n_tracks - n_val - n_test

    return n_train, n_val, n_test


def select_discrete_tracks(
    parquet_path: Path,
    *,
    max_tracks: int = DEFAULT_MAX_TRACKS,
    species: Optional[str] = None,
    min_observations: int = MIN_OBSERVATIONS,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[list[int], list[int], list[int]]:
    """
    Select the richest tracks, then split them deterministically by trajectory length.

    Shorter tracks are assigned to validation/test to keep the held-out splits
    slightly harder than the training set.
    """
    df = pd.read_parquet(parquet_path)
    if "valid_track" in df.columns:
        df = df[df["valid_track"]].copy()

    if species is not None and "species" in df.columns:
        df = df[df["species"] == species].copy()
        if df.empty:
            raise ValueError(f"No tracks found for species={species!r}.")

    counts = df.groupby("track_id").size()
    counts = counts[counts >= min_observations]
    if counts.empty:
        raise ValueError(
            f"No tracks with >= {min_observations} observations after filtering."
        )

    top_ids = counts.nlargest(max_tracks).index.tolist()
    n_train, n_val, n_test = _resolve_split_sizes(len(top_ids), val_frac, test_frac)

    sorted_ids = counts.loc[top_ids].sort_values(ascending=True).index.tolist()
    val_ids = sorted_ids[:n_val]
    test_ids = sorted_ids[n_val : n_val + n_test]
    train_ids = sorted_ids[n_val + n_test :]

    print(
        f"\n[train_baselines] Track selection"
        + (f" — species={species!r}" if species else "")
        + f"\n  Total selected : {len(top_ids)}"
        + f"\n  Train ({n_train}) : {train_ids}"
        + f"\n  Val   ({n_val}) : {val_ids}"
        + f"\n  Test  ({n_test}) : {test_ids}"
    )
    return train_ids, val_ids, test_ids


def build_baseline_datasets(
    parquet_path: Path,
    *,
    sigma_mode: str,
    max_tracks: int,
    species: Optional[str],
    val_frac: float,
    test_frac: float,
    min_observations: int = MIN_OBSERVATIONS,
    train_time_frac: float = 1.0,
) -> tuple[GaussianStateTransformer, PlantTrackStateDataset, PlantTrackStateDataset, PlantTrackStateDataset]:
    train_ids, val_ids, test_ids = select_discrete_tracks(
        parquet_path,
        max_tracks=max_tracks,
        species=species,
        min_observations=min_observations,
        val_frac=val_frac,
        test_frac=test_frac,
    )

    df = pd.read_parquet(parquet_path)
    if "valid_track" in df.columns:
        df = df[df["valid_track"]].copy()
    if species is not None and "species" in df.columns:
        df = df[df["species"] == species].copy()

    train_df = df[df["track_id"].isin(train_ids)].copy()
    val_df = df[df["track_id"].isin(val_ids)].copy()
    test_df = df[df["track_id"].isin(test_ids)].copy()
    ds_min_obs = min_observations
    if train_time_frac < 1.0:
        train_df = clip_tracks_to_time_frac(train_df, train_ids, train_time_frac)
        val_df = clip_tracks_to_time_frac(val_df, val_ids, train_time_frac)
        ds_min_obs = min(min_observations, 8)
        # Test stays full-length; prefix-trained models are scored on the
        # tail separately by eval/evaluate_extrap.py.
    transformer = GaussianStateTransformer(sigma_mode=sigma_mode)
    transformer.fit(train_df)

    train_ds = PlantTrackStateDataset(
        parquet_path,
        transformer,
        track_ids=train_ids,
        dataframe=train_df,
        valid_track_only=True,
        min_observations=ds_min_obs,
    )
    val_ds = PlantTrackStateDataset(
        parquet_path,
        transformer,
        track_ids=val_ids,
        dataframe=val_df,
        valid_track_only=True,
        min_observations=ds_min_obs,
    )
    test_ds = PlantTrackStateDataset(
        parquet_path,
        transformer,
        track_ids=test_ids,
        dataframe=test_df,
        valid_track_only=True,
        min_observations=min_observations,
    )
    return transformer, train_ds, val_ds, test_ds


def make_dataloader(
    dataset: PlantTrackStateDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    collate_fn = _collate_variable_length_tracks if batch_size > 1 else _collate_single_track
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


class _DatasetAuditAdapter:
    """Tiny adapter so the existing audit_z_coverage helper can be reused."""

    def __init__(self, dataset: PlantTrackStateDataset) -> None:
        self._dataset = dataset

    def train_dataloader(self):
        class _Loader:
            def __init__(self, dataset):
                self.dataset = dataset

        return _Loader(self._dataset)


class BaselineLightning(L.LightningModule):
    """Lightning wrapper for the discrete baseline suite."""

    def __init__(
        self,
        *,
        model_type: str,
        lr: float = 5e-4,
        n_context_frames: int = 3,
        geometry_size_loss: str = "log",
        log_size_eps: float = LOG_SIZE_EPS,
        linear_size_weight: float = LINEAR_SIZE_WEIGHT,
        size_loss_weight: float = 1.0,
        phys_loss_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model_type = model_type
        self.lr = lr
        self.n_context_frames = n_context_frames
        self.geometry_size_loss = geometry_size_loss
        self.log_size_eps = log_size_eps
        self.linear_size_weight = linear_size_weight
        self.size_loss_weight = size_loss_weight
        self.phys_loss_weight = phys_loss_weight

        self.model = build_baseline_model(
            model_type=model_type,
            n_context_frames=n_context_frames,
        )

    def _coerce_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states_5d = batch["states_5d"]
        t_absolute = batch["t_absolute"]
        color_mask = batch["color_mask"]
        track_lengths = batch.get("track_lengths")

        if states_5d.dim() == 2:
            states_5d = states_5d.unsqueeze(0)
            t_absolute = t_absolute.unsqueeze(0)
            color_mask = color_mask.unsqueeze(0)
            track_lengths = torch.tensor(
                [states_5d.shape[1]],
                dtype=torch.long,
                device=states_5d.device,
            )
        elif track_lengths is None:
            track_lengths = torch.full(
                (states_5d.shape[0],),
                states_5d.shape[1],
                dtype=torch.long,
                device=states_5d.device,
            )

        return states_5d, t_absolute, color_mask, track_lengths

    def _full_prediction_5d(self, pred_future: torch.Tensor, target_future: torch.Tensor) -> torch.Tensor:
        pred_full = torch.zeros_like(target_future)
        pred_full[..., TARGET_SLICE] = pred_future
        return pred_full

    def _step(
        self,
        batch: dict,
        prefix: str,
        *,
        teacher_forcing: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
        states_5d, t_absolute, color_mask, track_lengths = self._coerce_batch(batch)
        batch_size = states_5d.shape[0]

        if teacher_forcing:
            pred_future = self.model.teacher_forcing_rollout(
                states_5d, t_absolute, track_lengths
            )
        else:
            pred_future = self.model.autoregressive_rollout(
                states_5d, t_absolute, track_lengths
            )

        k = self.n_context_frames
        target_future = states_5d[:, k:, :]
        color_future = color_mask[:, k:]
        future_steps = target_future.shape[1]
        step_index = torch.arange(future_steps, device=states_5d.device).unsqueeze(0)
        pad_mask = step_index < (track_lengths - k).clamp(min=0).unsqueeze(1)

        pred_full = self._full_prediction_5d(pred_future, target_future)
        loss, geo_loss, phys_loss, geo_diag = hybrid_loss(
            pred_full,
            target_future,
            color_future,
            size_loss=self.geometry_size_loss,
            log_eps=self.log_size_eps,
            linear_size_weight=self.linear_size_weight,
            xy_loss_weight=0.0,
            size_loss_weight=self.size_loss_weight,
            phys_loss_weight=self.phys_loss_weight,
            pad_mask=pad_mask,
        )

        logs: dict[str, torch.Tensor] = {
            f"{prefix}_loss": loss,
            f"{prefix}_geo_loss": geo_loss,
            f"{prefix}_phys_loss": phys_loss,
        }
        for key, value in geo_diag.items():
            logs[f"{prefix}_{key}"] = value
        logs.update(
            per_channel_mse_logs(
                pred_full,
                target_future,
                color_future,
                prefix,
                pad_mask=pad_mask,
            )
        )

        if batch_size > 1 and future_steps > 0:
            with torch.no_grad():
                size_err = (pred_future[..., :2] - target_future[..., 2:4]).pow(2)
                denom = pad_mask.float().sum(dim=1).clamp(min=1.0)
                track_mse = (size_err * pad_mask.unsqueeze(-1)).sum(dim=(1, 2)) / (denom * 2.0)
                logs[f"{prefix}_track_mse_mean"] = track_mse.mean()
                logs[f"{prefix}_track_mse_std"] = track_mse.std()
                logs[f"{prefix}_track_mse_max"] = track_mse.max()

        return loss, logs, batch_size

    def training_step(self, batch, batch_idx):
        loss, logs, batch_size = self._step(batch, "train", teacher_forcing=True)
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs, batch_size = self._step(batch, "val", teacher_forcing=False)
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs, batch_size = self._step(batch, "test", teacher_forcing=False)
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the discrete baseline suite")
    parser.add_argument("--model_type", choices=("lstm", "gru", "transformer"), required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", default="latent-ode-baselines")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--success-geo-max", type=float, default=GEO_LOSS_SUCCESS_THRESHOLD)

    parser.add_argument("--max-tracks", type=int, default=DEFAULT_MAX_TRACKS)
    parser.add_argument("--species", type=str, default=None)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--sigma-mode", choices=("raw", "z_score"), default="z_score")
    parser.add_argument("--n-context-frames", type=int, default=3)

    parser.add_argument("--geometry-size-loss", choices=("linear", "log"), default="log")
    parser.add_argument("--log-size-eps", type=float, default=LOG_SIZE_EPS)
    parser.add_argument("--linear-size-weight", type=float, default=LINEAR_SIZE_WEIGHT)
    parser.add_argument("--size-loss-weight", type=float, default=1.0)
    parser.add_argument("--phys-loss-weight", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--train-time-frac",
        type=float,
        default=1.0,
        help="If <1, keep only the first fraction of each TRAIN/VAL track timeline.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    git = assert_clean_or_allowed(repo, args.allow_dirty)
    L.seed_everything(args.seed, workers=True)

    parquet_path = args.parquet.expanduser()
    if not parquet_path.is_absolute():
        parquet_path = Path(__file__).parent.parent / parquet_path
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Parquet not found: {args.parquet!r}\n  Resolved -> {parquet_path.resolve()}"
        )

    if args.name is None:
        tag = f"{args.model_type}-{args.max_tracks}t"
        if args.species:
            tag += f"-{args.species.lower()}"
        args.name = f"baseline-{tag}-s{args.seed}"
        if args.train_time_frac < 1.0:
            args.name += f"-extrap{int(round(args.train_time_frac * 100))}"

    transformer, train_ds, val_ds, test_ds = build_baseline_datasets(
        parquet_path,
        sigma_mode=args.sigma_mode,
        max_tracks=args.max_tracks,
        species=args.species,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        train_time_frac=args.train_time_frac,
    )
    audit_z_coverage(
        _DatasetAuditAdapter(train_ds),
        label=f"{args.model_type} max_tracks={args.max_tracks}",
        n_tracks=len(train_ds),
    )

    train_batch = min(args.batch_size, len(train_ds))
    val_batch = max(1, len(val_ds))
    test_batch = max(1, len(test_ds))
    train_loader = make_dataloader(
        train_ds,
        batch_size=train_batch,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=val_batch,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = make_dataloader(
        test_ds,
        batch_size=test_batch,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(
        f"\n[train_baselines] Config"
        f"\n  model={args.model_type!r}  seed={args.seed}  epochs={args.epochs}  lr={args.lr}"
        f"\n  sigma_mode={args.sigma_mode!r}  size_loss={args.geometry_size_loss!r}"
        f"\n  weights: size={args.size_loss_weight}  phys={args.phys_loss_weight}  xy=0.0"
        f"\n  train_batch={train_batch}  val_batch={val_batch}  test_batch={test_batch}"
        f"\n  context_frames={args.n_context_frames}"
        f"\n  train_time_frac={args.train_time_frac}"
        f"\n  git_hash={git['git_hash']} dirty={git['git_dirty']}"
    )

    model = BaselineLightning(
        model_type=args.model_type,
        lr=args.lr,
        n_context_frames=args.n_context_frames,
        geometry_size_loss=args.geometry_size_loss,
        log_size_eps=args.log_size_eps,
        linear_size_weight=args.linear_size_weight,
        size_loss_weight=args.size_loss_weight,
        phys_loss_weight=args.phys_loss_weight,
    )

    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb:
        logger = WandbLogger(project=args.project, name=args.name)
        try:
            logger.experiment.config.update(
                {
                    "git_hash": git["git_hash"],
                    "git_dirty": git["git_dirty"],
                    "seed": args.seed,
                    "model_type": args.model_type,
                    "train_time_frac": args.train_time_frac,
                },
                allow_val_change=True,
            )
        except Exception as exc:
            print(f"[train_baselines] wandb config update skipped: {exc}")
        ckpt_cb: Callback = WandbArtifactCheckpoint(
            monitor="val_geo_loss",
            mode="min",
            artifact_name=f"{args.model_type}_baseline_best_{args.name}",
        )
    else:
        logger = True
        ckpt_cb = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=f"{args.model_type}_baseline_best_{args.name}",
            monitor="val_geo_loss",
            mode="min",
            save_top_k=1,
        )
    best_val_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"{args.model_type}_valmse_{args.name}",
        monitor="val_track_mse_mean",
        mode="min",
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[ckpt_cb, best_val_cb],
        enable_progress_bar=True,
        accelerator=args.accelerator,
        devices=1,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if args.train_time_frac >= 1.0:
        test_ckpt = "best" if not args.wandb else None
        trainer.test(model, dataloaders=test_loader, ckpt_path=test_ckpt)
    else:
        print("[train_baselines] skip trainer.test — prefix-train; use eval/evaluate_extrap.py")

    final_geo = trainer.callback_metrics.get("val_geo_loss")
    final_geo = float("inf") if final_geo is None else float(final_geo)
    status = "PASS" if final_geo < args.success_geo_max else "WARN"
    print(
        f"\n[train_baselines] {status} Final val_geo_loss = {final_geo:.4g} "
        f"(threshold {args.success_geo_max})"
    )

    for cb in trainer.callbacks:
        if hasattr(cb, "best_model_path") and cb.best_model_path:
            print(f"[train_baselines] Best checkpoint -> {cb.best_model_path}")
            break

    if args.wandb:
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
