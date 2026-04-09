"""
PlantTrackDataModule: Lightning-style data module.

Splits by tray_id stratified by coverage hours; fits GaussianStateTransformer
on train set only; provides train/val/test dataloaders with batch_size=1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import PlantTrackStateDataset
from .transforms import GaussianStateTransformer


def _stratify_trays_by_coverage(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split tray_id into train/val/test so that coverage hours are stratified.
    Coverage per tray = max(time_since_germination_hours) in that tray.
    """
    if "time_since_germination_hours" not in df.columns:
        tray_coverage = df.groupby("tray_id").size()
    else:
        tray_coverage = df.groupby("tray_id")["time_since_germination_hours"].max()
    tray_ids = tray_coverage.index.tolist()
    if not tray_ids:
        return [], [], []

    # Sort by coverage and assign to bins so distribution is similar across splits
    sorted_trays = tray_coverage.sort_values().index.tolist()
    n = len(sorted_trays)
    n_train = max(1, int(n * train_frac))
    n_val = max(0, int(n * val_frac))
    n_test = max(0, n - n_train - n_val)

    # Interleave so low/medium/high coverage trays go to each split
    train_ids, val_ids, test_ids = [], [], []
    for i, tid in enumerate(sorted_trays):
        if i % 3 == 0 and len(train_ids) < n_train:
            train_ids.append(int(tid))
        elif i % 3 == 1 and len(val_ids) < n_val:
            val_ids.append(int(tid))
        elif len(test_ids) < n_test:
            test_ids.append(int(tid))
        elif len(train_ids) < n_train:
            train_ids.append(int(tid))
        elif len(val_ids) < n_val:
            val_ids.append(int(tid))
        else:
            test_ids.append(int(tid))

    return train_ids, val_ids, test_ids


class PlantTrackDataModule:
    """
    Lightning-style DataModule: load Parquet, split by tray (stratified),
    fit transformer on train only, expose train/val/test dataloaders.
    """

    def __init__(
        self,
        parquet_path: Path | str,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._transformer: Optional[GaussianStateTransformer] = None
        self._train_track_ids: Optional[list[int]] = None
        self._val_track_ids: Optional[list[int]] = None
        self._test_track_ids: Optional[list[int]] = None
        self._train_ds: Optional[PlantTrackStateDataset] = None
        self._val_ds: Optional[PlantTrackStateDataset] = None
        self._test_ds: Optional[PlantTrackStateDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        df = pd.read_parquet(self.parquet_path)
        if "valid_track" in df.columns:
            df = df[df["valid_track"]].copy()

        train_tray_ids, val_tray_ids, test_tray_ids = _stratify_trays_by_coverage(
            df, train_frac=self.train_frac, val_frac=self.val_frac
        )

        train_track_ids = df[df["tray_id"].isin(train_tray_ids)]["track_id"].unique().tolist()
        val_track_ids = df[df["tray_id"].isin(val_tray_ids)]["track_id"].unique().tolist()
        test_track_ids = df[df["tray_id"].isin(test_tray_ids)]["track_id"].unique().tolist()

        train_df = df[df["track_id"].isin(train_track_ids)]

        transformer = GaussianStateTransformer()
        transformer.fit(train_df)

        self._transformer = transformer
        self._train_track_ids = train_track_ids
        self._val_track_ids = val_track_ids
        self._test_track_ids = test_track_ids

        self._train_ds = PlantTrackStateDataset(
            self.parquet_path,
            transformer,
            track_ids=train_track_ids,
            valid_track_only=True,
        )
        self._val_ds = PlantTrackStateDataset(
            self.parquet_path,
            transformer,
            track_ids=val_track_ids,
            valid_track_only=True,
        )
        self._test_ds = PlantTrackStateDataset(
            self.parquet_path,
            transformer,
            track_ids=test_track_ids,
            valid_track_only=True,
        )

    @property
    def transformer(self) -> GaussianStateTransformer:
        if self._transformer is None:
            raise RuntimeError("Call setup() before accessing transformer")
        return self._transformer

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            raise RuntimeError("Call setup() before requesting dataloaders")
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_collate_single_track,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_ds is None:
            raise RuntimeError("Call setup() before requesting dataloaders")
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_single_track,
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_ds is None:
            raise RuntimeError("Call setup() before requesting dataloaders")
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_single_track,
        )


def _collate_single_track(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate a batch of one track each (batch_size=1); just take the single item."""
    assert len(batch) == 1
    return batch[0]
