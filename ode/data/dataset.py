"""
PlantTrackStateDataset: PyTorch Dataset that yields per-track 7D state sequences
and timestamps (and color_mask) for torchdiffeq.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import GaussianStateTransformer


class PlantTrackStateDataset(Dataset):
    """
    Dataset that returns one track per sample: (states, timestamps, color_mask).
    States shape (T, 7), timestamps (T,), color_mask (T,) bool.
    """

    def __init__(
        self,
        parquet_path: Path | str,
        transformer: GaussianStateTransformer,
        track_ids: Optional[list[int]] = None,
        valid_track_only: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        parquet_path : path to metrics Parquet
        transformer : fitted GaussianStateTransformer
        track_ids : if set, only include these track IDs (for train/val/test split)
        valid_track_only : if True, keep only rows with valid_track==True
        """
        self.parquet_path = Path(parquet_path)
        self.transformer = transformer
        self.valid_track_only = valid_track_only

        df = pd.read_parquet(self.parquet_path)
        if valid_track_only and "valid_track" in df.columns:
            df = df[df["valid_track"]].copy()
        if track_ids is not None:
            df = df[df["track_id"].isin(track_ids)].copy()

        # Require columns for 7D state
        required = [
            "track_id", "tray_id", "timestamp", "centroid_x", "centroid_y",
            "width", "height", "time_since_germination_hours", "edge_density",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet missing required columns: {missing}")

        # Drop rows with NaN in required fields (per track we could interpolate; for now drop)
        df = df.dropna(subset=["timestamp", "centroid_x", "centroid_y", "width", "height", "time_since_germination_hours"])
        df["edge_density"] = df["edge_density"].fillna(0.0)

        self._track_ids = df["track_id"].unique().tolist()
        self._df = df

    def __len__(self) -> int:
        return len(self._track_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        track_id = self._track_ids[idx]
        track_df = self._df[self._df["track_id"] == track_id].copy()
        if len(track_df) < 2:
            raise ValueError(f"Track {track_id} has fewer than 2 observations after filtering")

        states, timestamps, color_mask = self.transformer.transform_track(track_df)

        return {
            "states": torch.from_numpy(states).float(),
            "timestamps": torch.from_numpy(timestamps).float(),
            "color_mask": torch.from_numpy(color_mask),
            "track_id": torch.tensor(track_id, dtype=torch.long),
        }
