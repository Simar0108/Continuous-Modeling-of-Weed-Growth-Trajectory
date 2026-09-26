"""
PlantTrackStateDataset: PyTorch Dataset that yields per-track 5D state sequences,
absolute timestamps, and color_mask for the Neural ODE model.

Batch keys:
  states_5d  — (T, 5) float  [x, y, σ_w, σ_h, Z]
  t_absolute — (T,)   float  hours since germination
  color_mask — (T,)   bool   True where Z is physiologically anchored
  track_id   — scalar long
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
        dataframe: Optional[pd.DataFrame] = None,
        valid_track_only: bool = True,
        min_observations: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        parquet_path      : path to metrics Parquet
        transformer       : fitted GaussianStateTransformer
        track_ids         : if set, only include these track IDs
        valid_track_only  : if True, keep only rows with valid_track==True
        min_observations  : drop tracks with fewer than this many rows
        """
        self.parquet_path = Path(parquet_path)
        self.transformer = transformer
        self.valid_track_only = valid_track_only
        self.min_observations = min_observations

        if dataframe is not None:
            df = dataframe.copy()
        else:
            df = pd.read_parquet(self.parquet_path)
        if valid_track_only and "valid_track" in df.columns:
            df = df[df["valid_track"]].copy()
        if track_ids is not None:
            df = df[df["track_id"].isin(track_ids)].copy()

        required = [
            "track_id", "tray_id", "timestamp", "centroid_x", "centroid_y",
            "width", "height", "time_since_germination_hours", "edge_density",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet missing required columns: {missing}")

        df = df.dropna(subset=[
            "timestamp", "centroid_x", "centroid_y",
            "width", "height", "time_since_germination_hours",
        ])
        df["edge_density"] = df["edge_density"].fillna(0.0)

        # Filter tracks that don't meet the minimum observation threshold
        counts = df.groupby("track_id").size()
        valid_ids = counts[counts >= min_observations].index
        df = df[df["track_id"].isin(valid_ids)].copy()

        self._track_ids = df["track_id"].unique().tolist()
        self._df = df

    def __len__(self) -> int:
        return len(self._track_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        track_id = self._track_ids[idx]
        track_df = self._df[self._df["track_id"] == track_id].copy()
        if len(track_df) < self.min_observations:
            raise ValueError(
                f"Track {track_id} has only {len(track_df)} observations "
                f"(min_observations={self.min_observations})"
            )

        states_5d, t_absolute, color_mask = self.transformer.transform_track_5d(track_df)

        return {
            "states_5d":  torch.from_numpy(states_5d).float(),   # (T, 5)
            "t_absolute": torch.from_numpy(t_absolute).float(),  # (T,)
            "color_mask": torch.from_numpy(color_mask),          # (T,) bool
            "track_id":   torch.tensor(track_id, dtype=torch.long),
        }
