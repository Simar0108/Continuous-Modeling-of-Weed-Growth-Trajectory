"""
GaussianStateTransformer: map Parquet metrics rows to 7D Gaussian latent state.

State: [x, y, σ_w, σ_h, m, Z, τ]
- x, y: tray-normalized centroids [0, 1]
- σ_w, σ_h: width/4, height/4 then Z-scored
- m: morphology from edge_density only (blob → low m, rosette → high m); Z-scored
- Z: greenness * edge_density when available (Z-scored); else 0.0 and color_mask=False
- τ: time since germination, per-track normalized [0, 1]

Statistics (mean/std and per-tray extent) are computed in fit() from the provided
DataFrame only (e.g. train set only) to avoid look-ahead bias.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# Minimum std to avoid division by zero when Z-scoring
_EPS_STD = 1e-8


class GaussianStateTransformer:
    """
    Transforms per-observation metrics into 7D state vectors with reproducible
    scaling. Fit on training data only; transform uses stored statistics.
    """

    def __init__(self) -> None:
        self._tray_extent: dict[int, tuple[float, float]] = {}  # tray_id -> (max_x, max_y)
        self._mean_sigma_w: float = 0.0
        self._std_sigma_w: float = 1.0
        self._mean_sigma_h: float = 0.0
        self._std_sigma_h: float = 1.0
        self._mean_m: float = 0.0
        self._std_m: float = 1.0
        self._mean_Z: float = 0.0
        self._std_Z: float = 1.0
        self._fitted: bool = False

    def fit(self, df: pd.DataFrame) -> GaussianStateTransformer:
        """
        Compute and store scaling statistics from the given DataFrame only.
        Use only training data to avoid look-ahead bias.

        Parameters
        ----------
        df : pd.DataFrame
            Per-observation metrics with columns: tray_id, centroid_x, centroid_y,
            width, height, time_since_germination_hours, edge_density, greenness
            (optional). For tray extent, xmax/ymax are used if present; else
            derived from centroid and width/height.
        """
        df = df.copy()

        # Ensure xmax, ymax exist for tray extent (derive from centroid + width/height if missing)
        if "xmax" not in df.columns:
            df["xmax"] = df["centroid_x"] + df["width"] / 2.0
        if "ymax" not in df.columns:
            df["ymax"] = df["centroid_y"] + df["height"] / 2.0

        # Per-tray max extent
        tray_agg = df.groupby("tray_id").agg(
            max_x=("xmax", "max"),
            max_y=("ymax", "max"),
        ).reset_index()
        self._tray_extent = {
            int(row["tray_id"]): (float(row["max_x"]), float(row["max_y"]))
            for _, row in tray_agg.iterrows()
        }

        # Derived features for Z-scoring
        sigma_w = (df["width"] / 4.0).replace([np.inf, -np.inf], np.nan).dropna()
        sigma_h = (df["height"] / 4.0).replace([np.inf, -np.inf], np.nan).dropna()
        m_raw = df["edge_density"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        m_vals = m_raw.dropna()

        has_greenness = "greenness" in df.columns
        if has_greenness:
            z_anchor = (df["greenness"] * df["edge_density"].fillna(0.0)).replace(
                [np.inf, -np.inf], np.nan
            )
            z_vals = z_anchor.dropna()
        else:
            z_vals = pd.Series(dtype=float)

        self._mean_sigma_w = float(sigma_w.mean())
        self._std_sigma_w = float(sigma_w.std()) or _EPS_STD
        self._mean_sigma_h = float(sigma_h.mean())
        self._std_sigma_h = float(sigma_h.std()) or _EPS_STD
        self._mean_m = float(m_vals.mean()) if len(m_vals) else 0.0
        self._std_m = float(m_vals.std()) or _EPS_STD
        self._mean_Z = float(z_vals.mean()) if len(z_vals) else 0.0
        self._std_Z = float(z_vals.std()) or _EPS_STD

        self._fitted = True
        return self

    def transform_track(
        self,
        track_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform one track's rows into 7D state sequence, timestamps, and color_mask.

        Parameters
        ----------
        track_df : pd.DataFrame
            Rows for a single track, sorted by timestamp. Must contain tray_id,
            centroid_x, centroid_y, width, height, time_since_germination_hours,
            edge_density; optional greenness, xmax, ymax.

        Returns
        -------
        states : np.ndarray
            Shape (T, 7), dtype float. Order: x, y, σ_w, σ_h, m, Z, τ.
        timestamps : np.ndarray
            Shape (T,) — numeric (e.g. seconds or hours) for torchdiffeq.
        color_mask : np.ndarray
            Shape (T,) bool — True where Z is anchored by physiology (greenness * edge_density).
        """
        if not self._fitted:
            raise RuntimeError("GaussianStateTransformer must be fitted before transform_track")

        track_df = track_df.sort_values("timestamp").copy()
        if "xmax" not in track_df.columns:
            track_df["xmax"] = track_df["centroid_x"] + track_df["width"] / 2.0
        if "ymax" not in track_df.columns:
            track_df["ymax"] = track_df["centroid_y"] + track_df["height"] / 2.0

        tray_id = int(track_df["tray_id"].iloc[0])
        max_x, max_y = self._tray_extent.get(tray_id, (1.0, 1.0))
        if max_x <= 0:
            max_x = 1.0
        if max_y <= 0:
            max_y = 1.0

        x = (track_df["centroid_x"] / max_x).clip(0.0, 1.0).to_numpy(dtype=np.float64)
        y = (track_df["centroid_y"] / max_y).clip(0.0, 1.0).to_numpy(dtype=np.float64)

        sigma_w_raw = (track_df["width"] / 4.0).to_numpy(dtype=np.float64)
        sigma_h_raw = (track_df["height"] / 4.0).to_numpy(dtype=np.float64)
        sigma_w = (sigma_w_raw - self._mean_sigma_w) / self._std_sigma_w
        sigma_h = (sigma_h_raw - self._mean_sigma_h) / self._std_sigma_h

        # Morphology m: edge_density only (blob → low m, rosette → high m)
        m_raw = track_df["edge_density"].fillna(0.0).to_numpy(dtype=np.float64)
        m = (m_raw - self._mean_m) / self._std_m

        # Z: anchor = greenness * edge_density where valid; else 0.0, color_mask=False
        has_greenness = "greenness" in track_df.columns
        if has_greenness:
            g = track_df["greenness"].to_numpy(dtype=np.float64)
            ed = track_df["edge_density"].fillna(0.0).to_numpy(dtype=np.float64)
            valid = np.isfinite(g) & ~np.isnan(g)
            z_anchor = np.where(valid, g * ed, np.nan)
            z_scaled = np.full_like(z_anchor, 0.0, dtype=np.float64)
            np.place(z_scaled, valid, (z_anchor[valid] - self._mean_Z) / self._std_Z)
            color_mask = valid.copy()
        else:
            z_scaled = np.zeros(len(track_df), dtype=np.float64)
            color_mask = np.zeros(len(track_df), dtype=bool)

        # τ: per-track normalized [0, 1]
        tau_h = track_df["time_since_germination_hours"].fillna(0.0).to_numpy(dtype=np.float64)
        tau_max = tau_h.max()
        tau = (tau_h / tau_max) if tau_max > 0 else np.zeros_like(tau_h)

        states = np.column_stack([x, y, sigma_w, sigma_h, m, z_scaled, tau])

        # Timestamps: use numeric (e.g. hours since first observation) for ODE solvers
        ts = track_df["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(ts):
            t0 = ts.iloc[0]
            timestamps = (ts - t0).dt.total_seconds().to_numpy(dtype=np.float64) / 3600.0
        else:
            timestamps = np.arange(len(track_df), dtype=np.float64)

        return states, timestamps, color_mask

    def save_stats(self, path: Path | str) -> None:
        """Save fitted statistics to JSON for reproducibility."""
        path = Path(path)
        obj = {
            "tray_extent": {str(k): list(v) for k, v in self._tray_extent.items()},
            "mean_sigma_w": self._mean_sigma_w,
            "std_sigma_w": self._std_sigma_w,
            "mean_sigma_h": self._mean_sigma_h,
            "std_sigma_h": self._std_sigma_h,
            "mean_m": self._mean_m,
            "std_m": self._std_m,
            "mean_Z": self._mean_Z,
            "std_Z": self._std_Z,
            "fitted": self._fitted,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    def load_stats(self, path: Path | str) -> GaussianStateTransformer:
        """Load statistics from JSON and set fitted state."""
        path = Path(path)
        with open(path) as f:
            obj = json.load(f)
        self._tray_extent = {int(k): tuple(v) for k, v in obj["tray_extent"].items()}
        self._mean_sigma_w = obj["mean_sigma_w"]
        self._std_sigma_w = obj["std_sigma_w"]
        self._mean_sigma_h = obj["mean_sigma_h"]
        self._std_sigma_h = obj["std_sigma_h"]
        self._mean_m = obj["mean_m"]
        self._std_m = obj["std_m"]
        self._mean_Z = obj["mean_Z"]
        self._std_Z = obj["std_Z"]
        self._fitted = obj["fitted"]
        return self

    @property
    def fitted(self) -> bool:
        return self._fitted
