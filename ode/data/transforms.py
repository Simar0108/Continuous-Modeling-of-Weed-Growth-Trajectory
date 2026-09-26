"""
GaussianStateTransformer: map Parquet metrics rows to state vectors.

5D state (transform_track_5d — used by model):  [x, y, σ_w, σ_h, Z]
  x, y   — tray-normalized centroids [0, 1]
  σ_w/σ_h — width/4, height/4; raw pixel units OR Z-scored depending on sigma_mode
  Z      — greenness * edge_density (Z-scored); 0.0 and color_mask=False when unavailable

7D state (transform_track — kept for compatibility):  [x, y, σ_w, σ_h, m, Z, τ]
  additionally includes m (morphology, Z-scored) and τ (per-track time [0,1])

sigma_mode='raw'     → σ_w = width/4  (pixel units; original behaviour)
sigma_mode='z_score' → σ_w = (width/4 - μ) / σ  (all dims ≈ O(1); recommended for run6+)

Statistics are fit on train data only to avoid look-ahead bias.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# Minimum std to avoid division by zero when Z-scoring
_EPS_STD = 1e-8


# ── Monotonicity audit ────────────────────────────────────────────────────────

def audit_monotonicity_violations(
    df: pd.DataFrame,
    *,
    track_col:  str = "track_id",
    time_col:   str = "time_since_germination_hours",
    width_col:  str = "width",
    height_col: str = "height",
    verbose:    bool = True,
) -> dict:
    """
    Audit how often σ_w / σ_h *decrease* between consecutive observations.

    Shrinkage events arise from two sources:
      1. Genuine measurement noise / segmentation error — harmless, should NOT
         be penalised (they average out over the trajectory).
      2. True biological "shrinkage" — leaf wilt, occlusion.  Even real plants
         occasionally shrink; a monotonicity penalty should therefore be soft.

    Parameters
    ----------
    df         : raw Parquet DataFrame (before any Z-scoring)
    track_col  : column identifying individual plant tracks
    time_col   : column with absolute time (hours)
    width_col  : pixel width column
    height_col : pixel height column
    verbose    : print a summary table

    Returns
    -------
    dict with keys:
      violation_rate_w     : fraction of consecutive pairs where width shrinks
      violation_rate_h     : fraction of consecutive pairs where height shrinks
      tracks_with_any      : number of tracks that contain ≥1 shrinkage event
      total_tracks         : total tracks examined
      recommended_mono_w   : suggested monotonicity_weight (float)
      per_track            : DataFrame with per-track violation rates

    Recommended mono weight heuristic
    -----------------------------------
    violation_rate > 0.20  →  0.0    (data is too noisy; penalty causes collapse)
    0.10 < rate ≤ 0.20     →  0.005  (very soft — only extreme shrinkage penalised)
    0.05 < rate ≤ 0.10     →  0.01
    rate ≤ 0.05            →  0.05   (data is fairly monotonic; enforce it)
    """
    rows = []
    all_pairs_w = 0
    all_viol_w  = 0
    all_pairs_h = 0
    all_viol_h  = 0

    for track_id, grp in df.groupby(track_col, sort=False):
        grp = grp.sort_values(time_col)
        w   = grp[width_col].to_numpy(dtype=float)
        h   = grp[height_col].to_numpy(dtype=float)

        if len(w) < 2:
            continue

        diff_w = np.diff(w)
        diff_h = np.diff(h)
        nv_w   = int((diff_w < 0).sum())
        nv_h   = int((diff_h < 0).sum())
        n      = len(w) - 1

        all_pairs_w += n
        all_viol_w  += nv_w
        all_pairs_h += n
        all_viol_h  += nv_h

        rows.append({
            "track_id":    track_id,
            "n_steps":     n,
            "viol_w":      nv_w,
            "viol_h":      nv_h,
            "rate_w":      nv_w / n,
            "rate_h":      nv_h / n,
            "has_any":     (nv_w + nv_h) > 0,
        })

    per_track = pd.DataFrame(rows)
    if per_track.empty:
        return {
            "violation_rate_w": 0.0, "violation_rate_h": 0.0,
            "tracks_with_any": 0, "total_tracks": 0,
            "recommended_mono_w": 0.0, "per_track": per_track,
        }

    rate_w = all_viol_w  / max(all_pairs_w, 1)
    rate_h = all_viol_h  / max(all_pairs_h, 1)
    worst  = max(rate_w, rate_h)

    if worst > 0.20:
        rec_w = 0.0
    elif worst > 0.10:
        rec_w = 0.005
    elif worst > 0.05:
        rec_w = 0.01
    else:
        rec_w = 0.05

    tracks_with_any = int(per_track["has_any"].sum())
    total_tracks    = len(per_track)

    if verbose:
        print("=" * 60)
        print("  Monotonicity Audit")
        print("=" * 60)
        print(f"  Tracks examined        : {total_tracks}")
        print(f"  Tracks with shrinkage  : {tracks_with_any}  "
              f"({100*tracks_with_any/max(total_tracks,1):.1f}%)")
        print(f"  Overall shrink rate σ_w: {rate_w:.3f}  "
              f"({all_viol_w}/{all_pairs_w} pairs)")
        print(f"  Overall shrink rate σ_h: {rate_h:.3f}  "
              f"({all_viol_h}/{all_pairs_h} pairs)")
        print(f"\n  Recommended monotonicity_weight: {rec_w}")
        if rec_w == 0.0:
            print("  ⚠  High noise — disable monotonicity penalty (weight=0.0)")
        elif rec_w <= 0.005:
            print("  ⚠  Moderate noise — very soft penalty only")
        print("=" * 60)

    return {
        "violation_rate_w":   rate_w,
        "violation_rate_h":   rate_h,
        "tracks_with_any":    tracks_with_any,
        "total_tracks":       total_tracks,
        "recommended_mono_w": rec_w,
        "per_track":          per_track,
    }


class GaussianStateTransformer:
    """
    Transforms per-observation metrics into 7D state vectors with reproducible
    scaling. Fit on training data only; transform uses stored statistics.
    """

    def __init__(self, sigma_mode: str = "raw") -> None:
        if sigma_mode not in ("raw", "z_score"):
            raise ValueError(f"sigma_mode must be 'raw' or 'z_score', got {sigma_mode!r}")
        self.sigma_mode = sigma_mode
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

    def transform_track_5d(
        self,
        track_df:       pd.DataFrame,
        z_smooth_window: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform one track into the 5D state used by the Neural ODE model.

        The Z channel (greenness × edge_density) is smoothed with a centred
        rolling mean (window = z_smooth_window, min_periods=1) to reduce
        single-frame noise in the physiological signal.  This gives the model
        a cleaner sigmoidal target without discarding any observations.

        Parameters
        ----------
        track_df        : single-track DataFrame sorted by timestamp
        z_smooth_window : rolling mean window for Z (default 3; set 1 to disable)

        Returns
        -------
        states_5d  : np.ndarray shape (T, 5)  — [x, y, σ_w, σ_h, Z]
        t_absolute : np.ndarray shape (T,)    — hours since germination
        color_mask : np.ndarray shape (T,) bool — True where Z is physiologically anchored
        """
        if not self._fitted:
            raise RuntimeError("GaussianStateTransformer must be fitted before transform_track_5d")

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
        if self.sigma_mode == "z_score":
            sigma_w = (sigma_w_raw - self._mean_sigma_w) / self._std_sigma_w
            sigma_h = (sigma_h_raw - self._mean_sigma_h) / self._std_sigma_h
        else:
            sigma_w = sigma_w_raw
            sigma_h = sigma_h_raw

        has_greenness = "greenness" in track_df.columns
        if has_greenness:
            g  = track_df["greenness"].to_numpy(dtype=np.float64)
            ed = track_df["edge_density"].fillna(0.0).to_numpy(dtype=np.float64)
            valid    = np.isfinite(g) & ~np.isnan(g)
            z_anchor = np.where(valid, g * ed, np.nan)

            z_scaled_raw = np.full_like(z_anchor, 0.0, dtype=np.float64)
            np.place(z_scaled_raw, valid, (z_anchor[valid] - self._mean_Z) / self._std_Z)

            # Rolling mean smoothing on Z — reduces single-frame noise so the
            # ODE targets the sigmoidal trend rather than fitting measurement
            # artefacts.  NaN entries (no colour) are excluded from the window
            # but do not break the rolling calculation.
            if z_smooth_window > 1 and valid.any():
                z_series = pd.Series(
                    np.where(valid, z_scaled_raw, np.nan)
                ).rolling(window=z_smooth_window, center=True, min_periods=1).mean()
                # Only overwrite positions where we have valid data
                z_scaled = np.where(valid, z_series.to_numpy(dtype=np.float64), 0.0)
            else:
                z_scaled = z_scaled_raw

            color_mask = valid.copy()
        else:
            z_scaled   = np.zeros(len(track_df), dtype=np.float64)
            color_mask = np.zeros(len(track_df), dtype=bool)

        states_5d  = np.column_stack([x, y, sigma_w, sigma_h, z_scaled])
        t_absolute = track_df["time_since_germination_hours"].fillna(0.0).to_numpy(dtype=np.float64)

        return states_5d, t_absolute, color_mask

    def save_stats(self, path: Path | str) -> None:
        """Save fitted statistics to JSON for reproducibility."""
        path = Path(path)
        obj = {
            "sigma_mode": self.sigma_mode,
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
        self.sigma_mode = obj.get("sigma_mode", "raw")
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
