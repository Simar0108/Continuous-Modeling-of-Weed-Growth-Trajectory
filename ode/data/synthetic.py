"""
Synthetic plant track data for validating the GaussianStateTransformer.

Generates DataFrames with Parquet-like columns so the transformer and
dataset can be tested without real data. Used to verify that e.g. linear
growth in width/height produces monotonically increasing σ_w, σ_h in the 7D state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .transforms import GaussianStateTransformer


def generate_linear_growth_tracks(
    n_tracks: int = 3,
    n_timestamps: int = 100,
    tray_size: tuple[float, float] = (1000.0, 1000.0),
    width_range: tuple[float, float] = (20.0, 200.0),
    height_range: tuple[float, float] = (20.0, 200.0),
    edge_density_range: Optional[tuple[float, float]] = None,
    with_color: bool = True,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic tracks where width and height grow linearly over time.

    This is the main preset for validating the transformer: after transform,
    σ_w and σ_h should increase monotonically (or steadily) over the sequence.

    Parameters
    ----------
    n_tracks : int
        Number of tracks to generate.
    n_timestamps : int
        Number of time steps per track (e.g. 100).
    tray_size : tuple
        (width, height) of the tray in pixels; used for centroid placement and extent.
    width_range : tuple
        (min, max) width over time; linear interpolation from first to last step.
    height_range : tuple
        (min, max) height over time; linear interpolation from first to last step.
    edge_density_range : tuple or None
        If set, (min, max) for edge_density (linear over time). If None, use constant 0.5.
    with_color : bool
        If True, set greenness to a constant (0.4); if False, greenness is NaN.
    seed : int or None
        Random seed for centroid jitter (optional).

    Returns
    -------
    pd.DataFrame
        One row per (track_id, timestamp) with columns aligned to the metrics Parquet:
        track_id, tray_id, timestamp, centroid_x, centroid_y, width, height,
        time_since_germination_hours, edge_density, greenness, valid_track,
        xmin, ymin, xmax, ymax, area.
    """
    rng = np.random.default_rng(seed)
    tray_w, tray_h = tray_size
    records = []

    for track_idx in range(n_tracks):
        track_id = 9000 + track_idx
        tray_id = 100 + (track_idx % 2)

        # Centroid fixed near center of tray with small jitter
        cx = tray_w / 2 + (rng.uniform(-50, 50) if seed is not None else 0.0)
        cy = tray_h / 2 + (rng.uniform(-50, 50) if seed is not None else 0.0)

        for t in range(n_timestamps):
            frac = t / max(1, n_timestamps - 1)

            width = width_range[0] + (width_range[1] - width_range[0]) * frac
            height = height_range[0] + (height_range[1] - height_range[0]) * frac

            xmin = cx - width / 2
            ymin = cy - height / 2
            xmax = cx + width / 2
            ymax = cy + height / 2

            time_since_germ_hours = frac * 100.0

            if edge_density_range is not None:
                edge_density = edge_density_range[0] + (edge_density_range[1] - edge_density_range[0]) * frac
            else:
                edge_density = 0.5

            greenness = 0.4 if with_color else np.nan

            # Timestamp: simple numeric or datetime; use datetime for compatibility
            ts = pd.Timestamp("2021-01-01") + pd.Timedelta(hours=time_since_germ_hours)

            records.append({
                "track_id": track_id,
                "tray_id": tray_id,
                "timestamp": ts,
                "centroid_x": cx,
                "centroid_y": cy,
                "width": width,
                "height": height,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "area": width * height,
                "time_since_germination_hours": time_since_germ_hours,
                "edge_density": edge_density,
                "greenness": greenness,
                "valid_track": True,
            })

    return pd.DataFrame(records)


def validate_transformer_on_synthetic(
    transformer: GaussianStateTransformer,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """
    Run the transformer on synthetic linear-growth data and assert that
    σ_w and σ_h increase monotonically over each track.

    Returns True if assertions pass (data foundation is solid).
    """
    if df is None:
        df = generate_linear_growth_tracks(
            n_tracks=2,
            n_timestamps=100,
            width_range=(30.0, 180.0),
            height_range=(30.0, 180.0),
            with_color=True,
            seed=42,
        )
    if not transformer.fitted:
        transformer.fit(df)

    for track_id, group in df.groupby("track_id"):
        states, timestamps, color_mask = transformer.transform_track(group)
        sigma_w = states[:, 2]
        sigma_h = states[:, 3]
        # Allow small numerical noise: require strictly increasing in sense of no backward step
        dw = np.diff(sigma_w)
        dh = np.diff(sigma_h)
        if not (np.all(dw >= -1e-6) and np.all(dh >= -1e-6)):
            raise AssertionError(
                f"Track {track_id}: sigma_w or sigma_h did not increase over time. "
                f"dw min={dw.min():.6f}, dh min={dh.min():.6f}"
            )
    return True
