"""
Phase 3: Temporal & Growth Analysis for ACHMI.

This script reuses the Phase 1 foundations (`explore_achmi.py`) to derive per-track
growth time series and summary statistics that will seed subsequent modeling
work (e.g., Neural ODE fitting).
"""

from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color as skcolor, feature, io as skio

import explore_achmi as achmi


RANDOM_SEED = 42
TRACK_SAMPLE_COUNT = 3
SMOOTH_CONFIG = {"window": 7, "polyorder": 2}
GERMINATION_AREA_THRESH = 500.0
LARGE_GAP_MINUTES = 24 * 60.0

# Phase 3.3 configuration
ENABLE_COLOR_EXTRACTION = True
COLOR_TRACK_LIMIT = 300
COLOR_CACHE_DIR = Path("Thesis/exploration/color_cache")
EDGE_SIGMA = 1.0
SAVE_METRICS_TO_DISK = True
METRICS_OUTPUT_PATH = Path("Thesis/exploration/metrics_with_features.parquet")
# Color coverage threshold: minimum fraction of frames with valid color features
# Set to 0.0 to include all tracks, or higher (e.g., 0.5) to filter low-coverage tracks
COLOR_COVERAGE_THRESHOLD = 0.0  # 0.0 = no filtering, 0.5 = require 50% coverage


@dataclass
class TrackSummary:
    track_id: int
    tray_id: int
    n_frames: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    coverage_hours: float
    start_area: float
    end_area: float
    total_growth_pct: float
    max_growth_rate: float
    mean_gap_minutes: float


def attach_geometric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment annotations with derived geometric stats.

    Phase 3.3A enhancements:
    - width, height, area, aspect_ratio, centroid_x/y (base)
    - perimeter = 2 * (width + height)
    - area_width_ratio = area / width (leaf spread proxy)
    - centroid displacement metrics computed per-track in compute_track_metrics
    """
    df = df.copy()
    df["width"] = df["xmax"] - df["xmin"]
    df["height"] = df["ymax"] - df["ymin"]
    df["area"] = df["width"] * df["height"]
    df["aspect_ratio"] = df["width"] / df["height"].replace({0: math.nan})
    df["centroid_x"] = (df["xmin"] + df["xmax"]) / 2.0
    df["centroid_y"] = (df["ymin"] + df["ymax"]) / 2.0
    
    # Phase 3.3A: Additional geometric features
    df["perimeter"] = 2.0 * (df["width"] + df["height"])
    df["area_width_ratio"] = df["area"] / df["width"].replace({0: math.nan})
    # Elongation: meaningful shape descriptor for bounding boxes
    df["elongation"] = np.maximum(df["width"], df["height"]) / np.minimum(
        df["width"].replace({0: math.nan}), df["height"].replace({0: math.nan})
    )
    
    return df


def compute_track_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a tidy per-observation table with timestamps, gaps, growth rates, and centroid displacement.
    
    Phase 3.3A: Adds centroid displacement metrics (dx, dy, displacement, velocity, cumulative_disp).
    """
    records: List[pd.DataFrame] = []
    for track_id, track_df in df.groupby("track_id"):
        track_df = track_df.sort_values("timestamp")
        if track_df["timestamp"].isna().any():
            continue
        track_df = track_df.assign(
            dt_minutes=lambda d: d["timestamp"].diff().dt.total_seconds() / 60.0,
            area_growth=lambda d: d["area"].diff(),
            area_growth_rate=lambda d: d["area_growth"] / d["dt_minutes"].replace({0: math.nan}),
        )
        
        # Phase 3.3A: Centroid displacement metrics
        track_df = track_df.assign(
            centroid_dx=lambda d: d["centroid_x"].diff(),
            centroid_dy=lambda d: d["centroid_y"].diff(),
            centroid_displacement=lambda d: np.sqrt(
                d["centroid_dx"] ** 2 + d["centroid_dy"] ** 2
            ),
            centroid_velocity=lambda d: d["centroid_displacement"] / d["dt_minutes"].replace({0: math.nan}),
        )
        track_df["cumulative_centroid_displacement"] = track_df["centroid_displacement"].fillna(0).cumsum()
        
        records.append(track_df)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def _savgol(series: pd.Series, window: int, polyorder: int) -> pd.Series:
    """
    Apply Savitzky–Golay smoothing with guardrails for short series.
    """
    from scipy.signal import savgol_filter

    length = len(series)
    if length < 3:
        return series.astype(float)

    window = min(window, length if length % 2 == 1 else length - 1)
    if window < 3:
        window = 3 if length >= 3 else length
    if window % 2 == 0:
        window = max(3, window - 1)
    if window <= polyorder:
        polyorder = max(1, window - 1)
    try:
        smoothed = savgol_filter(series.to_numpy(dtype=float), window_length=window, polyorder=polyorder)
    except ValueError:
        return series.astype(float)
    return pd.Series(smoothed, index=series.index)


def smooth_track_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Add smoothed area/rate, relative growth, germination detection, gap flags, and noise residuals.
    """
    records: List[pd.DataFrame] = []
    for track_id, track_df in metrics.groupby("track_id"):
        track_df = track_df.sort_values("timestamp").copy()

        track_df["area_smooth"] = _savgol(
            track_df["area"], SMOOTH_CONFIG["window"], SMOOTH_CONFIG["polyorder"]
        )
        rate_series = track_df["area_growth_rate"].ffill().bfill()
        track_df["growth_rate_smooth"] = _savgol(
            rate_series, SMOOTH_CONFIG["window"], SMOOTH_CONFIG["polyorder"]
        )
        # Replace inf with nan before division
        growth_rate_clean = track_df["growth_rate_smooth"].replace([np.inf, -np.inf], math.nan)
        area_smooth_clean = track_df["area_smooth"].replace({0: math.nan})
        track_df["relative_growth_rate"] = growth_rate_clean / area_smooth_clean
        track_df["area_noise"] = track_df["area"] - track_df["area_smooth"]

        germ_mask = track_df["area_smooth"] >= GERMINATION_AREA_THRESH
        if germ_mask.any():
            germ_index = germ_mask.idxmax()
        else:
            germ_index = track_df.index.min()
        germ_ts = track_df.loc[germ_index, "timestamp"]
        track_df["germination_timestamp"] = germ_ts
        track_df["time_since_germination_hours"] = (
            track_df["timestamp"] - germ_ts
        ).dt.total_seconds() / 3600.0
        track_df["germination_index"] = germ_index

        track_df["large_gap"] = track_df["dt_minutes"] > LARGE_GAP_MINUTES
        records.append(track_df)

    if not records:
        return metrics
    return pd.concat(records, ignore_index=True)


def _load_image_crop(tray_id: int, filename: str, xmin: int, ymin: int, xmax: int, ymax: int) -> Optional[np.ndarray]:
    """
    Load an image and crop to the bounding box region.
    Returns None if image is missing or load fails.
    """
    image_stem = Path(filename).name
    image_path = achmi.IMAGE_ROOT / str(tray_id) / f"{image_stem}.jpeg"
    if not image_path.exists():
        return None
    try:
        image = skio.imread(image_path)
        # Ensure bbox is within image bounds
        h, w = image.shape[:2]
        xmin = max(0, int(xmin))
        ymin = max(0, int(ymin))
        xmax = min(w, int(xmax))
        ymax = min(h, int(ymax))
        if xmax <= xmin or ymax <= ymin:
            return None
        crop = image[ymin:ymax, xmin:xmax]
        return crop
    except Exception:
        return None


def _extract_color_features_from_crop(crop: np.ndarray) -> Dict[str, float]:
    """
    Extract RGB, HSV, greenness, texture variance, and edge density from a crop.
    """
    if crop is None or crop.size == 0:
        return {
            "mean_R": math.nan,
            "mean_G": math.nan,
            "mean_B": math.nan,
            "mean_H": math.nan,
            "mean_S": math.nan,
            "mean_V": math.nan,
            "greenness": math.nan,
            "texture_var": math.nan,
            "edge_density": math.nan,
        }
    
    # RGB means
    if len(crop.shape) == 3 and crop.shape[2] == 3:
        mean_R = float(np.mean(crop[:, :, 0]))
        mean_G = float(np.mean(crop[:, :, 1]))
        mean_B = float(np.mean(crop[:, :, 2]))
    else:
        # Grayscale image
        mean_R = mean_G = mean_B = float(np.mean(crop))
    
    # HSV conversion
    if len(crop.shape) == 3 and crop.shape[2] == 3:
        hsv = skcolor.rgb2hsv(crop)
        mean_H = float(np.mean(hsv[:, :, 0]))
        mean_S = float(np.mean(hsv[:, :, 1]))
        mean_V = float(np.mean(hsv[:, :, 2]))
    else:
        mean_H = mean_S = 0.0
        mean_V = float(np.mean(crop)) / 255.0 if crop.dtype == np.uint8 else float(np.mean(crop))
    
    # Greenness index
    rgb_sum = mean_R + mean_G + mean_B + 1e-6
    greenness = mean_G / rgb_sum
    
    # Texture variance (grayscale)
    if len(crop.shape) == 3:
        gray = skcolor.rgb2gray(crop)
    else:
        gray = crop.astype(float) / 255.0 if crop.dtype == np.uint8 else crop.astype(float)
    texture_var = float(np.var(gray))
    
    # Edge density
    try:
        edges = feature.canny(gray, sigma=EDGE_SIGMA)
        edge_density = float(np.mean(edges))
    except Exception:
        edge_density = math.nan
    
    return {
        "mean_R": mean_R,
        "mean_G": mean_G,
        "mean_B": mean_B,
        "mean_H": mean_H,
        "mean_S": mean_S,
        "mean_V": mean_V,
        "greenness": greenness,
        "texture_var": texture_var,
        "edge_density": edge_density,
    }


def extract_color_features_for_track(track_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract color features for all frames in a track.
    Returns a DataFrame with color features aligned to track_df rows.
    Also tracks image_path_valid and missing_image_count.
    """
    color_records = []
    for _, row in track_df.iterrows():
        crop = _load_image_crop(
            row["tray_id"],
            row["filename"],
            row["xmin"],
            row["ymin"],
            row["xmax"],
            row["ymax"],
        )
        image_valid = crop is not None
        if crop is None:
            print(f"[warn] Missing image for track {row['track_id']} frame at timestamp {row['timestamp']}")
            features = _extract_color_features_from_crop(None)
        else:
            features = _extract_color_features_from_crop(crop)
        features["image_path_valid"] = image_valid
        color_records.append(features)
    
    color_df = pd.DataFrame(color_records, index=track_df.index)
    return color_df


def select_tracks_for_color_extraction(summary_df: pd.DataFrame, limit: Optional[int] = None) -> List[int]:
    """
    Select top-N tracks by coverage hours for color extraction.
    """
    if limit is None or limit >= len(summary_df):
        return summary_df["track_id"].tolist()
    top_tracks = summary_df.nlargest(limit, "coverage_hours")
    return top_tracks["track_id"].tolist()


def extract_color_features_with_cache(
    metrics: pd.DataFrame, selected_track_ids: List[int]
) -> pd.DataFrame:
    """
    Extract color features for selected tracks, using disk cache.
    Returns metrics DataFrame with color features merged in.
    """
    COLOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    metrics = metrics.copy()
    # Initialize color columns with NaN
    color_columns = [
        "mean_R", "mean_G", "mean_B",
        "mean_H", "mean_S", "mean_V",
        "greenness", "texture_var", "edge_density",
    ]
    for col in color_columns:
        if col not in metrics.columns:
            metrics[col] = math.nan
    
    # Initialize image path tracking columns (use NaN to indicate "not checked")
    if "image_path_valid" not in metrics.columns:
        metrics["image_path_valid"] = pd.NA
    if "missing_image_count" not in metrics.columns:
        metrics["missing_image_count"] = 0
    
    for track_id in selected_track_ids:
        cache_path = COLOR_CACHE_DIR / f"{track_id}.pkl"
        track_df = metrics[metrics["track_id"] == track_id].sort_values("timestamp")
        
        if track_df.empty:
            continue
        
        # Try to load from cache
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached_features = pickle.load(f)
                # Check if cache has valid color features
                has_valid_color = False
                for features in cached_features.values():
                    greenness = features.get("greenness")
                    if greenness is not None:
                        try:
                            if not (isinstance(greenness, float) and (math.isnan(greenness) or pd.isna(greenness))):
                                has_valid_color = True
                                break
                        except (TypeError, ValueError):
                            has_valid_color = True
                            break
                if has_valid_color:
                    print(f"[info] Loading cached color features for track {track_id} ({len(track_df)} frames)...")
                else:
                    print(f"[warn] Cache exists for track {track_id} but contains no valid color features. Re-extracting...")
                    # Delete bad cache and continue to extraction
                    cache_path.unlink()
                # Merge cached features into metrics
                for idx, features in cached_features.items():
                    if idx in track_df.index:
                        for col in color_columns + ["image_path_valid"]:
                            if col in features:
                                metrics.loc[idx, col] = features[col]
                if has_valid_color:
                    continue
            except Exception as e:
                print(f"[warn] Failed to load cache for track {track_id}: {e}")
        
        # Extract color features
        print(f"[info] Extracting color features for track {track_id} ({len(track_df)} frames)...")
        color_df = extract_color_features_for_track(track_df)
        
        # Merge into metrics
        for col in color_columns + ["image_path_valid"]:
            if col in color_df.columns:
                metrics.loc[track_df.index, col] = color_df[col]
        
        # Cache results
        try:
            cached_features = color_df.to_dict(orient="index")
            with open(cache_path, "wb") as f:
                pickle.dump(cached_features, f)
        except Exception as e:
            print(f"[warn] Failed to cache track {track_id}: {e}")
    
    # Initialize brightness_stability column
    if "brightness_stability" not in metrics.columns:
        metrics["brightness_stability"] = math.nan
    
    # Compute brightness stability (mean absolute frame-to-frame change of V channel) per track
    for track_id in selected_track_ids:
        track_df = metrics[metrics["track_id"] == track_id].sort_values("timestamp")
        if len(track_df) > 1 and "mean_V" in track_df.columns:
            v_diff = track_df["mean_V"].diff().abs()
            brightness_stability = v_diff.mean()
            metrics.loc[track_df.index, "brightness_stability"] = brightness_stability
    
    return metrics


def segment_growth_phases(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Add growth phase labels: germination, early_growth, exponential, slowdown.
    Detects turning points in dA/dt (growth_rate_smooth).
    """
    metrics = metrics.copy()
    if "growth_phase" not in metrics.columns:
        metrics["growth_phase"] = "unknown"
    
    for track_id, track_df in metrics.groupby("track_id"):
        track_df = track_df.sort_values("timestamp")
        if len(track_df) < 3 or "growth_rate_smooth" not in track_df.columns:
            continue
        
        # Find turning points in growth rate
        growth_rate = track_df["growth_rate_smooth"].fillna(0)
        area_smooth = track_df["area_smooth"]
        
        # Simple phase detection based on growth rate and area
        phases = []
        for i in range(len(track_df)):
            if i == 0:
                phases.append("germination")
            elif i < len(track_df) * 0.1:  # First 10% of track
                phases.append("early_growth")
            elif growth_rate.iloc[i] > growth_rate.iloc[i-1] * 1.1:  # Accelerating
                phases.append("exponential")
            elif growth_rate.iloc[i] < growth_rate.iloc[i-1] * 0.9:  # Decelerating
                phases.append("slowdown")
            else:
                # Continue previous phase or default to exponential if growing
                if i > 0:
                    phases.append(phases[-1] if phases else "exponential")
                else:
                    phases.append("exponential")
        
        # Assign phases
        metrics.loc[track_df.index, "growth_phase"] = phases
    
    return metrics


def compute_track_quality_score(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Compute track quality score and valid_track flag.
    Quality based on: missing image rate, noise ratio, large gaps, continuity, color features.
    """
    metrics = metrics.copy()
    
    # Initialize columns
    if "track_quality_score" not in metrics.columns:
        metrics["track_quality_score"] = 0.0
    if "valid_track" not in metrics.columns:
        metrics["valid_track"] = False
    if "missing_image_count" not in metrics.columns:
        metrics["missing_image_count"] = 0
    
    for track_id, track_df in metrics.groupby("track_id"):
        track_df = track_df.sort_values("timestamp")
        
        # Compute missing image rate (only for tracks that had color extraction)
        if "image_path_valid" in track_df.columns:
            # Only count if image_path_valid was actually checked (not NA)
            checked_mask = track_df["image_path_valid"].notna()
            if checked_mask.any():
                # Only compute for frames that were checked
                checked_df = track_df[checked_mask]
                missing_count = (~checked_df["image_path_valid"]).sum()
                missing_rate = missing_count / len(checked_df) if len(checked_df) > 0 else 1.0
            else:
                # No color extraction attempted for this track
                missing_count = 0
                missing_rate = 0.0  # Don't penalize tracks without color extraction
        else:
            missing_count = 0
            missing_rate = 0.0
        
        # Compute other quality metrics
        end_area = track_df["area"].iloc[-1] if len(track_df) > 0 else 0
        
        # Compute noise ratio on the fly
        if "area_noise" in track_df.columns and "area" in track_df.columns:
            area_noise_var = track_df["area_noise"].var()
            area_var = track_df["area"].var()
            noise_ratio = area_noise_var / area_var if area_var not in (0, None) and not pd.isna(area_var) else math.nan
        else:
            noise_ratio = math.nan
        
        # Count large gaps
        large_gap_count = track_df["large_gap"].sum() if "large_gap" in track_df.columns else 0
        large_gap_rate = large_gap_count / len(track_df) if len(track_df) > 0 else 1.0
        
        # Check continuity of smooth area (no flat signals)
        if "area_smooth" in track_df.columns and len(track_df) > 1:
            area_var = track_df["area_smooth"].var()
            is_flat = area_var < 100.0  # Very low variance = flat signal
        else:
            is_flat = True
        
        # Check for valid color features
        has_color_features = False
        if "greenness" in track_df.columns:
            has_color_features = track_df["greenness"].notna().any()
        
        # Compute quality score (0-1, higher is better)
        score = 1.0
        score -= missing_rate * 0.4  # Penalize missing images
        score -= large_gap_rate * 0.2  # Penalize large gaps
        if is_flat:
            score -= 0.3  # Penalize flat signals
        if pd.isna(noise_ratio) or noise_ratio > 0.3:
            score -= 0.1  # Penalize high noise
        score = max(0.0, score)  # Clamp to [0, 1]
        
        # Valid track criteria
        # Note: missing_rate check only applies if color extraction was attempted
        # (if image_path_valid was never checked, missing_rate = 0.0, so this passes)
        valid = (
            missing_rate < 0.5  # Only checked for tracks with color extraction
            and end_area > 5000
            and (pd.isna(noise_ratio) or noise_ratio < 0.3)
            and not is_flat
        )
        
        # Assign to all rows in track
        metrics.loc[track_df.index, "track_quality_score"] = score
        metrics.loc[track_df.index, "valid_track"] = valid
        metrics.loc[track_df.index, "missing_image_count"] = missing_count
    
    return metrics


def analyze_tray_image_availability(metrics: pd.DataFrame, sample_per_tray: int = 10) -> None:
    """
    Analyze image availability at the tray level to understand data completeness.
    """
    print("\n=== Tray-level Image Availability Analysis ===")
    
    tray_stats = []
    for tray_id in metrics["tray_id"].unique():
        tray_df = metrics[metrics["tray_id"] == tray_id]
        unique_filenames = tray_df["filename"].unique()
        
        # Sample some filenames to check
        sample_size = min(sample_per_tray, len(unique_filenames))
        sample_filenames = np.random.choice(unique_filenames, size=sample_size, replace=False)
        
        images_exist = 0
        for filename in sample_filenames:
            image_stem = Path(filename).name
            image_path = achmi.IMAGE_ROOT / str(tray_id) / f"{image_stem}.jpeg"
            if image_path.exists():
                images_exist += 1
        
        availability_rate = images_exist / sample_size if sample_size > 0 else 0.0
        n_tracks = tray_df["track_id"].nunique()
        n_frames = len(tray_df)
        
        tray_stats.append({
            "tray_id": tray_id,
            "n_tracks": n_tracks,
            "n_frames": n_frames,
            "sample_checked": sample_size,
            "images_found": images_exist,
            "availability_rate": availability_rate,
        })
    
    tray_df = pd.DataFrame(tray_stats).sort_values("availability_rate", ascending=False)
    
    print(f"\nTrays with images (availability > 0%): {(tray_df['availability_rate'] > 0).sum()}/{len(tray_df)}")
    print(f"Trays with all images (availability = 100%): {(tray_df['availability_rate'] == 1.0).sum()}/{len(tray_df)}")
    print(f"Trays with no images (availability = 0%): {(tray_df['availability_rate'] == 0.0).sum()}/{len(tray_df)}")
    
    print("\nTop 5 trays by image availability:")
    print(tray_df.head(5)[["tray_id", "n_tracks", "n_frames", "availability_rate"]].to_string(index=False))
    
    print("\nBottom 5 trays by image availability:")
    print(tray_df.tail(5)[["tray_id", "n_tracks", "n_frames", "availability_rate"]].to_string(index=False))
    
    return tray_df


def diagnose_missing_color_features(metrics: pd.DataFrame, top_n: int = 5) -> None:
    """
    Diagnose why top tracks are missing color features.
    Checks image paths and cache status for top N tracks by coverage.
    """
    summaries = summarize_tracks(metrics)
    top_tracks = summaries.head(top_n)
    
    print(f"\n=== Diagnosing Missing Color Features (Top {top_n} tracks) ===")
    for _, row in top_tracks.iterrows():
        track_id = row["track_id"]
        track_df = metrics[metrics["track_id"] == track_id].sort_values("timestamp")
        
        has_color = track_df["greenness"].notna().any() if "greenness" in track_df.columns else False
        
        print(f"\nTrack {track_id} (coverage: {row['coverage_hours']:.1f} hours, {row['n_frames']} frames):")
        print(f"  Has color features: {has_color}")
        
        # Check cache
        cache_path = COLOR_CACHE_DIR / f"{track_id}.pkl"
        print(f"  Cache exists: {cache_path.exists()}")
        
        # Check image paths for first few frames
        if len(track_df) > 0:
            sample_frames = track_df.head(3)
            images_found = 0
            for _, frame in sample_frames.iterrows():
                image_stem = Path(frame["filename"]).name
                image_path = achmi.IMAGE_ROOT / str(frame["tray_id"]) / f"{image_stem}.jpeg"
                exists = image_path.exists()
                if exists:
                    images_found += 1
                print(f"    Frame {frame.name}: {image_path.name} -> {'EXISTS' if exists else 'MISSING'}")
            
            # Check image_path_valid if available
            if "image_path_valid" in track_df.columns:
                checked_mask = track_df["image_path_valid"].notna()
                if checked_mask.any():
                    checked_df = track_df[checked_mask]
                    valid_count = checked_df["image_path_valid"].sum()
                    total_checked = len(checked_df)
                    print(f"  Image validation: {valid_count}/{total_checked} valid (from cache/extraction)")
                else:
                    print(f"  Image validation: Not checked (no color extraction attempted)")


def print_color_feature_coverage(metrics: pd.DataFrame) -> None:
    """
    Print summary of color feature coverage across tracks.
    """
    if "greenness" not in metrics.columns:
        print("\n[info] No color features found in metrics.")
        return
    
    track_ids = metrics["track_id"].unique()
    total_tracks = len(track_ids)
    
    tracks_with_color = 0
    tracks_with_nan_color = 0
    tracks_high_missing = 0
    
    # Compute color coverage per track
    color_coverages = []
    for track_id in track_ids:
        track_df = metrics[metrics["track_id"] == track_id]
        has_color = track_df["greenness"].notna().any()
        
        if has_color:
            tracks_with_color += 1
            # Compute coverage for tracks with color
            coverage = track_df["greenness"].notna().sum() / len(track_df) if len(track_df) > 0 else 0.0
            color_coverages.append(coverage)
        else:
            tracks_with_nan_color += 1
            color_coverages.append(0.0)
        
        # Check missing image rate (only for tracks that had color extraction)
        if "image_path_valid" in track_df.columns:
            checked_mask = track_df["image_path_valid"].notna()
            if checked_mask.any():
                checked_df = track_df[checked_mask]
                missing_rate = (~checked_df["image_path_valid"]).sum() / len(checked_df) if len(checked_df) > 0 else 0
                if missing_rate > 0.5:
                    tracks_high_missing += 1
    
    print("\n=== Color Feature Coverage Report ===")
    print(f"Total tracks: {total_tracks}")
    print(f"Tracks with valid color features: {tracks_with_color}/{total_tracks} ({100*tracks_with_color/total_tracks:.1f}%)")
    print(f"Tracks with NaN color features: {tracks_with_nan_color}/{total_tracks} ({100*tracks_with_nan_color/total_tracks:.1f}%)")
    print(f"Tracks with >50% missing images: {tracks_high_missing}/{total_tracks}")
    
    if color_coverages:
        coverages_array = np.array(color_coverages)
        print(f"\nColor Coverage Statistics (for tracks with any color features):")
        print(f"  Mean coverage: {coverages_array[coverages_array > 0].mean():.2%}" if (coverages_array > 0).any() else "  Mean coverage: N/A")
        print(f"  Median coverage: {np.median(coverages_array[coverages_array > 0]):.2%}" if (coverages_array > 0).any() else "  Median coverage: N/A")
        print(f"  Min coverage: {coverages_array[coverages_array > 0].min():.2%}" if (coverages_array > 0).any() else "  Min coverage: N/A")
        print(f"  Max coverage: {coverages_array.max():.2%}")
        
        # Show filtering impact at different thresholds
        print(f"\nFiltering Impact (tracks remaining at different color_coverage thresholds):")
        for threshold in [0.0, 0.25, 0.5, 0.75, 0.9]:
            n_remaining = (coverages_array >= threshold).sum()
            pct_remaining = 100 * n_remaining / total_tracks
            print(f"  Threshold {threshold:.0%}: {n_remaining}/{total_tracks} tracks ({pct_remaining:.1f}%)")


def summarize_tracks(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-frame metrics into per-track summaries.
    """
    summaries: List[TrackSummary] = []
    for track_id, track_df in metrics.groupby("track_id"):
        start = track_df["timestamp"].min()
        end = track_df["timestamp"].max()
        coverage = (end - start).total_seconds() / 3600.0
        start_area = track_df.iloc[0]["area"]
        end_area = track_df.iloc[-1]["area"]
        growth_pct = ((end_area - start_area) / start_area) * 100.0 if start_area else math.nan
        max_growth_rate = track_df["area_growth_rate"].max()
        mean_gap = track_df["dt_minutes"].dropna().mean()
        tray_id = track_df.iloc[0]["tray_id"]
        summaries.append(
            TrackSummary(
                track_id=track_id,
                tray_id=tray_id,
                n_frames=len(track_df),
                start_time=start,
                end_time=end,
                coverage_hours=coverage,
                start_area=start_area,
                end_area=end_area,
                total_growth_pct=growth_pct,
                max_growth_rate=max_growth_rate,
                mean_gap_minutes=mean_gap,
            )
        )
    summary_df = pd.DataFrame([s.__dict__ for s in summaries])
    if summary_df.empty:
        return summary_df

    def _pull_metric(track_id: int, column: str, reducer="max"):
        subset = metrics.loc[metrics["track_id"] == track_id, column]
        if subset.empty:
            return math.nan
        if reducer == "max":
            return subset.max()
        if reducer == "mean":
            return subset.mean()
        return subset.iloc[0]

    summary_df["germination_timestamp"] = [
        _pull_metric(row.track_id, "germination_timestamp", reducer="first")
        for row in summary_df.itertuples()
    ]
    summary_df["max_relative_growth_rate"] = [
        _pull_metric(row.track_id, "relative_growth_rate", reducer="max")
        for row in summary_df.itertuples()
    ]
    summary_df["noise_ratio"] = [
        (
            metrics.loc[metrics["track_id"] == row.track_id, "area_noise"].var()
            / metrics.loc[metrics["track_id"] == row.track_id, "area"].var()
        )
        if metrics.loc[metrics["track_id"] == row.track_id, "area"].var() not in (0, None)
        else math.nan
        for row in summary_df.itertuples()
    ]
    
    # Phase 3.3B: Color summary statistics
    color_summary_cols = ["mean_greenness", "mean_brightness", "mean_edge_density", "mean_texture_variance"]
    for col in color_summary_cols:
        summary_df[col] = math.nan
    
    for row in summary_df.itertuples():
        track_metrics = metrics[metrics["track_id"] == row.track_id]
        if not track_metrics.empty:
            if "greenness" in track_metrics.columns:
                summary_df.loc[summary_df["track_id"] == row.track_id, "mean_greenness"] = (
                    track_metrics["greenness"].mean()
                )
            if "mean_V" in track_metrics.columns:
                summary_df.loc[summary_df["track_id"] == row.track_id, "mean_brightness"] = (
                    track_metrics["mean_V"].mean()
                )
            if "edge_density" in track_metrics.columns:
                summary_df.loc[summary_df["track_id"] == row.track_id, "mean_edge_density"] = (
                    track_metrics["edge_density"].mean()
                )
            if "texture_var" in track_metrics.columns:
                summary_df.loc[summary_df["track_id"] == row.track_id, "mean_texture_variance"] = (
                    track_metrics["texture_var"].mean()
                )
    
    # Add quality metrics
    if "track_quality_score" in metrics.columns:
        summary_df["track_quality_score"] = [
            metrics.loc[metrics["track_id"] == row.track_id, "track_quality_score"].iloc[0]
            if len(metrics.loc[metrics["track_id"] == row.track_id]) > 0
            else 0.0
            for row in summary_df.itertuples()
        ]
        summary_df["valid_track"] = [
            metrics.loc[metrics["track_id"] == row.track_id, "valid_track"].iloc[0]
            if len(metrics.loc[metrics["track_id"] == row.track_id]) > 0
            else False
            for row in summary_df.itertuples()
        ]
        summary_df["missing_image_count"] = [
            metrics.loc[metrics["track_id"] == row.track_id, "missing_image_count"].iloc[0]
            if len(metrics.loc[metrics["track_id"] == row.track_id]) > 0
            else 0
            for row in summary_df.itertuples()
        ]
        # Missing image rate
        summary_df["missing_image_rate"] = (
            summary_df["missing_image_count"] / summary_df["n_frames"]
        ).fillna(0.0)
    
    # Compute color coverage (fraction of frames with valid color features)
    if "greenness" in metrics.columns:
        summary_df["color_coverage"] = [
            (
                metrics.loc[metrics["track_id"] == row.track_id, "greenness"].notna().sum()
                / len(metrics.loc[metrics["track_id"] == row.track_id])
            )
            if len(metrics.loc[metrics["track_id"] == row.track_id]) > 0
            else 0.0
            for row in summary_df.itertuples()
        ]
    else:
        summary_df["color_coverage"] = 0.0
    
    return summary_df.sort_values("coverage_hours", ascending=False)


def sample_tracks(metrics: pd.DataFrame, n: int) -> List[int]:
    track_ids = metrics["track_id"].unique().tolist()
    if not track_ids:
        return []
    random.seed(RANDOM_SEED)
    if len(track_ids) <= n:
        return track_ids
    return random.sample(track_ids, n)


def plot_track_growth(metrics: pd.DataFrame, track_id: int) -> None:
    track_df = metrics[metrics["track_id"] == track_id].sort_values("timestamp")
    if track_df.empty:
        print(f"[warn] No metrics for track {track_id}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(track_df["timestamp"], track_df["area"], marker="o", alpha=0.6, label="area (raw)")
    axes[0].plot(track_df["timestamp"], track_df["area_smooth"], linewidth=2, label="area (smooth)")
    axes[0].set_ylabel("Area (px^2)")
    axes[0].set_title(f"Track {track_id}: area trajectory")
    axes[0].legend()

    axes[1].plot(
        track_df["timestamp"],
        track_df["area_growth_rate"],
        marker="o",
        linestyle="--",
        color="tab:orange",
        alpha=0.6,
        label="dA/dt (raw)",
    )
    axes[1].plot(
        track_df["timestamp"],
        track_df["growth_rate_smooth"],
        marker="o",
        color="tab:red",
        label="dA/dt (smooth)",
    )
    axes[1].set_ylabel("dA/dt (px^2/min)")
    axes[1].set_title("Growth rate")
    axes[1].legend()

    axes[2].plot(
        track_df["timestamp"],
        track_df["relative_growth_rate"],
        marker="o",
        color="tab:green",
    )
    axes[2].set_ylabel("Relative growth (1/min)")
    axes[2].set_xlabel("Timestamp")
    axes[2].set_title("Relative growth rate")

    germ_ts = track_df["germination_timestamp"].iloc[0]
    if pd.notna(germ_ts):
        for ax in axes:
            ax.axvline(germ_ts, color="purple", linestyle=":", linewidth=1.5)
            ax.text(
                germ_ts,
                ax.get_ylim()[1],
                "germination",
                color="purple",
                rotation=90,
                va="top",
                ha="right",
            )

    gaps = track_df.index[track_df["large_gap"].fillna(False)]
    for idx in gaps:
        if idx - 1 in track_df.index:
            start_ts = track_df.loc[idx - 1, "timestamp"]
            end_ts = track_df.loc[idx, "timestamp"]
            axes[0].axvspan(start_ts, end_ts, color="grey", alpha=0.2)

    plt.tight_layout()


def aggregate_tray_stats(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    agg = (
        summary_df.groupby("tray_id")
        .agg(
            n_tracks=("track_id", "count"),
            median_growth_pct=("total_growth_pct", "median"),
            mean_growth_pct=("total_growth_pct", "mean"),
            median_coverage_hours=("coverage_hours", "median"),
        )
        .reset_index()
    )
    return agg.sort_values("median_growth_pct", ascending=False)


def main() -> None:
    random.seed(RANDOM_SEED)

    df = achmi.load_annotations()
    df = achmi.add_timestamp(df)
    df = attach_geometric_features(df)

    metrics = compute_track_metrics(df)
    if metrics.empty:
        print("[warn] No track metrics computed (missing timestamps?)")
        return
    metrics = smooth_track_metrics(metrics)
    
    # Add growth phase segmentation
    metrics = segment_growth_phases(metrics)

    # Phase 3.3B: Color extraction (optional)
    if ENABLE_COLOR_EXTRACTION:
        summaries_pre_color = summarize_tracks(metrics)
        selected_track_ids = select_tracks_for_color_extraction(
            summaries_pre_color, limit=COLOR_TRACK_LIMIT
        )
        print(f"\n[info] Extracting color features for {len(selected_track_ids)} tracks...")
        metrics = extract_color_features_with_cache(metrics, selected_track_ids)
        print("[info] Color extraction complete.")
        
        # Print color feature coverage report
        print_color_feature_coverage(metrics)
        
        # Analyze tray-level image availability
        analyze_tray_image_availability(metrics, sample_per_tray=10)
        
        # Diagnose missing color features for top tracks
        diagnose_missing_color_features(metrics, top_n=5)
    
    # Compute track quality scores
    metrics = compute_track_quality_score(metrics)
    
    # Print quality summary
    summaries = summarize_tracks(metrics)
    if "valid_track" in summaries.columns:
        n_valid = summaries["valid_track"].sum()
        n_total = len(summaries)
        print(f"\n[info] Track quality: {n_valid}/{n_total} tracks marked as valid for ODE training")
    print("\n=== Track Summary (top 10 by coverage) ===")
    display_cols = [
        "track_id",
        "tray_id",
        "n_frames",
        "coverage_hours",
        "total_growth_pct",
        "max_growth_rate",
        "max_relative_growth_rate",
        "noise_ratio",
        "mean_gap_minutes",
    ]
    # Add quality metrics if available
    if "valid_track" in summaries.columns:
        display_cols.extend(["valid_track", "track_quality_score", "missing_image_rate"])
    # Add color summary columns if available
    if ENABLE_COLOR_EXTRACTION:
        color_cols = ["mean_greenness", "mean_brightness", "mean_edge_density", "color_coverage"]
        display_cols.extend([col for col in color_cols if col in summaries.columns])
    
    # Apply color coverage filter if threshold is set
    if ENABLE_COLOR_EXTRACTION and COLOR_COVERAGE_THRESHOLD > 0.0:
        if "color_coverage" in summaries.columns:
            n_before = len(summaries)
            summaries = summaries[summaries["color_coverage"] >= COLOR_COVERAGE_THRESHOLD]
            n_after = len(summaries)
            print(f"\n[info] Applied color_coverage threshold {COLOR_COVERAGE_THRESHOLD:.0%}: {n_before} -> {n_after} tracks ({100*n_after/n_before:.1f}% retained)")
    
    print(summaries[display_cols].head(10))

    tray_stats = aggregate_tray_stats(summaries)
    if not tray_stats.empty:
        print("\n=== Tray-level Growth Summary ===")
        print(tray_stats.head(10))

    # Save unified metrics to disk if enabled
    if SAVE_METRICS_TO_DISK:
        METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            metrics.to_parquet(METRICS_OUTPUT_PATH, index=False)
            print(f"\n[info] Saved unified metrics to {METRICS_OUTPUT_PATH}")
        except Exception as e:
            print(f"[warn] Failed to save metrics to parquet: {e}")
            # Fallback to CSV
            csv_path = METRICS_OUTPUT_PATH.with_suffix(".csv")
            metrics.to_csv(csv_path, index=False)
            print(f"[info] Saved unified metrics to {csv_path} (CSV fallback)")

    sampled_tracks = sample_tracks(metrics, TRACK_SAMPLE_COUNT)
    print(f"\n[info] Plotting {len(sampled_tracks)} sampled tracks: {sampled_tracks}")
    for track_id in sampled_tracks:
        plot_track_growth(metrics, track_id)

    plt.show()


if __name__ == "__main__":
    main()

