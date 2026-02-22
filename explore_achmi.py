import io
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


GT_CSV_PATH = Path("Thesis/data/gt.csv")
IMAGE_ROOT = Path("Thesis/data/jpegs/ACHMI")
TARGET_LABEL = "ACHMI"
TOP_N = 5
DATE_PATTERN = re.compile(r"(?P<year>\d{4})Y(?P<month>\d{2})M(?P<day>\d{2})D")
TIME_PATTERN = re.compile(r"(?P<hour>\d{2})H(?P<minute>\d{2})M(?P<second>\d{2})S")


def load_annotations() -> pd.DataFrame:
    """
    Load the ground-truth CSV and filter down to the target label.
    """
    df = pd.read_csv(GT_CSV_PATH)
    if TARGET_LABEL is not None:
        df = df[df["label_id"] == TARGET_LABEL].copy()
    return df


def print_structure_summary(df: pd.DataFrame) -> None:
    """
    Print quick structural diagnostics so we know what columns exist
    and whether any obvious null issues pop up.
    """
    print("\n=== Data Preview (first 5 rows) ===")
    print(df.head())

    buffer = io.StringIO()
    df.info(buf=buffer)
    print("\n=== DataFrame Info ===")
    print(buffer.getvalue())

    print("=== Null Counts ===")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0] if (null_counts > 0).any() else "No null values detected.")


def print_scope_summary(df: pd.DataFrame) -> None:
    """
    Report high-level counts that describe the scope of the dataset.
    """
    total_records = len(df)
    unique_trays = df["tray_id"].nunique()
    unique_tracks = df["track_id"].nunique()
    unique_filenames = df["filename"].nunique()

    print("\n=== Scope Summary ===")
    print(f"Total records: {total_records:,}")
    print(f"Unique trays: {unique_trays:,}")
    print(f"Unique track IDs: {unique_tracks:,}")
    print(f"Unique filenames: {unique_filenames:,}")

    print(f"\nTop {TOP_N} trays by annotation count:")
    print(df["tray_id"].value_counts().head(TOP_N))

    print(f"\nTop {TOP_N} track IDs by annotation count:")
    print(df["track_id"].value_counts().head(TOP_N))


def summarize_tray_distribution(df: pd.DataFrame) -> None:
    """
    Provide additional statistics for tray-level coverage.
    """
    tray_counts = df["tray_id"].value_counts()
    print("\n=== Tray Distribution Summary ===")
    print(f"Min annotations per tray: {tray_counts.min()}")
    print(f"Median annotations per tray: {tray_counts.median():.1f}")
    print(f"Mean annotations per tray: {tray_counts.mean():.1f}")
    print(f"Max annotations per tray: {tray_counts.max()}")
    print(f"10th percentile: {tray_counts.quantile(0.10):.1f}")
    print(f"90th percentile: {tray_counts.quantile(0.90):.1f}")

    print(f"\nBottom {TOP_N} trays by annotation count:")
    print(tray_counts.tail(TOP_N))


def parse_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract a datetime from filename strings of the form
    ACHMI/133801/ACHMI_133801_2021Y07M28D_00H49M09S_img.
    """
    stem = Path(filename).name
    parts = stem.split("_")
    if len(parts) < 5:
        return None

    date_part, time_part = parts[2], parts[3]
    date_match = DATE_PATTERN.fullmatch(date_part)
    time_match = TIME_PATTERN.fullmatch(time_part)
    if not date_match or not time_match:
        return None

    try:
        return datetime(
            year=int(date_match.group("year")),
            month=int(date_match.group("month")),
            day=int(date_match.group("day")),
            hour=int(time_match.group("hour")),
            minute=int(time_match.group("minute")),
            second=int(time_match.group("second")),
        )
    except ValueError:
        return None


def add_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    timestamps = df["filename"].apply(parse_timestamp_from_filename)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(timestamps, errors="coerce")
    return df


def check_temporal_coherence(df: pd.DataFrame) -> None:
    """
    Verify that timestamps exist, are monotonic per track, and inspect gap statistics.
    """
    df = add_timestamp(df)
    missing_timestamp = df["timestamp"].isna().sum()
    print("\n=== Timestamp Coverage ===")
    print(f"Rows missing timestamp: {missing_timestamp}")

    duplicate_tracks = []
    max_gap = (None, pd.Timedelta(0))
    gap_series = []

    for track_id, track_df in df.groupby("track_id"):
        track_df = track_df.dropna(subset=["timestamp"]).sort_values("timestamp")
        diffs = track_df["timestamp"].diff().dropna()
        if diffs.empty:
            continue
        if (diffs == pd.Timedelta(0)).any():
            duplicate_tracks.append(track_id)
        gap_series.append(diffs)
        track_max = diffs.max()
        if track_max > max_gap[1]:
            max_gap = (track_id, track_max)

    if gap_series:
        all_gaps = pd.concat(gap_series)
        gap_minutes = all_gaps.dt.total_seconds() / 60.0
        print("\n=== Gap Statistics Across Tracks ===")
        print(f"Median gap (minutes): {gap_minutes.median():.2f}")
        print(f"Mean gap (minutes): {gap_minutes.mean():.2f}")
        print(f"95th percentile gap (minutes): {gap_minutes.quantile(0.95):.2f}")
        print(f"Max gap (minutes): {gap_minutes.max():.2f}")
        if max_gap[0] is not None:
            print(f"Largest gap occurs in track {max_gap[0]} ({max_gap[1]})")
    else:
        print("No gaps computed (insufficient timestamp data).")

    if duplicate_tracks:
        print("\nTracks with duplicate timestamps detected:")
        print(sorted(duplicate_tracks))
    else:
        print("\nNo duplicate timestamps detected within tracks.")


def main() -> None:
    df = load_annotations()
    print_structure_summary(df)
    print_scope_summary(df)
    summarize_tray_distribution(df)
    check_temporal_coherence(df)


if __name__ == "__main__":
    main()