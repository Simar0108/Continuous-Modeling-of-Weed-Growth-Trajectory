import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from skimage import io as skio

import explore_achmi as achmi


TRAY_SAMPLE_COUNT = 6
TRACK_SAMPLE_MIN_FRAMES = 8
TRACK_SAMPLE_DISPLAY_FRAMES = 6
RANDOM_SEED = 42


def sample_trays(df: pd.DataFrame, n: int) -> List[int]:
    trays = df["tray_id"].unique()
    if n >= len(trays):
        return trays.tolist()
    random.seed(RANDOM_SEED)
    return random.sample(trays.tolist(), n)


def sample_images_for_tray(
    df: pd.DataFrame, tray_id: int, n_images: int = 2
) -> List[pd.Series]:
    tray_rows = df[df["tray_id"] == tray_id]
    if tray_rows.empty:
        return []
    random.seed(RANDOM_SEED + tray_id)
    sampled_rows = tray_rows.sample(
        n=min(n_images, len(tray_rows)), replace=False, random_state=RANDOM_SEED
    )
    return list(sampled_rows.to_dict(orient="records"))


def select_track_with_min_frames(
    df: pd.DataFrame, min_frames: int
) -> Optional[int]:
    candidates = (
        df["track_id"]
        .value_counts()
        .loc[lambda s: s >= min_frames]
        .index.tolist()
    )
    if not candidates:
        return None
    random.seed(RANDOM_SEED)
    return random.choice(candidates)


def load_image(tray_id: int, filename: str) -> Optional[Tuple[Path, "skio.Image"]]:
    image_stem = Path(filename).name  # e.g. ACHMI_133801_...._img
    image_path = achmi.IMAGE_ROOT / str(tray_id) / f"{image_stem}.jpeg"
    if not image_path.exists():
        print(f"[warn] Missing image: {image_path}")
        return None
    try:
        image = skio.imread(image_path)
        return image_path, image
    except Exception as exc:  # pragma: no cover - diagnostic
        print(f"[warn] Failed to load {image_path}: {exc}")
        return None


def plot_image_with_bboxes(
    ax: plt.Axes, image, rows: Sequence[pd.Series], title: str
) -> None:
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
    for row in rows:
        xmin, ymin = row["xmin"], row["ymin"]
        width = row["xmax"] - xmin
        height = row["ymax"] - ymin
        rect = patches.Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)


def show_tray_samples(df: pd.DataFrame, tray_ids: Iterable[int], n_images: int = 2):
    for tray_id in tray_ids:
        records = df[df["tray_id"] == tray_id]
        images = sample_images_for_tray(records, tray_id, n_images=n_images)
        if not images:
            print(f"[info] No images found for tray {tray_id}")
            continue

        fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 6))
        if len(images) == 1:
            axes = [axes]

        for ax, row in zip(axes, images):
            result = load_image(tray_id=row["tray_id"], filename=row["filename"])
            if result is None:
                ax.axis("off")
                continue
            image_path, image = result
            row_df = records[records["filename"] == row["filename"]]
            plot_image_with_bboxes(
                ax,
                image,
                row_df.to_dict(orient="records"),
                title=f"Tray {tray_id}\n{image_path.name}",
            )
        fig.suptitle(f"Tray {tray_id} samples")
        fig.tight_layout()


def show_track_timeline(
    df: pd.DataFrame,
    track_id: int,
    n_frames: int = TRACK_SAMPLE_DISPLAY_FRAMES,
) -> None:
    track_df = achmi.add_timestamp(df[df["track_id"] == track_id])
    track_df = track_df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if track_df.empty:
        print(f"[info] Track {track_id} has no timestamped images.")
        return

    selected = track_df.iloc[:: max(len(track_df) // n_frames, 1)].head(n_frames)
    fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 5))
    if len(selected) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, selected.iterrows()):
        result = load_image(tray_id=row["tray_id"], filename=row["filename"])
        if result is None:
            ax.axis("off")
            continue
        image_path, image = result
        plot_image_with_bboxes(
            ax,
            image,
            track_df[track_df["filename"] == row["filename"]].to_dict(orient="records"),
            title=row["timestamp"].strftime("%Y-%m-%d %H:%M"),
        )
        ax.set_xlabel(f"{image_path.name}")

    fig.suptitle(f"Track {track_id} timeline")
    fig.tight_layout()


def main():
    df = achmi.load_annotations()
    # Phase 1 summaries (reused for context)
    achmi.print_structure_summary(df)
    achmi.print_scope_summary(df)
    achmi.summarize_tray_distribution(df)
    achmi.check_temporal_coherence(df)

    trays = sample_trays(df, TRAY_SAMPLE_COUNT)
    print(f"\n[info] Displaying {len(trays)} sampled trays: {trays}")
    show_tray_samples(df, trays, n_images=2)

    track_id = select_track_with_min_frames(df, TRACK_SAMPLE_MIN_FRAMES)
    if track_id is None:
        print("[warn] No track found with enough frames for timeline visualization.")
        return
    print(f"\n[info] Displaying timeline for track {track_id}")
    show_track_timeline(df, track_id, TRACK_SAMPLE_DISPLAY_FRAMES)

    plt.show()


if __name__ == "__main__":
    main()

