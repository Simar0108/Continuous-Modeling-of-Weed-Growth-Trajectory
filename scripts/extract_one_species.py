#!/usr/bin/env python3
"""Run the verified ACHMI extractor on one MFWD label_id.

Does not apply lock-parity postprocess. Writes parquet + COMPLETE under
the output directory. All rows must have label_id == --species.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--species", required=True)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--image-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    args = p.parse_args()

    species = args.species
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "metrics_with_features.parquet"
    cache_dir = out_dir / "color_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    achmi = load_module("explore_achmi", data_dir / "explore_achmi.py")
    growth = load_module("explore_achmi_growth", data_dir / "explore_achmi_growth.py")

    achmi.GT_CSV_PATH = args.gt
    achmi.IMAGE_ROOT = args.image_root
    achmi.TARGET_LABEL = species
    growth.achmi = achmi
    growth.METRICS_OUTPUT_PATH = parquet_path
    growth.COLOR_CACHE_DIR = cache_dir
    growth.COLOR_TRACK_LIMIT = 10**9
    growth.TRACK_SAMPLE_COUNT = 0
    growth.APPLY_LOCK_PARITY = False
    growth.ENABLE_COLOR_EXTRACTION = True
    growth.SAVE_METRICS_TO_DISK = True

    growth.main()

    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if df.empty:
        raise SystemExit(f"{species}: empty parquet")
    if not (df["label_id"] == species).all():
        bad = sorted(df.loc[df["label_id"] != species, "label_id"].unique().tolist())
        raise SystemExit(f"{species}: label_id leak {bad}")
    df["species"] = species
    n_dup = int(df.duplicated(["track_id", "bbox_id"]).sum())
    if n_dup:
        raise SystemExit(f"{species}: duplicate track_id/bbox_id {n_dup}")
    df.to_parquet(parquet_path, index=False)

    n_tracks = int(df["track_id"].nunique())
    n_ge15 = int((df.groupby("track_id").size() >= 15).sum())
    n_valid = int(df.groupby("track_id")["valid_track"].first().sum()) if "valid_track" in df.columns else 0
    n_usable = int(
        ((df.groupby("track_id").size() >= 15) & df.groupby("track_id")["valid_track"].first()).sum()
    ) if "valid_track" in df.columns else 0
    n_valid_frames = int(df["valid_track"].sum()) if "valid_track" in df.columns else 0
    summary = {
        "species": species,
        "n_rows": int(len(df)),
        "n_tracks": n_tracks,
        "n_ge15": n_ge15,
        "n_valid_track": n_valid,
        "n_usable": n_usable,
        "n_valid_frames": n_valid_frames,
        "parquet": str(parquet_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "COMPLETE").write_text(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
