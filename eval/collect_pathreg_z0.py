"""Fit-end z0 cosine on pinned pathreg checkpoints (no training)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ode._pyc_bootstrap import bootstrap

bootstrap()

import torch

from ode.data.datamodule import PlantTrackDataModule
from ode.data.dataset import PlantTrackStateDataset
from ode.pathreg import collect_z0_and_dz, load_ode_module
from ode.train_baselines import select_discrete_tracks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, default=REPO / "checkpoints")
    p.add_argument("--run-tag", type=str, required=True)
    p.add_argument("--ckpt-name", type=str, required=True)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--extrap", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    parquet = REPO / "metrics_with_features.parquet"
    if not parquet.is_file():
        parquet = REPO / "Thesis" / "metrics_with_features.parquet"
    train_ids, _, _ = select_discrete_tracks(
        parquet, max_tracks=100, species="Maize",
        min_observations=15, val_frac=0.15, test_frac=0.15,
    )
    dm = PlantTrackDataModule(
        parquet_path=parquet, batch_size=32, num_workers=0, sigma_mode="z_score",
    )
    dm.setup()
    ds = PlantTrackStateDataset(
        dm.parquet_path, dm.transformer, track_ids=train_ids,
        valid_track_only=True, min_observations=8,
    )
    per_seed = []
    for seed in range(args.n_seeds):
        stem = f"{args.run_tag}{seed}_extrap60" if args.extrap else f"{args.run_tag}{seed}"
        path = args.ckpt_dir / stem / args.ckpt_name
        if not path.is_file():
            print(f"[z0] missing {path}")
            continue
        module = load_ode_module(path, device, strict=False)
        report = collect_z0_and_dz(module, ds, device)
        report.pop("z0", None)
        report.pop("dz", None)
        report["seed"] = seed
        report["path"] = str(path)
        per_seed.append(report)
        print(
            f"[z0] seed={seed} cosine={report['z0_cosine']['mean_offdiag']:.6f} "
            f"dz_std={report['train_dz_dt_size_std']:.6f} "
            f"dz_norm={report['dz_dt_mean_norm']:.6f}"
        )
    mean_z0 = None
    if per_seed:
        mean_z0 = sum(r["z0_cosine"]["mean_offdiag"] for r in per_seed) / len(per_seed)
    payload = {"mean_offdiag": mean_z0, "seeds": per_seed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"[z0] wrote {args.out} mean={mean_z0}")


if __name__ == "__main__":
    main()
