"""D-019 Probe 2: z0 cosine on the no-embedding stab seed (eval only).

Does not retrain. Does not write h1_final_best. Stab already has
use_track_embed=False (D-019).
"""

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

H1_SHA = "2498d033ea02e2df1e312a58179226b649a9d3d5c7143b2d040fc9341ef86222"
LOCK_COSINE = 0.996
DROP_BAR = 0.05


def _parquet() -> Path:
    p = REPO / "metrics_with_features.parquet"
    return p if p.is_file() else REPO / "Thesis" / "metrics_with_features.parquet"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=REPO / "checkpoints" / "h1_stab_seed0" / "best.ckpt")
    p.add_argument("--out-dir", type=Path, default=REPO / "figures" / "encoder_probes")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if "h1_final_best" in str(args.ckpt):
        raise SystemExit("REFUSE: Probe 2 must not load h1_final_best as the ablation")
    ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters") or {}
    embed_on = bool(hp.get("use_track_embed", False))
    emb_keys = [k for k in ckpt["state_dict"] if "track_embedding" in k or "track_embed" in k]
    print(f"[probe2] ckpt={args.ckpt} epoch={ckpt.get('epoch')} use_track_embed={embed_on} embed_tensors={len(emb_keys)}")
    if embed_on or emb_keys:
        raise SystemExit(
            "REFUSE: stab ckpt has embeddings; D-019 said do not invent a new recipe. "
            "A no-embed retrain would be a new decision, not this probe."
        )
    device = torch.device(args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    module = load_ode_module(args.ckpt, device, strict=False)
    parquet = _parquet()
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
    report = collect_z0_and_dz(module, ds, device)
    report.pop("z0", None)
    report.pop("dz", None)
    cosine = float(report["z0_cosine"]["mean_offdiag"])
    dropped = (LOCK_COSINE - cosine) >= DROP_BAR
    payload = {
        "label": "D-019 Probe 2. Existing h1_stab_seed0, no retrain.",
        "ckpt": str(args.ckpt),
        "best_epoch": int(ckpt.get("epoch", -1)),
        "use_track_embed": embed_on,
        "embed_tensors": emb_keys,
        "z0_cosine_mean": cosine,
        "z0_cosine": report["z0_cosine"],
        "train_dz_dt_size_std": report["train_dz_dt_size_std"],
        "lock_cosine_reference": LOCK_COSINE,
        "material_drop_bar": DROP_BAR,
        "material_drop": dropped,
        "h1_final_best_sha256_untouched": H1_SHA,
        "retrained": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "probe2.json").write_text(json.dumps(payload, indent=2))
    print(
        f"[probe2] z0_cosine={cosine:.6f} material_drop={dropped} "
        f"dz_std={report['train_dz_dt_size_std']:.6f}"
    )


if __name__ == "__main__":
    main()
