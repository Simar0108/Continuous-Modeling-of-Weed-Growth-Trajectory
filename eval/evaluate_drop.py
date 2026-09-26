"""Step 10 drop protocol: 20/40/60% frame drop × 3 drop seeds.

Size MSE on dropped observations only. Context is the first K kept
frames; future ground truth is never leaked into the query.
"""

from __future__ import annotations

import argparse
import csv
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wilcoxon

from ode.compare_models import _build_context_only_query
from ode.train_baselines import BaselineLightning, build_baseline_datasets, select_discrete_tracks
from ode.train_multi import MultiTrackLightning

OUT_DIR = REPO / "figures" / "h1_lock"
DROP_FRACS = (0.20, 0.40, 0.60)
DROP_SEEDS = (0, 1, 2)
K = 3


def _parquet() -> Path:
    p = REPO / "metrics_with_features.parquet"
    return p if p.is_file() else REPO / "Thesis" / "metrics_with_features.parquet"


def _keep_mask(length: int, track_id: int, drop_seed: int, drop_frac: float) -> np.ndarray:
    rng = np.random.default_rng(int(drop_seed) * 1_000_003 + int(track_id))
    keep = np.ones(length, dtype=bool)
    if length <= 1:
        return keep
    n_drop = min(int(round((length - 1) * drop_frac)), max(0, length - K))
    candidates = np.arange(1, length)
    if n_drop > 0 and candidates.size:
        chosen = rng.choice(candidates, size=min(n_drop, candidates.size), replace=False)
        keep[chosen] = False
    if keep.sum() < K:
        keep[:K] = True
    keep[0] = True
    return keep


def _pred_ode(module, query_states, query_times, track_ids=None):
    pred = module.model(query_states.unsqueeze(0), query_times, return_aux=False, track_ids=track_ids)
    if pred.dim() == 3:
        pred_b = pred[:, 0, :] if pred.shape[0] == query_states.shape[0] else pred[0]
    else:
        pred_b = pred
    return pred_b


def _pred_baseline(module, query_states, query_times):
    length = torch.tensor([query_states.shape[0]], device=query_states.device)
    pred = module.model.autoregressive_rollout(query_states.unsqueeze(0), query_times, length)
    return pred.squeeze(0)


def _size_mse_dropped(pred_future, target_states, dropped_mask):
    if dropped_mask.sum() == 0:
        return float("nan")
    # ODE pred is 5-D; baseline pred is 3-D [sw, sh, Z]
    if pred_future.shape[-1] >= 4:
        pred_wh = pred_future[:, 2:4]
    else:
        pred_wh = pred_future[:, :2]
    target_wh = target_states[:, 2:4]
    n = min(pred_wh.shape[0], target_wh.shape[0], dropped_mask.shape[0])
    err = (pred_wh[:n] - target_wh[:n]).pow(2).mean(dim=-1)
    mask = torch.as_tensor(dropped_mask[:n], device=err.device, dtype=torch.bool)
    if mask.sum() == 0:
        return float("nan")
    return float(err[mask].mean().item())


def _score_track(predict_fn, states, t_abs, keep, device):
    keep_idx = torch.nonzero(torch.as_tensor(keep, device=device), as_tuple=False).squeeze(1)
    if keep_idx.numel() < K:
        return float("nan")
    context = keep_idx[:K]
    future = torch.arange(int(context[-1].item()) + 1, states.shape[0], device=device)
    if future.numel() == 0:
        return float("nan")
    q_states, q_times = _build_context_only_query(
        states, t_abs, context_indices=context, prediction_indices=future,
    )
    pred = predict_fn(q_states, q_times)
    dropped = ~keep[future.detach().cpu().numpy()]
    return _size_mse_dropped(pred, states[future], dropped)


def _discover_ode_seeds(ckpt_dir: Path) -> list[tuple[str, Path]]:
    found = []
    lock = ckpt_dir / "h1_final_best" / "best.ckpt"
    if lock.is_file():
        found.append(("ode_lock", lock))
    for seed in (0, 1, 2):
        d = ckpt_dir / f"h1_seed{seed}"
        cands = sorted(d.glob("best*.ckpt")) if d.is_dir() else []
        if cands:
            found.append((f"ode_s{seed}", cands[0]))
    return found


def _discover_baselines(ckpt_dir: Path) -> list[tuple[str, Path]]:
    found = []
    for p in sorted(ckpt_dir.glob("*_valmse_baseline-*.ckpt")):
        if "-extrap" in p.stem:
            continue
        name = p.stem
        model = "lstm"
        for m in ("lstm", "gru", "transformer"):
            if f"{m}_valmse" in name or name.startswith(m):
                model = m
                break
        seed = 0
        if "-s" in name:
            try:
                seed = int(name.rsplit("-s", 1)[-1].split("-")[0])
            except ValueError:
                seed = 0
        found.append((f"{model}_s{seed}", p))
    return found


def _wilcoxon(a_rows, b_rows):
    a = {r["track_id"]: r["size_mse"] for r in a_rows if np.isfinite(r["size_mse"])}
    b = {r["track_id"]: r["size_mse"] for r in b_rows if np.isfinite(r["size_mse"])}
    ids = sorted(set(a) & set(b))
    if len(ids) < 6:
        return {"n": len(ids), "p": float("nan")}
    xa = np.array([a[i] for i in ids])
    xb = np.array([b[i] for i in ids])
    stat, p = wilcoxon(xa, xb, alternative="less")
    return {
        "n": len(ids),
        "p": float(p),
        "stat": float(stat),
        "ode_mean": float(xa.mean()),
        "other_mean": float(xb.mean()),
        "mean_diff": float((xa - xb).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--ckpt-dir", type=Path, default=REPO / "checkpoints")
    p.add_argument("--split", default="both", choices=("val", "test", "both"))
    args = p.parse_args()
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    parquet = _parquet()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _, _, val_ds, test_ds = build_baseline_datasets(
        parquet, sigma_mode="z_score", max_tracks=100, species="Maize",
        val_frac=0.15, test_frac=0.15,
    )
    splits = {}
    if args.split in ("val", "both"):
        splits["val"] = val_ds
    if args.split in ("test", "both"):
        splits["test"] = test_ds

    models = []
    for key, path in _discover_ode_seeds(args.ckpt_dir):
        try:
            mod = MultiTrackLightning.load_from_checkpoint(str(path), map_location=device, strict=False)
            mod.eval().to(device)
            mod.horizon_start_frac = 1.0
            models.append((key, lambda qs, qt, m=mod: _pred_ode(m, qs, qt)))
            print(f"[drop] loaded {key} {path}")
        except Exception as exc:
            print(f"[drop] skip {path}: {exc}")
    for key, path in _discover_baselines(args.ckpt_dir):
        try:
            mod = BaselineLightning.load_from_checkpoint(str(path), map_location=device, strict=False)
            mod.eval().to(device)
            models.append((key, lambda qs, qt, m=mod: _pred_baseline(m, qs, qt)))
            print(f"[drop] loaded {key} {path}")
        except Exception as exc:
            print(f"[drop] skip {path}: {exc}")

    long_rows = []
    for model_key, predict_fn in models:
        for split, ds in splits.items():
            for drop_frac in DROP_FRACS:
                for drop_seed in DROP_SEEDS:
                    for sample in ds:
                        states = sample["states_5d"].to(device)
                        t_abs = sample["t_absolute"].to(device)
                        tid = int(sample["track_id"].item())
                        if states.shape[0] <= K:
                            continue
                        keep = _keep_mask(states.shape[0], tid, drop_seed, drop_frac)
                        with torch.no_grad():
                            mse = _score_track(predict_fn, states, t_abs, keep, device)
                        long_rows.append({
                            "model": model_key, "split": split,
                            "drop_frac": drop_frac, "drop_seed": drop_seed,
                            "track_id": tid, "size_mse": mse,
                        })

    long_path = OUT_DIR / "drop_per_track.csv"
    with long_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()) if long_rows else
                           ["model", "split", "drop_frac", "drop_seed", "track_id", "size_mse"])
        w.writeheader()
        w.writerows(long_rows)

    # Aggregate: mean over tracks, then mean±std over drop seeds
    summary = []
    by_key: dict[tuple, list[float]] = {}
    for r in long_rows:
        if not np.isfinite(r["size_mse"]):
            continue
        key = (r["model"], r["split"], r["drop_frac"], r["drop_seed"])
        by_key.setdefault(key, []).append(r["size_mse"])
    seed_means: dict[tuple, float] = {k: float(np.mean(v)) for k, v in by_key.items()}
    grouped: dict[tuple, list[float]] = {}
    for (model, split, frac, _ds), mean in seed_means.items():
        grouped.setdefault((model, split, frac), []).append(mean)
    for (model, split, frac), vals in sorted(grouped.items()):
        summary.append({
            "model": model, "split": split, "drop_frac": frac,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_drop_seeds": len(vals),
        })
    sum_path = OUT_DIR / "drop_summary.csv"
    with sum_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "split", "drop_frac", "mean", "std", "n_drop_seeds"])
        w.writeheader()
        w.writerows(summary)

    # Wilcoxon: mean over drop seeds per track, ODE seeds vs LSTM/GRU/Transformer
    wilcox = []
    for split in splits:
        for frac in DROP_FRACS:
            def rows_for(model):
                by_track: dict[int, list[float]] = {}
                for r in long_rows:
                    if r["model"] == model and r["split"] == split and r["drop_frac"] == frac:
                        if np.isfinite(r["size_mse"]):
                            by_track.setdefault(r["track_id"], []).append(r["size_mse"])
                return [{"track_id": t, "size_mse": float(np.mean(v))} for t, v in by_track.items()]

            ode_keys = [m for m, _ in models if m.startswith("ode_s")]
            if not ode_keys:
                ode_keys = [m for m, _ in models if m == "ode_lock"]
            others = [m for m, _ in models if not m.startswith("ode")]
            for ok in ode_keys:
                for other in others:
                    block = _wilcoxon(rows_for(ok), rows_for(other))
                    block.update({"split": split, "drop_frac": frac, "ode": ok, "vs": other})
                    wilcox.append(block)
    w_path = OUT_DIR / "drop_wilcoxon.csv"
    if wilcox:
        with w_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(wilcox[0].keys()))
            w.writeheader()
            w.writerows(wilcox)

    # Curves
    fig, axes = plt.subplots(1, max(1, len(splits)), figsize=(5 * max(1, len(splits)), 4), squeeze=False)
    families = {}
    for row in summary:
        fam = row["model"].split("_s")[0] if "_s" in row["model"] else row["model"]
        families.setdefault((fam, row["split"]), {}).setdefault(row["drop_frac"], []).append(row["mean"])
    for ax, split in zip(axes[0], splits):
        for fam in sorted({f for f, s in families if s == split}):
            fracs = sorted(families[(fam, split)])
            means = [float(np.mean(families[(fam, split)][fr])) for fr in fracs]
            ax.plot(fracs, means, marker="o", label=fam)
        ax.set_xlabel("drop fraction")
        ax.set_ylabel("size MSE (dropped frames)")
        ax.set_title(f"{split} drop protocol")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "drop_curves.png", dpi=160)
    plt.close(fig)

    (OUT_DIR / "drop_headline.json").write_text(json.dumps({
        "summary": summary, "wilcoxon": wilcox,
    }, indent=2, default=str))
    print(f"[drop] wrote {sum_path} {w_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
