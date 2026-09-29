"""Step 10 extrapolation: train on first 60% of timeline, score last 40%.

Loads prefix-trained checkpoints (h1_seed*_extrap60 and *-extrap60
baselines). Context is observations with t ≤ t0 + 0.6·span.
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
from ode.train_baselines import BaselineLightning, build_baseline_datasets
from ode.train_multi import MultiTrackLightning

OUT_DIR = REPO / "figures" / "h1_lock"
TRAIN_FRAC = 0.6
K = 3


def _parquet() -> Path:
    p = REPO / "metrics_with_features.parquet"
    return p if p.is_file() else REPO / "Thesis" / "metrics_with_features.parquet"


def _pred_ode(module, query_states, query_times):
    pred = module.model(query_states.unsqueeze(0), query_times, return_aux=False)
    if pred.dim() == 3:
        pred_b = pred[:, 0, :] if pred.shape[0] == query_states.shape[0] else pred[0]
    else:
        pred_b = pred
    return pred_b


def _pred_baseline(module, query_states, query_times):
    length = torch.tensor([query_states.shape[0]], device=query_states.device)
    pred = module.model.autoregressive_rollout(query_states.unsqueeze(0), query_times, length)
    return pred.squeeze(0)


def _tail_mse(pred_future, target_states):
    if pred_future.shape[-1] >= 4:
        pred_wh = pred_future[:, 2:4]
    else:
        pred_wh = pred_future[:, :2]
    target_wh = target_states[:, 2:4]
    n = min(pred_wh.shape[0], target_wh.shape[0])
    if n == 0:
        return float("nan")
    return float((pred_wh[:n] - target_wh[:n]).pow(2).mean().item())


def _score_track(predict_fn, states, t_abs, device):
    t = t_abs.detach().cpu().numpy()
    cut = float(t[0] + TRAIN_FRAC * (t[-1] - t[0]))
    ctx_idx = np.where(t <= cut + 1e-9)[0]
    if ctx_idx.size < K:
        ctx_idx = np.arange(min(K, states.shape[0]))
    tail_idx = np.where(t > cut + 1e-9)[0]
    if tail_idx.size == 0:
        return float("nan")
    context = torch.as_tensor(ctx_idx, device=device, dtype=torch.long)
    future = torch.as_tensor(tail_idx, device=device, dtype=torch.long)
    q_states, q_times = _build_context_only_query(
        states, t_abs, context_indices=context, prediction_indices=future,
    )
    pred = predict_fn(q_states, q_times)
    return _tail_mse(pred, states[future])


def _ckpt_epoch(path: Path) -> int:
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(path), map_location="cpu")
    return int(ckpt.get("epoch", -1))


def _discover(ckpt_dir: Path, run_tag: str = "h1_seed", n_seeds: int = 3, fail_before_epoch: int = 20) -> list[tuple[str, str, Path]]:
    found = []
    for seed in range(int(n_seeds)):
        d = ckpt_dir / f"{run_tag}{seed}_extrap60"
        cands = [p for p in d.glob("best*.ckpt") if p.is_file()] if d.is_dir() else []
        if cands:
            cands.sort(key=lambda p: (_ckpt_epoch(p), p.stat().st_mtime), reverse=True)
            epoch = _ckpt_epoch(cands[0])
            if epoch < fail_before_epoch:
                print(f"[extrap] TRAINING FAILURE skip ode_s{seed} epoch={epoch} {cands[0]}")
                continue
            found.append(("ode", f"ode_s{seed}", cands[0]))
    for p in sorted(ckpt_dir.glob("*_valmse_baseline-*-extrap60.ckpt")):
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
        found.append((model, f"{model}_s{seed}", p))
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
    denom = len(ids) * (len(ids) + 1)
    r_rb = float(1.0 - 2.0 * stat / denom) if denom else float("nan")
    sd = float(np.std(xa - xb, ddof=1))
    d = float(np.mean(xa - xb) / sd) if sd > 1e-12 else 0.0
    return {
        "n": len(ids), "p": float(p), "stat": float(stat),
        "ode_mean": float(xa.mean()), "other_mean": float(xb.mean()),
        "mean_diff": float((xa - xb).mean()),
        "rank_biserial": r_rb, "paired_cohens_d": d,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--ckpt-dir", type=Path, default=REPO / "checkpoints")
    p.add_argument("--run-tag", type=str, default="h1_seed")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--fail-before-epoch", type=int, default=20)
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
    splits = {"val": val_ds, "test": test_ds}

    models = []
    for family, key, path in _discover(
        args.ckpt_dir, args.run_tag, args.n_seeds, args.fail_before_epoch,
    ):
        try:
            if family == "ode":
                mod = MultiTrackLightning.load_from_checkpoint(str(path), map_location=device, strict=False)
                mod.eval().to(device)
                mod.horizon_start_frac = 1.0
                predict = lambda qs, qt, m=mod: _pred_ode(m, qs, qt)
            else:
                mod = BaselineLightning.load_from_checkpoint(str(path), map_location=device, strict=False)
                mod.eval().to(device)
                predict = lambda qs, qt, m=mod: _pred_baseline(m, qs, qt)
            models.append((key, predict))
            print(f"[extrap] loaded {key} {path}")
        except Exception as exc:
            print(f"[extrap] skip {path}: {exc}")

    long_rows = []
    for key, predict_fn in models:
        for split, ds in splits.items():
            for sample in ds:
                states = sample["states_5d"].to(device)
                t_abs = sample["t_absolute"].to(device)
                tid = int(sample["track_id"].item())
                if states.shape[0] <= K:
                    continue
                with torch.no_grad():
                    mse = _score_track(predict_fn, states, t_abs, device)
                long_rows.append({"model": key, "split": split, "track_id": tid, "size_mse": mse})

    long_path = OUT_DIR / "extrap_per_track.csv"
    with long_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "split", "track_id", "size_mse"])
        w.writeheader()
        w.writerows(long_rows)

    summary = []
    grouped: dict[tuple, list[float]] = {}
    for r in long_rows:
        if np.isfinite(r["size_mse"]):
            grouped.setdefault((r["model"], r["split"]), []).append(r["size_mse"])
    for (model, split), vals in sorted(grouped.items()):
        arr = np.array(vals)
        summary.append({
            "model": model, "split": split, "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr)),
        })
    sum_path = OUT_DIR / "extrap_summary.csv"
    with sum_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "split", "n", "mean", "std", "median"])
        w.writeheader()
        w.writerows(summary)

    wilcox = []
    by_model: dict[str, dict[str, list]] = {}
    for r in long_rows:
        by_model.setdefault(r["model"], {}).setdefault(r["split"], []).append(r)
    ode_keys = [m for m in by_model if m.startswith("ode_s")]
    others = [m for m in by_model if not m.startswith("ode")]
    # 3-seed mean per track as the model of record
    if ode_keys:
        for split in ("val", "test"):
            by_tid: dict[int, list[float]] = {}
            for ok in ode_keys:
                for r in by_model.get(ok, {}).get(split, []):
                    if np.isfinite(r["size_mse"]):
                        by_tid.setdefault(int(r["track_id"]), []).append(r["size_mse"])
            by_model.setdefault("ode_clean", {})[split] = [
                {"track_id": t, "size_mse": float(np.mean(v))} for t, v in by_tid.items()
            ]
        ode_keys = ["ode_clean"] + ode_keys
    for ok in ode_keys:
        for other in others:
            for split in ("val", "test"):
                if split not in by_model[ok] or split not in by_model[other]:
                    continue
                block = _wilcoxon(by_model[ok][split], by_model[other][split])
                block.update({"split": split, "ode": ok, "vs": other})
                wilcox.append(block)
    w_path = OUT_DIR / "extrap_wilcoxon.csv"
    if wilcox:
        with w_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(wilcox[0].keys()))
            w.writeheader()
            w.writerows(wilcox)

    (OUT_DIR / "extrap_headline.json").write_text(json.dumps({
        "train_time_frac": TRAIN_FRAC, "summary": summary, "wilcoxon": wilcox,
    }, indent=2, default=str))

    fig, ax = plt.subplots(figsize=(7, 3.6))
    families = {}
    for row in summary:
        fam = row["model"].split("_s")[0] if "_s" in row["model"] else row["model"]
        if fam == "ode_clean":
            continue
        families.setdefault((fam, row["split"]), []).append(row["mean"])
    cats = ["val", "test"]
    names = sorted({f for f, _s in families})
    x = np.arange(len(names))
    width = 0.35
    for i, split in enumerate(cats):
        vals = [float(np.mean(families.get((n, split), [np.nan]))) for n in names]
        ax.bar(x + (i - 0.5) * width, vals, width, label=f"{split} tail 40%")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("size MSE")
    ax.set_title("Extrapolation: train first 60% of timeline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "extrap_curves.png", dpi=160)
    plt.close(fig)

    print(f"[extrap] wrote {sum_path} {w_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
