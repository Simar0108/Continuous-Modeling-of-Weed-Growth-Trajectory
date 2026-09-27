"""H1 lock eval: frozen Neural ODE vs LSTM/GRU/Transformer and Richards NLS.

Full-horizon per-track size MSE on the official Maize 100-track 70/15/15
split. No synthetic frame drop. Wilcoxon + effect sizes on val and test.
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
import pandas as pd
import torch
from scipy.stats import wilcoxon

from ode.data.datamodule import _collate_variable_length_tracks
from ode.data.dataset import PlantTrackStateDataset
from ode.train_baselines import BaselineLightning, build_baseline_datasets, select_discrete_tracks
from ode.train_multi import MultiTrackLightning
from tests.diagnose_growth_form import fit_richards, richards_curve

H1_CKPT = REPO / "checkpoints" / "h1_final_best" / "best.ckpt"
H1_SHA256 = "2498d033ea02e2df1e312a58179226b649a9d3d5c7143b2d040fc9341ef86222"
OUT_DIR = REPO / "figures" / "h1_lock"


def _parquet() -> Path:
    p = REPO / "metrics_with_features.parquet"
    return p if p.is_file() else REPO / "Thesis" / "metrics_with_features.parquet"


def _t_vec(batch: dict) -> torch.Tensor:
    t_abs = batch["t_absolute"]
    lengths = batch.get("track_lengths")
    if lengths is not None and t_abs.dim() > 1:
        return t_abs[int(lengths.argmax().item())]
    return t_abs[0] if t_abs.dim() > 1 else t_abs


def _per_track_size_mse_ode(
    module, dataset, device: torch.device, *, post_context: bool = False, k: int = 3,
) -> list[dict]:
    module.eval()
    rows = []
    with torch.no_grad():
        for sample in dataset:
            states = sample["states_5d"].to(device)
            t_abs = sample["t_absolute"].to(device)
            tid = int(sample["track_id"].item())
            track_ids = sample.get("track_ids")
            if track_ids is None and "track_id" in sample:
                track_ids = sample["track_id"].reshape(1)
            track_ids = track_ids.to(device) if torch.is_tensor(track_ids) else None
            pred = module.model(states.unsqueeze(0), t_abs, return_aux=False, track_ids=track_ids)
            if pred.dim() == 3:
                # (T, B, D) or (B, T, D)
                if pred.shape[0] == states.shape[0]:
                    pred_b = pred[:, 0, :]
                else:
                    pred_b = pred[0]
            else:
                pred_b = pred
            err = (pred_b[:, 2:4] - states[:, 2:4]).pow(2)
            if post_context and states.shape[0] > k:
                err = err[k:]
            mse = float(err.mean().item())
            rows.append({"track_id": tid, "n_frames": int(states.shape[0]), "size_mse": mse})
    return rows


def _per_track_size_mse_baseline(module, dataset, device: torch.device) -> list[dict]:
    module.eval()
    k = int(getattr(module, "n_context_frames", 3))
    rows = []
    with torch.no_grad():
        for sample in dataset:
            states = sample["states_5d"].to(device)
            t_abs = sample["t_absolute"].to(device)
            tid = int(sample["track_id"].item())
            length = torch.tensor([states.shape[0]], device=device)
            pred = module.model.autoregressive_rollout(states.unsqueeze(0), t_abs, length)
            pred = pred.squeeze(0)
            target = states[k:, 2:4]
            if pred.shape[0] != target.shape[0]:
                n = min(pred.shape[0], target.shape[0])
                pred, target = pred[:n], target[:n]
            mse = float((pred[:, :2] - target).pow(2).mean().item())
            rows.append({"track_id": tid, "n_frames": int(states.shape[0]), "size_mse": mse})
    return rows


def _zscore_sigma(transformer, sw: np.ndarray, sh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sw_z = (sw - transformer._mean_sigma_w) / transformer._std_sigma_w
    sh_z = (sh - transformer._mean_sigma_h) / transformer._std_sigma_h
    return sw_z, sh_z


def _per_track_nls(parquet: Path, transformer, track_ids: list[int]) -> list[dict]:
    df = pd.read_parquet(parquet)
    if "valid_track" in df.columns:
        df = df[df["valid_track"]]
    rows = []
    for tid in track_ids:
        g = df[df["track_id"] == tid].sort_values("time_since_germination_hours")
        if g.empty:
            continue
        t = g["time_since_germination_hours"].to_numpy(dtype=float)
        sw = (g["width"].to_numpy(dtype=float) / 4.0)
        sh = (g["height"].to_numpy(dtype=float) / 4.0)
        fw = fit_richards(t, sw)
        fh = fit_richards(t, sh)
        t0 = t - t[0]
        pred_w = richards_curve(t0, fw["K"], fw["r"], fw["nu"], fw["z0"]) if fw["ok"] else np.full_like(sw, np.nan)
        pred_h = richards_curve(t0, fh["K"], fh["r"], fh["nu"], fh["z0"]) if fh["ok"] else np.full_like(sh, np.nan)
        sw_z, sh_z = _zscore_sigma(transformer, sw, sh)
        pw_z, ph_z = _zscore_sigma(transformer, pred_w, pred_h)
        mse = float(np.nanmean(np.stack([(pw_z - sw_z) ** 2, (ph_z - sh_z) ** 2], axis=-1)))
        rows.append({
            "track_id": int(tid), "n_frames": int(len(t)), "size_mse": mse,
            "nls_ok": bool(fw["ok"] and fh["ok"]),
        })
    return rows


def _summarize(rows: list[dict]) -> dict:
    xs = np.array([r["size_mse"] for r in rows], dtype=float)
    xs = xs[np.isfinite(xs)]
    return {
        "n": int(xs.size),
        "mean": float(np.mean(xs)) if xs.size else float("nan"),
        "std": float(np.std(xs, ddof=1)) if xs.size > 1 else 0.0,
        "median": float(np.median(xs)) if xs.size else float("nan"),
        "max": float(np.max(xs)) if xs.size else float("nan"),
    }


def _rank_biserial(stat: float, n: int) -> float:
    # Wilcoxon W is the sum of ranks of positive diffs (ours - ref) when
    # alternative='less' uses the standard scipy convention. Rank-biserial
    # r = 1 - 2W / (n(n+1)). Negative r means ODE worse if W is large.
    denom = n * (n + 1)
    if denom == 0:
        return float("nan")
    return float(1.0 - 2.0 * stat / denom)


def _paired_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    sd = float(np.std(diff, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diff) / sd)


def _wilcoxon_block(ode_rows: list[dict], other_rows: list[dict], split: str, other: str) -> dict:
    ode = {r["track_id"]: r["size_mse"] for r in ode_rows}
    alt = {r["track_id"]: r["size_mse"] for r in other_rows}
    ids = sorted(set(ode) & set(alt))
    a = np.array([ode[i] for i in ids], dtype=float)
    b = np.array([alt[i] for i in ids], dtype=float)
    n = len(ids)
    report = {
        "split": split,
        "vs": other,
        "n": n,
        "ode_mean": float(a.mean()) if n else float("nan"),
        "other_mean": float(b.mean()) if n else float("nan"),
        "mean_diff_ode_minus_other": float((a - b).mean()) if n else float("nan"),
    }
    if n >= 6:
        stat, p = wilcoxon(a, b, alternative="less")
        report.update({
            "wilcoxon_stat": float(stat),
            "p": float(p),
            "rank_biserial": _rank_biserial(float(stat), n),
            "paired_cohens_d": _paired_d(a, b),
        })
    else:
        report.update({
            "wilcoxon_stat": float("nan"), "p": float("nan"),
            "rank_biserial": float("nan"), "paired_cohens_d": float("nan"),
        })
    return report


def _discover_baselines(ckpt_dir: Path) -> list[tuple[str, int, Path]]:
    found: list[tuple[str, int, Path]] = []
    for p in sorted(ckpt_dir.glob("*_valmse_baseline-*.ckpt")):
        if "-extrap" in p.stem:
            continue
        # lstm_valmse_baseline-lstm-100t-maize-s0.ckpt
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
        found.append((model, seed, p))
    return found


def _discover_ode_seeds(ckpt_dir: Path) -> list[tuple[str, Path]]:
    """Clean 70/15/15 ODE seeds. Does not include h1_final_best."""
    found: list[tuple[str, Path]] = []
    for seed in (0, 1, 2):
        d = ckpt_dir / f"h1_seed{seed}"
        if not d.is_dir():
            continue
        cands = sorted(d.glob("best*.ckpt"))
        if cands:
            found.append((f"ode_s{seed}", cands[0]))
    return found


def _load_ode(path: Path, device: torch.device):
    module = MultiTrackLightning.load_from_checkpoint(
        str(path), map_location=device, strict=False,
    )
    module.eval().to(device)
    module.horizon_start_frac = 1.0
    module.horizon_ramp_start = 0
    module.horizon_ramp_end = 0
    return module


def _plot_sampling(parquet: Path, track_ids: list[int], out: Path) -> None:
    df = pd.read_parquet(parquet)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, tid in enumerate(track_ids[:3]):
        g = df[df["track_id"] == tid].sort_values("time_since_germination_hours")
        t = g["time_since_germination_hours"].to_numpy(dtype=float)
        ax.eventplot(t, lineoffsets=i, linelengths=0.7)
    ax.set_yticks(range(min(3, len(track_ids))))
    ax.set_yticklabels([f"track {tid}" for tid in track_ids[:3]])
    ax.set_xlabel("hours since germination")
    ax.set_title("Irregular sampling (3 tracks)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_curves(
    parquet: Path,
    transformer,
    ode_module,
    baseline_module,
    baseline_name: str,
    dataset,
    track_ids: list[int],
    device: torch.device,
    out: Path,
) -> None:
    df = pd.read_parquet(parquet)
    k = int(getattr(baseline_module, "n_context_frames", 3)) if baseline_module is not None else 3
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    axes = axes.ravel()
    samples = {int(s["track_id"].item()): s for s in dataset}
    for ax, tid in zip(axes, track_ids[:4]):
        g = df[df["track_id"] == tid].sort_values("time_since_germination_hours")
        t = g["time_since_germination_hours"].to_numpy(dtype=float)
        sw = (g["width"].to_numpy(dtype=float) / 4.0)
        sw_z, _ = _zscore_sigma(transformer, sw, sw)
        ax.plot(t, sw_z, "k.", ms=4, label="actual")
        sample = samples.get(int(tid))
        if sample is not None:
            states = sample["states_5d"].to(device)
            t_abs = sample["t_absolute"].to(device)
            with torch.no_grad():
                pred = ode_module.model(states.unsqueeze(0), t_abs, return_aux=False)
                if pred.dim() == 3:
                    pred_b = pred[:, 0, :] if pred.shape[0] == states.shape[0] else pred[0]
                else:
                    pred_b = pred
            ax.plot(t, pred_b[:, 2].cpu().numpy(), label="Neural ODE")
            if baseline_module is not None:
                length = torch.tensor([states.shape[0]], device=device)
                bpred = baseline_module.model.autoregressive_rollout(
                    states.unsqueeze(0), t_abs, length,
                ).squeeze(0)
                ax.plot(
                    t[k:k + bpred.shape[0]],
                    bpred[:, 0].detach().cpu().numpy(),
                    label=baseline_name,
                )
        ax.set_title(f"track {tid}")
        ax.set_ylabel("z-scored σ_w")
        ax.set_xlabel("hours")
    axes[0].legend(fontsize=8)
    fig.suptitle("Growth curves, horizon 1.0")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _average_tracks(row_lists: list[list[dict]]) -> list[dict]:
    by_id: dict[int, list[dict]] = {}
    for rows in row_lists:
        for r in rows:
            by_id.setdefault(int(r["track_id"]), []).append(r)
    out = []
    for tid, rs in sorted(by_id.items()):
        out.append({
            "track_id": tid,
            "n_frames": int(rs[0]["n_frames"]),
            "size_mse": float(np.mean([x["size_mse"] for x in rs])),
        })
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--ckpt-dir", type=Path, default=REPO / "checkpoints")
    args = p.parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    parquet = _parquet()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids, test_ids = select_discrete_tracks(
        parquet, max_tracks=100, species="Maize",
        min_observations=15, val_frac=0.15, test_frac=0.15,
    )
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise SystemExit(f"REFUSE: train/test overlap {sorted(overlap)}")
    transformer, train_ds, val_ds, test_ds = build_baseline_datasets(
        parquet, sigma_mode="z_score", max_tracks=100, species="Maize",
        val_frac=0.15, test_frac=0.15,
    )
    splits = {"train": (train_ids, train_ds), "val": (val_ids, val_ds), "test": (test_ids, test_ds)}
    print(f"[evaluateh1] split train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    print(f"[evaluateh1] test n_frames={[int(s['states_5d'].shape[0]) for s in test_ds]}")

    models: dict[str, dict] = {}
    ode_full: dict[str, dict] = {}
    ode_post: dict[str, dict] = {}

    leaked = _load_ode(H1_CKPT, device)
    for split, (_ids, ds) in splits.items():
        ode_full.setdefault("ode_leaked", {})[split] = _per_track_size_mse_ode(leaked, ds, device)
        ode_post.setdefault("ode_leaked", {})[split] = _per_track_size_mse_ode(
            leaked, ds, device, post_context=True,
        )
        models.setdefault("ode_leaked", {})[split] = ode_full["ode_leaked"][split]

    ode_seed_ckpts = _discover_ode_seeds(args.ckpt_dir)
    print(f"[evaluateh1] discovered {len(ode_seed_ckpts)} clean ODE seeds")
    if len(ode_seed_ckpts) < 3:
        raise SystemExit(f"REFUSE: need 3 clean ODE seeds, found {len(ode_seed_ckpts)}")
    first_clean = None
    for key, path in ode_seed_ckpts:
        try:
            ode_mod = _load_ode(path, device)
        except Exception as exc:
            print(f"[evaluateh1] skip {path}: {exc}")
            continue
        if first_clean is None:
            first_clean = ode_mod
        for split, (_ids, ds) in splits.items():
            full = _per_track_size_mse_ode(ode_mod, ds, device)
            post = _per_track_size_mse_ode(ode_mod, ds, device, post_context=True)
            models.setdefault(key, {})[split] = full
            ode_full.setdefault(key, {})[split] = full
            ode_post.setdefault(key, {})[split] = post

    seed_keys = [m for m in models if m.startswith("ode_s")]
    for split in splits:
        models.setdefault("ode_clean", {})[split] = _average_tracks(
            [models[k][split] for k in seed_keys if split in models[k]]
        )

    nls_rows = {}
    for split, (ids, _ds) in splits.items():
        nls_rows[split] = _per_track_nls(parquet, transformer, ids)
        models.setdefault("nls_insample", {})[split] = nls_rows[split]

    baselines = _discover_baselines(args.ckpt_dir)
    print(f"[evaluateh1] discovered {len(baselines)} baseline ckpts")
    for model, seed, path in baselines:
        key = f"{model}_s{seed}"
        try:
            bl = BaselineLightning.load_from_checkpoint(str(path), map_location=device, strict=False)
            bl.eval().to(device)
        except Exception as exc:
            print(f"[evaluateh1] skip {path}: {exc}")
            continue
        for split, (_ids, ds) in splits.items():
            models.setdefault(key, {})[split] = _per_track_size_mse_baseline(bl, ds, device)
    for family in ("lstm", "gru", "transformer"):
        fam_keys = [m for m in models if m.startswith(f"{family}_s")]
        if not fam_keys:
            continue
        for split in splits:
            models.setdefault(f"{family}_clean", {})[split] = _average_tracks(
                [models[k][split] for k in fam_keys if split in models[k]]
            )

    table1 = []
    long_rows = []
    for model, by_split in models.items():
        seed = "clean" if model.endswith("_clean") else (
            "leaked" if model == "ode_leaked" else (model.split("_s")[-1] if "_s" in model else "nls")
        )
        for split, rows in by_split.items():
            sm = _summarize(rows)
            table1.append({"model": model, "split": split, "seed": seed, **sm})
            for r in rows:
                long_rows.append({"model": model, "split": split, **r})
    # Across-seed mean±std of seed-level split means (ode_clean extra row kind)
    for split in ("val", "test", "train"):
        seed_stats = [_summarize(models[k][split]) for k in seed_keys if split in models[k]]
        if not seed_stats:
            continue
        means = [s["mean"] for s in seed_stats]
        medians = [s["median"] for s in seed_stats]
        table1.append({
            "model": "ode_clean_seedmean", "split": split, "seed": "0-2",
            "n": len(means),
            "mean": float(np.mean(means)),
            "std": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
            "median": float(np.mean(medians)),
            "max": float("nan"),
        })

    t1_path = OUT_DIR / "table1_v2.csv"
    _write_csv(t1_path, table1)
    _write_csv(OUT_DIR / "table1_model_split.csv", table1)
    _write_csv(OUT_DIR / "per_track_size_mse.csv", long_rows)

    # Table 2 v2: 3-seed-mean-per-track vs each baseline seed + families + NLS
    table2 = []
    others = [m for m in models if not m.startswith("ode")]
    for other in others:
        for split in ("val", "test"):
            if "ode_clean" not in models or split not in models["ode_clean"] or split not in models[other]:
                continue
            block = _wilcoxon_block(models["ode_clean"][split], models[other][split], split, other)
            block["ode"] = "ode_clean"
            table2.append(block)
    t2_path = OUT_DIR / "table2_v2.csv"
    _write_csv(t2_path, table2)

    table2_per_seed = []
    for ode_key in seed_keys + (["ode_leaked"] if "ode_leaked" in models else []):
        for other in others:
            for split in ("val", "test"):
                if split not in models[ode_key] or split not in models[other]:
                    continue
                block = _wilcoxon_block(models[ode_key][split], models[other][split], split, other)
                block["ode"] = ode_key
                table2_per_seed.append(block)
    _write_csv(OUT_DIR / "table2_per_seed.csv", table2_per_seed)
    _write_csv(OUT_DIR / "table2_wilcoxon.csv", table2)

    # Matched-rollout: full minus post-context (negative => context frames easier)
    matched = []
    for key, by_split in ode_full.items():
        for split, full_rows in by_split.items():
            post_rows = ode_post[key][split]
            full_m = {r["track_id"]: r["size_mse"] for r in full_rows}
            post_m = {r["track_id"]: r["size_mse"] for r in post_rows}
            deltas = [{"track_id": tid, "size_mse": full_m[tid] - post_m[tid]} for tid in full_m if tid in post_m]
            sm = _summarize(deltas)
            fsm = _summarize(full_rows)
            psm = _summarize(post_rows)
            matched.append({
                "model": key, "split": split,
                "full_mean": fsm["mean"], "full_median": fsm["median"],
                "post_context_mean": psm["mean"], "post_context_median": psm["median"],
                "delta_full_minus_post_mean": sm["mean"],
                "delta_full_minus_post_median": sm["median"],
                "n": sm["n"],
            })
    _write_csv(OUT_DIR / "matched_rollout.csv", matched)

    stochastic = [m for m in others if not m.startswith("nls") and not m.endswith("_clean")]
    seed_means = {}
    for split in ("val", "test"):
        vals = [_summarize(models[k][split])["mean"] for k in seed_keys if split in models[k]]
        seed_means[split] = {
            "n_seeds": len(vals),
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "median": float(np.median(vals)) if vals else float("nan"),
        }
    leaked_means = {
        split: _summarize(models["ode_leaked"][split])
        for split in ("val", "test") if "ode_leaked" in models and split in models["ode_leaked"]
    }
    headline = {
        "beats_every_baseline": True,
        "notes": [],
        "ode_leaked_excluded_from_mean": True,
        "ode_3seed": seed_means,
        "ode_leaked": leaked_means,
        "contamination_val": leaked_means.get("val", {}).get("mean", float("nan")) - seed_means.get("val", {}).get("mean", float("nan")),
        "contamination_test": leaked_means.get("test", {}).get("mean", float("nan")) - seed_means.get("test", {}).get("mean", float("nan")),
        "matched_rollout": matched,
    }
    for other in stochastic:
        for split in ("val", "test"):
            block = next((b for b in table2 if b.get("ode") == "ode_clean" and b["vs"] == other and b["split"] == split), None)
            if block is None or not (block.get("p", 1) < 0.05 and block.get("mean_diff_ode_minus_other", 1) < 0):
                headline["beats_every_baseline"] = False
                headline["notes"].append(f"fail ode_clean vs {other} {split}")
    (OUT_DIR / "headline.json").write_text(json.dumps({"h1_sha256": H1_SHA256, **headline, "table2": table2}, indent=2, default=str))

    _plot_sampling(parquet, test_ids[:3], OUT_DIR / "irregular_sampling.png")
    tes = sorted(models["ode_clean"]["test"], key=lambda r: r["size_mse"])
    pick = [tes[0]["track_id"], tes[len(tes) // 3]["track_id"], tes[2 * len(tes) // 3]["track_id"], tes[-1]["track_id"]]
    best_bl_name, best_bl = None, None
    best_mean = float("inf")
    for model, seed, path in baselines:
        key = f"{model}_s{seed}"
        if key in models and "val" in models[key]:
            m = _summarize(models[key]["val"])["mean"]
            if m < best_mean:
                best_mean = m
                best_bl_name = key
                try:
                    best_bl = BaselineLightning.load_from_checkpoint(str(path), map_location=device, strict=False)
                    best_bl.eval().to(device)
                except Exception:
                    best_bl = None
    _plot_curves(
        parquet, transformer, first_clean or leaked, best_bl, best_bl_name or "baseline",
        test_ds, pick, device, OUT_DIR / "growth_curves.png",
    )
    print(f"[evaluateh1] table1_v2 → {t1_path}")
    print(f"[evaluateh1] table2_v2 → {t2_path}")
    print(f"[evaluateh1] matched_rollout → {OUT_DIR / 'matched_rollout.csv'}")
    print(f"[evaluateh1] headline beats_every_baseline={headline['beats_every_baseline']}")
    print(json.dumps(table1, indent=2))


if __name__ == "__main__":
    main()
