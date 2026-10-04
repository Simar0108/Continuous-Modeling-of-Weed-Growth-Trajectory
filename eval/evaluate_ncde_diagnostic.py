"""PARTIAL NCDE diagnostic: score COMPLETE seeds only. Not D-017.

Writes figures/ncdediagnostic/. Does not write figures/ncde/ or figures/h1_lock.
No 5-seed means, no Wilcoxon, no beats_every_baseline.
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

import importlib.util

from ode.data.datamodule import PlantTrackDataModule
from ode.data.dataset import PlantTrackStateDataset
from ode.pathreg import collect_z0_and_dz, load_ode_module
from ode.repro import is_complete_ckpt
from ode.train_baselines import build_baseline_datasets, select_discrete_tracks


def _load_eval(name: str):
    path = REPO / "eval" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"REFUSE: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_h1 = _load_eval("evaluateh1")
_ex = _load_eval("evaluate_extrap")
_per_track_size_mse_ode = _h1._per_track_size_mse_ode
_summarize = _h1._summarize
_pred_ode = _ex._pred_ode
_score_track = _ex._score_track

LABEL = "PARTIAL n=3/n=2 diagnostic, not pre-registered protocol."
RUN_TAG = "h1_ncde_seed"


def _parquet() -> Path:
    p = REPO / "metrics_with_features.parquet"
    return p if p.is_file() else REPO / "Thesis" / "metrics_with_features.parquet"


def _complete_seeds(ckpt_dir: Path, extrap: bool) -> list[tuple[int, Path]]:
    found = []
    for seed in range(5):
        stem = f"{RUN_TAG}{seed}_extrap60" if extrap else f"{RUN_TAG}{seed}"
        d = ckpt_dir / stem
        if is_complete_ckpt(d):
            found.append((seed, d / "best.ckpt"))
        else:
            print(f"[ncde-diag] skip {'prefix-60' if extrap else 'in-window'} seed={seed}")
    return found


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {LABEL}\n")
        return
    with path.open("w", newline="") as f:
        f.write(f"# {LABEL}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _nfe_rows(ckpt_dir: Path, seed: int, extrap: bool) -> dict:
    stem = f"{RUN_TAG}{seed}_extrap60" if extrap else f"{RUN_TAG}{seed}"
    hist = ckpt_dir / stem / "nfe_history.jsonl"
    rows = []
    if hist.is_file():
        for line in hist.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return {
            "label": LABEL, "arm": "prefix-60" if extrap else "in-window",
            "seed": seed, "n_epochs": 0,
        }
    peaks = [float(r["peak"]) for r in rows]
    means = [float(r["ode_nfe_mean"]) for r in rows]
    streak = max_streak = 0
    for p in peaks:
        if p > 150:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return {
        "label": LABEL,
        "arm": "prefix-60" if extrap else "in-window",
        "seed": seed,
        "n_epochs": len(rows),
        "last_mean": means[-1],
        "last_peak": peaks[-1],
        "max_peak": max(peaks),
        "max_mean": max(means),
        "n_epochs_peak_gt_150": int(sum(1 for p in peaks if p > 150)),
        "max_consecutive_peak_gt_150": int(max_streak),
    }


def _plot_nfe(ckpt_dir: Path, out: Path, in_seeds: list[int], ex_seeds: list[int]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.4), sharex=False)
    specs = (
        (axes[0], in_seeds, False, "in-window"),
        (axes[1], ex_seeds, True, "prefix-60"),
    )
    for ax, seeds, extrap, title in specs:
        for seed in seeds:
            stem = f"{RUN_TAG}{seed}_extrap60" if extrap else f"{RUN_TAG}{seed}"
            hist = ckpt_dir / stem / "nfe_history.jsonl"
            if not hist.is_file():
                continue
            rows = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()]
            ep = [r["epoch"] for r in rows]
            ax.plot(ep, [r["ode_nfe_mean"] for r in rows], label=f"s{seed} mean")
            ax.plot(ep, [r["peak"] for r in rows], ls="--", alpha=0.7, label=f"s{seed} peak")
        ax.axhline(150, color="k", lw=1, ls=":", label="budget 150")
        ax.set_ylabel("ode_nfe")
        ax.set_title(f"{title} COMPLETE seeds")
        ax.legend(fontsize=7, ncol=2)
    axes[1].set_xlabel("epoch")
    fig.suptitle(LABEL, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--ckpt-dir", type=Path, default=REPO / "checkpoints")
    p.add_argument("--out-dir", type=Path, default=REPO / "figures" / "ncdediagnostic")
    args = p.parse_args()
    if args.out_dir.resolve() == (REPO / "figures" / "ncde").resolve():
        raise SystemExit("REFUSE: diagnostic must not write figures/ncde/")
    if "h1_lock" in str(args.out_dir):
        raise SystemExit("REFUSE: diagnostic must not write figures/h1_lock")
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "PARTIAL.txt").write_text(LABEL + "\n")
    print(f"[ncde-diag] {LABEL}")
    print(f"[ncde-diag] device={device} out={out}")

    in_ckpts = _complete_seeds(args.ckpt_dir, extrap=False)
    ex_ckpts = _complete_seeds(args.ckpt_dir, extrap=True)
    print(f"[ncde-diag] in-window COMPLETE {[s for s, _ in in_ckpts]}")
    print(f"[ncde-diag] prefix-60 COMPLETE {[s for s, _ in ex_ckpts]}")
    if not in_ckpts and not ex_ckpts:
        raise SystemExit("REFUSE: no COMPLETE NCDE seeds")

    parquet = _parquet()
    train_ids, _, _ = select_discrete_tracks(
        parquet, max_tracks=100, species="Maize",
        min_observations=15, val_frac=0.15, test_frac=0.15,
    )
    _transformer, _train_ds, val_ds, test_ds = build_baseline_datasets(
        parquet, sigma_mode="z_score", max_tracks=100, species="Maize",
        val_frac=0.15, test_frac=0.15,
    )
    splits = {"val": val_ds, "test": test_ds}
    dm = PlantTrackDataModule(
        parquet_path=parquet, batch_size=32, num_workers=0, sigma_mode="z_score",
    )
    dm.setup()
    z0_ds = PlantTrackStateDataset(
        dm.parquet_path, dm.transformer, track_ids=train_ids,
        valid_track_only=True, min_observations=8,
    )

    in_rows = []
    z0_rows = []
    for seed, path in in_ckpts:
        module = load_ode_module(path, device, strict=False)
        per_split = {}
        for split, ds in splits.items():
            rows = _per_track_size_mse_ode(module, ds, device)
            sm = _summarize(rows)
            per_split[split] = sm
            print(f"[ncde-diag] in-window seed={seed} {split} mean={sm['mean']:.6f} n={sm['n']}")
        diag_path = path.parent / "diagnostics.json"
        diag = json.loads(diag_path.read_text()) if diag_path.is_file() else {}
        z0_fit = collect_z0_and_dz(module, z0_ds, device)
        z0_fit.pop("z0", None)
        z0_fit.pop("dz", None)
        in_rows.append({
            "label": LABEL,
            "arm": "in-window",
            "seed": seed,
            "path": str(path),
            "val_mse": per_split["val"]["mean"],
            "val_std": per_split["val"]["std"],
            "val_n": per_split["val"]["n"],
            "test_mse": per_split["test"]["mean"],
            "test_std": per_split["test"]["std"],
            "test_n": per_split["test"]["n"],
            "train_dz_dt_size_std": z0_fit.get("train_dz_dt_size_std"),
            "z0_cosine_mean": z0_fit.get("z0_cosine", {}).get("mean_offdiag"),
            "fit_train_dz_dt_size_std": diag.get("train_dz_dt_size_std"),
            "fit_z0_cosine_mean": (diag.get("z0_cosine") or {}).get("mean_offdiag"),
        })
        z0_rows.append({
            "label": LABEL, "arm": "in-window", "seed": seed,
            "z0_cosine_mean": z0_fit.get("z0_cosine", {}).get("mean_offdiag"),
            "train_dz_dt_size_std": z0_fit.get("train_dz_dt_size_std"),
            "dz_dt_mean_norm": z0_fit.get("dz_dt_mean_norm"),
            "n_tracks": z0_fit.get("n_tracks"),
        })

    ex_rows = []
    for seed, path in ex_ckpts:
        module = load_ode_module(path, device, strict=False)
        predict = lambda qs, qt, m=module: _pred_ode(m, qs, qt)
        per_split = {}
        for split, ds in splits.items():
            mses = []
            for sample in ds:
                states = sample["states_5d"].to(device)
                t_abs = sample["t_absolute"].to(device)
                if states.shape[0] <= 3:
                    continue
                with torch.no_grad():
                    mse = _score_track(predict, states, t_abs, device)
                if np.isfinite(mse):
                    mses.append(mse)
            arr = np.array(mses, dtype=float)
            sm = {
                "n": int(arr.size),
                "mean": float(arr.mean()) if arr.size else float("nan"),
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            }
            per_split[split] = sm
            print(
                f"[ncde-diag] prefix-60 TAIL seed={seed} {split} "
                f"mean={sm['mean']:.6f} n={sm['n']}"
            )
        diag_path = path.parent / "diagnostics.json"
        diag = json.loads(diag_path.read_text()) if diag_path.is_file() else {}
        z0_fit = collect_z0_and_dz(module, z0_ds, device)
        z0_fit.pop("z0", None)
        z0_fit.pop("dz", None)
        ex_rows.append({
            "label": LABEL,
            "arm": "prefix-60-tail",
            "seed": seed,
            "path": str(path),
            "val_tail_mse": per_split["val"]["mean"],
            "val_tail_std": per_split["val"]["std"],
            "val_n": per_split["val"]["n"],
            "test_tail_mse": per_split["test"]["mean"],
            "test_tail_std": per_split["test"]["std"],
            "test_n": per_split["test"]["n"],
            "train_dz_dt_size_std": z0_fit.get("train_dz_dt_size_std"),
            "z0_cosine_mean": z0_fit.get("z0_cosine", {}).get("mean_offdiag"),
            "fit_train_dz_dt_size_std": diag.get("train_dz_dt_size_std"),
            "fit_z0_cosine_mean": (diag.get("z0_cosine") or {}).get("mean_offdiag"),
        })
        z0_rows.append({
            "label": LABEL, "arm": "prefix-60", "seed": seed,
            "z0_cosine_mean": z0_fit.get("z0_cosine", {}).get("mean_offdiag"),
            "train_dz_dt_size_std": z0_fit.get("train_dz_dt_size_std"),
            "dz_dt_mean_norm": z0_fit.get("dz_dt_mean_norm"),
            "n_tracks": z0_fit.get("n_tracks"),
        })

    nfe_rows = (
        [_nfe_rows(args.ckpt_dir, s, False) for s, _ in in_ckpts]
        + [_nfe_rows(args.ckpt_dir, s, True) for s, _ in ex_ckpts]
    )
    _write_csv(out / "inwindow_per_seed.csv", in_rows)
    _write_csv(out / "extrap_tail_per_seed.csv", ex_rows)
    _write_csv(out / "z0_dz_per_seed.csv", z0_rows)
    _write_csv(out / "nfe_per_seed.csv", nfe_rows)
    _plot_nfe(
        args.ckpt_dir, out / "nfe_trajectories.png",
        [s for s, _ in in_ckpts], [s for s, _ in ex_ckpts],
    )

    payload = {
        "label": LABEL,
        "protocol": "not D-017",
        "in_window_complete": [s for s, _ in in_ckpts],
        "prefix60_complete": [s for s, _ in ex_ckpts],
        "in_window": in_rows,
        "prefix60_tail": ex_rows,
        "z0_dz": z0_rows,
        "nfe": nfe_rows,
        "wilcoxon": None,
        "five_seed_mean": None,
        "beats_every_baseline": None,
        "lstm_prefix60_tail_reference_not_a_test": {
            "note": "D-017 pre-registered LSTM clean tail mean; not recomputed here",
            "per_seed": [0.969, 1.133, 0.940],
            "mean": 1.014,
        },
        "lstm_inwindow_test_band_reference_not_a_test": [0.194, 0.246],
    }
    (out / "headline.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"[ncde-diag] wrote {out}")
    print(json.dumps({
        "label": LABEL,
        "in_window": [
            {"seed": r["seed"], "val": r["val_mse"], "test": r["test_mse"]}
            for r in in_rows
        ],
        "prefix60_tail": [
            {"seed": r["seed"], "val_tail": r["val_tail_mse"], "test_tail": r["test_tail_mse"]}
            for r in ex_rows
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
