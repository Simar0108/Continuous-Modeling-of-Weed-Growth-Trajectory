"""D-019 Probe 1: does K=3 context contain track identity?

No ODE training. Writes figures/encoder_probes/. Chance bars are in D-019.
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

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ode.data.datamodule import PlantTrackDataModule
from ode.data.dataset import PlantTrackStateDataset
from ode.train_baselines import select_discrete_tracks

K = 3
LABEL = "D-019 encoder probe. Not a lock headline."


def _parquet() -> Path:
    p = REPO / "metrics_with_features.parquet"
    return p if p.is_file() else REPO / "Thesis" / "metrics_with_features.parquet"


def context_feature(states: np.ndarray, t: np.ndarray, start: int = 0) -> np.ndarray:
    """K frames of [sigma_w, sigma_h, delta_t_hours]."""
    sl = slice(start, start + K)
    sw = states[sl, 2]
    sh = states[sl, 3]
    tt = t[sl]
    dt = np.zeros(K, dtype=np.float64)
    dt[1:] = np.diff(tt)
    return np.stack([sw, sh, dt], axis=1).reshape(-1)


def late_targets(states: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    size = 0.5 * (states[:, 2] + states[:, 3])
    final_size = float(size[-1])
    if size.shape[0] < 2:
        return final_size, 0.0
    inc = np.diff(size)
    i = int(np.argmax(inc)) + 1
    span = float(t[-1] - t[0])
    t_norm = float((t[i] - t[0]) / span) if span > 1e-8 else 0.0
    return final_size, t_norm


def _collect(ds) -> list[dict]:
    rows = []
    for sample in ds:
        states = sample["states_5d"].numpy()
        t = sample["t_absolute"].numpy()
        tid = int(sample["track_id"].item())
        if states.shape[0] < K:
            continue
        feat = context_feature(states, t, 0)
        final_size, t_max = late_targets(states, t)
        n = int(states.shape[0])
        cut = max(K, int(np.ceil(0.20 * n)))
        windows = []
        for start in range(0, max(1, cut - K + 1)):
            windows.append(context_feature(states, t, start))
        rows.append({
            "track_id": tid,
            "feat": feat,
            "final_size": final_size,
            "t_max_growth": t_max,
            "windows": windows,
            "split": None,
        })
    return rows


def _fit_cls(X, y, seed: int, kind: str):
    if kind == "logreg":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=400, solver="lbfgs",
                random_state=seed,
            ),
        )
    else:
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32,), activation="relu",
                max_iter=400, random_state=seed,
            ),
        )
    clf.fit(X, y)
    return clf


def _fit_reg(X, y, seed: int, kind: str):
    if kind == "ridge":
        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    else:
        reg = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(32,), activation="relu",
                max_iter=400, random_state=seed,
            ),
        )
    reg.fit(X, y)
    return reg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=REPO / "figures" / "encoder_probes")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    parquet = _parquet()
    train_ids, val_ids, test_ids = select_discrete_tracks(
        parquet, max_tracks=100, species="Maize",
        min_observations=15, val_frac=0.15, test_frac=0.15,
    )
    split_of = {int(i): "train" for i in train_ids}
    split_of.update({int(i): "val" for i in val_ids})
    split_of.update({int(i): "test" for i in test_ids})
    all_ids = list(train_ids) + list(val_ids) + list(test_ids)
    dm = PlantTrackDataModule(
        parquet_path=parquet, batch_size=32, num_workers=0, sigma_mode="z_score",
    )
    dm.setup()
    ds = PlantTrackStateDataset(
        dm.parquet_path, dm.transformer, track_ids=all_ids,
        valid_track_only=True, min_observations=8,
    )
    rows = _collect(ds)
    for r in rows:
        r["split"] = split_of.get(int(r["track_id"]), "train")
    n_tracks = len(rows)
    chance = 1.0 / n_tracks if n_tracks else float("nan")
    print(f"[probe1] n_tracks={n_tracks} chance={chance:.4f} {LABEL}")

    X_off = np.stack([r["feat"] for r in rows])
    y_id = np.array([r["track_id"] for r in rows])
    id_in = {}
    for kind in ("logreg", "mlp"):
        clf = _fit_cls(X_off, y_id, args.seed, kind)
        acc = float(accuracy_score(y_id, clf.predict(X_off)))
        id_in[kind] = acc
        print(f"[probe1] 1a official in-sample {kind} acc={acc:.4f} chance={chance:.4f}")

    win_X, win_y = [], []
    for r in rows:
        for w in r["windows"]:
            win_X.append(w)
            win_y.append(r["track_id"])
    win_X = np.stack(win_X)
    win_y = np.array(win_y)
    Xtr, Xte, ytr, yte = train_test_split(
        win_X, win_y, test_size=0.30, random_state=args.seed, stratify=win_y,
    )
    id_early = {}
    for kind in ("logreg", "mlp"):
        clf = _fit_cls(Xtr, ytr, args.seed, kind)
        tr = float(accuracy_score(ytr, clf.predict(Xtr)))
        te = float(accuracy_score(yte, clf.predict(Xte)))
        id_early[kind] = {"train": tr, "test": te, "n_train": int(len(ytr)), "n_test": int(len(yte))}
        print(f"[probe1] 1a early-window {kind} train={tr:.4f} test={te:.4f}")

    def split_xy(target: str):
        X = {s: [] for s in ("train", "val", "test")}
        y = {s: [] for s in ("train", "val", "test")}
        for r in rows:
            X[r["split"]].append(r["feat"])
            y[r["split"]].append(r[target])
        return {s: np.stack(X[s]) for s in X}, {s: np.array(y[s], dtype=np.float64) for s in y}

    late = {}
    for target in ("final_size", "t_max_growth"):
        Xs, ys = split_xy(target)
        late[target] = {}
        for kind in ("ridge", "mlp"):
            reg = _fit_reg(Xs["train"], ys["train"], args.seed, kind)
            block = {}
            for split in ("train", "val", "test"):
                pred = reg.predict(Xs[split])
                block[split] = {
                    "r2": float(r2_score(ys[split], pred)),
                    "n": int(ys[split].size),
                }
            late[target][kind] = block
            print(
                f"[probe1] 1b {target} {kind} "
                f"val_R2={block['val']['r2']:.4f} test_R2={block['test']['r2']:.4f}"
            )

    best_in = max(id_in.values())
    best_early_te = max(v["test"] for v in id_early.values())
    a_succeed = best_in >= 0.20 or best_early_te >= 0.10
    a_fail = best_in < 0.10 and best_early_te < 0.05
    best_val_r2 = max(
        late[t][k]["val"]["r2"] for t in late for k in late[t]
    )
    b_succeed = best_val_r2 >= 0.20
    b_fail = all(late[t][k]["val"]["r2"] <= 0.05 for t in late for k in late[t])
    p1_succeed = bool(a_succeed or b_succeed)
    p1_chance = bool(a_fail and b_fail)
    verdict = "PROBE1_CHANCE_FAIL" if p1_chance else (
        "PROBE1_SUCCEED" if p1_succeed else "PROBE1_INCONCLUSIVE"
    )
    payload = {
        "label": LABEL,
        "n_tracks": n_tracks,
        "chance_1a": chance,
        "official_in_sample_acc": id_in,
        "early_window": id_early,
        "late_stage": late,
        "best_official_in_sample_acc": best_in,
        "best_early_window_test_acc": best_early_te,
        "best_val_r2": best_val_r2,
        "1a_succeed": a_succeed,
        "1a_chance_fail": a_fail,
        "1b_succeed": b_succeed,
        "1b_chance_fail": b_fail,
        "probe1_succeed": p1_succeed,
        "probe1_chance_fail": p1_chance,
        "verdict": verdict,
        "d019_bars": {
            "1a_succeed": "in_sample>=0.20 or early_test>=0.10",
            "1a_fail": "in_sample<0.10 and early_test<0.05",
            "1b_succeed": "val_R2>=0.20 on either target",
            "1b_fail": "val_R2<=0.05 on both targets both models",
        },
    }
    (out / "probe1.json").write_text(json.dumps(payload, indent=2))
    print(f"[probe1] {verdict} wrote {out / 'probe1.json'}")


if __name__ == "__main__":
    main()
