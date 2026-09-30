"""Print the pathreg stopping table: z0 cosine, tail MSE, in-window test MSE."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REF_Z0 = 0.996
REF_TAIL = 1.869
REF_INWINDOW = 0.245

LAMBDAS = ("0.01", "0.1", "1.0")


def _lambda_tag(lam: str) -> str:
    from ode.pathreg import lambda_tag
    return lambda_tag(float(lam))


def _mean_z0(ckpt_root: Path, run_tag: str) -> float | None:
    vals = []
    for seed in range(5):
        path = ckpt_root / f"{run_tag}{seed}" / "diagnostics.json"
        if not path.is_file():
            continue
        blob = json.loads(path.read_text())
        vals.append(float(blob["z0_cosine"]["mean_offdiag"]))
    return sum(vals) / len(vals) if vals else None


def _table1_test(out_dir: Path) -> float | None:
    path = out_dir / "table1_v2.csv"
    if not path.is_file():
        return None
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("model") == "ode_clean" and row.get("split") == "test":
                return float(row["mean"])
    return None


def _extrap_test(out_dir: Path) -> float | None:
    path = out_dir / "extrap_summary.csv"
    if not path.is_file():
        return None
    by_seed = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("split") == "test" and str(row.get("model", "")).startswith("ode_s"):
                by_seed.append(float(row["mean"]))
    return sum(by_seed) / len(by_seed) if by_seed else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=REPO / "figures" / "pathreg")
    p.add_argument("--ckpt-dir", type=Path, default=REPO / "checkpoints")
    args = p.parse_args()
    rows = []
    print(
        f"{'lambda':>8}  {'z0_cos':>10}  {'inwindow':>10}  {'tail':>10}  "
        f"{'z0<0.99':>8}  {'tail<1.869':>11}  {'inwin ok':>8}"
    )
    for lam in LAMBDAS:
        tag = f"h1_pathreg_l{_lambda_tag(lam)}_seed"
        out_dir = args.out_root / tag
        z0 = _mean_z0(args.ckpt_dir, tag)
        inn = _table1_test(out_dir)
        tail = _extrap_test(out_dir)
        z0_ok = z0 is not None and z0 < 0.99
        tail_ok = tail is not None and tail < REF_TAIL
        inn_ok = inn is not None and abs(inn - REF_INWINDOW) <= 0.05
        def fmt(x):
            return f"{x:.4f}" if x is not None else "  n/a "
        print(
            f"{lam:>8}  {fmt(z0):>10}  {fmt(inn):>10}  {fmt(tail):>10}  "
            f"{str(z0_ok):>8}  {str(tail_ok):>11}  {str(inn_ok):>8}"
        )
        rows.append({
            "lambda": lam, "run_tag": tag,
            "z0_cosine_mean": z0, "inwindow_test_mse": inn, "tail_test_mse": tail,
            "ref_z0_cosine": REF_Z0, "ref_inwindow": REF_INWINDOW, "ref_tail": REF_TAIL,
        })
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "stopping_table.json").write_text(json.dumps(rows, indent=2))
    print(f"reference locked arm: z0~{REF_Z0}  in-window test {REF_INWINDOW}  tail {REF_TAIL}")
    print(f"wrote {args.out_root / 'stopping_table.json'}")


if __name__ == "__main__":
    main()
