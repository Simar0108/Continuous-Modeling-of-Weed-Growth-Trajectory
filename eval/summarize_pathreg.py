"""Print the pathreg stopping table: z0 cosine, tail MSE, in-window test MSE."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ode._pyc_bootstrap import bootstrap

bootstrap()

from ode.pathreg import lambda_tag

REF_Z0 = 0.996
REF_TAIL = 1.869
REF_INWINDOW = 0.245

LAMBDAS = ("0.01", "0.1", "1.0")


def _mean_z0(ckpt_root: Path, run_tag: str, out_dir: Path | None = None) -> float | None:
    if out_dir is not None:
        pinned = out_dir / "z0_from_bestv2.json"
        if pinned.is_file():
            blob = json.loads(pinned.read_text())
            if blob.get("mean_offdiag") is not None:
                return float(blob["mean_offdiag"])
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


def _load_recovered(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    blob = json.loads(path.read_text())
    return {str(row["lambda"]): row for row in blob.get("rows", blob if isinstance(blob, list) else [])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=REPO / "figures" / "pathreg")
    p.add_argument("--ckpt-dir", type=Path, default=REPO / "checkpoints")
    p.add_argument("--recovered", type=Path, default=REPO / "logs" / "pathreg_recovered.json")
    p.add_argument("--table-out", type=Path, default=REPO / "figures" / "pathreg" / "stopping_table.json")
    args = p.parse_args()
    recovered = _load_recovered(args.recovered)
    rows = []
    print(
        f"{'lambda':>8}  {'z0_cos':>10}  {'inwindow':>10}  {'tail':>10}  "
        f"{'z0<0.99':>8}  {'tail<1.869':>11}  {'inwin ok':>8}  source"
    )
    for lam in LAMBDAS:
        tag = f"h1_pathreg_l{lambda_tag(float(lam))}_seed"
        out_dir = args.out_root / tag
        extrap_dir = args.out_root / f"{tag}_extrap60"
        rec = recovered.get(lam, {})
        z0 = _mean_z0(args.ckpt_dir, tag, out_dir)
        inn = _table1_test(out_dir)
        tail = _extrap_test(extrap_dir)
        src = []
        if z0 is None and rec.get("z0_cosine_mean") is not None:
            z0 = float(rec["z0_cosine_mean"])
            src.append("z0:log")
        else:
            src.append("z0:diag" if z0 is not None else "z0:miss")
        if inn is None and rec.get("inwindow_test_mse") is not None:
            inn = float(rec["inwindow_test_mse"])
            src.append("in:log")
        else:
            src.append("in:csv" if inn is not None else "in:miss")
        if tail is None and rec.get("tail_test_mse") is not None:
            tail = float(rec["tail_test_mse"])
            src.append("tail:log")
        else:
            src.append("tail:csv" if tail is not None else "tail:miss")
        z0_ok = z0 is not None and z0 < 0.99
        tail_ok = tail is not None and tail < REF_TAIL
        inn_ok = inn is not None and abs(inn - REF_INWINDOW) <= 0.05

        def fmt(x):
            return f"{x:.4f}" if x is not None else "  n/a "

        print(
            f"{lam:>8}  {fmt(z0):>10}  {fmt(inn):>10}  {fmt(tail):>10}  "
            f"{str(z0_ok):>8}  {str(tail_ok):>11}  {str(inn_ok):>8}  {','.join(src)}"
        )
        rows.append({
            "lambda": lam, "run_tag": tag,
            "z0_cosine_mean": z0, "inwindow_test_mse": inn, "tail_test_mse": tail,
            "ref_z0_cosine": REF_Z0, "ref_inwindow": REF_INWINDOW, "ref_tail": REF_TAIL,
            "source": ",".join(src),
        })
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lock_is_model_of_record": True,
        "lambda_0p1_is_ablation": True,
        "rows": rows,
    }
    args.table_out.write_text(json.dumps(payload, indent=2))
    (args.out_root / "stopping_table.json").write_text(json.dumps(payload, indent=2))
    print(f"reference locked arm: z0~{REF_Z0}  in-window test {REF_INWINDOW}  tail {REF_TAIL}")
    print(f"wrote {args.table_out}")


if __name__ == "__main__":
    main()
