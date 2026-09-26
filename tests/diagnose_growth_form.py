"""Step 7 diagnostic: empirical derivatives, Richards NLS, inflection alignment.

No training. Uses the 100 richest valid tracks (same pool as the 70/15/15 split).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PARQUET = REPO / "metrics_with_features.parquet"
OUT_DIR = REPO / "logs" / "step7_diagnostics"
FIG_DIR = REPO / "figures" / "step7"
GRID_DT = 2.0  # hours
SG_WINDOW_H = 48.0
SG_POLY = 3
MIN_OBS = 15


def _load_tracks(max_tracks: int = 100) -> pd.DataFrame:
    path = PARQUET if PARQUET.is_file() else REPO / "Thesis" / "metrics_with_features.parquet"
    df = pd.read_parquet(path)
    if "valid_track" in df.columns:
        df = df[df["valid_track"]].copy()
    if "species" in df.columns:
        maize = df[df["species"] == "Maize"]
        if len(maize):
            df = maize
    counts = df.groupby("track_id").size()
    counts = counts[counts >= MIN_OBS]
    top = counts.nlargest(max_tracks).index.tolist()
    df = df[df["track_id"].isin(top)].copy()
    return df


def _series(grp: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = grp.sort_values("time_since_germination_hours")
    t = g["time_since_germination_hours"].to_numpy(dtype=float)
    sw = (g["width"].to_numpy(dtype=float) / 4.0)
    sh = (g["height"].to_numpy(dtype=float) / 4.0)
    return t, sw, sh


def _sg_on_grid(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if len(t) < MIN_OBS or t.max() - t.min() < SG_WINDOW_H:
        return None
    t_grid = np.arange(t.min(), t.max() + GRID_DT, GRID_DT)
    if len(t_grid) < 9:
        return None
    y_grid = PchipInterpolator(t, y, extrapolate=False)(t_grid)
    finite = np.isfinite(y_grid)
    if finite.sum() < 9:
        return None
    t_grid, y_grid = t_grid[finite], y_grid[finite]
    win = int(round(SG_WINDOW_H / GRID_DT))
    if win % 2 == 0:
        win += 1
    win = min(win, (len(y_grid) // 2) * 2 - 1)
    if win < SG_POLY + 2:
        return None
    y_s = savgol_filter(y_grid, window_length=win, polyorder=SG_POLY, mode="interp")
    dy = savgol_filter(y_grid, window_length=win, polyorder=SG_POLY, deriv=1, delta=GRID_DT, mode="interp")
    return t_grid, y_s, dy


def richards_curve(t: np.ndarray, K: float, r: float, nu: float, z0: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    z0 = np.clip(z0, 1e-3, K - 1e-3)
    nu = np.clip(nu, 0.25, 4.0)
    ratio = np.clip((K / z0) ** nu - 1.0, 1e-8, 1e8)
    inner = 1.0 + ratio * np.exp(-r * nu * t)
    inner = np.clip(inner, 1e-12, None)
    return K * inner ** (-1.0 / nu)


def fit_richards(t: np.ndarray, y: np.ndarray) -> dict:
    t0 = t - t[0]
    y = np.clip(y, 1e-3, None)
    ymax = float(np.nanmax(y))
    p0 = (max(ymax * 1.5, ymax + 1.0), 0.005, 1.0, float(y[0]))
    lo = (ymax * 1.01, 1e-5, 0.25, max(1e-3, float(y[0]) * 0.3))
    hi = (max(ymax * 8.0, 50.0), 0.2, 4.0, float(y[0]) * 2.0 + 1.0)
    try:
        popt, _ = curve_fit(
            richards_curve, t0, y, p0=p0, bounds=(lo, hi), maxfev=20000
        )
        pred = richards_curve(t0, *popt)
        mse = float(np.mean((pred - y) ** 2))
        K, r, nu, z0 = [float(v) for v in popt]
        # inflection: z* = K (1+nu)^(-1/nu)
        z_star = K * (1.0 + nu) ** (-1.0 / nu)
        arg = ((K / max(z0, 1e-6)) ** nu - 1.0) / max(nu, 1e-8)
        t_inf = float("nan")
        if arg > 0 and r * nu > 0:
            t_inf = float(t[0] + np.log(arg) / (r * nu))
        return {
            "ok": True, "K": K, "r": r, "nu": nu, "z0": z0,
            "t_inflection_h": t_inf, "z_inflection": float(z_star), "mse": mse,
        }
    except Exception as exc:
        return {
            "ok": False, "K": np.nan, "r": np.nan, "nu": np.nan, "z0": np.nan,
            "t_inflection_h": np.nan, "z_inflection": np.nan, "mse": np.nan,
            "error": str(exc),
        }


def _rgr_verdict(rgr: np.ndarray) -> str:
    rgr = rgr[np.isfinite(rgr)]
    if rgr.size < 8:
        return "unknown"
    n = len(rgr)
    early, late = rgr[: n // 3].mean(), rgr[-n // 3 :].mean()
    mid = rgr[n // 3 : 2 * n // 3].mean()
    # slope of RGR vs index
    slope = np.polyfit(np.arange(n), rgr, 1)[0]
    if slope > 1e-5 and late > early * 1.05:
        return "increasing"
    if slope < -1e-5 and late < early * 0.95:
        return "decreasing"
    return "constant"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_tracks(100)
    ids = df["track_id"].unique().tolist()
    print(f"[step7] tracks={len(ids)} rows={len(df)}")

    deriv_rows = []
    richards_rows = []
    t_peak_w = []
    t_peak_h = []
    verdicts_w = []
    verdicts_h = []

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plotted = 0

    for tid, grp in df.groupby("track_id"):
        t, sw, sh = _series(grp)
        richards_rows.append({"track_id": int(tid), "channel": "sigma_w", **fit_richards(t, sw)})
        richards_rows.append({"track_id": int(tid), "channel": "sigma_h", **fit_richards(t, sh)})

        for ch, y in (("sigma_w", sw), ("sigma_h", sh)):
            out = _sg_on_grid(t, y)
            if out is None:
                continue
            tg, ys, dy = out
            rgr = dy / np.clip(ys, 1e-3, None)
            v = _rgr_verdict(rgr)
            if ch == "sigma_w":
                verdicts_w.append(v)
                t_peak_w.append(float(tg[int(np.nanargmax(dy))]))
            else:
                verdicts_h.append(v)
                t_peak_h.append(float(tg[int(np.nanargmax(dy))]))
            deriv_rows.append({
                "track_id": int(tid),
                "channel": ch,
                "rgr_verdict": v,
                "rgr_mean": float(np.nanmean(rgr)),
                "rgr_early": float(np.nanmean(rgr[: max(1, len(rgr) // 3)])),
                "rgr_late": float(np.nanmean(rgr[-max(1, len(rgr) // 3) :])),
                "abs_growth_mean": float(np.nanmean(dy)),
                "t_peak_abs_growth_h": float(tg[int(np.nanargmax(dy))]),
                "t_span_h": float(tg[-1] - tg[0]),
            })
            if plotted < 12 and ch == "sigma_w":
                axes[0, 0].plot(tg, ys, lw=0.8, alpha=0.6)
                axes[0, 1].plot(tg, dy, lw=0.8, alpha=0.6)
                axes[1, 0].plot(tg, rgr, lw=0.8, alpha=0.6)
                plotted += 1

    axes[0, 0].set_title("SG σ_w (first 12 tracks)")
    axes[0, 0].set_ylabel("σ_w (px/4)")
    axes[0, 1].set_title("instantaneous dσ_w/dt")
    axes[1, 0].set_title("relative growth rate (1/σ) dσ/dt")
    axes[1, 0].set_xlabel("hours")
    if t_peak_w:
        axes[1, 1].hist(t_peak_w, bins=20, color="steelblue", alpha=0.8)
        axes[1, 1].set_title("time of max dσ_w/dt")
        axes[1, 1].set_xlabel("hours")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "empirical_derivatives.png", dpi=140)
    plt.close(fig)

    deriv_df = pd.DataFrame(deriv_rows)
    rich_df = pd.DataFrame(richards_rows)
    deriv_df.to_csv(OUT_DIR / "empirical_derivatives.csv", index=False)
    rich_df.to_csv(OUT_DIR / "richards_nls.csv", index=False)

    def _count(vs):
        return {k: int((np.array(vs) == k).sum()) for k in ("increasing", "constant", "decreasing", "unknown")}

    t_peak_w = np.asarray(t_peak_w)
    spread_w = float(t_peak_w.max() - t_peak_w.min()) if t_peak_w.size else float("nan")
    ok_w = rich_df[(rich_df.channel == "sigma_w") & (rich_df.ok == True)]
    summary = {
        "n_tracks": len(ids),
        "rgr_verdict_sigma_w": _count(verdicts_w),
        "rgr_verdict_sigma_h": _count(verdicts_h),
        "majority_rgr_sigma_w": max(_count(verdicts_w), key=_count(verdicts_w).get) if verdicts_w else None,
        "t_peak_sigma_w_min_h": float(t_peak_w.min()) if t_peak_w.size else None,
        "t_peak_sigma_w_max_h": float(t_peak_w.max()) if t_peak_w.size else None,
        "t_peak_sigma_w_spread_h": spread_w,
        "t_peak_spread_gt_200h": bool(spread_w > 200) if np.isfinite(spread_w) else None,
        "richards_n_ok_sigma_w": int(len(ok_w)),
        "richards_r_mean": float(ok_w.r.mean()) if len(ok_w) else None,
        "richards_K_mean": float(ok_w.K.mean()) if len(ok_w) else None,
        "richards_nu_mean": float(ok_w.nu.mean()) if len(ok_w) else None,
        "richards_t_inf_mean_h": float(ok_w.t_inflection_h.mean()) if len(ok_w) else None,
        "richards_mse_mean": float(ok_w.mse.mean()) if len(ok_w) else None,
        "richards_mse_median": float(ok_w.mse.median()) if len(ok_w) else None,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[step7] wrote {OUT_DIR} and {FIG_DIR / 'empirical_derivatives.png'}")


if __name__ == "__main__":
    main()
