# Step 7 receipt — functional-form ablation

Recorded 2026-09-25. Diagnostics ran **before** any Step 7 training jobs.

## Step 6 arm → run name (confirmed, not inferred)

| Arm | W&B name | Run id | Full-horizon `val_track_mse_mean` |
|---|---|---|---|
| (i) bounds only | `h1-s6-bounds` | `wvfixj9e` | 1.933 |
| (ii) +embedding | `h1-s6-bounds-embed` | `n0shekkw` | 1.889 |
| (iii) +embed +head-lr 10 | `h1-s6-bounds-embed-headlr10` | `b3xxmswi` | 1.856 |
| (iv) +hinge | `h1-s6-bounds-embed-headlr10-hinge` | `wan6vch4` | 2.233 |

Arm (iii) is the high `rate_pre_std` run. Hinge (iv) is worse than (iii). Bounds hold (`satfrac=0`). Lag-gated full-horizon val stays 1.6–2.2 vs pre-lag 0.396.

## Empirical derivatives (100 richest valid tracks)

Savitzky–Golay on PCHIP-interpolated 2 h grid, 48 h window, poly 3. `σ = width/4` (raw px). Figure: `figures/step7/empirical_derivatives.png`. CSV: `logs/step7_diagnostics/`.

| | increasing | constant (slope rule) | decreasing |
|---|---|---|---|
| RGR `σ_w` | 1 | 78 | 21 |
| RGR `σ_h` | 1 | 70 | 29 |

Mean RGR early vs late (`σ_w`): **0.0046 → 0.0026** (late/early median **0.57**). So RGR is **mildly decreasing**, not a logistic parked at K, and not still accelerating in relative terms. Absolute growth still has a late peak.

**Alignment:** time of max `dσ_w/dt` spans **248 h – 1845 h** (spread **1597 h > 200 h**). Per-track time offsets are motivated.

## Richards NLS baseline (observed `σ_w`, all 100 tracks fit)

| | mean | p10 | p50 | p90 |
|---|---|---|---|---|
| r | 0.0053 | 0.0030 | 0.0044 | 0.0080 |
| K (px/4) | 624 | 163 | 279 | 1545 |
| ν | 2.29 | 0.25 (bound) | 3.00 | 4.00 (bound) |
| t_inflection (h) | 1080 | 555 | 963 | 1660 |
| MSE | 141 | 29 | 105 | 312 |

ν hits 0.25 on 15 tracks and 4.0 on 22 tracks. Logistic (ν=1) is not the typical fit. Inflection is inside the window for most tracks even while RGR only declines mildly — the NLS K is often extrapolated.

## Architecture for the jobs

- Lag gate **disabled** on every arm.
- Bounded heads + track embedding + `head-lr-mult 10` kept.
- Hinge diversity **off**.
- `normalize_z0` ON. `h1_final_best` not touched.
- Split: 100 tracks, **70/15/15** via `select_discrete_tracks`.
- Horizon **1.0** from epoch 0 (no 30% curriculum).
- 300 epochs, 3 seeds (0,1,2).

| Arm | RHS |
|---|---|
| S1 | current `rate·σ(sat−z)`, lag off |
| S2 | Richards `r z (1−(z/K)^ν)`, ν∈[0.25,4] |
| S3 | `r(t)=r0 exp(β t)`, K=10 fixed |
| S4 | Richards + per-track constant Z (a priori; S3+Z can follow if S3 wins val) |

## Per-arm per-seed gate table

**Not yet.** Jobs were submitted after this diagnostic. Fill after epoch 299.

Gates (all 3 seeds must pass; report mean±std): `r_std>0.01`, `train_dz_dt_size_std>0.05`, `accel>1.0`, `val_track_mse_mean<0.40` on 15 val tracks, `satfrac<0.5`. Wilcoxon vs pre-lag per-track MSE: **not run yet**.

## What was NOT verified

- Wilcoxon signed-rank vs `h1_final_best` per-track MSE.
- Whether S2/S3 `ode_nfe` stays under 60.
- Whether latent-space Richards/exp-rate (softplus `z`) matches the *observed*-σ NLS numbers (different spaces).
- GP-Matern derivatives (SG used instead).
- Test-split (15 tracks) scores — gates use val only.
- Source restoration of deleted `ode/model.py` etc. Jobs load Step-6 **bytecode** via `ode/_pyc_bootstrap.py`. The `.py` files were gone from disk on 2026-09-25 ~15:08; last good pyc is 2026-09-24 16:46 (overfit_test 2026-09-25 09:50).
- S4 is Richards+const-Z, not a data-driven “winner of S2 vs S3”.
