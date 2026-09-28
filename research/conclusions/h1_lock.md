# H1 lock — rewritten 2026-09-26 (Step 10)

**All Step 9 pairwise claims are superseded.** Step 9 compared a leaked
80/20 ODE checkpoint to three clean baseline seeds. The H1 model of
record is now the **clean 3-seed mean** on Maize 70/15/15
(`checkpoints/h1_seed{0,1,2}/best.ckpt`). `h1_final_best` is a leaked
reference only (D-009, D-010). Val/test are **short-track transfer**
(test tracks 53–65 frames).

Interpretability is closed (D-008).

## Banner

The revised H1 sentence (parity with LSTM; significantly better test
generalization than GRU; stable val→test where Transformer degrades)
is **not supported** by the clean 3-seed mean.

| Clause | Status | Statistic |
|---|---|---|
| LSTM parity on short-track transfer | **unsupported** | ode_clean test mean 0.582 vs lstm_clean 0.218; Wilcoxon p=0.9997, d=+1.46 (ODE worse) |
| GRU test win | **unsupported** | ode_clean test 0.582 vs gru_clean 0.370; p=0.99997, d=+2.14 (ODE worse) |
| Stable transfer vs Transformer | **unsupported** for the 3-seed mean | ode_clean val 0.398 → test 0.582; transformer_clean val 0.159 → test 0.306. ODE gap is larger. |
| Drop-protocol advantage | **unsupported** | 3-seed ODE test drop-20/40/60 means 0.60 / 0.60 / 0.63 vs LSTM ~0.22 / 0.22 / 0.25 |
| Prefix-60% / tail-40% advantage | **unsupported** | ode_clean test tail 2.27 vs LSTM 0.97–1.13; p=0.99997, d≈+2.0 |

Caveat that does **not** rescue the headline: two of three `best.ckpt`
files are **epoch 0** (curriculum selected the first val step). Seed 0
is epoch 19 (same selection rule as the leaked lock, which is epoch 20).
Last-epoch weights are not on disk.

## Model of record vs leaked reference

| Artifact | Role | Val mean (median) | Test mean (median) |
|---|---|---|---|
| ode_clean (3-seed mean-per-track) | **H1 model of record** | 0.398 (0.337) | 0.582 (0.439) |
| ode_clean_seedmean (mean of 3 seed means) | variance of the method | 0.398±0.176 | 0.582±0.335 |
| ode_s0 (epoch 19) | one clean seed | 0.197 (0.162) | 0.197 (0.141) |
| ode_s1 (epoch 0) | one clean seed | 0.520 (0.439) | 0.808 (0.644) |
| ode_s2 (epoch 0) | one clean seed | 0.478 (0.454) | 0.740 (0.550) |
| ode_leaked (`h1_final_best`, epoch 20, 80/20) | contamination reference | 0.222 (0.172) | 0.215 (0.149) |

Contamination effect (leaked − clean seed-mean): val **−0.177**, test
**−0.366**. The leaked file looks better because it saw test IDs and
because seeds 1–2 are untrained `best.ckpt`s.

Lock SHA256 still
`2498d033ea02e2df1e312a58179226b649a9d3d5c7143b2d040fc9341ef86222`.
Do not overwrite.

## Table 1 v2 — size MSE, means and medians

Source: `figures/h1_lock/table1_v2.csv`. Full-horizon $(\sigma_w,\sigma_h)$
MSE. n=15 val, n=15 test unless noted.

| Model | Val mean / median | Test mean / median |
|---|---|---|
| ode_clean | 0.398 / 0.337 | 0.582 / 0.439 |
| lstm_clean | 0.204 / 0.121 | 0.218 / 0.126 |
| gru_clean | 0.242 / 0.203 | 0.370 / 0.278 |
| transformer_clean | 0.159 / 0.104 | 0.306 / 0.199 |
| ode_leaked | 0.222 / 0.172 | 0.215 / 0.149 |
| NLS in-sample | 0.012 / 0.010 | 0.017 / 0.015 |

LSTM s0/s1/s2 test means 0.194 / 0.215 / 0.246 (medians 0.086 / 0.117 /
0.174). Track 6883 remains a heavy tail (leaked ODE test 0.668).

## Table 2 v2 — Wilcoxon, ode_clean vs other (alternative=less)

Source: `figures/h1_lock/table2_v2.csv`. Per-track average of 3 ODE
seeds, then signed-rank vs each baseline. **Zero comparisons have
p<0.05 in the ODE’s favor.** Every discrete baseline on val and test
has p≥0.96 (ODE worse). NLS p=1.0.

Per-seed pairs: `figures/h1_lock/table2_per_seed.csv`.

## Matched-rollout (D-013)

Source: `figures/h1_lock/matched_rollout.csv`. Full-horizon minus
post-context (frames after index 2). Negative means context frames are
easier.

Leaked test: full 0.215 vs post 0.226, Δ=−0.011. Seed 0 test Δ=−0.009.
The scoring asymmetry is ~0.01 MSE, not the Step 9 GRU gap (~0.15).
Do not retune.

## Drop protocol (D-011)

Source: `figures/h1_lock/drop_summary.csv`, `drop_curves.png`.
20/40/60% × 3 drop seeds. No imputation; LSTM/GRU get kept frames with
Δt; Transformer also gets t.

3-seed ODE does **not** win. ode_s0 test stays ~0.18–0.20 across drop
fractions (comparable to LSTM). ode_s1/s2 stay ~0.77–0.88. Protocol was
not retuned.

## Extrapolation (D-012)

Source: `figures/h1_lock/extrap_summary.csv`, `extrap_curves.png`.
Train first 60% of timeline; score last 40%.

ode_clean test tail mean 2.27 vs LSTM 0.97–1.13, GRU 0.89–1.24,
Transformer 1.44–1.65. Wilcoxon vs every LSTM seed: p=0.99997, d≈+2.0.
**The ODE loses.** That sentence belongs in the paper.

## Figures

- `figures/h1_lock/growth_curves.png`
- `figures/h1_lock/irregular_sampling.png`
- `figures/h1_lock/drop_curves.png`
- `figures/h1_lock/extrap_curves.png`

## Interpretability

Closed. D-008. 0/6 early-ckpt gates.

## What Step 9 still is

A comparison of **one leaked early-curriculum checkpoint** (epoch 20)
to three clean baselines. Those numbers (`ode_leaked` test 0.215 vs
LSTM 0.218 vs GRU 0.370) must be labeled leaked. They are not the
3-seed H1 result.
