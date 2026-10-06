# H1 lock — rewritten 2026-09-29 (Step 11c)

**All Step 9 and Step 10 pairwise claims are superseded.** Step 10’s
3-seed mean (test 0.582) mixed two epoch-0 `best.ckpt` files into the
architecture comparison (D-014). Those numbers are quarantined in
`figures/h1_lock/step10_quarantined/`. The H1 model of record is now
the **clean 5-seed mean** on Maize 70/15/15
(`checkpoints/h1_stab_seed{0..4}/`, EMA `state_dict`, selected on
`val_full_horizon_mse`). `h1_final_best` is a leaked 80/20 reference
only (D-009, D-010). Val/test are **short-track transfer** (test tracks
53–65 frames).

Interpretability is closed (D-008). Encoder-collapse probes are
closed (D-019 outcome, gate 3): H-enc-1 rejected, no contrastive
arm; see `research/conclusions/encoder.md`. The lock tables below
are unchanged.

## Banner

The original H1 sentence (parity with LSTM; significantly better test
generalization than GRU; stable val→test where Transformer degrades)
is **not supported as written**.

| Clause | Status | Statistic |
|---|---|---|
| Convergence (≥4/5, best epoch ≥20) | **pass (final)** | 5/5, all epoch 399 (`run_validity.csv`) |
| LSTM parity on short-track transfer | **unsupported (final)** | ode_clean test 0.245 vs lstm_clean 0.218. One-sided p_less=0.979 (cannot claim ODE smaller). **Two-sided p=0.048**, d=+0.30 (ODE worse, n=15). |
| GRU test win | **supported (final)** | ode_clean test 0.245 vs gru_clean 0.370; p_less=3.05e-5, p_two=6.1e-5, r=1.0, d=−1.39 |
| Stable transfer vs Transformer | **partial (final)** | ODE val 0.156 → test 0.245 (Δ+0.090); Transformer 0.159 → 0.306 (Δ+0.147); LSTM 0.204 → 0.218 (Δ+0.014). Better than Transformer, worse than LSTM. Test vs Transformer p_two=0.68 (ns). |
| Drop-protocol advantage | **unsupported (final)** | 5-seed ODE test drop-20/40/60 means 0.277 / 0.278 / 0.291 vs LSTM 0.217 / 0.222 / 0.249 |
| Prefix-60% / tail-40% advantage | **unsupported (final)** | Lock ODE test tail **1.869** vs LSTM **1.014** (`extrap_summary.csv` ode_s0–s4 / lstm_s0–s2). ODE loses. Step 10 3-seed tails (~2.27) stay quarantined. |

`beats_every_baseline` in `headline.json` is **false**.

The evidence matrix is **complete** as of 2026-10-04 (D-017-close).
Every banner cell is final. `figures/ncdediagnostic/` stays PARTIAL
and is not a headline source.

## Model of record vs leaked reference

| Artifact | Role | Val mean (median) | Test mean (median) |
|---|---|---|---|
| ode_clean (5-seed mean-per-track) | **H1 model of record** | 0.156 (0.115) | 0.245 (0.226) |
| ode_converged_seedmean | variance of the method | 0.156±0.022 | 0.245±0.061 |
| ode_s0 (epoch 399) | one clean seed | 0.147 (0.104) | 0.240 (0.216) |
| ode_s1 (epoch 399) | one clean seed | 0.141 (0.101) | **0.182 (0.123)** |
| ode_s2 (epoch 399) | one clean seed | 0.168 (0.115) | 0.286 (0.167) |
| ode_s3 (epoch 399, restage) | one clean seed | **0.135 (0.112)** | 0.194 (0.146) |
| ode_s4 (epoch 399, restage) | one clean seed | 0.189 (0.142) | 0.325 (0.184) |
| ode_leaked (`h1_final_best`, epoch 20, 80/20) | contamination reference | 0.222 (0.172) | 0.215 (0.149) |

LSTM test seed range 0.194–0.246. ODE seeds 2 and 4 (0.286, 0.325) sit
outside that band. Seed 1 is the only ODE seed clearly inside it on
both mean and median.

Lock SHA256 still
`2498d033ea02e2df1e312a58179226b649a9d3d5c7143b2d040fc9341ef86222`.
Do not overwrite.

## Table 1 v2 — size MSE, means and medians

Source: `figures/h1_lock/table1_v2.csv`. Full-horizon $(\sigma_w,\sigma_h)$
MSE. n=15 val, n=15 test unless noted. Eval job 29203599.

| Model | Val mean / median | Test mean / median |
|---|---|---|
| ode_clean | 0.156 / 0.115 | 0.245 / 0.226 |
| lstm_clean | 0.204 / 0.121 | 0.218 / 0.126 |
| gru_clean | 0.242 / 0.203 | 0.370 / 0.278 |
| transformer_clean | 0.159 / 0.104 | 0.306 / 0.199 |
| ode_leaked | 0.222 / 0.172 | 0.215 / 0.149 |
| NLS in-sample | 0.012 / 0.010 | 0.017 / 0.015 |

Train size MSE: ODE 0.528 vs LSTM 0.412 (ODE is not the better
interpolator). Track 6883 remains a heavy tail (leaked ODE test 0.668).

## Table 2 v2 — Wilcoxon, ode_clean vs other

Source: `figures/h1_lock/table2_v2.csv`. Per-track average of 5
converged ODE seeds, then signed-rank vs each baseline. Column `p` /
`p_less` = alternative “less” (ODE smaller). Column `p_two_sided` is
the two-sided test. LSTM parity uses two-sided.

| vs | Split | Δ (ODE−other) | p_less | p_two_sided | d |
|---|---|---|---|---|---|
| lstm_clean | val | −0.048 | 0.076 | 0.151 | −0.43 |
| lstm_clean | test | **+0.027** | 0.979 | **0.048** | **+0.30** |
| gru_clean | val | −0.086 | 0.0042 | 0.0084 | −0.79 |
| gru_clean | test | −0.125 | 3.1e-5 | 6.1e-5 | −1.39 |
| transformer_clean | val | −0.003 | 0.42 | 0.85 | −0.03 |
| transformer_clean | test | −0.061 | 0.34 | 0.68 | −0.22 |
| NLS in-sample | test | +0.228 | 1.0 | 6.1e-5 | +1.18 |

Per-seed pairs: `figures/h1_lock/table2_per_seed.csv`.

## Matched-rollout (D-013)

Source: `figures/h1_lock/matched_rollout.csv`. Full-horizon minus
post-context (frames after index 2). Negative means context frames are
easier.

Stab seeds, test Δ ≈ −0.009 to −0.017. Scoring asymmetry is ~0.01 MSE,
not the GRU gap (~0.12). Do not retune.

## Drop protocol (D-011)

Source: `figures/h1_lock/drop_summary.csv`, `drop_curves.png`.
20/40/60% × 3 drop seeds. No imputation; LSTM/GRU get kept frames with
Δt; Transformer also gets t.

5-seed ODE test means 0.277 / 0.278 / 0.291 vs LSTM 0.217 / 0.222 /
0.249. Seed 1 is LSTM-like (~0.20); seeds 2 and 4 are closer to GRU
(~0.33–0.39). **No drop-protocol advantage.**

## Extrapolation (D-012) — final

Source: `figures/h1_lock/extrap_summary.csv` (D-016). Lock ODE
`h1_stab_seed{0..4}_extrap60` test tails 1.867 / 1.850 / 1.917 /
1.850 / 1.861, **mean 1.869**. LSTM 0.969 / 1.133 / 0.940, **mean
1.014**. GRU ~0.89–1.24. Transformer ~1.44–1.65. **ODE loses.** This
is the headline tail cell.

Step 10 3-seed ODE tails (~2.27) stay in
`figures/h1_lock/step10_quarantined/`. Do not put them in the paper.

## Neural CDE (D-017) — closed

**D-017-close (2026-10-04).** Arm closed. No D-018. Not a lock
headline.

Oct 2 (`73cbaf6`): 10 trainers died on `late_head` before epoch 0.
Oct 4 (`05e01d3`): 5/10 `COMPLETE` (in-window 0–2, prefix-60 0–1);
5/10 `NFE_PAUSE`. PARTIAL diagnostic
(`figures/ncdediagnostic/`, labeled, not promoted):

- prefix-60 test tails **1.873 / 1.868** (n=2) vs LSTM 1.014 and vs
  lock ODE **1.869**. Endpoint miss. Tail parity with the lock is
  structural (control degenerates to the time channel; see
  `research/conclusions/ncdearm.md`).
- in-window test 0.198 / 0.269 / 0.240 (n=3). Not a 5-seed secondary.
- No 5-seed mean, no Wilcoxon, no `beats_every_baseline`.

`figures/ncde/` is the Oct 2 baseline-only leftover. Do not put NCDE
plots in the paper Results.

## EMA vs raw weights

Lightning `EMAWeightAveraging.on_save_checkpoint` writes the averaged
weights to `state_dict` and the raw (non-EMA) weights to
`current_model_state`. `evaluateh1` / `evaluate_extrap` call
`load_from_checkpoint` with no EMA callback, so they load `state_dict`
= EMA. All five stab files contain both keys (epoch 399). Mean cosine
EMA vs raw ≈ 0.994–0.996; they are not identical (`logs/ema_state_dict_audit.json`).

W&B `val_full_horizon_mse` (400 rows/seed): the curve minimum is epoch
399 on every seed, including restaged 3–4 (`duhgh0nj`, `jj50h369`).
Last = best.

## Figures

- `figures/h1_lock/growth_curves.png`
- `figures/h1_lock/irregular_sampling.png`
- `figures/h1_lock/drop_curves.png`
- `figures/h1_lock/extrap_curves.png` (lock ODE + baselines; final)
- `figures/ncdediagnostic/` (PARTIAL n=3/n=2; **not** a lock figure)

## Interpretability

Closed. D-008. 0/6 early-ckpt gates. Latent size dynamics remain
collapsed on the stab runs (`train_dz_dt_size_std` ~0.001–0.003).

## What Step 9 and Step 10 still are

Step 9 compared **one leaked early-curriculum checkpoint** (epoch 20,
80/20) to three clean baselines. Step 10 compared three clean seeds
whose `best.ckpt` files were selected on ramping short-horizon val
(epochs 19/0/0). Neither is the 5-seed H1 result.
