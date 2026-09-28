# Research decisions

## D-008 — Per-track parameter identifiability abandoned for H1

**Date.** 2026-09-26 (before Step 9 runs).

**Decision.** Per-track interpretable parameters (r, K, μ, λ, ν, t0) are
abandoned for Hypothesis 1. They are a documented negative result and
future work, not part of the H1 claim.

**H1 is.** `checkpoints/h1_final_best/best.ckpt` (pre-lag Neural ODE)
+ discrete baselines on the same irregular sampling + statistics
(Wilcoxon on per-track size MSE, val and test).

**H1 is not.** Identifiable per-track kinetic parameters.

**Why.** Six rounds (Steps 3–8) failed with distinct, measured root
causes: shared-constant heads, gradient starvation / z0 cosine ~0.996,
bounded ceiling vs unbounded explosion, lag-gate 5× regression,
K-in-window unidentifiable, Zwietering sat-on-init. Wilcoxon vs
`h1_final_best` on Step 8 T4: p 0.98–0.998, worse on every seed.

**Pointer.** `research/log/negativeresults.md`, `logs/step8_receipt.md`,
`logs/step7_receipt.md`.

**Invariants unchanged.** `hybrid_loss` color_mask, `xy_loss_weight=0`,
affine decoder freeze/scale, `normalize_z0` ON, `ode_nfe<60`, do not
overwrite `h1_final_best`. Track embeddings remain H1-only.

---

## D-009 — H1 weights frozen; identity is checksum

**Date.** 2026-09-26.

**Decision.** Do not retrain or overwrite `h1_final_best`. The working
tree has no usable git HEAD (`git_hash=NO_REPO`, dirty=True). Training
jobs must pass `--allow-dirty`. The durable identity of the locked
model is the file checksum, not a commit.

| Field | Value |
|---|---|
| Path | `checkpoints/h1_final_best/best.ckpt` |
| Bytes | 5,863,020 |
| SHA256 | `2498d033ea02e2df1e312a58179226b649a9d3d5c7143b2d040fc9341ef86222` |
| Twin | byte-identical to `checkpoints/h1final_epoch60_artifact/best.ckpt` |
| git_hash | `NO_REPO` |
| Best-val | this file **is** `best.ckpt`; no last-epoch twin is on disk |

`multi_best_valmse_maize-100-h1-final2.ckpt` is not the lock.

---

## D-010 — Three clean ODE seeds; lock excluded from the mean

**Date.** 2026-09-26.

**Decision.** Train three new `StableSigmoidal` seeds on Maize 70/15/15
using the *checkpoint* hyperparameters of `h1_final_best` (`phys_loss_weight=0.2`,
no track embedding, `normalize_z0` on, horizon 0.3→1.0 over epochs
100–250, 400 epochs). Write under `checkpoints/h1_seed{0,1,2}/`. Do not
overwrite `h1_final_best`.

The locked file remains a named artifact and checksum identity. It is
**not** one of the three seeds and is **not** in the 3-seed mean: that
run used an 80/20 split and the current test tracks were in its train
set.

**Why.** Step 9 Table 1 compared one contaminated ODE checkpoint to
three clean baseline seeds. Seed-matched ODE variance is required before
any H1 generalization sentence.

**Outcomes.** Jobs 29114600–602 completed. `best.ckpt` epochs: seed0=19,
seed1=0, seed2=0 (all before horizon ramp at epoch 100). Eval job
29115924. Clean 3-seed mean (model of record): val 0.398±0.176, test
0.582±0.335. Leaked reference: val 0.222, test 0.215. Contamination
(leaked − clean): val −0.177, test −0.366. Wilcoxon: ode_clean is
**worse** than LSTM/GRU/Transformer on val and test (all p>0.96).
Tables: `figures/h1_lock/table1_v2.csv`, `table2_v2.csv`.

---

## D-011 — Drop-protocol headline is eval-only

**Date.** 2026-09-26.

**Decision.** The 20/40/60% × 3 drop-seed size-MSE suite is evaluation
only (`eval/evaluate_drop.py`). It reuses the context-only query from
`ode/compare_models.py` so dropped-frame ground truth never enters the
encoder. It is not the lock table. If the ODE shows no advantage, report
that; do not retune.

**Outcomes.** 3-seed ODE has no drop-protocol advantage (test means
~0.60–0.63 vs LSTM ~0.22–0.25). ode_s0 alone is comparable to LSTM
(~0.18–0.20 across 20/40/60%). Protocol not retuned.
`figures/h1_lock/drop_summary.csv`, `drop_curves.png`.

---

## D-012 — Extrapolation is prefix-train / tail-eval

**Date.** 2026-09-26.

**Decision.** Extrapolation trains on the first 60% of each train (and
val-monitor) timeline (`--train-time-frac 0.6`) and scores the last 40%
of val/test. Twelve jobs: ODE × 3 seeds + LSTM/GRU/Transformer × 3.
Wilcoxon on per-track tail size MSE. This is not the lock table. If the
ODE loses, that goes in the paper.

**Outcomes.** Jobs 29114603–614 completed. ode_clean test tail mean 2.27
vs LSTM 0.97–1.13 (p=0.99997, d≈+2.0). **ODE loses.** Paper sentence
required. `figures/h1_lock/extrap_summary.csv`, `extrap_curves.png`.

---

## D-013 — Matched-rollout scoring asymmetry is small

**Date.** 2026-09-26.

**Decision.** Re-score ODE seeds post-context only (frames after index
2) to match baseline AR scoring. Report full − post-context. Do not
retune the headline protocol.

**Outcomes.** Leaked test: full 0.215 vs post 0.226, Δ=−0.011. Seed 0
test Δ=−0.009. The ~0.01 gap does not explain Step 9’s GRU test
difference (~0.15). `figures/h1_lock/matched_rollout.csv`.
