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

---

## D-014 — Checkpoint on full-horizon val; epoch<20 is TRAINING FAILURE

**Date.** 2026-09-28.

**Decision.** Jobs 29114600–602 selected `best.ckpt` on ramping
`val_track_mse_mean` (`_get_horizon_fraction` → slice `[:horizon_T]`).
No separate full-horizon val was logged. Seed best epochs 19/0/0 are
short-horizon minima (hfrac=0.30), not optimizer blow-ups. Reconstructing
the first epoch with `val_horizon_frac=1.0` (epoch 250) gives val MSE
0.296/0.265/0.263, then worse to 0.393/0.363/0.358 at epoch 399.

Going forward: log `val_full_horizon_mse` every epoch (horizon forced
to 1.0; training curriculum unchanged), checkpoint on that, evaluate
EMA weights (decay 0.999). Any run with best-val epoch < 20 is a
TRAINING FAILURE and is excluded from architecture comparisons.

**Stab jobs** write `checkpoints/h1_stab_seed{0..4}/`. Do not overwrite
`h1_final_best` or `h1_seed{0,1,2}`.

**Outcomes (2026-09-29, Step 11c).** Five seeds converged (best epoch
399/399/399/399/399). Restage jobs 29203514/515 recovered seeds 3–4
after a W&B artifact quota kill; eval loaded `best-v1.ckpt` (highest
epoch). `evaluateh1` job 29203599: ode_clean val 0.156 / test 0.245 vs
lstm_clean 0.204 / 0.218, gru_clean 0.242 / 0.370. Two-sided Wilcoxon
vs LSTM test **p=0.048** (ODE worse, d=+0.30); one-sided p_less=0.979
does not support an ODE win. GRU test win stands (p_less=3.05e-5,
d=−1.39). Eval loaded EMA `state_dict` (Lightning writes raw weights
to `current_model_state`). W&B `val_full_horizon_mse` minimum is epoch
399 on all five runs, including restage. Prefix-60 ODE trains are not
in this outcome (dirty-tree refuse 29203633). Step 10 tables live in
`figures/h1_lock/step10_quarantined/`. Gate: do not lock the original
H1 sentence. Pointer: `research/conclusions/h1_lock.md`,
`logs/step11c_receipt.md`.

---

## D-016 — Lock remains model of record; pathreg λ=0.1 is an ablation

**Date.** 2026-10-01.

**Decision.** `h1_stab_seed{0..4}` (EMA, full-horizon selection, 5/5
epoch 399) remains the H1 model of record. In-window: ode_clean val
0.156 / test 0.245 vs lstm_clean 0.204 / 0.218, two-sided Wilcoxon vs
LSTM test p=0.048 (ODE worse, d=+0.30). Prefix-60 tail: ODE 1.869 vs
LSTM ~1.014. Do not replace the lock with a pathreg checkpoint.

Pathreg λ=0.1 in-window test 0.201 is **test-selected post hoc** after
seeing the three-λ stopping table. It is reported as an ablation only
and is **not eligible as the headline ODE**. Identity collapse is not
broken (z0 cosine 0.970 / 0.977 / 0.941 vs lock ~0.996). Do not tune λ.

Jobs 29301136–41 collided on empty `lambda_tag` (`h1_pathreg_l_seed`).
Versioned files on disk: `best.ckpt`=λ=0.01, `best-v1`=λ=0.1,
`best-v2`=λ=1.0. Step 12 re-scores λ=1.0 in-window (`best-v2`) and
λ=0.1 prefix-60 (`best-v1`) into unique `--out-dir`s. Wrapper tags are
inline `.4g` (no `ode.pathreg` import). Launch graph is in-window then
`afterok` extrap per λ.

**Invariants unchanged.** `h1_final_best` read-only. `figures/h1_lock`
not written by pathreg repair.

---

## D-017 — NCDE pre-registration (written before training)

**Date.** 2026-10-01. **Kill date.** 2026-10-25.

**Arm.** Neural CDE: `dz/dt = f_θ(z, X(t)) X'(t)` with cubic Hermite
X over `[σ_w, σ_h, t_norm]`. Knots are the **context frames only**
plus a **hold knot at t_norm=1**. Size is held after the last
observation; the time channel is the identity `X_t=t`, `X'_t=1` on
the full odeint span so the tail cannot freeze (`dz/dt = f·0`).
Encoder, affine decoder, `hybrid_loss`
color_mask, `xy_loss_weight=0`, `normalize_z0` ON, dopri5 1e-6 reused.
Writes `checkpoints/h1_ncde_seed{0..4}/` and `*_extrap60`. Does not
write `h1_final_best` or `h1_stab`.

**Protocol.** Maize 70/15/15, 5 seeds, EMA 0.999, checkpoint on
`val_full_horizon_mse`, fail if best epoch < 20. Same H1_HP as stab
except the RHS.

**Primary endpoint (pre-registered).** Prefix-60 tail test MSE beats
LSTM (~1.014): 5-seed ODE/NCDE mean < LSTM clean tail mean, reported
with two-sided Wilcoxon. This is the only success criterion for
replacing the lock on extrapolation.

**Secondary endpoint (pre-registered).** In-window test mean inside
the LSTM seed band [0.194, 0.246] (lstm_s0/s1/s2 test means).

**Report regardless of outcome.** If not converged by 2026-10-25
(best-val epoch < 20 on any seed, or jobs still running), stop and
document. Do not retune the CDE after seeing the endpoints.

**Pointer.** `ode/ncde.py`, `ode/train_ncde.py`, `scripts/run_ncde.sh`.

### D-017 outcome — 2026-10-02 launch (jobs 29349037–29349049)

**Verdict. Endpoints not scored.** All 10 training jobs (5 in-window +
5 prefix-60) launched clean (`git_hash=73cbaf6`, `dirty=False`) on
A100 and died in `configure_optimizers` before epoch 0. No
`best.ckpt`, no `nfe_history.jsonl`, no `ode_nfe`. Eval jobs 29349048
and 29349049 ran because `run_ncde.sh` / `run_ncde_extrap.sh` lack
`set -e` and exit 0 after the Python crash; they discovered **0 NCDE
seeds**. Wilcoxon, z0 cosine, and both D-017 endpoints are therefore
undefined. Kill date remains 2026-10-25. This is a plumbing miss, not
a CDE result — do not treat it as a failed endpoint.

**Crash.** `AttributeError: 'NeuralCDEFunc' object has no attribute
'late_head'` at `OverfitLightning.configure_optimizers` (bytecode
`ode/__pycache__/overfit_test.cpython-310.pyc` line 152). The
three-group Adam split unconditionally reads `ode.late_head`.
`StableSigmoidalODEFunc` / Zwietering expose that alias;
`NeuralCDEFunc` is a matrix field and does not.

**Not written.** `h1_final_best` and `h1_stab` untouched. Lock SHA256
unchanged.

---

## D-017a — Restore sourceless Step-6 modules; NCDE two-group Adam

**Date.** 2026-10-04.

**Hygiene.** `ode/__pycache__` had 10 orphaned `.pyc` files (no sibling
`.py`): `model`, `training_loop`, `overfit_test`, `train_multi` (3.10
+ 3.9) and `hypothesis_one_final` (3.10 + 3.9). The four 3.10 files
were the live trainer (`_pyc_bootstrap`). They were never in git.
`decompyle3` / `uncompyle6` cannot read 3.9/3.10 here; `pycdc` gave a
broken skeleton. Sources were reconstructed from 3.10 disassembly +
inspect signatures + the pycdc docs, then checked against the archived
bytecode.

**Restored (committed).** `ode/model.py`, `ode/training_loop.py`,
`ode/overfit_test.py`, `ode/train_multi.py`. CLI `main()` on overfit /
train_multi is a refuse stub; classes used by H1/NCDE/pathreg are
source-complete. `hybrid_loss` color_mask / pad_mask gating matches
the archived pyc (oracle test). Affine scale/bias still start frozen.
`xy_loss_weight` default stays 0. `normalize_z0` default stays True.

**Archived, not imported.**
`research/archive/step6_bytecode/{model,training_loop,overfit_test,train_multi}.cpython-310.pyc`
+ `SHA256SUMS`.

**Deleted, not restored.** `hypothesis_one_final` 3.10/3.9 pyc (May
2022 / 13:08 2026-09-24, not on the H1 path). `scripts/run_hypothesis_one.sh`
now refuses. Stale 3.9 copies of the four restored modules deleted.

**Launcher guard.** `assert_no_sourceless_bytecode()` runs from
`assert_clean_or_allowed`. `_pyc_bootstrap` imports `.py` only.

**Sentinel.** Train writes `COMPLETE` only after `best.ckpt` exists and
NFE did not pause. Eval wrappers `set -euo pipefail` and refuse unless
every seed has `COMPLETE` + `best.ckpt`. Mid-run kill leaves no
`COMPLETE`; eval refuses.

**Optimizer (D-017a).** `NCDELightning.configure_optimizers`: group 1 =
encoder/decoder/affine (+ non-net CDE params) at `lr`; group 2 =
`NeuralCDEFunc.net` at `lr`. No `late_head` group. Do not alias `net`
as `late_head`.

**Not a CDE retune.** Kill date still 2026-10-25. D-017 endpoints still
unscored until this relaunch finishes.
