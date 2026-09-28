# H1 identity-collapse grid — pre-registered interpretation

**Written: 2026-09-24, before grid results were inspected.**

This document locks the meaning of each outcome pattern *before* numbers exist.
It exists to block post-hoc storytelling. Google's ML experimentation guidance
is blunt about the surrounding risks: small evaluation sets produce uneven
quality estimates, and initialization / data-order noise can mimic real
effects, so a result needs replication before it is trusted.

The current 10-track training grid uses a 8/2 train/val split
(`val_frac=0.2`). Two validation tracks make `val_track_mse_mean`
statistically fragile by construction. That metric is **recorded**, not used
as a pass/fail gate on this grid.

## Frozen pass criteria (gates)

An arm **passes** if and only if all three frozen H1 gates hold on the
evaluation harness (`eval/evaluateh1.py`), not on the 2-track val split:

| Gate | Threshold | Collapse reference |
|---|---|---|
| `train_dz_dt_size_std` (eval equivalent: per-track `dz/dt` size std at `z0`) | **> 0.05** | epoch-290: 0.0067 |
| `z_traj/acceleration_ratio` | **> 1.0** | epoch-290: 0.53 |
| per-track `t_lag` variance | **> 1e-8** | LagHead zero-init is a shared constant |

Secondary diagnostic (not a gate, recorded for every arm):

- mean pairwise `z0` cosine. Step-4 collapse receipt: mean 0.9958, median 0.9989.
  A drop below 0.99 means encoder directions are no longer near-parallel.

`val_track_mse_mean` on 2 val tracks is logged only. It does not decide the
tree below.

## Arms (this grid)

Five isolated jobs, 10 tracks, 300 epochs. One edit pass; flags only.

| Arm | Name | Flags | What it tests |
|---|---|---|---|
| (i) a+b | `h1-ab-nonorm-z0scale` | `--no-normalize-z0 --z0-reg-scale 0.1` | Encoder magnitude + weaker `z0` penalty, no head intervention |
| (ii) d | `h1-d-headlr50` | `--head-lr-mult 50` | Optimizer geometry: 50× LR on `rate_net`, `saturation_net`, `LagHead` |
| (c) | `h1-c-div001` | `--diversity-weight 0.01` | Explicit JS diversity on per-track r / sat / `t_lag` |
| (c)+(d) | `h1-cd-div-headlr` | `--diversity-weight 0.01 --head-lr-mult 50` | Both interventions together |
| (b)+(c)+(d) | `h1-bcd-z0scale-div-headlr` | `--z0-reg-scale 0.1 --diversity-weight 0.01 --head-lr-mult 50` | Weaker `z0` penalty + both interventions |

"Only arms with (c) pass" means (c), (c)+(d), and (b)+(c)+(d) pass, and
(i) and (ii) d fail. "(c) and (d) both pass independently" means (c) alone
and (ii) d alone both pass, regardless of the combination arms.

## Decision tree

| Outcome pattern | Interpretation | Next action |
|---|---|---|
| Only (ii) d passes | Optimizer pathology; heads starved by LR geometry, not architecture | Scale winners to 100-track, 3 seeds |
| Only arms with (c) pass | Explicit diversity regularization required; methodological contribution | Ablate diversity-weight magnitude |
| (c) and (d) both pass independently | Either fix sufficient; prefer (d) for simplicity | Report both, recommend (d) |
| Everything passes including (i) | Collapse was fragile; any perturbation breaks it | Suspicious. Re-run with seeds before believing |
| Nothing passes | Loss landscape or solver-level issue, not conditioning | Stiffness/timescale analysis (see below) |

### Combination-arm tie-breakers (also pre-registered)

- If (c)+(d) passes and neither singleton does: the two fixes are complementary;
  scale the combination, then ablate which term can be dropped at 100-track.
- If (b)+(c)+(d) is the only passing arm: weaker `z0` regularization is
  necessary *in addition to* a head intervention. Do not credit (c) or (d)
  alone.
- If a combination fails while a singleton passes: the extra term is harmful
  at this scale. Carry only the singleton forward.

## Early-abort (do not wait passively for 300 epochs)

Pre-activation logs (`head/{rate,sat,lag}_pre_{std,satfrac}`) are sufficient
to kill a doomed arm. Check W&B at epoch ~50.

An arm has already answered its question if **either** holds by epoch 50:

1. any head `satfrac > 0.5` (`|pre-activation| > 5` on more than half the
   units), or
2. any head pre-activation std collapsing to ~0.

Killed arms are **recorded**, not discarded. Unsuccessful experiments are
evidence about which approaches do not work.

## Why `val_track_mse_mean` is not a gate here

The 10-track grid holds out 2 tracks. A two-point mean can flip from
"best run" to "worst run" under initialization noise alone. The frozen
gates above are within-batch dynamical properties (derivative diversity,
acceleration, lag specificity). They can fail or pass on 10 tracks
without relying on a 2-point val mean.

Confirmatory quality claims wait for the 100-track split
(train 70 / val 15 / test 15) and the Wilcoxon harness in
`eval/evaluateh1.py`.

## If nothing passes: stiffness / timescale analysis

Do **not** invent a sixth architecture flag. The next measurement is
whether the solver is seeing a stiff or multi-timescale vector field:

1. Log `ode_nfe` vs epoch. NFE climbing toward the 60-cap while gates stay
   dead is a solver-level symptom.
2. Compare `rate_net` and `saturation_net` timescales at `z0` (mean rate vs
   remaining headroom `sat − z`).
3. Evaluate the same checkpoint under `rk4` / tighter `(rtol, atol)` without
   retraining. If gates appear only after a solver change, the collapse is
   numerical, not representational.
4. Only after (1)–(3) consider a stiffness-aware dynamics change, and only
   as a new pre-registered experiment.

## Replication rule (applies to every "scale winners" action)

Small-N and single-seed results are not believed. The first confirmatory
step after this grid is:

```
100-track Maize, train70 / val15 / test15, 3 seeds
eval/evaluateh1.py <winner> --vs <reference> --seeds 3
```

Wilcoxon signed-rank is on **per-track size MSE**, paired by `track_id`.
A winner is claimed only if the gate trio still holds and the Wilcoxon
test on the 15-track test split is significant in the expected direction.

## What this document is not

It is not a results table. Results go below this line, in a later dated
section, after the grid is scored against the tree above.

---

## Results (filled after the grid — do not edit the tree above)

The 10-track identity-collapse grid did not produce a passing
interpretability arm. That thread is closed in
`research/log/negativeresults.md` (D-008). H1 does not use those heads.

---

## H1 lock (2026-09-26, Step 10 rewrite)

Full write-up: `research/conclusions/h1_lock.md`. D-008–D-013 in
`research/log/decisions.md`. **Step 9 pairwise claims are superseded.**

**Model of record.** Clean 3-seed mean on Maize 70/15/15
(`checkpoints/h1_seed{0,1,2}/best.ckpt`). `h1_final_best` is a leaked
80/20 reference only (SHA256
`2498d033ea02e2df1e312a58179226b649a9d3d5c7143b2d040fc9341ef86222`).

**Headline.** The clean 3-seed ODE does **not** match LSTM or beat GRU
on short-track transfer (test mean 0.582 vs LSTM 0.218 vs GRU 0.370;
Wilcoxon p>0.96). Two of three `best.ckpt` files are epoch 0.
Extrapolation (train 60% / eval last 40%): ODE loses (test tail 2.27 vs
LSTM ~1.0). Drop protocol: no 3-seed advantage.

Tables: `figures/h1_lock/table1_v2.csv`, `table2_v2.csv`.
Receipt: `logs/step10_receipt.md`.
