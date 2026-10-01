# Step 12 receipt — pathreg eval repair + NCDE pre-registration

**Date.** 2026-10-01. **D-016 / D-017** written before any NCDE train.

## Part A — pathreg repair (no training)

Tag collision from jobs 29301136–41: empty `lambda_tag` wrote every λ
into `checkpoints/h1_pathreg_l_seed{0..4}/`. Lightning versions:

| file | λ | in-window epoch |
|---|---|---|
| `best.ckpt` | 0.01 | 399 |
| `best-v1.ckpt` | 0.1 | 399 |
| `best-v2.ckpt` | 1.0 | 167–206 |

Fixes in wrappers: inline `.4g` tag (no `ode.pathreg` import), unique
`--out-dir` per λ and per window, `afterok` in-window → extrap,
`train_pathreg` refuses a run-tag that omits `l{lambda_tag}`.
`evaluateh1` / `evaluate_extrap` accept `--ckpt-name`.

Repair job re-scores only:

- λ=1.0 in-window: `best-v2.ckpt` → `figures/pathreg/h1_pathreg_l1_seed/`
- λ=0.1 prefix-60: `best-v1.ckpt` → `figures/pathreg/h1_pathreg_l0p1_seed_extrap60/`

Does not write `figures/h1_lock` or `checkpoints/h1_final_best`.

Partial table (log-recovered; n/a filled by repair GPU):

| λ | z0 cosine | in-window test | prefix-60 tail |
|---|---|---|---|
| 0.01 | 0.970 | 0.292 | 1.873 |
| 0.1 | 0.977 | 0.201 (ablation, not headline) | pending repair |
| 1.0 | 0.941 (fit-end) | pending repair | 1.588 |

Lock remains model of record (D-016). Do not tune λ.

## Part B — NCDE (kill date 2026-10-25)

Pre-registered in D-017 **before** `sbatch`:

- Primary: prefix-60 tail test MSE beats LSTM (~1.014)
- Secondary: in-window test inside LSTM seed band [0.194, 0.246]
- Report regardless of outcome

Control path is cubic Hermite on context knots `[σ_w, σ_h, t_norm]`
plus a hold knot at `t_norm=1`. Size is held after the last observation;
the time channel is identity (`X_t=t`, `X'_t=1`) so the prefix-60 tail
cannot freeze. CPU test `test_time_channel_prevents_frozen_tail` covers
this. Jobs 29331496/7 were cancelled ~1 min in and resubmitted after
this fix.

## What was not verified

- Repair-eval numbers (λ=1.0 in-window, λ=0.1 tail) until the GPU job
  finishes.
- NCDE per-seed results vs either endpoint (training not done).
- `ode_nfe < 60` on the CDE (last-layer-zero init is the only NFE
  control; not measured on a GPU fit).
- `z_traj/acceleration_ratio` and per-track r/K under NCDE.
- EMA `state_dict` vs `current_model_state` on NCDE checkpoints.
- Drop-protocol for NCDE beyond the script that will run after in-window
  seeds.
- Cubic Hermite vs `torchcde` (library not installed; local interpolator
  unit-tested on CPU only).
