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

Protocol matches stab: 70/15/15, 5 seeds, EMA 0.999, dopri5 1e-6,
`val_full_horizon_mse`, fail if best epoch < 20. Control path is cubic
Hermite on context knots `[σ_w, σ_h, t_norm]` only.

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
