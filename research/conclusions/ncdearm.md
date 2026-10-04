# Neural CDE arm — closed (D-017-close)

**Date.** 2026-10-04. **Status. Final.** Not a headline H1 result.

D-017 pre-registered a 5-seed Neural CDE
(`dz/dt = f_θ(z, X(t)) X'(t)`, cubic Hermite on `[σ_w, σ_h, t_norm]`)
against LSTM prefix-60 tail ~1.014 and the in-window LSTM band
[0.194, 0.246]. The 5-seed protocol never completed. The arm is closed
as an endpoint miss on the surviving seeds plus an NFE-budget
incompatibility on the rest. D-018 (smoothed control) was not launched.

`figures/ncdediagnostic/` is a PARTIAL n=3 / n=2 diagnostic. Do not
promote it to `figures/h1_lock/` or the paper Results.

## Three established facts

### 1. In-window dz/dt revived (on COMPLETE seeds)

Locked `StableSigmoidal` stab runs stay collapsed on size velocity:
`train_dz_dt_size_std` ~0.001–0.003 (`research/conclusions/h1_lock.md`).
NCDE COMPLETE in-window seeds recovered an order of magnitude:

| Seed | Fit-end `train_dz_dt_size_std` | Eval-time (70 train tracks) |
|---|---|---|
| 0 | 0.053 | 0.053 |
| 1 | 0.033 | 0.033 |
| 2 | 0.029 | 0.029 |

Seed 0 crosses the old interpretability gate (>0.05). Seeds 1–2 sit
under it but are not the lock's ~0.002. This is a real in-window
dynamics change. It is not an H1 win: the secondary endpoint is test
MSE, and the n=3 diagnostic tests are 0.198 / 0.269 / 0.240.

### 2. z0 collapse persisted

Encoder directions stayed near-parallel on every scored seed.

| Arm | Seed | z0 cosine (off-diag mean) |
|---|---|---|
| in-window | 0 | 0.981 |
| in-window | 1 | 0.980 |
| in-window | 2 | 0.979 |
| prefix-60 | 0 | 0.959 |
| prefix-60 | 1 | 0.959 |

Lock / pathreg notes sit at ~0.94–0.996. Track embeddings remain an
H1 known-track symmetry breaker only (D-008). The CDE control did not
solve encoder identifiability.

### 3. Tail degeneration (structural parity with the lock)

After the last context frame the size spline is held
(`X'_w = X'_h = 0`) and the time channel is forced to the identity
(`X_t = t`, `X'_t = 1`; `ode/ncde.py` `apply_identity_time_channel`).
The product `f_θ(z, X) X'` then has a nonzero contribution only from
the time column. Tail dynamics are a latent ODE in `z`.

Measured tails:

| Model | Test tail MSE | Provenance |
|---|---|---|
| LSTM s0 / s1 / s2 | 0.969 / 1.133 / 0.940 (mean 1.014) | `figures/h1_lock/extrap_summary.csv` |
| Lock ODE s0–s4 | 1.867 / 1.850 / 1.917 / 1.850 / 1.861 (**mean 1.869**) | same file; D-016 |
| NCDE COMPLETE s0 / s1 | **1.873 / 1.868** | `figures/ncdediagnostic/extrap_tail_per_seed.csv` |

Parity with the lock (1.868 / 1.873 vs 1.869) is structural, not
noise. It is also why a smoother in-window control (D-018) was not
worth another 5+5: the tail has already thrown the size path away.

## Partial-seed provenance

Score only `COMPLETE` + `best.ckpt`. In-window 0–2 and prefix-60 0–1
finished 400 epochs. In-window 3–4 and prefix-60 2–4 wrote
`NFE_PAUSE` and are **not** in the diagnostic tables.

Do not compute a 5-seed mean, a Wilcoxon, or `beats_every_baseline`
from this set. Do not treat n=3 / n=2 as the D-017 protocol. Seed 4
in-window is a TRAINING FAILURE (best epoch 0) even before the pause.

## Infrastructure lessons (keep)

Oct 2 (`73cbaf6`): all 10 trainers died on `late_head`; wrappers
lacked `set -e`, so `afterok` released an eval that scored 0 NCDE
seeds.

Oct 4 (`05e01d3` + D-017a): two-group Adam, `COMPLETE` withheld on
NFE pause, `set -e`. Five trainers exited 1. `afterok` then sat at
`DependencyNeverSatisfied` (jobs 29396502 / 29396503, cancelled).

What stays in the launch graph:

- **`afterany`** for eval, not `afterok`. All-or-nothing dependencies
  cannot express “score what survived.”
- **`COMPLETE` sentinels.** Train writes the file only after
  `best.ckpt` exists and NFE did not pause. A paused `best.ckpt` is
  not a finished seed.
- **`--require-complete`.** Discoveries skip dirs without the
  sentinel. Official 5/5 still writes `figures/ncde/`. Survivors
  under 5 write `figures/ncdediagnostic/` and stay labeled PARTIAL.

## Pointers

- Decision: `research/log/decisions.md` D-017, D-017a, D-017-close.
- Lock tables (headline): `research/conclusions/h1_lock.md`.
- Diagnostic (not headline): `figures/ncdediagnostic/`.
