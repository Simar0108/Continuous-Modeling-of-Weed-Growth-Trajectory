# Step 10 receipt

Started 2026-09-26 after user chose **1A** (reconnect git) and **2A**
(three clean ODE seeds on Maize 70/15/15; `h1_final_best` frozen and
out of the 3-seed mean).

## Git (1A)

- `.git` had objects/refs dirs but no `HEAD` or `config` (not a repo).
- `git init -b main` in `/rhome/ssing226/MastersThesis`.
- `.gitignore` excludes wandb, checkpoints, parquet, slurm logs.
- Remote: not recovered from disk (`gh` missing; no URL in leftover
  `.git`). Named repo: Continuous-Modeling-of-Weed…. Add `origin` when
  the URL is known.

## Clean ODE seeds (2A)

Launcher: `ode/train_h1_seed.py` + `scripts/run_h1_seed.sh`.
Hyperparameters from `h1_final_best/best.ckpt` (`phys_loss_weight=0.2`,
no track embed, curriculum 0.3@[100,250], 400 epochs). Writes
`checkpoints/h1_seed{N}/` only.

| Seed | Job | Notes |
|---|---|---|
| 0 | pending | |
| 1 | pending | |
| 2 | pending | |

Do not overwrite `checkpoints/h1_final_best/`.

## Later (after seeds)

- Re-score Table 1/2 with 3-seed mean (lock excluded) — D-010
- Drop protocol 20/40/60 × 3 — D-011
- Extrapolation `--train-time-frac 0.6` × 12 jobs — D-012
- Update `research/conclusions/h1_lock.md` and paper Results skeleton
