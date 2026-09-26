# Step 10 receipt

Started 2026-09-26 after user chose **1A** (reconnect git) and **2A**
(three clean ODE seeds on Maize 70/15/15; `h1_final_best` frozen and
out of the 3-seed mean).

## Git (1A)

- `.git` had objects/refs dirs but no `HEAD` or `config` (not a repo).
- `git init -b main` in `/rhome/ssing226/MastersThesis`.
- `.gitignore` excludes wandb, checkpoints, parquet, slurm logs.
- Remote: `origin` =
  `https://github.com/Simar0108/Continuous-Modeling-of-Weed-Growth-Trajectory.git`
  (not pushed). First commit `87a8ee3`.

## Clean ODE seeds (2A)

Launcher: `ode/train_h1_seed.py` + `scripts/run_h1_seed.sh`.
Hyperparameters from `h1_final_best/best.ckpt` (`phys_loss_weight=0.2`,
no track embed, curriculum 0.3@[100,250], 400 epochs). Writes
`checkpoints/h1_seed{N}/` only.

| Seed | Job | Notes |
|---|---|---|
| 0 | 29114600 | full horizon |
| 1 | 29114601 | full horizon |
| 2 | 29114602 | full horizon |

Lock SHA256 still
`2498d033ea02e2df1e312a58179226b649a9d3d5c7143b2d040fc9341ef86222`.

## Extrapolation jobs (D-012)

| Job | Name |
|---|---|
| 29114603–605 | ODE prefix 60% seeds 0–2 |
| 29114606–608 | LSTM extrap s0–2 |
| 29114609–611 | GRU extrap s0–2 |
| 29114612–614 | Transformer extrap s0–2 |

Drop-protocol eval waits on 29114600–602.

## Later (after seeds)

- Re-score Table 1/2 with 3-seed mean (lock excluded) — D-010
- Drop protocol 20/40/60 × 3 — D-011
- Extrapolation `--train-time-frac 0.6` × 12 jobs — D-012
- Update `research/conclusions/h1_lock.md` and paper Results skeleton
