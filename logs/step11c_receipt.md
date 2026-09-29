# Step 11c receipt — fixed-horizon + EMA 5-seed lock eval

## Jobs

| Job | What | Result |
|---|---|---|
| 29180363–365 | stab seeds 0–2 | COMPLETED, best epoch 399, `best.ckpt` |
| 29180366–367 | stab seeds 3–4 | FAILED EDQUOT (`~/.cache/wandb` ~25G) |
| 29203514 / 515 | restage seeds 3–4 (no per-epoch artifacts) | COMPLETED, epoch 399, `best-v1.ckpt` |
| 29203599 | evaluateh1 + drop, `--run-tag h1_stab_seed --n-seeds 5` | COMPLETED 00:04:10 |
| 29203633 | sequential prefix-60 + evaluate_extrap | REFUSE dirty tree (eval had already written `figures/h1_lock/`) |

Git at restage train: `c4e2bc9`. Eval wrote tables on a dirty tree
after `07918ac`. This receipt is committed with the 11c tables and a
two-sided Wilcoxon column so the next extrap job can start clean.

## Convergence

`figures/h1_lock/run_validity.csv`: 5/5 converged, fail-before-epoch 20.
Seeds 3–4: discover-by-max-epoch picked `best-v1.ckpt` over stale
`best.ckpt` from the quota crash.

## Headline (full-horizon size MSE)

ode_clean val 0.156 / 0.115 median, test 0.245 / 0.226.
lstm_clean val 0.204 / 0.121, test 0.218 / 0.126.
gru_clean test 0.370. transformer_clean test 0.306.
`beats_every_baseline=false`.

## Wilcoxon vs LSTM (the missing two-sided cell)

`table2_v2.csv` columns: `p` = `p_less` (ODE smaller), `p_two_sided`.

- val vs lstm_clean: Δ=−0.048, p_less=0.076, p_two=0.151, d=−0.43
- test vs lstm_clean: Δ=+0.027, p_less=0.979, **p_two=0.048**, d=+0.30

Two-sided test rejects equality on test at α=0.05, ODE worse. Parity
is not supported. GRU test remains a win (p_two=6.1e-5, d=−1.39).

## EMA / which state_dict eval loaded

Lightning `EMAWeightAveraging.on_save_checkpoint` copies the module
into `current_model_state` and writes AveragedModel weights to
`state_dict`. `MultiTrackLightning.load_from_checkpoint` (no EMA
callback) loads `state_dict` → **EMA**. All five scored files have
`current_model_state` and `averaging_state`. Mean cosine EMA vs raw
0.994–0.996; sum L2 16–20 (`logs/ema_state_dict_audit.json`). They are
not the same tensor.

## Restage last = best

W&B history, key `val_full_horizon_mse`, 400 rows each:

| Seed | run | min epoch | min | last | last=min |
|---|---|---|---|---|---|
| 0 | zndieicv | 399 | 0.183355 | 0.183355 | yes |
| 1 | s3zr7wci | 399 | 0.128696 | 0.128696 | yes |
| 2 | go6522sr | 399 | 0.220047 | 0.220047 | yes |
| 3 restage | duhgh0nj | 399 | 0.137929 | 0.137929 | yes |
| 4 restage | jj50h369 | 399 | 0.222977 | 0.222977 | yes |

Gate `[h1_gate] best_epoch=399` matches the curve minimum.

## Drop / matched-rollout / extrap

Drop and matched-rollout scored on the five EMA ckpts (see
`h1_lock.md`). Extrap ODE **missing** until the post-commit job
finishes. Step 10 ODE extrap stays quarantined.

## NOT verified

- Non-EMA `current_model_state` re-score (not loaded by evaluateh1).
- `ode_nfe` of the extra full-horizon val pass beyond the 40–58 band.
- Prefix-60 ODE tails (job not started at receipt write).
- H2. `h1_final_best` not written.

## Docs

`research/conclusions/h1_lock.md` and D-014 outcomes match these
tables. Paper Results are **not** written until extrap returns.
