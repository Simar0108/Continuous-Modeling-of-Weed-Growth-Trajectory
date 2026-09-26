# Step 9 receipt — close interpretability, lock H1

Recorded 2026-09-26.

## Decision log

- `research/log/negativeresults.md` — six-round chain + early-ckpt close-out
- `research/log/decisions.md` — D-008 (abandon per-track params), D-009 (freeze + SHA256)
- Early-ckpt table: `logs/step8_diagnostics/early_ckpt_gates.csv` — **0/6 pass, thread closed**

## H1 freeze

SHA256 `2498d033ea02e2df1e312a58179226b649a9d3d5c7143b2d040fc9341ef86222`
matches `h1final_epoch60_artifact`. `git_hash=NO_REPO`. Not retrained.

## Baselines

9/9 finished (jobs 29114341–349, ~2–3 min each). Best-val ckpts:

- `checkpoints/lstm_valmse_baseline-lstm-100t-maize-s{0,1,2}.ckpt`
- `checkpoints/gru_valmse_baseline-gru-100t-maize-s{0,1,2}.ckpt`
- `checkpoints/transformer_valmse_baseline-transformer-100t-maize-s{0,1,2}.ckpt`

`--seed` + local `val_track_mse_mean` checkpoint added to
`ode/train_baselines.py`. `--allow-dirty` required.

## Tables / figures

- `figures/h1_lock/table1_model_split.csv`
- `figures/h1_lock/table2_wilcoxon.csv`
- `figures/h1_lock/per_track_size_mse.csv`
- `figures/h1_lock/irregular_sampling.png`
- `figures/h1_lock/growth_curves.png` (job 29114442)
- `research/conclusions/h1_lock.md`
- `paper/draft.md`

## H1 headline

**Fail.** ODE does not beat every baseline on irregular sampling with
statistics. Comparable to LSTM; better than GRU on test only;
Transformer often better on val, not significant on test.

## What was NOT verified

- Git-clean launches (no HEAD; `--allow-dirty`)
- Drop-protocol (30% context / 50% drop) not re-run as headline
- `h1_final_best` last-epoch weights (not on disk)
- H2
- Wilcoxon vs a correctly re-wrapped Step 8 T1 (load bug not re-litigated)
- NLS is in-sample, not a forecast
- ODE is a single locked seed
- Playbook retro (deferred)
