# Encoder-collapse probes (D-019)

**Date.** 2026-10-04 / logged 2026-10-06. **Status. Final for this
thread.** Not a lock headline. Contrastive-z0 was not launched.

z0 off-diagonal cosine is ~0.996 on the lock ODE. D-019 tested three
hypotheses without a hyperparameter sweep.

## Gate

Pre-registered in `research/log/decisions.md` (D-019) before any
probe numbers. Outcome is **gate 3**: Probe 1 succeeds, Probe 2 does
not drop cosine by ≥ 0.05 → reject H-enc-1, do not launch
contrastive z0, do not retune the lock.

## Probe 1 — does K=3 context contain identity?

Features: first three frames as `[σ_w, σ_h, Δt]` (9-D). Maize
100-track official split. Chance for 100-way ID = 0.01.

| Call | Number |
|---|---|
| 1a official in-sample (logreg / MLP) | 1.00 / 1.00 |
| 1a early-window test (logreg / MLP) | 0.050 / **0.118** |
| 1b val R² final size (ridge / MLP) | −1.58 / −12.7 |
| 1b val R² t_norm of max Δsize (ridge / MLP) | −0.54 / −17.7 |

1a **succeeds** (in-sample ≥ 0.20, and early-window MLP test ≥ 0.10).
Official in-sample 1.00 is one vector per class in 9-D and is not
evidence of a usable encoder. The generalization number is early-window
MLP test 0.118 vs chance 0.01, just over the pre-registered 0.10 bar.

1b **fails at chance**. Held-out late size and growth timing are not
in the official K=3 context (negative val R² on both targets).

## Probe 2 — embeddings already off

`checkpoints/h1_stab_seed0/best.ckpt`: `use_track_embed=False`, no
embedding tensors. D-019 forbade retraining an identical 400-epoch
seed. Eval-only z0:

| Quantity | Value |
|---|---|
| z0 cosine mean / median | 0.964 / 0.981 |
| drop vs lock 0.996 | 0.032 (bar 0.05) |
| material drop | **false** |
| `train_dz_dt_size_std` | 0.004 |
| seed-0 val (lock table) | 0.147 |
| retrained | false |

`h1_final_best` was not loaded. SHA256
`2498d033ea02e2df1e312a58179226b649a9d3d5c7143b2d040fc9341ef86222`.

## What this means for H2

H-enc-1 (track embeddings starve the encoder) is **rejected**. The
lock already runs without embeddings and z0 stays collapsed.

H-enc-3 (K=3 has no identity) is **weakly falsified** for track ID
inside early windows and **supported** for late-stage phenotype.

H-enc-2 (the hybrid loss does not need a unique z0; the decoder
absorbs a shared initial state) remains open. A richer encoder or a
contrastive z0 loss is not the next H1 move. H2 stays
species-as-input conditioning on a new split, not an encoder rewrite
of the lock.
