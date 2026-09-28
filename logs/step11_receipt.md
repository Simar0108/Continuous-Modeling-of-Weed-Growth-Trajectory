# Step 11 receipt — STOPPED after Step 1

**STOP.** Horizon curriculum **was present** on jobs 29114600–602.
Steps 2–4 were not executed.

The hypothesis that failed seeds trained at horizon 1.0 from epoch 0 is
**false**. All three runs logged `val_horizon_frac=0.300` at the start
and `1.000` at the end (~3993 val logs each). Checkpoint
`hyper_parameters` for seeds 0/1/2 are **identical** to `h1_final_best`
on the curriculum fields.

## Config diff (`h1_final_best` vs 29114600–602)

| Knob | Lock `best.ckpt` | Clean seeds (all 3) | Match? |
|---|---|---|---|
| `horizon_start_frac` | 0.3 | 0.3 | yes |
| `horizon_ramp_start` | 100 | 100 | yes |
| `horizon_ramp_end` | 250 | 250 | yes |
| Runtime `val_horizon_frac` | (lock era) 0.30 early | 0.300 → 1.000 | yes |
| `lr` | 5e-4 | 5e-4 | yes |
| LR schedule / warmup | none in HP | none | yes |
| Gradient clip | Trainer 1.0 (launcher) | `gradient_clip_val=1.0` | yes |
| Solver | dopri5, rtol/atol 1e-6 | same | yes |
| `phys_loss_weight` | 0.2 | 0.2 | yes |
| `normalize_z0` | True | True | yes |
| latent/hidden | 64 / 256 | 64 / 256 | yes |
| Affine freeze / scale | 50 / 6.0 | 50 / 6.0 | yes |
| Batch | 32 (typical maize-100) | `min(32, 70)` train; val=15 | yes |
| Split | 80/20 leaked | 70/15/15 clean | **no — intentional** |
| Best-ckpt epoch | 20 | 19 / **0** / **0** | seed 1–2 fail selection |

Full HP vector compared via `torch.load`: seeds 0/1/2 vs lock =
**IDENTICAL** on every saved hyperparameter key listed in the snapshot.

## Code state of the failed jobs

`git show 4d13054 --stat`: only `logs/step10_receipt.md` and
`research/log/decisions.md` (job-ID bookkeeping). **Zero diff** in
`ode/train_h1_seed.py` vs `87a8ee3` (the commit that introduced the
launcher).

Logs:

```
[repro] git_hash=4d13054ded53695a8cba08b3a881643ebda93f7c dirty=True
```

`--allow-dirty` was hard-coded in `scripts/run_h1_seed.sh`. Dirty tree
was untracked figure archives, not a different trainer.

## Verdict

Clean seeds **did replicate the lock training recipe** (including
horizon 0.3→1.0 over epochs 100–250, clip 1.0, no warmup). They did
**not** train at full horizon from epoch 0.

Epoch-0 `best.ckpt` on seeds 1–2 is therefore **not** a missing-curriculum
bug. Likely causes still open: val-selected checkpoint at `min_epoch=0`
(epoch-0 val can win if later full-horizon val is worse), optimizer
noise on a new 70/15/15 split, or both. Distinguishing those is Step 11
Step 2+ and was **not** run.

## NOT verified / not done

- Steps 2–4 (validity gate, 5-seed retrain, re-eval)
- Whether lock training used an LR scheduler not stored in `hyper_parameters`
- `ode_nfe` on these runs
- Last-epoch weights (not on disk)
- H2
