# Step 11b receipt — selection-metric audit + stab retrain

## Code path (val is ramping-horizon)

`validation_step` → `LatentODELightning._step(batch, "val")` →
`horizon_frac = self._get_horizon_fraction()` →
`horizon_T = max(n_context+1, int(T_full * horizon_frac))` →
loss / `val_track_mse_mean` on `pred[:, :horizon_T]` only.

`_get_horizon_fraction` (bytecode `ode/training_loop.py`):

- epoch < `horizon_ramp_start` (100) → `horizon_start_frac` (0.3)
- 100 ≤ epoch < 250 → linear ramp to 1.0
- epoch ≥ 250 → 1.0

If `horizon_start_frac >= 1.0`, it returns 1.0 immediately.

`EpochGatedModelCheckpoint(min_epoch=0)` monitors `val_track_mse_mean`
(the ramping metric). There is **no** second val pass at horizon 1.0.

## W&B jobs 29114600–602

Runs: seed0 `td89ush5`, seed1 `tr12kmij`, seed2 `y2ehuo61`.
400 val rows each. Horizon-like keys: only `val_horizon_frac`,
`train_horizon_frac_{step,epoch}`. **Full-horizon val was not logged
separately.** That is the finding. Curves below use the ramping metric
and subset rows where that metric reached 1.0.

| Seed | job | best any-horizon | hfrac | mse | first/best full (ep≥250) | last (ep 399) | selected ckpt epoch |
|---|---|---|---|---|---|---|---|
| 0 | 29114600 | ep 19 | 0.30 | 0.0262 | ep 250 / **0.296** | 0.393 | **19** |
| 1 | 29114601 | ep 0 | 0.30 | 0.0194 | ep 250 / **0.265** | 0.363 | **0** |
| 2 | 29114602 | ep 0 | 0.30 | 0.0403 | ep 250 / **0.263** | 0.358 | **0** |

CSV: `logs/step11b_diagnostics/`.

## Verdict

**Selection artifact.** Seeds 1–2 are not genuine “never improved”
training failures. Their `best.ckpt` won because short-horizon val
(~0.02–0.04 at hfrac=0.30) is incomparable to later full-horizon val
(~0.26). Seed 0 “converged” the same way (epoch 19 still hfrac=0.30).

Under the epoch<20 gate, **all three Step 10 seeds are TRAINING FAILURE**
(epochs 19/0/0), including the LSTM-parity seed 0.

At *fixed* horizon 1.0 (ep 250–399) val **gets worse** after the first
full-horizon epoch (~+0.10 MSE). That late drift is real, not a
selection bug; EMA is the response, not LR warmup (warmup is the
genuine-divergence fork, which this audit did not take).

## Patch (no architecture change)

- `FixedHorizonValCallback`: every epoch, extra val `_step` with
  horizon forced to 1.0; log `val_full_horizon_mse`.
- Checkpoint / W&B artifact monitor that metric.
- `EMAWeightAveraging(decay=0.999)`; Lightning saves EMA into
  `state_dict`, so eval loads EMA weights.
- `BestEpochGateCallback`: print TRAINING FAILURE if best epoch < 20.
- `eval/evaluateh1.py`: same gate; Wilcoxon and `ode_clean` mean use
  **converged seeds only**.
- Jobs: `scripts/run_h1_stab.sh` → `checkpoints/h1_stab_seed{0..4}/`.
  Git-clean, no `--allow-dirty`. Curriculum/clip/LR unchanged.
  No LR warmup.

## Retrain

Five seeds launched after this commit (job IDs below after sbatch).
Eval/Wilcoxon/drop/extrap filled after jobs finish.

## NOT verified

- Bytecode line numbers vs original `.py` (source deleted; path from
  disassembly).
- Whether lock-era W&B logged a hidden full-horizon metric under another
  name (none in history keys requested).
- Last-epoch non-EMA weights (Lightning ckpt `current_model_state`).
- `ode_nfe` of the extra val pass (one extra full solve / epoch).
- H2. `h1_final_best` not written.
