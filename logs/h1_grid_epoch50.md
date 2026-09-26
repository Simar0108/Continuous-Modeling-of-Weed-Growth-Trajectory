# H1 grid — epoch-50 early-abort record

Recorded 2026-09-24 while the 10-track / 300-epoch grid was in flight.
This is **not** a scoring of the pre-registered decision tree. The tree in
`research/conclusions/conclusions.md` stays locked until the evaluation
harness is run on the surviving checkpoints.

Abort rule (pre-registered): by epoch 50, kill if any head `satfrac > 0.5`
or any head pre-activation std collapses to ~0.

| Arm | W&B | State at check | Epoch 50 satfrac (rate / sat / lag) | Epoch 50 pre-std | Abort? | Action |
|---|---|---|---|---|---|---|
| (i) a+b `h1-ab-nonorm-z0scale` | `1tgveh65` | finished epoch 299 before monitor | 0 / 0 / 0 | 1.5e-3 / 3.4e-4 / 4.5e-3 | no | keep; already finished |
| (ii) d `h1-d-headlr50` | `95o91oc8` | finished epoch 299 before monitor | **1 / 1 / 1** | 0.90 / 1.08 / 2.41 | **yes (satfrac)** | would have killed at 50; finished anyway |
| (c) `h1-c-div001` | `fnx0kgt5` | finished epoch 299 before monitor | 0 / 0 / 0 | 0.076 / 0.0055 / 0.038 | no | keep; already finished |
| (c)+(d) `h1-cd-div-headlr` | `52afq1wy` | finished epoch 299 before monitor | **1 / 1 / 1** | 0.20 / 0.32 / 0.66 | **yes (satfrac)** | would have killed at 50; finished anyway |
| (b)+(c)+(d) `h1-bcd-z0scale-div-headlr` | `41e33o9y` job 29059755 | running, epoch **100** when checked | satfrac **1 / 1 / 1** (epoch 100; same pattern as other `head-lr-mult 50` arms) | std 3.43 / 1.17 / 22.2 (not collapsed) | **yes (satfrac)** | **scancel 29059755** |

Notes:

- The two `head-lr-mult 50` arms that finished (`d`, `c+d`) were already
  saturated at epoch 50. The third (`b+c+d`) showed the same saturation
  by epoch 100 and was killed. Partial outcome: 50× head LR drives
  pre-activations through the `|x|>5` ceiling; that is an answer, not a
  discarded run.
- (i) and (c) did **not** trip abort. Their heads stayed unsaturated
  with non-zero pre-std. They are the arms that still need harness
  scoring against the frozen gates.
- `val_track_mse_mean` is not used here.
