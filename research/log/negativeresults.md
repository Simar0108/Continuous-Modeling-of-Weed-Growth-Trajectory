# Negative results — per-track interpretability (Steps 3–8)

Recorded 2026-09-26 **before** Step 9 training or re-score runs.
This is the durable write-up of the six-round failure. H1 does not depend
on these heads. Evidence pointers are W&B run names / ids in project
`latent-ode-maize-100` unless noted. Receipt: `logs/step8_receipt.md`.

Decision: see D-008 in `research/log/decisions.md`.

---

## Round 1 — Steps 3/4: head collapse to shared constants

**Hypothesis.** Per-track rate / saturation / lag heads would break identity
collapse and give track-specific r, K, t_lag.

**Arm config.** Pre-lag `StableSigmoidalODEFunc` plus LagHead; 10-track
identity-collapse grid (`h1-ab-nonorm-z0scale`, `h1-c-div001`,
`h1-d-headlr50`, `h1-cd-div-headlr`, `h1-bcd-z0scale-div-headlr`) and the
100-track lag-gate probe `h1-lag-gate`.

**Gate outcome.** Fail. Heads collapsed to shared constants:
`r_std ~ 8e-6`, `t_lag` variance `1.76e-11`. Frozen gates
(`train_dz_dt_size_std > 0.05`, `accel > 1.0`, per-track `t_lag` var > 1e-8)
did not pass. See `figures/h1_eval/h1_eval_gates.csv` on
`h1_final_best`: `t_lag` var = 0, `r_std` on val = 0.0068 (still a
near-constant), accel mean 0.79, `dz/dt` std 0.0007.

**Measured root cause.** Encoder directions are near-parallel
(mean pairwise z0 cosine ~0.996 on the collapse receipt; 0.91–0.95 on the
later 100-track eval). Heads see almost the same z0 and output one shared
(r, K, t_lag).

**W&B evidence.**
- `h1-ab-nonorm-z0scale` `1tgveh65`
- `h1-c-div001` `fnx0kgt5`
- `h1-d-headlr50` `95o91oc8`
- `h1-cd-div-headlr` `52afq1wy`
- `h1-bcd-z0scale-div-headlr` `41e33o9y` (scancelled epoch ~100)
- `h1-z0-mag` (job 29059653)
- `h1-lag-gate` `8ue55805` (full-horizon val 1.840)

---

## Round 2 — Step 4: gradient starvation

**Hypothesis.** Raising head LR or weakening z0 regularization would let
LagHead / rate heads receive gradient.

**Arm config.** Grid arms with `--head-lr-mult 50` and/or `--z0-reg-scale 0.1`
(`h1-d-headlr50`, `h1-ab-nonorm-z0scale`, `h1-bcd-z0scale-div-headlr`).

**Gate outcome.** Fail. LagHead gradient ~3.5e-5 vs encoder ~2.47.
z0 cosine stayed ~0.996. Ablation `a` / `a+b` without `normalize_z0`
collapsed z0 norms to ~0.09 and was discarded; `normalize_z0` stays ON.

**Measured root cause.** Gradient mass sits in the encoder/decoder.
Heads are starved. 50× head LR does not fix identifiability; it drives
pre-activations through the `|x|>5` ceiling (see Round 3).

**W&B evidence.** Same grid ids as Round 1; epoch-50 abort record
`logs/h1_grid_epoch50.md`.

---

## Round 3 — Step 5: bounded ceiling vs unbounded explosion

**Hypothesis.** Bounded r∈(0,2), K∈(0.3,2) would keep parameters in a
physiological range; unbounded heads would find useful rates.

**Arm config.** Bounded sigmoid heads vs unbounded / high head-LR arms
(`h1-d-headlr50`, `h1-cd-div-headlr`, `h1-bcd-z0scale-div-headlr`).

**Gate outcome.** Fail both ways.
- Bounded: pinned at ceiling (`r=2.0`, `satfrac=1.0`).
- Unbounded / 50× head LR: escaped to exponential rates (`r` in 8–28
  range; satfrac 1 by epoch 50 on `d` and `c+d`).

**Measured root cause.** The loss does not identify an interior r.
The optimizer either saturates the sigmoid or leaves the intended range.
`logs/h1_grid_epoch50.md`: `h1-d-headlr50` satfrac 1/1/1 at epoch 50;
`h1-bcd` same pattern by epoch 100 (job 29059755 scancelled).

**W&B evidence.** `95o91oc8`, `52afq1wy`, `41e33o9y`.

---

## Round 4 — Step 6: bounds hold, lag gate is a 5× regression

**Hypothesis.** Keep bounds, add track embeddings (H1-only symmetry
breaker) and head-LR ×10; lag gate might restore late growth.

**Arm config.** 100-track, lag gate ON.
- (i) bounds only — `h1-s6-bounds` `wvfixj9e` val 1.933
- (ii) +embed — `h1-s6-bounds-embed` `n0shekkw` val 1.889
- (iii) +embed +head-lr 10 — `h1-s6-bounds-embed-headlr10` `b3xxmswi` val 1.856
- (iv) +hinge — `h1-s6-bounds-embed-headlr10-hinge` `wan6vch4` val 2.233
- Prior lag-only: `h1-lag-gate` `8ue55805` val 1.840

**Gate outcome.** Fail. Bounds hold (`satfrac=0` on the 10-track probe).
Embed is inert. Head-LR 10 is active. Hinge is harmful. Full-horizon val
1.6–2.2 vs pre-lag `h1_final_best` 0.396. Accel always < 1. z sits near K.

**Measured root cause.** The multiplicative lag gate is a 5× reconstruction
regression. It is not a biological lag. Track embeddings do not solve
encoder identifiability.

**W&B evidence.** Names/ids above; `logs/step7_receipt.md` (Step 6 table).

---

## Round 5 — Step 7: K-in-window unidentifiable; RHS ablation

**Hypothesis.** After removing the lag gate, a Richards or time-varying
rate (K=10 out of window) would accelerate and beat `h1_final_best`.

**Arm config.** 100-track 70/15/15, horizon 1.0, 3 seeds, lag off.
- S1 `h1-s7-S1-stable-s{0,1,2}` (`vqpun6y8`, …) current `rate·σ(sat−z)`
- S2 `h1-s7-S2-richards-s{0,1,2}` (`v80lnxgu`, …) Richards ν∈[0.25,4]
- S3 `h1-s7-S3-exprate-s{0,1,2}` (`t0vmh63l`, …) `r(t)=r0 exp(βt)`, K=10
- S4 `h1-s7-S4-richards-zconst-s{0,1,2}` Richards + per-track const Z

Diagnostics (before jobs): RGR late/early 0.57; NLS K median 279 px/4 vs
latent bound 2; peak `dσ_w/dt` spread 1597 h.
`logs/step7_diagnostics/summary.json`.

**Gate outcome.** 0/12 pass. r pinned at 2.0, satfrac=1 on S1–S3.
S3 is the only arm with accel>1 on all seeds (1.23–1.36) and the worst
val (0.594±0.08). S4-s2 val 0.378 (only seed <0.40); 3-seed fail.
const-Z `val_mse_Z` identically 1.066 on unseen val tracks.

**Measured root cause.** The observation window is pre-saturation.
K-in-window is unidentifiable. Exp-rate can accelerate but does not
fit. Const-Z is an H1 (known-track) table and does not transfer.

**W&B evidence.** Names above; `logs/step7_receipt.md`,
`logs/step7_diagnostics/wandb_summaries.json`.

---

## Round 6 — Step 8: Zwietering sat on init

**Hypothesis.** Zwietering–Richards (μ, λ, ν), K=10, λ as an IC (not a
gate), optional t0∈[−0.3, 0.3], would put lag in range (~0.15–0.25) and
register the 1597 h peak spread.

**Arm config.** Same 100-track split as Step 7 (`species=None`), 3 seeds.
- T1 `h1-s8-T1-s3-s{0,1,2}` (`qojb6fmp`, `to2872c1`, `5tr2e8ph`) S3 control
- T2 `h1-s8-T2-zwiet-s{0,1,2}` (`izhdszlw`, `6sgbs7zq`, `ziegwivm`)
- T3 `h1-s8-T3-zwiet-t0-s{0,1,2}` (`3prx48mi`, `dwd8jbho`, `dbmln7qx`)
- T4 `h1-s8-T4-zwiet-t0-wilcox-s{0,1,2}` (`nnmm9gkv`, `nf119cvz`, `xs4oam2u`)
  (same train as T3 + Wilcoxon)

**Gate outcome.** 0/12 last-epoch pass. T2 last-epoch val 0.362±0.040
(passes val gate only). `near_bound_λ=1` every Zwietering seed.
`train_dz_dt_size_std` 0.005–0.006 (fail). Accel>1 all seeds (pass).
T1 r≡2.0. Wilcoxon vs `h1_final_best` (ref 0.152): p=0.982–0.998,
worse on every seed. vs-T1 column is not trustworthy (load without
`ExpRateODEFunc`).

**Measured root cause.** μ moved +0.07 from `softplus(0)`; ν moved −0.005
and collapsed (std~0.002); λ≈0.71 stayed above the 0.6 reporting range
from epoch 0; t0≈0.02 (~30–46 h) vs 1597 h misalignment. Heads sat on
init. `val_mse_Z` identically 1.066 (const-Z, unseen val).

**W&B evidence.** `logs/step8_receipt.md`,
`logs/step8_diagnostics/wandb_summaries.json`,
`logs/step8_diagnostics/wilcoxon_h1-s8-T4-*.json`.

---

## Early-ckpt re-score (Step 9 Task 2) — thread closed

Job 29114340. Table: `logs/step8_diagnostics/early_ckpt_gates.csv`.
Best-val T2/T3 checkpoints (not last epoch). Split = Step 8 (`species=None`).

| Run | val MSE | train dz/dt std | accel | near_bound_any | pass |
|---|---|---|---|---|---|
| T2-s0 | 0.179 | 0.0031 | 0.670 | 1 | no |
| T2-s1 | 0.185 | 0.0008 | 0.731 | 1 | no |
| T2-s2 | 0.197 | 0.0023 | 1.307 | 1 | no |
| T3-s0 | 0.144 | 0.0043 | 0.996 | 1 | no |
| T3-s1 | 0.160 | 0.0030 | 0.945 | 1 | no |
| T3-s2 | 0.178 | 0.0006 | 0.840 | 1 | no |

Val gate (<0.40) holds on every early ckpt — last-epoch scoring did understate fit.
Dynamics gates fail: `near_bound_any=1` (λ still ~0.70–0.72), `train_dz_dt_size_std`
0.0006–0.0043 (<<0.05). Accel>1 only on T2-s2. **0/6 pass. 3-seed fail.**

μ/λ/ν/t0 are still at init (μ≈0.76, λ≈0.71, ν≈0.939, t0≈0.02). Early-best
val is a decoder/affine fit, not identified kinetics.

**This thread is closed.** No further per-track head / RHS work for H1.
