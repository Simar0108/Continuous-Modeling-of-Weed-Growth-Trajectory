# Continuous-time Neural ODE for irregular plant growth trajectories

Draft Methods, Experiments, Results, and a short Discussion. Results
numbers are taken only from `figures/h1_lock/` and
`research/conclusions/h1_lock.md`. Do not write a Results sentence that
claims Wilcoxon vs LSTM as an ODE win. Neural CDE numbers live in
`research/conclusions/ncdearm.md` and stay out of Results.

## Methods

### Data and split

MFWD tracks from `metrics_with_features.parquet`. Provenance (D-021): the
dataset is MFWD yarrow (*Achillea millefolium*, ACHMI), 105 usable tracks,
H1 cap 100; the label “maize-100” is a mislabel and all comparisons were
internal to the same 100 tracks. Official H1 lock uses ACHMI only, the
100 richest valid tracks (`≥15` observations), split deterministically by
trajectory length into train 70 / val 15 / test 15
(`ode.train_baselines.select_discrete_tracks`). `--species Maize` was a
no-op (no `species` column). Shorter
tracks go to val/test. State is 5-D: tray-normalized $(x,y)$, z-scored
$\sigma_w=\mathrm{width}/4$, $\sigma_h=\mathrm{height}/4$, and physiology $Z$
(greenness × edge density) with a `color_mask`. The `GaussianStateTransformer`
is fit on train tracks only.

Observation times are the camera's irregular hours-since-germination. No
imputation is applied for the Neural ODE. No synthetic frame-drop is used
in the lock protocol (the older 30% context / 50% drop suite in
`figures/hypothesis_one_summary.csv` is a different protocol and is not
the headline table).

### Neural ODE

A latent Neural ODE (`dopri5`, rtol/atol $10^{-6}$) encodes the first $K=3$
frames with `ContextEncoderMLP`, integrates $\mathrm{d}h/\mathrm{d}t$ on the
observed time grid (normalized per track to $[0,1]$), and decodes to the
5-D state. Default dynamics are `StableSigmoidalODEFunc`:
$\mathrm{d}z/\mathrm{d}t = r\,\sigma(K-z)$. `normalize_z0` is on. The
geometry loss is log-size on $(\sigma_w,\sigma_h)$; $xy$ weight is 0; $Z$
enters only on valid `color_mask` frames. Affine decoder scale/bias start
frozen (`output_scale` init 6, freeze 50 epochs). Locked weights:
`checkpoints/h1_final_best/best.ckpt`, SHA256
`2498d033ea02e2df1e312a58179226b649a9d3d5c7143b2d040fc9341ef86222`.
Do not retrain.

### Discrete baselines

LSTM, GRU, and a 2-layer causal Transformer (`ode/baselines.py`) consume
consecutive *observed* frames only. There is **no imputation** onto a
regular grid. Each track’s observation times are independently rescaled
to $[0,1]$. The recurrent models take
$[\sigma_w,\sigma_h,Z,\Delta t_{\mathrm{norm}}]$ at each step; the
Transformer also receives the target timestamp $t_{\mathrm{norm}}$.
Training is teacher-forced on the observed sequence; validation and test
are fully autoregressive from the $K=3$ context frames. They predict
$[\sigma_w,\sigma_h,Z]$ only. Same ACHMI 70/15/15 split, 300 epochs,
three seeds $\{0,1,2\}$, best-val checkpoint (`val_track_mse_mean`).

This is the irregular-sampling handling for the discrete suite: time
enters as a feature ($\Delta t$, plus $t$ for the Transformer), not as a
fixed step. The Neural ODE instead integrates on the raw observed times
via `dopri5`. Neither family fills missing camera frames.

### Step 10 add-on protocols (not the lock table)

- **Clean ODE seeds.** Three new `StableSigmoidal` runs on ACHMI
  70/15/15 with the locked checkpoint hyperparameters
  (`phys_loss_weight=0.2`, no track embedding, horizon 0.3$\to$1.0 over
  epochs 100–250). `h1_final_best` stays frozen and is excluded from the
  3-seed mean (it was trained 80/20 and saw the current test tracks).
- **Drop protocol.** Eval-only. Randomly drop 20/40/60% of frames after
  the first observation (three drop seeds), keep $\ge K$ frames, encode
  the first $K$ retained frames, score size MSE on dropped future
  frames. Future ground truth is never placed in the query. The ODE
  integrates on the remaining observed times. LSTM/GRU see only the
  kept frames as $[\sigma_w,\sigma_h,Z,\Delta t_{\mathrm{norm}}]$;
  the Transformer also gets $t_{\mathrm{norm}}$. No family imputes
  dropped camera frames onto a regular grid.
- **Extrapolation.** Retrain on the first 60% of each train/val
  timeline; score size MSE on the last 40% of val/test.

### Classical baseline

Per-track Richards NLS on observed $\sigma_w$ and $\sigma_h$ (pixel/4),
then scored as z-scored size MSE with the same train-fit transformer.
This fit sees the whole track (in-sample curve fit, not a forecast).

### Evaluation

`eval/evaluateh1.py`. Headline metric: full-horizon per-track mean squared
error on $(\sigma_w,\sigma_h)$, reported as mean **and** median (errors are
heavy-tailed; e.g. track 6883). Val/test are **short-track transfer**
(test tracks have 53–65 frames). A matched-rollout sensitivity also
scores the ODE on frames after index $K-1$ only. Wilcoxon signed-rank,
alternative “less” (ODE better) for all pairs, plus two-sided Wilcoxon
vs LSTM for the parity claim, $n=15$ val and $n=15$ test, paired by
`track_id`, using the converged-seed-mean ODE per track. Effect sizes:
mean paired difference, rank-biserial $r=1-2W/(n(n+1))$, paired Cohen’s
$d$.
The leaked `h1_final_best` row is shown only as a contamination
reference and is excluded from the 5-seed mean.

## Experiments

### Protocol

1. Freeze `h1_final_best` (checksum above). Identity is the file hash
   plus a reconnected git repo (D-009/D-010). Do not overwrite the lock.
2. Document Steps 3–8 as a negative result (`research/log/negativeresults.md`,
   D-008). Re-score Step 8 early-best T2/T3 checkpoints once. Close the
   interpretability thread (0/6 pass).
3. Train LSTM / GRU / Transformer × 3 seeds on the lock split.
4. Score Table 1 (model × split × size MSE mean/std/max) and Table 2
   (Wilcoxon + effect sizes). Figures: four test-track growth curves;
   irregular sampling for three tracks.

Tables and figures: `figures/h1_lock/table1_model_split.csv`,
`table2_wilcoxon.csv`, `growth_curves.png`, `irregular_sampling.png`.
Headline verdict: `research/conclusions/h1_lock.md`.

### Negative-result gates (not the H1 lock)

Interpretability required all of: no parameter in the outer 5% of its
reporting range, `train_dz_dt_size_std > 0.05`, acceleration ratio $>1$,
`val_track_mse_mean < 0.40`, three seeds. No arm in Steps 6–8 passed.
Early-best Step 8 T2/T3 improved val (0.14–0.20) but still failed
`near_bound` and `dz/dt` std. See Limitations.

## Limitations

Per-track kinetic parameters $(r,K,\mu,\lambda,\nu,t_0)$ were not
identifiable on this window. Six rounds (collapse to shared constants;
gradient starvation / $z_0$ cosine $\approx 0.996$; bounded ceiling vs
unbounded explosion; lag-gate 5× regression; $K$ out of window / NLS
$K\sim 279$ px vs latent bound 2; Zwietering sat on init, $t_0\sim 40$ h
vs 1597 h peak-spread) are in `research/log/negativeresults.md`. Const-$Z$
tables do not transfer to unseen tracks (`val_mse_Z` identically 1.066).
The observation window covers the accelerating / pre-saturation phase
only. H2 is out of scope for this lock.

The H1 100-track split is plant/`track_id`-disjoint but not tray-disjoint:
train and test share 12 trays. Same-tray plants share treatment,
microclimate, and imaging conditions. Fairness is preserved (all
baselines saw the identical split); generalization claims are
correspondingly bounded. H2 will use tray-aware splitting (D-020).

The evaluation battery ran many comparisons (3 baselines × 2 splits ×
protocols). Headline claims rest on tests that survive conservative
correction: the GRU test win ($p_{\mathrm{two}}=6.1\times 10^{-5}$) does;
the LSTM test difference ($p_{\mathrm{two}}=0.048$) is the marginal one
and is reported as such.

Scope is a single species (yarrow / ACHMI). Per-track encoder
identifiability is a closed negative result
(`research/conclusions/encoder.md`): D-019 rejected the embedding
shortcut, did not launch contrastive $z_0$, and left $z_0$ cosine
collapsed ($\sim$0.96–0.996). Track embeddings remain an H1 known-track
device.

The locked file `h1_final_best` is a leaked 80/20 seed (epoch 20,
D-009). The H1 model of record is the clean 5-seed EMA mean
(`table1_v2.csv`): test 0.245 vs LSTM 0.218, GRU 0.370, Transformer
0.306. Two-sided Wilcoxon vs LSTM test does not favor the ODE
(p=0.048, d=+0.30). Prefix-60 / tail-40: lock ODE test tail 1.869 vs
LSTM 1.014 (**ODE loses**). Richards NLS is in-sample. Step 9 and
Step 10 pairwise claims are superseded
(`research/conclusions/h1_lock.md`).

## Results

Numbers in this section are the lock tables. Every cell is final
(`research/conclusions/h1_lock.md`). `beats_every_baseline` is false.

### In-window size MSE (Table 1 v2)

Full-horizon $(\sigma_w,\sigma_h)$ MSE on ACHMI 70/15/15. Val and test
are short-track transfer (test tracks 53–65 frames). Source:
`figures/h1_lock/table1_v2.csv`. Eval job 29203599.

| Model | Val mean / median | Test mean / median |
|---|---|---|
| ode_clean (5-seed) | 0.156 / 0.115 | 0.245 / 0.226 |
| lstm_clean | 0.204 / 0.121 | 0.218 / 0.126 |
| gru_clean | 0.242 / 0.203 | 0.370 / 0.278 |
| transformer_clean | 0.159 / 0.104 | 0.306 / 0.199 |
| ode_leaked (reference only) | 0.222 / 0.172 | 0.215 / 0.149 |
| NLS in-sample | 0.012 / 0.010 | 0.017 / 0.015 |

The 5-seed method variance is val 0.156±0.022, test 0.245±0.061
(`ode_converged_seedmean`). LSTM test seeds span 0.194–0.246. ODE
seeds 2 and 4 (0.286, 0.325) sit outside that band. Train size MSE is
ODE 0.528 vs LSTM 0.412: the ODE is not the better interpolator.

### Wilcoxon (Table 2 v2)

Per-track average of the five converged ODE seeds, then signed-rank
versus each baseline. Source: `figures/h1_lock/table2_v2.csv`. Column
`p_less` is alternative “ODE smaller.” LSTM parity uses two-sided.

| vs | Split | Δ (ODE−other) | p_less | p_two_sided | d |
|---|---|---|---|---|---|
| lstm_clean | val | −0.048 | 0.076 | 0.151 | −0.43 |
| lstm_clean | test | **+0.027** | 0.979 | **0.048** | **+0.30** |
| gru_clean | val | −0.086 | 0.0042 | 0.0084 | −0.79 |
| gru_clean | test | −0.125 | 3.1e-5 | 6.1e-5 | −1.39 |
| transformer_clean | val | −0.003 | 0.42 | 0.85 | −0.03 |
| transformer_clean | test | −0.061 | 0.34 | 0.68 | −0.22 |

The original H1 sentence (parity with LSTM; significantly better test
generalization than GRU; stable val→test where Transformer degrades)
is not supported as written. GRU test is the one pairwise win. LSTM
test two-sided p=0.048 is an ODE loss, not a tie. Transfer vs
Transformer is in the same direction (ODE Δ+0.090, Transformer
Δ+0.147, LSTM Δ+0.014) but the test comparison is not significant
(p_two=0.68).

### Drop protocol

20/40/60% frame drop × 3 drop seeds. Size MSE on dropped observations
only. Source: `figures/h1_lock/drop_summary.csv`.

5-seed ODE test means 0.277 / 0.278 / 0.291 vs LSTM 0.217 / 0.222 /
0.249. **No drop-protocol advantage.** Seed 1 is LSTM-like (~0.20);
seeds 2 and 4 sit nearer GRU (~0.33–0.39).

### Prefix-60 / tail-40

Train on the first 60% of each timeline; score the last 40% of
val/test. Source: `figures/h1_lock/extrap_summary.csv`.

Lock ODE test tails 1.867 / 1.850 / 1.917 / 1.850 / 1.861 (mean
**1.869**). LSTM 0.969 / 1.133 / 0.940 (mean **1.014**). **ODE
loses.** Step 10’s 3-seed tail (~2.27) is quarantined and is not this
result.

### What the lock is not

`h1_final_best` is excluded from every mean (D-009). Interpretability
is closed (D-008). Neural CDE is closed (D-017-close) and is not in
this table.

## Discussion and future work

The lock ODE is a continuous-time interpolator that transfers to
short tracks better than GRU and worse than LSTM, and it does not
extrapolate the frozen tail. Matched-rollout asymmetry is ~0.01 MSE
and does not explain the GRU gap. Encoder identifiability (z0 cosine
~0.98–0.996) is unsolved; track embeddings stay H1-only. D-019
rejected the embedding-shortcut hypothesis: the lock already has
embeddings off and eval-time cosine on seed 0 is still 0.964.

### Neural CDE (not a Results claim)

A Neural CDE with cubic Hermite control on `[σ_w, σ_h, t]` was
pre-registered (D-017) to beat the LSTM tail. The 5-seed protocol
did not finish: five of ten trainers hit the NFE budget (peak >150
for 3 epochs). The COMPLETE survivors (in-window n=3, prefix-60 n=2)
were scored only as a PARTIAL diagnostic
(`figures/ncdediagnostic/`). Prefix-60 test tails 1.873 / 1.868 sit
on the lock ODE tail 1.869, not on LSTM 1.014. That parity is
structural: after the last observation the size path is held and the
control reduces to the identity time channel, so the tail is a
latent ODE (`research/conclusions/ncdearm.md`). The arm is closed
(D-017-close). A smoothed-control relaunch (D-018) was not run.

The documented scalability fix, if this RHS is revisited after the
thesis, is a **Linear Neural CDE** (Kidger–Morrill line / Log-ODE
reduction): the vector field is linear in $z$, so the integral is a
linear functional of the control and NFE does not track interpolant
roughness. That is future work. It is not a claim of this lock.

## COMMENTS

Task 3 Results audit (2026-10-08). Allowed sources for this pass:
`figures/h1_lock/table1_v2.csv`, `table2_v2.csv`, lock CSVs cited in
Results, and receipts in `research/log/decisions.md`. No checkpoints
were scored.

Unsourced numbers:

- Results / In-window: “test tracks 53–65 frames.” Not present in
  `table1_v2.csv` or `decisions.md`. Left as written; no source
  invented.
