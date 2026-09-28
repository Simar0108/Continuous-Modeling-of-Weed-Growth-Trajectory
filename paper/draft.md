# Continuous-time Neural ODE for irregular plant growth trajectories

Draft Methods and Experiments for Hypothesis 1. Results numbers live in
`figures/h1_lock/` and `research/conclusions/h1_lock.md`. Do not write a
Results narrative that overclaims Wilcoxon vs LSTM.

## Methods

### Data and split

MFWD tracks from `metrics_with_features.parquet`. Official H1 lock uses
**Maize** only, the 100 richest valid tracks (`≥15` observations), split
deterministically by trajectory length into train 70 / val 15 / test 15
(`ode.train_baselines.select_discrete_tracks`, `species="Maize"`). Shorter
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
$[\sigma_w,\sigma_h,Z]$ only. Same Maize 70/15/15 split, 300 epochs,
three seeds $\{0,1,2\}$, best-val checkpoint (`val_track_mse_mean`).

This is the irregular-sampling handling for the discrete suite: time
enters as a feature ($\Delta t$, plus $t$ for the Transformer), not as a
fixed step. The Neural ODE instead integrates on the raw observed times
via `dopri5`. Neither family fills missing camera frames.

### Step 10 add-on protocols (not the lock table)

- **Clean ODE seeds.** Three new `StableSigmoidal` runs on Maize
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
alternative “less” (ODE better), $n=15$ val and $n=15$ test, paired by
`track_id`, using the 3-seed-mean ODE per track. Effect sizes: mean
paired difference, rank-biserial $r=1-2W/(n(n+1))$, paired Cohen’s $d$.
The leaked `h1_final_best` row is shown only as a contamination
reference and is excluded from the 3-seed mean.

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
only. Encoder identifiability is unsolved; track embeddings are an H1
known-track device. H2 is out of scope for this lock.

The locked ODE file is a leaked 80/20 seed (epoch 20). The H1 model of
record is the clean 3-seed mean (`table1_v2.csv`): test 0.582 vs LSTM
0.218, GRU 0.370, Transformer 0.306. Wilcoxon does not favor the ODE.
Prefix-60%/tail-40% extrapolation: ODE test tail 2.27 vs LSTM ~1.0
(**ODE loses**). Richards NLS is in-sample. Best checkpoints for seeds
1–2 are epoch 0 (curriculum). Step 9 pairwise claims are superseded
(`research/conclusions/h1_lock.md`).
