# H1 lock — 2026-09-26

## Verdict

Hypothesis 1 is **locked on** `checkpoints/h1_final_best/best.ckpt`
(SHA256 `2498d033ea02e2df1e312a58179226b649a9d3d5c7143b2d040fc9341ef86222`).
Interpretability is **not** part of the claim (D-008).

The headline “Neural ODE beats every discrete baseline on irregular
sampling, with statistics” is **false** on the lock protocol
(full-horizon size MSE, Maize 70/15/15, n=15 val and n=15 test).

What the statistics support:

- vs **LSTM**: comparable. No Wilcoxon p<0.05 in ODE’s favor on val or
  test for any seed. Test means: ODE 0.215 vs LSTM 0.194 / 0.215 / 0.246.
- vs **GRU**: ODE is significantly better on **test** (p=0.003–0.004,
  all three seeds; paired d ≈ −0.64 to −0.69). Val is not significant
  (p≈0.34–0.47).
- vs **Transformer**: Transformer has lower val means (0.14–0.17 vs
  ODE 0.222) but is not significantly worse or better on test
  (p=0.09–0.74). Seed 0 test mean 0.412 (unstable).
- vs **Richards NLS** (in-sample): NLS wins by a large margin (val 0.012,
  test 0.017). This is a full-track curve fit, not a forecast.

Bring these tables here before drafting paper Results prose.

## Locked model

| Field | Value |
|---|---|
| Path | `checkpoints/h1_final_best/best.ckpt` |
| SHA256 | `2498d033ea02e2df1e312a58179226b649a9d3d5c7143b2d040fc9341ef86222` |
| Twin | `checkpoints/h1final_epoch60_artifact/best.ckpt` (byte-identical) |
| git_hash | `NO_REPO` (`--allow-dirty`) |
| Split | Maize, 100 tracks, 70/15/15 |
| Protocol | full-horizon size MSE; no synthetic drop |

## Table 1 — size MSE (mean / std / max)

Source: `figures/h1_lock/table1_model_split.csv`.

| Model | Val mean±std (max) | Test mean±std (max) |
|---|---|---|
| Neural ODE (lock) | 0.222±0.143 (0.595) | 0.215±0.162 (0.668) |
| LSTM s0 | 0.227±0.201 (0.546) | 0.194±0.192 (0.568) |
| LSTM s1 | 0.228±0.210 (0.577) | 0.215±0.207 (0.501) |
| LSTM s2 | 0.156±0.134 (0.395) | 0.246±0.199 (0.665) |
| GRU s0 | 0.220±0.174 (0.564) | 0.360±0.267 (1.011) |
| GRU s1 | 0.249±0.180 (0.632) | 0.375±0.277 (1.052) |
| GRU s2 | 0.257±0.195 (0.666) | 0.376±0.289 (1.070) |
| Transformer s0 | 0.169±0.115 (0.386) | 0.412±0.545 (1.931) |
| Transformer s1 | 0.163±0.137 (0.409) | 0.296±0.283 (0.924) |
| Transformer s2 | 0.143±0.096 (0.368) | 0.210±0.203 (0.721) |
| Richards NLS (in-sample) | 0.012±0.007 (0.031) | 0.017±0.011 (0.034) |

ODE train mean 0.384 (n=70). LSTM 3-seed val mean 0.204, test 0.218.
GRU 3-seed val 0.242, test 0.370. Transformer 3-seed val 0.158, test 0.306.

## Table 2 — Wilcoxon (ODE vs other, alternative=less)

Source: `figures/h1_lock/table2_wilcoxon.csv`. Rank-biserial
$r=1-2W/(n(n+1))$ as implemented in `eval/evaluateh1.py`.

Significant ODE-better (p<0.05): **test vs GRU s0/s1/s2 only**.
All LSTM and Transformer comparisons, and all val GRU, fail that bar.
NLS: p=1.0 (ODE worse).

## Figures

- `figures/h1_lock/growth_curves.png` — 4 test tracks, actual vs ODE vs best-val baseline
- `figures/h1_lock/irregular_sampling.png` — observation times of 3 test tracks

## Interpretability

Closed. Early-ckpt re-score 0/6 (`logs/step8_diagnostics/early_ckpt_gates.csv`).
`research/log/negativeresults.md`, D-008.

## Limitations (honest)

Encoder identifiability (z0 cosine still high). Const-Z non-transfer.
Window is pre-saturation / accelerating phase. No-repo checksum identity.
NLS is in-sample. The lock ODE is one contaminated 80/20 seed; three
clean 70/15/15 seeds are training under `checkpoints/h1_seed{0,1,2}/`
(jobs 29114600–602) and are excluded from this table until they finish.
Drop-protocol appendix (`figures/hypothesis_one_summary.csv`: ODE extrap
0.497 vs LSTM 0.546, p=0.30) is not this table.
