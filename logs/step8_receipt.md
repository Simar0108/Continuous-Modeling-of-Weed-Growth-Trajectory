# Step 8 receipt — Zwietering–Richards + alignment

Recorded 2026-09-26 after all 12 jobs finished (Slurm 29099660–671, ~17:12–18:34 PDT 2026-09-25). W&B project `latent-ode-maize-100`.

## Established (not relitigated)

- K-in-window unidentifiable (RGR late/early 0.57, NLS K=279 px, latent bound 2).
- S3 (K=10, exp rate) is the only Step 7 arm that accelerated on all seeds.
- Peak dσ_w/dt spread = 1597 h.
- Lag is a parameter (λ), not a gate.

## What launched

100-track 70/15/15, 300 epochs, 3 seeds, horizon 1.0, `--allow-dirty`
(no git repo on disk; jobs log `git_hash=NO_REPO`).

| Arm | Flags | W&B names / ids |
|---|---|---|
| T1 | S3 control: `--rhs exprate` | `h1-s8-T1-s3-s{0,1,2}` (`qojb6fmp`, `to2872c1`, `5tr2e8ph`) |
| T2 | Zwietering μ/λ/ν, K=10, const Z, no t0 | `h1-s8-T2-zwiet-s{0,1,2}` (`izhdszlw`, `6sgbs7zq`, `ziegwivm`) |
| T3 | T2 + t0 ∈ [-0.3, 0.3] | `h1-s8-T3-zwiet-t0-s{0,1,2}` (`3prx48mi`, `dwd8jbho`, `dbmln7qx`) |
| T4 | T3 + Wilcoxon vs `h1_final_best` and vs T1 | `h1-s8-T4-zwiet-t0-wilcox-s{0,1,2}` (`nnmm9gkv`, `nf119cvz`, `xs4oam2u`) |

T3 and T4 are the same training (identical last-epoch metrics per seed). T4 only adds the Wilcoxon dump. λ and t0 alias as (λ+t0) in the Zwietering IC.

## Gates (revised) — 0 / 12 pass

All three seeds must pass. Last-epoch numbers (horizon 1.0):

| Gate | T1 | T2 | T3 = T4 |
|---|---|---|---|
| `near_bound_any == 0` (T1: r not at sigmoid ceiling) | **fail** (r ≡ 2.0, satfrac=1) | **fail** (`near_bound_λ=1`) | **fail** (`near_bound_λ=1`) |
| `train_dz_dt_size_std > 0.05` | **fail** (0.031±0.006) | **fail** (0.0048±0.0007) | **fail** (0.0060±0.0014) |
| `accel > 1.0` | pass (1.31±0.06) | pass (1.38±0.22) | pass (1.40±0.05) |
| `val_track_mse_mean < 0.40` | **fail** (0.552±0.071) | pass (0.362±0.040) | **fail** (0.402±0.020; s2=0.429) |
| 3-seed rule | **no** | **no** | **no** |

**H1 remains unlocked.** No arm beats `h1_final_best` (Wilcoxon per-track size MSE 0.152).

## Per-arm last-epoch table

| Run | val MSE | dz/dt std | accel | NFE | notes |
|---|---|---|---|---|---|
| T1-s0 | 0.562 | 0.030 | 1.230 | 40 | r=2.000, K=10, satfrac=1 |
| T1-s1 | 0.634 | 0.039 | 1.364 | 40 | same pin |
| T1-s2 | 0.461 | 0.025 | 1.345 | 40 | same pin |
| T2-s0 | 0.384 | 0.0054 | 1.261 | 28 | λ near-bound |
| T2-s1 | 0.397 | 0.0038 | 1.194 | 22 | λ near-bound |
| T2-s2 | **0.307** | 0.0052 | 1.698 | 34 | best last-epoch val |
| T3-s0 | 0.381 | 0.0064 | 1.372 | 28 | = T4-s0 |
| T3-s1 | 0.397 | 0.0075 | 1.360 | 28 | = T4-s1 |
| T3-s2 | 0.429 | 0.0041 | 1.469 | 28 | = T4-s2 |

Early-training minima (epoch ~12–27) are much lower (T1 0.19–0.25, T2 0.14–0.18, T3 0.13–0.17) then degrade. Best-ckpt is not the number the gates use; gates are last-epoch.

`ode_nfe` 22–40, under 60 on every run.

## μ / λ / ν / t0 (epoch 299)

Init of the softplus heads is `softplus(0)+1e-4 ≈ 0.693` and `ν = 0.25+softplus(0) ≈ 0.943`. Reporting ranges: μ∈[1e-3,8], λ∈[1e-3,0.6], ν∈[0.25,10], t0∈[-0.3,0.3]. Outer 5% of λ is λ>0.570.

| | T2 mean±std (across seeds) | T3 mean±std | vs init | vs range |
|---|---|---|---|---|
| μ | 0.762±0.004 | 0.776±0.011 | +0.07 | interior (near_bound_μ=0) |
| λ | 0.711±0.007 | 0.720±0.004 | +0.02 | **above 0.6** → near_bound_λ=1 |
| ν | 0.9395±0.0007 | 0.9384±0.0005 | −0.005 | interior but **collapsed** (std~0.002 within seed) |
| t0 | 0 (off) | 0.020±0.004 | from 0 | interior (near_bound_t0=0); within-seed std 0.007–0.013 |

λ plausibility window was ~0.15–0.25 of normalized time (~277–461 h if the window is ~1845 h). Observed λ≈0.71 is ~1300 h — most of the window, not a lag. t0≈0.02 is only ~30–46 h, vs the 1597 h peak-growth spread. t0 does not absorb registration.

λ+t0 alias: T3 (λ+t0) ≈ 0.736–0.750 vs T2 λ ≈ 0.703–0.720. The extra name did not help last-epoch val (T3 0.402 > T2 0.362).

`phase/r_mean≈1.0` and `phase/K_mean≈1.15` on T2/T3 are leftover unused default heads, not μ/K.

## Constant-Z

`val_mse_Z` is **identically 1.066** on all 9 T2/T3/T4 runs — the same unseen-val number as Step 7 const-Z. The per-track Z table is an H1 (known-track) device; val tracks are unseen, so Z reconstruction is a constant miss. Size channels can still score 0.30–0.43.

## Wilcoxon (T4, n=15 val tracks, alternative="less")

| Seed | ours | vs h1_final_best (ref 0.152) | vs T1 (loaded ckpt) |
|---|---|---|---|
| 0 | 0.381 | p=0.995 | p=0.932 (ref 0.221) |
| 1 | 0.397 | p=0.982 | p=0.979 (ref 0.145) |
| 2 | 0.429 | p=0.998 | p=0.994 (ref 0.151) |

T4 is **worse** than `h1_final_best` on every seed. The vs-T1 column is **not trustworthy**: T1 was loaded with `strict=False` without reinstalling `ExpRateODEFunc`, and the reported T1 refs (0.15–0.22) match `h1_final_best` (~0.152), not T1’s logged val (0.46–0.63). On logged last-epoch `val_track_mse_mean`, T2/T3 actually beat T1.

## What was NOT verified

- Git-clean launches (no repo; `--allow-dirty` used).
- Separate identifiability of λ vs t0 (they alias; T2 vs T3 is only a registration check).
- Test-split scores.
- Wilcoxon vs a correctly re-wrapped T1 checkpoint.
- Whether early-min checkpoints (epoch 12–27, val 0.13–0.18) would pass dynamics gates — they were not the monitored last-epoch state.
- Restored `.py` sources for `ode/model.py` (bytecode bootstrap still in use).
