# H2 context notes (preserve for D-020; do not act)

**Date.** 2026-10-06. **Status.** Notes only. No D-020. No training.

Source: Phase 0b `gt.csv` audit (`logs/h2_data_auditfull.md`). Validate
all of this **after** extraction. Do not pre-register from length-only
upper bounds.

## Draft LOSO five (post-extraction validation)

Chosen from EPPO families, not from H1 results:

| EPPO | Family | Role |
|---|---|---|
| ARTVU | Asteraceae | eudicot |
| GALAP | Rubiaceae | eudicot |
| CHEAL | Amaranthaceae | eudicot |
| STEME | Caryophyllaceae | eudicot |
| ALOMY | Poaceae | grass / monocot |

Four eudicot families + one grass tests taxonomy-initialized
zero-shot transfer across the monocot/eudicot divide. Revisit if
extraction drops any of these below the inclusion bar.

## Inclusion bar

`≥120` usable tracks applies **after** `valid_track` + `≥15` frames,
not on `n_ge15` upper bounds. ACHMI: 122 `≥15` → **105** usable
(~14% cut). PLAMA (138) and PULDY (130) may fall below if that rate
repeats. ACHMI itself is 105 `<` 120; D-020 must revisit the bar or
its justification with the full extracted table.

## Generic `Weed`

161 tracks with `≥15` frames. Unresolved label, not a species.
Exclude from the main battery. Note in the audit.

## Maize / sorghum crops

ZEAKJ 75, ZEALP 43 (`n_ge15`). The crop in the dataset name will not
be in the H2 main battery under the current bar. SORFR (Freya) 137 is
the only crop variety above 120 on length alone.

## GA3 treatment

Tray-id digit 4 inferred as `8` = minus, `9` = plus (barcode encodes
species, GA3 +/−, replicate). Verify against official MFWD docs
before D-020. Record treatment as a per-track covariate. Stratify;
do not pool GA3+ and GA3− blindly (hormone injects a bimodal
treatment effect into “species dynamics”).

## H1 tray leakage

Official H1 length split is `track_id`-disjoint but not tray-disjoint
(train∩test = 12 shared trays). Stated limitation. D-020 must specify
tray-aware or tray-disjoint splitting for H2.

## Download

JPEG pull on scratch (`pid` 2561724 at launch, ~10 s/zip, 640 zips).
Do not interrupt. Verify manifest completeness when it finishes.
Image type verified JPEG (`FF D8`) on ACHMI `133801.zip`.

**2026-10-06 follow-up (skylark).** `/scratch` is node-local. This
host does not have pid 2561724 or the JPEG tree. Re-pull to a
shared path (or the same node) before color verification.

## Extractor (recovered into the local tree)

GitHub [`Simar0108/Continuous-Modeling-of-Weed-Growth-Trajectory`](https://github.com/Simar0108/Continuous-Modeling-of-Weed-Growth-Trajectory)
`origin/main` commit `a538b36` (“adding exploration processes of ACHMI growth”).
Remote HEAD at fetch: `3b41644` (unrelated 6-commit history vs local `f5a8c7b`).

Checked out at repo root (original CWD paths):
- `explore_achmi.py` — `gt.csv` + ACHMI timestamps (`Thesis/data/gt.csv`, `Thesis/data/jpegs/ACHMI`)
- `explore_achmi_growth.py` — writes `Thesis/exploration/metrics_with_features.parquet` via `metrics.to_parquet`
- `explore_achmi_visuals.py`
- `phase3_3_analysis.md`

SHA256 `explore_achmi_growth.py`:
`a829c6acba0abd9fe349cd45213d889f4a5af140c9339714d624d753d5d95ee3`.

`valid_track`: `missing_rate<0.5` and `end_area>5000` and `noise_ratio<0.3` and not flat (`area_smooth` var ≥ 100). Greenness = mean_G/(R+G+B). `sigma_w`/`sigma_h` are **not** in the extractor; they are `width/4`, `height/4` in `ode/data/transforms.py`. Color-mask channels live there too (`greenness * edge_density`).

No-color rerun vs lock parquet: 8,953 rows, 153 tracks, 122 `≥15`,
**105 `valid_track` IDs identical**, geometry/timestamps 1:1.
`track_quality_score` does **not** match (color missing-rate term).
Full parquet byte match needs JPEGs + skimage. Not committed to
`data/` until that run matches. `*.parquet` stays gitignored.

