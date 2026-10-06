# H2 Phase 0 / 0b — MFWD data audit

**Date.** 2026-10-06. **Status. Metadata complete. Extractor missing.
No D-020. No training.** Full-28 table:
`logs/h2_data_auditfull.csv` + `logs/h2_data_auditfull.md`. Provenance:
D-021 (H1 lock is ACHMI, not maize).

**gt.csv totals match the paper:** 200,148 records, 5,068 plants.
On `≥15` frames, 12 sown weeds + `Weed` + sorghum Freya clear 120
(upper bound). ACHMI H1-full usable remains **105** (`valid_track`).
`valid_track` is unknown for every other label until the thesis
extractor is restored.

Phase 0 parquet-only notes below are kept for ACHMI geometry / plant
ID checks. They are not the 28-species inventory.

---

# Phase 0 parquet-only (ACHMI) — retained

**Parquet identity.** `metrics_with_features.parquet` =
`Thesis/metrics_with_features.parquet`, 1,601,399 bytes, SHA256
`97d6c35ddb755fedeab7bf61c2364eaa3cd2a16de1ba777b9e061308ef9ad286`.

**Pipeline used (H1, no new extraction).** Same filters as
`ode.train_baselines.select_discrete_tracks` +
`PlantTrackStateDataset`: `valid_track==True`; dropna on
`timestamp`, `centroid_x/y`, `width`, `height`,
`time_since_germination_hours`; usable = tracks with `≥15` frames.
State extraction is unchanged: `σ_w = width/4`, `σ_h = height/4`,
`Δt` from consecutive `time_since_germination_hours` (encoder uses
normalized context `Δt`; ODE integrates on absolute hours). No
training.

CSV: `logs/h2_dataaudit_species.csv`.

---

## Headline

The on-disk trajectory table contains **one** of 28 MFWD weed
species: **ACHMI** (*Achillea millefolium*). **105 usable tracks.**
That is **below the ≥120 inclusion bar.** The other **27 species are
absent** (0 tracks). H2 cannot be pre-registered from this parquet
until the remaining species are extracted through the same pipeline.

H1 lock docs say “Maize 70/15/15” and trainers pass `--species Maize`.
The parquet has **no `species` column**, so that filter is a silent
no-op (`datamodule.py` / `select_discrete_tracks`). Every row has
`label_id=ACHMI`. The locked 100-track run is ACHMI, not *Zea mays*.

---

## Inventory (28 weed species)

MFWD = Moving Fields Weed Dataset (Genze et al., Sci Data 2024;
https://doi.org/10.1038/s41597-024-02945-6). Twenty-three sown weeds
(Table 2) + five volunteer dicots (Table 4) = 28. Crops in Table 3
(SORVU 6 varieties, ZEAMX 2 maize varieties) are **not** in the 28
and are also absent from the parquet.

| EPPO | Species | Family | In parquet | Usable tracks (≥15) | <120 |
|---|---|---|---|---|---|
| ACHMI | *Achillea millefolium* | Asteraceae | yes | **105** | **yes** |
| AGRRE | *Elymus repens* | Poaceae | no | 0 | yes |
| ALOMY | *Alopecurus myosuroides* | Poaceae | no | 0 | yes |
| ARTVU | *Artemisia vulgaris* | Asteraceae | no | 0 | yes |
| CHEAL | *Chenopodium album* | Amaranthaceae | no | 0 | yes |
| CIRAR | *Cirsium arvense* | Asteraceae | no | 0 | yes |
| CONAR | *Convolvulus arvensis* | Convolvulaceae | no | 0 | yes |
| ECHCG | *Echinochloa crus-galli* | Poaceae | no | 0 | yes |
| GALAP | *Galium aparine* | Rubiaceae | no | 0 | yes |
| GASPA | *Galinsoga parviflora* | Asteraceae | no | 0 | yes |
| LAMAL | *Lamium album* | Lamiaceae | no | 0 | yes |
| MATCH | *Matricaria chamomilla* | Asteraceae | no | 0 | yes |
| PLAMA | *Plantago major* | Plantaginaceae | no | 0 | yes |
| POAAN | *Poa annua* | Poaceae | no | 0 | yes |
| POLCO | *Fallopia convolvulus* | Polygonaceae | no | 0 | yes |
| POROL | *Portulaca oleracea* | Portulacaceae | no | 0 | yes |
| PULDY | *Pulicaria dysenterica* | Asteraceae | no | 0 | yes |
| SOLNI | *Solanum nigrum* | Solanaceae | no | 0 | yes |
| SSYOF | *Sisymbrium officinale* | Brassicaceae | no | 0 | yes |
| STEME | *Stellaria media* | Caryophyllaceae | no | 0 | yes |
| THLAR | *Thlaspi arvense* | Brassicaceae | no | 0 | yes |
| VEROF | *Veronica officinalis* | Plantaginaceae | no | 0 | yes |
| VIOAR | *Viola arvensis* | Violaceae | no | 0 | yes |
| AETCY | *Aethusa cynapium* | Apiaceae | no | 0 | yes |
| GERMO | *Geranium molle* | Geraniaceae | no | 0 | yes |
| POLAM | *Persicaria amphibia* | Polygonaceae | no | 0 | yes |
| POLAV | *Polygonum aviculare* | Polygonaceae | no | 0 | yes |
| VICVI | *Vicia villosa* | Fabaceae | no | 0 | yes |

**Main battery if H2 launched on this parquet: empty.** ACHMI fails
`≥120`. Supplemental pooled analysis would be ACHMI-only (same as H1).

Literature image/tray counts (paper, not pipeline) are in the CSV.
Volunteer literature plant individuals: AETCY 4, GERMO 4, POLAM 55,
POLAV 2, VICVI 1. Those five cannot reach 120 even after extraction
unless the parquet disagrees with the paper.

---

## ACHMI through the H1 extractor

| Filter | Tracks | Frames |
|---|---|---|
| All rows | 153 | 8,953 |
| `valid_track==False` (dropped) | 48 | 734 |
| `valid_track==True` | 105 | 8,219 |
| + dropna required cols | 105 | 8,219 |
| + `≥15` frames (**usable**) | **105** | 8,219 |
| H1 cap `max_tracks=100` | 100 | (richest 100 of 105) |

`valid_track` is constant within a track (0 mixed). Invalid tracks
are short (median 13 frames, min 1, max 86). Every valid track already
has ≥19 frames, so min-8 vs min-15 does not change the usable set.

### Frame-count distribution (105 usable)

mean 78.3, std 37.8, min 19, p25 58, median 70, p75 85, p90 146, max 214.

| Bin | n tracks |
|---|---|
| 15–19 | 2 |
| 20–29 | 5 |
| 40–49 | 9 |
| 50–59 | 11 |
| 60–79 | 46 |
| 80–99 | 17 |
| 100–199 | 14 |
| 200+ | 1 |

H1 length-split of the richest 100: train 70 tracks (65–214 frames),
val 15 (29–53), test 15 (53–65). Same short-track transfer as the lock.

### Δt and size (usable)

- `Δt` hours: mean 13.51, median 11.88, min 0.10, max 70.99. No
  negative or zero steps.
- Horizon span hours: mean 1044, median 1020, min 228, max 2305.
- `σ_w` px/4: mean 74.9, std 76.3, range 2.5–421.8.
- `σ_h` px/4: mean 72.1, std 71.7, range 2.75–374.3.
- Greenness finite on 40.2% of usable frames. 54/105 tracks have
  **zero** Z frames; 51/105 have Z on every frame. Physiology is
  already gated by `color_mask` (do not change `hybrid_loss`).

### Imbalance

With one species, species-level imbalance is undefined (ACHMI = 100%
of usable tracks). Within ACHMI, tray occupancy is 2–8 plants/tray
(median 4) across 25 trays. Frame-count CV = 37.8/78.3 ≈ 0.48.

---

## Plant-level identity

**Exposed key = `track_id`.** There is no `plant_id` column. MFWD
`gt.csv` defines `track_id` as the individual time series connecting
bboxes of one plant in one tray (Genze et al., Table 6). On this
parquet:

| Check | Result |
|---|---|
| `plant_id` column | **absent** |
| `species` column | **absent** (species = `label_id` EPPO) |
| Tracks spanning >1 tray | 0 |
| Tracks spanning >1 label | 0 |
| `bbox_id` spanning >1 track | 0 |
| Mixed `valid_track` within a track | 0 |
| Global uniqueness of `track_id` on this file | yes (153 distinct) |

Every plant is partitionable by `track_id`. A track-level 70/15/15
has empty train/val/test plant-ID intersection **by construction**
(verified on the H1 100-track length split: 0 ∩ 0 ∩ 0).

**Not the same as tray-level.** The H1 length split leaks trays:
train∩val 8 trays, train∩test 12, val∩test 5. Multiple plants share
a tray (2–8). Phase 1 must say whether “plant-level” means `track_id`
only (H1) or also tray-disjoint. Anti-contamination SHA256 of split
manifests still applies.

**Not verified for 28 species.** `track_id` uniqueness is only shown
inside ACHMI. When other species are added, Phase 1 must prove
`track_id` is globally unique (or namespace by `label_id`) so one
plant cannot land in two splits.

---

## Where the other 27 species are (not)

Searched, not present:

- `MFWD/data` and `MFWD/models` — empty dirs; submodule not populated.
- `Thesis/data/jpegs/` — `ACHMI/` only.
- `/bigdata/mcgiffenlab/ssing226` and `.../shared` — empty.
- No other `*.parquet` under the repo.

H2 extraction of the remaining species is **blocked on data access**
(MFWD download: https://github.com/grimmlab/MFWD,
https://mediatum.ub.tum.de/1717366). That is data ingest, not
training, and is outside this Phase 0 pass.

---

## Numbering conflict

`research/log/decisions.md` already has **D-019 = encoder-collapse
probes**, outcome logged 2026-10-06 (H-enc-1 rejected; no contrastive
arm). The H2 protocol in the query reused D-019. H2 pre-registration
should be **D-020**. Do not overwrite D-019.

Step 15 is **done** on disk (`research/conclusions/encoder.md`,
`figures/encoder_probes/`). H2 **training** still waits on explicit
go-ahead. This audit is not that go-ahead.

---

## Blockers for Phase 1 (do not skip)

1. **27/28 species missing** from the trajectory table.
2. **ACHMI 105 < 120** → no species enters the main battery under the
   stated inclusion rule.
3. **Cannot pre-specify five LOSO species from this audit.** LOSO
   “chosen from Phase-0 counts before training” has no five species
   with usable tracks. Do not pick LOSO from literature image counts.
4. **H1 “Maize” label is wrong** for this parquet. Register ACHMI
   (or ingest ZEAMX) before writing the claim.
5. **`species` column missing.** Conditioning needs an explicit
   species field (`label_id` is the present EPPO).

---

## NOT-verified

- Per-species usable-track counts for any EPPO except ACHMI.
- Global `track_id` uniqueness across species.
- Whether ingested MFWD `gt.csv` plant counts will exceed 120 for
  any sown species.
- Taxonomy-neighbor distances for frozen embeddings (Phase 1).
- E1/E2/E3, NFE, z0 cosine, drop/prefix-60 (no H2 training).
- Implementation diffs (Phase 2, gated).

---

## Receipt (Phase 0 only)

| Item | Status |
|---|---|
| Phase-0 audit numbers | this file + CSV |
| D-019/D-020 H2 text | **not written** (wait) |
| Implementation plan | **not written** (wait) |
| Per-job git hashes | n/a (no jobs) |
| E1/E2/E3 | unmeasured |
| Per-species tables | CSV; 27 species = 0 |
| H1 artifacts | untouched |
