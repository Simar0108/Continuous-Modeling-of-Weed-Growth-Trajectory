# H2 Phase 0b — MFWD gt.csv metadata audit (Step 16 Task 0)

**Date.** 2026-10-06. **Status. Totals match the paper. Download
scope unlocked.** No training. No D-020. H1 checkpoints untouched.

**gt.csv.** FTP `dataserv.ub.tum.de` as `m1717366` (public MFWD).
Saved `/scratch/ssing226/mfwd/gt.csv` (19,289,551 bytes). SHA256
`787d0ec82d218c0b70511e66deb070aa283e2851df91a114b62dee1fdd72c5c1`.
Also `varieties.md` (maize/sorghum tray map).

**CSV.** `logs/h2_data_auditfull.csv` (37 rows: 23 sown weeds, 5
volunteers, 6 sorghum varieties, 2 maize varieties, `Weed`).

---

## Totals vs published

| Quantity | Paper (Genze 2024) | `gt.csv` | Call |
|---|---|---|---|
| Records (bboxes) | 200,148 | **200,148** | match |
| Plant individuals (`track_id`) | 5,068 | **5,068** | match |
| Distinct `label_id` | 28 weeds + crops + extras | 37 | see below |
| Trays in `gt.csv` | — | 640 | — |

No wild deviation. Task 1 download is allowed.

`label_id` breakdown: 23 sown weeds + 5 Table-4 volunteers + generic
`Weed` (305 tracks) + 6 sorghum variety codes + 2 maize variety
codes = 37. Crops are stored as `SORFR`/`ZEALP`/… not `SORVU`/`ZEAMX`;
the filename folder is the EPPO (`SORVU/`, `ZEAMX/`).

`track_id` is **globally unique** (5,068 tracks = 5,068
label–track pairs; 0 tracks span two labels or two trays). No EPPO
prefix needed for uniqueness on this `gt.csv`.

---

## H1 filter on metadata

`valid_track` is **not** in `gt.csv`. It is a thesis extractor flag.
ACHMI is the only species where it is known (parquet):

| ACHMI | n |
|---|---|
| All tracks | 153 |
| `≥15` frames | 122 |
| `valid_track` and `≥15` (H1 usable) | **105** |
| Dropped by `valid_track` among `≥15` | 17 |

ACHMI `gt.csv` rows/tracks/trays match the parquet exactly (8,953 /
153 / 25). Geometry (`width`/`height`/centroid) is bbox-derived
(`xmax-xmin`). Color/`greenness` is finite on 43.4% of ACHMI frames.
`image_path_valid` is never True in the parquet (False 5,069; None
3,884) even on frames with color — the original JPEG lookup was
partial.

**Inclusion on `≥15` only (upper bound; `valid_track` will cut).**
Species/varieties with `n_ge15 ≥ 120`:

| label | origin | tracks | n_ge15 |
|---|---|---|---|
| ARTVU | weed | 377 | **281** |
| GALAP | weed | 265 | **211** |
| SOLNI | weed | 290 | **190** |
| ALOMY | weed | 533 | **186** |
| POAAN | weed | 277 | **186** |
| CHEAL | weed | 229 | **172** |
| Weed | generic | 305 | **161** |
| STEME | weed | 205 | **159** |
| ECHCG | weed | 284 | **145** |
| PLAMA | weed | 151 | **138** |
| AGRRE | weed | 177 | **132** |
| PULDY | weed | 274 | **130** |
| ACHMI | weed (H1) | 153 | **122** (H1-full **105**, fail) |
| SORFR | sorghum Freya | 139 | **137** |

Near misses: VEROF 119, VIOAR 117, LAMAL 104, THLAR 95, MATCH 92,
POROL 81. Volunteers AETCY 4, GERMO 4, POLAM 53, POLAV 2, VICVI 1.
Maize: ZEAKJ 75, ZEALP 43 (194 raw tracks; most are short). Other
sorghum varieties 17–59.

If ACHMI’s 122→105 (`valid_track`) rate repeats (~14% cut), PLAMA
138 and PULDY 130 can fall below 120. **Do not lock the ≥120 list
until extraction.**

---

## Treatment (GA3)

Inferred from the 4th digit of the 6-digit `tray_id`: `8` =
`ga3_minus` (525 trays), `9` = `ga3_plus` (115 trays). Not in
official MFWD docs; consistent with the paper’s “species / GA3 /
replicate” barcode and with GA3-heavy species (THLAR 94/103 plus,
VEROF 124/138 plus, VIOAR 131/174 plus). All crop trays are digit
`8`. ACHMI is 153/153 `ga3_minus`. Pool vs stratify is a Phase 1
decision. Column in CSV: `n_ga3_minus_tracks`, `n_ga3_plus_tracks`.

---

## Image type (measurement consistency)

ACHMI extraction path on disk is `Thesis/data/jpegs/ACHMI/` (empty
tray dirs; images not retained). FTP layout has `jpegs/` and `pngs/`.
**Download `img_type=jpegs`.** JPEG set ~114 GB; PNG ~814 GB. Scratch
node has ~1.5 T free. Never write images under `$HOME`.

**Verified on first ACHMI zip** (`133801.zip`, 107,803,768 bytes): 75
files, all `.jpeg`, SOI magic `FF D8 FF E0`. Filename stems match the
parquet (`ACHMI_133801_..._img`). Sequential downloader:
`/scratch/ssing226/mfwd/download_jpegs.py` (640 unique tray zips,
ACHMI first). Manifest `/scratch/ssing226/mfwd/download_manifest.csv`.

Unique labeled tray zips in `gt.csv`: **640** (ACHMI 25). Volunteer
plants live inside the sown-species zip (e.g. SOLNI inside
`ACHMI/133801`). Downloading each unique `sown/tray.zip` once pulls
all 37 labels.

---

## Extractor status (Task 2 gate)

The thesis trajectory extractor (greenness, edge_density,
`track_quality_score`, `valid_track`, germination time) is **not in
the repo**. `Thesis/` contains only the ACHMI parquet. Task 2 cannot
run “the exact ACHMI code path” until that module is restored. Do
not invent a replacement extractor. Geometry for ACHMI is already
recoverable from `gt.csv` bboxes; `valid_track` is not.

---

## Download scope (unlocked)

Sequential FTP, `jpegs`, all unique 640 tray zips from `gt.csv`
(28 weeds + maize + sorghum folders). Target
`/scratch/ssing226/mfwd/jpegs/`. Manifest on scratch. No concurrent
FTP.

---

## NOT-verified

- `valid_track` / H1-usable counts for any label except ACHMI.
- Official meaning of tray-id digit 4 (GA3 inference).
- JPEG pixel parity with the original ACHMI color features.
- Tray-disjoint split design (D-020, gated).
- E1/E2/E3.
