# H1 archival manifest (`v-h1-final`)

**Date.** 2026-10-08. Reference only: no checkpoint was copied or moved.
D-020 is unwritten. Species is yarrow (ACHMI); D-021.

## Tag

| Field | Value |
|---|---|
| Tag | `v-h1-final` (annotated) |
| Message | `H1 model of record, textual finalization` |
| Tagged commit | `72d3009ada7730ed281bab0de187be59bdfe9d94` |
| Tag object SHA1 | `538c2eeb3d4ab6de00e55ee5b0f59952445cd014` |
| `v-thesis-freeze-candidate` | `c563024c7e6d0ce119c5fcba33951f2e07c1dada` |
| `h1-lock` | `17fdb1f5ddeb8fecf0b0ffadb31e3fb5c725a0a2` |

The tagged commit contains H1 history through D-021 and the Task 1–3
textual pass. This manifest and D-022 are committed after the tag.

## Model-of-record checkpoints (`h1_stab_seed{0..4}`)

Scored files match `figures/h1_lock/run_validity.csv` (seeds 3–4:
`best-v1.ckpt`, epoch 399). Stale `best.ckpt` on seeds 3–4 is the
quota-crash leftover (D-014); hashed, not scored.

| Seed | Path (not copied) | Bytes | SHA256 | Role |
|---|---|---|---|---|
| 0 | `checkpoints/h1_stab_seed0/best.ckpt` | 7817594 | `f1eefab27541907d266a860c1db8a134fdcce880c044008ef7c27e07febd6d5e` | scored |
| 1 | `checkpoints/h1_stab_seed1/best.ckpt` | 7817594 | `932abcc59f03043ded0fb068ec76c3b75ad04ea3d4e6c665eef1e878c99df848` | scored |
| 2 | `checkpoints/h1_stab_seed2/best.ckpt` | 7817594 | `c1ed251e35ad77c6ae69e8d531646fd8bd3bf776d28c1bef770bbdb6950bbb75` | scored |
| 3 | `checkpoints/h1_stab_seed3/best-v1.ckpt` | 7818296 | `f6b45144b43ffd36db5daec818ad280fe7165bfdd98a290360a0e4aa00d35579` | scored |
| 3 | `checkpoints/h1_stab_seed3/best.ckpt` | 7817530 | `aadbb444993d555fd53b12266042b9475e2d6e8304e504fe1c480d56b32936b8` | stale, not scored |
| 4 | `checkpoints/h1_stab_seed4/best-v1.ckpt` | 7818296 | `ce4ec778aee837a4756a45740e4f8c0167ee7af58f0ee3b001debc3a53e16d27` | scored |
| 4 | `checkpoints/h1_stab_seed4/best.ckpt` | 7817530 | `3d01d30d51788c595d0d2b9b1320a8031a2625b4868dc26c480608d723b471c1` | stale, not scored |

Leaked lock (excluded from all H1 means; read-only, D-009):

| Path | SHA256 |
|---|---|
| `checkpoints/h1_final_best/best.ckpt` | `2498d033ea02e2df1e312a58179226b649a9d3d5c7143b2d040fc9341ef86222` |

Prefix-60 retrains (D-016 tail; referenced, not copied):

| Seed | Path | SHA256 |
|---|---|---|
| 0 | `checkpoints/h1_stab_seed0_extrap60/best.ckpt` | `655a9742a3c2f4362ad2a7c3d5f81c253accac788bffae0ed38c2afded9e4d5c` |
| 1 | `checkpoints/h1_stab_seed1_extrap60/best.ckpt` | `f9b4780c2fb9195477062bb8bd3eeb2bbe75ce7d4ce57e2d9e65e9865f86cb29` |
| 2 | `checkpoints/h1_stab_seed2_extrap60/best.ckpt` | `29ad4af272fb043e32b4f84939dcf3f3d05fe1d4840bfa5ed3c42ae1f0e73d8f` |
| 3 | `checkpoints/h1_stab_seed3_extrap60/best.ckpt` | `f19bf4b2c554884cc1a3bb8e63f9485d142200a9b9a1532888f2176e89acb315` |
| 4 | `checkpoints/h1_stab_seed4_extrap60/best.ckpt` | `5157c64844c0f71939f1d881c574ef9bf21cfc03e1d716b22092f7c647065d01` |

## Split

No standalone split JSON exists in the tree. The official H1 70/15/15 is
`ode.train_baselines.select_discrete_tracks` on the 100 richest valid
ACHMI tracks. Lock inventory: `figures/h1_lock/per_track_size_mse.csv`
(`ode_clean` rows; SHA256 below). Track-id disjoint; not tray-disjoint
(train∩test = 12 trays, D-021).

**train (n=70).** 5145, 5146, 5147, 5148, 5149, 5150, 5151, 5152, 5153,
5154, 5155, 5156, 5157, 5158, 5159, 5160, 5161, 5162, 5163, 5164, 5165,
5166, 5167, 5168, 5169, 5170, 5171, 5172, 5173, 5174, 5178, 5181, 5182,
5183, 5184, 5186, 5187, 5188, 5189, 5191, 5192, 5193, 5194, 5196, 5197,
5198, 5200, 5201, 5202, 5203, 5204, 5205, 5207, 5208, 5209, 5210, 5211,
5220, 5229, 5230, 5231, 5235, 5242, 5253, 6301, 6302, 6303, 6304, 6305,
6916.

**val (n=15).** 5176, 5185, 5199, 5215, 5217, 5221, 5222, 5227, 5232,
5236, 5243, 6917, 6922, 6923, 6924.

**test (n=15).** 5175, 5177, 5179, 5180, 5190, 5195, 5206, 5212, 5216,
5228, 5233, 5234, 6300, 6306, 6883.

## Lock tables (`figures/h1_lock/`)

| File | SHA256 |
|---|---|
| `table1_v2.csv` | `1a08e467e2c4873446fa0811e764768326d52c3ffdf6e48d8bda5a15f0e19e12` |
| `table2_v2.csv` | `1e48c338e54b86682ee198d18567c12113b85b74252d372b70d2a5202b1ca1d5` |
| `table2_per_seed.csv` | `54a8f3d43f9bdabe5b71cfd3ad9366fb8c848f3b6e1f67a372f0f9a6e239cf89` |
| `drop_summary.csv` | `5b156465bb3c2d0c90263a4efb64d2c5bf8b64ca6a9d8600b7b3a4241a26e71c` |
| `extrap_summary.csv` | `8a8e47dac99888d3ecd65e4f22a223a77760770b161e8b124f6b915db9903640` |
| `headline.json` | `6e22e49995267050c7ecd51a9eb5d71c32237e89a66cd3b25cd24a9ff3424268` |
| `per_track_size_mse.csv` | `5c90abe415fd1e904636f9b847efd4e452ac78680fa19d73e3dfd99913c2355a` |
| `run_validity.csv` | `1f6370f82af4af7a8696103e8045d692aac60a2e3433db37cc84c5e811a9d218` |
| `matched_rollout.csv` | `69800bda3df26532eeda1b43fb28367a5d4026b61e5c177a04949f5f6ba8b2ac` |

## Decision log and parquet

| Artifact | SHA256 at `v-h1-final` |
|---|---|
| `research/log/decisions.md` (through D-021; before D-022) | `fe02c17a2d3d3a3b735ce316684005085a47772c7159426dd0e029006ee359b1` |
| `metrics_with_features.parquet` | `97d6c35ddb755fedeab7bf61c2364eaa3cd2a16de1ba777b9e061308ef9ad286` |

## Not copied / not written

- No file under `checkpoints/` was copied, moved, or overwritten.
- D-020 (H2 pre-registration) was not written.
- Maize-named checkpoint and log filenames were not renamed.
