# Step 10 receipt (amended)

**HEADER — contradiction of the revised H1.** The clean 3-seed mean is
worse than LSTM, GRU, and Transformer on short-track val and test
(Wilcoxon alternative=less, all p>0.96). LSTM parity, GRU test win, and
“stable transfer vs Transformer” are **unsupported** as 3-seed claims.
Two of three `best.ckpt` files are epoch 0. Drop-protocol and
prefix-60%/tail-40% also fail for the 3-seed mean. The ODE loses
extrapolation; that belongs in the paper.

## Task 0 — Version control

- `git init -b main` already done (`87a8ee3`).
- Tag `h1-lock` = `17fdb1f` (annotated tag object `06bc7142`).
- HEAD after eval commit: `b3f9ec2`.
- `.cursor/rules/research-ode.mdc`: `h1_final_best/` read-only.
- Origin set, **not pushed**.
- Verify: `git rev-parse HEAD` is a real hash; lock SHA256 unchanged.

## Task 1 — Clean ODE seeds (headline)

Jobs 29114600–602 COMPLETED. No retrain. Eval 29115924 COMPLETED 4m18s.

| Seed | Job | best.ckpt epoch | Val mean | Test mean | Test median |
|---|---|---|---|---|---|
| 0 | 29114600 | 19 | 0.197 | 0.197 | 0.141 |
| 1 | 29114601 | 0 | 0.520 | 0.808 | 0.644 |
| 2 | 29114602 | 0 | 0.478 | 0.740 | 0.550 |
| ode_clean | — | — | 0.398 | 0.582 | 0.439 |
| ode_leaked | — | 20 | 0.222 | 0.215 | 0.149 |

ode_clean_seedmean: val 0.398±0.176, test 0.582±0.335.
lstm_clean: val 0.204 / 0.121, test 0.218 / 0.126.
gru_clean: val 0.242 / 0.203, test 0.370 / 0.278.
transformer_clean: val 0.159 / 0.104, test 0.306 / 0.199.

Wilcoxon ode_clean vs lstm_clean test: p=0.9997, d=+1.46.
vs gru_clean test: p=0.99997, d=+2.14.
vs transformer_clean test: p=0.995, d=+0.67.

Split verified: 70/15/15, train∩test empty. Test IDs
[5216, 5190, 5234, 6306, 5206, 5228, 5180, 5177, 5175, 6300, 5179, 5195, 5212, 5233, 6883].

**Curriculum caveat (all best epochs < 100).** Same selection rule as
the leaked lock (epoch 20). Seed 0 last-epoch val=0.393 (W&B td89ush5,
epoch 399). Last-epoch weights not on disk.

Training diagnostics (seed 0, mid-run epoch ~27): `train_dz_dt_size_std≈0.002`,
`z_traj/acceleration_ratio≈0.42`, `val_track_mse_mean≈0.144`, r/K nearly
shared (r_std≈0.01). Affine scale still ~6.

Artifacts: `figures/h1_lock/table1_v2.csv`, `table2_v2.csv`,
`table2_per_seed.csv`, `headline.json`.

## Task 2 — Matched-rollout

`figures/h1_lock/matched_rollout.csv`. Leaked test Δ(full−post)=−0.011.
ode_s0 test Δ=−0.009. Asymmetry is small. Not retuned.

## Task 3 — Drop protocol

`drop_summary.csv`, `drop_curves.png`, `drop_wilcoxon.csv`.
3-seed ODE test ~0.60–0.63 vs LSTM ~0.22–0.25. No advantage. Not retuned.
ode_s0 alone ~0.18–0.20 (LSTM-like).

## Task 4 — Baseline Methods

Verified against `ode/baselines.py` `_future_inputs`. Paper
`paper/draft.md`: no imputation, per-track t→[0,1], LSTM/GRU
[σw,σh,Z,Δt], Transformer +t, K=3, TF train / AR val-test. Drop
paragraph added.

## Task 5 — Extrapolation

Jobs 29114603–614 COMPLETED. `extrap_summary.csv`, `extrap_curves.png`.
ode_clean test tail 2.27 vs LSTM 0.97–1.13, p=0.99997. **ODE loses.**

## Task 6 — Claim rewrite

`research/conclusions/h1_lock.md` rewritten. D-010–D-013 outcomes filled.
Every headline sentence cites a statistic. Step 9 pairwise claims marked
superseded.

## Revised H1 clauses

1. LSTM parity (clean 3-seed, means and medians) — **unsupported**
2. GRU test significance — **unsupported**
3. Stable val→test vs Transformer — **unsupported** (3-seed mean)
4. Drop-protocol advantage — **unsupported** (fail-open, not retuned)
5. Prefix-60% / tail-40% advantage — **unsupported** (ODE loses)

Seed 0 alone still looks like Step 9’s leaked row (test 0.197 vs LSTM
0.218). That is **provisional / not the model of record**.

## NOT verified

- Git-clean launches of 29114600–614 (`--allow-dirty`, hash `4d13054`)
- Last-epoch weights (only `best.ckpt` on disk)
- Playbook retro, H2, `git push`
- Whether selecting epoch ≥250 would change Table 1 (would be a new
  protocol; not done)
- ode_nfe on the scored checkpoints
