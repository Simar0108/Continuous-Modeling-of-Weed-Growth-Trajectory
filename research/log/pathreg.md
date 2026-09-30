# D-015 draft — pathreg arm (not the H1 lock)

**Date.** 2026-09-30.

Coarse 3-point path-length sweep (λ ∈ {0.01, 0.1, 1.0}) on the locked
StableSigmoidal architecture, 5 seeds, EMA 0.999, `val_full_horizon_mse`
selection, `dopri5` 1e-6. Writes `checkpoints/h1_pathreg_l{λ}_seed{0..4}/`
only. Does not write `h1_final_best` or `h1_stab`.

Stopping table: z0 cosine vs ~0.996, in-window test MSE vs 0.245,
prefix-60/tail-40 vs 1.869. Do not tune λ further until that table is
reviewed. `eval/summarize_pathreg.py`.
