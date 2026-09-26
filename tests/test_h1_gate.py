"""H1 gates after Step 8: utilization replaces r_std.

Hard fail if val_horizon_frac != 1.0.

Gates:
  - no Zwietering parameter in the outer 5% of its reporting range
    (zwiet/near_bound_any == 0); T1 control uses r-at-ceiling as the analog
  - train_dz_dt_size_std > 0.05
  - accel > 1.0
  - val_track_mse_mean < 0.40
"""

from __future__ import annotations

print("Step 8 gate helpers live in ode/rhs_families.param_near_bound_frac")
print("Train-time logs: zwiet/near_bound_{mu,lambda,nu,t0,any}")
