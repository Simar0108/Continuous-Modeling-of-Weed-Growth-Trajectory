"""Re-score Step 8 T2/T3 best-val checkpoints against the interpretability gates.

Loads each ``step8_valmse_h1-s8-T{2,3}-*.ckpt``, reinstalls the Zwietering
RHS (so weights are not dropped), evaluates at horizon 1.0 on the Step 8
split (``species=None``), and writes a gate table. Does not train.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ode._pyc_bootstrap import bootstrap

bootstrap()

import torch
from lightning import Trainer
from torch.utils.data import DataLoader

from ode.data.datamodule import PlantTrackDataModule, _collate_variable_length_tracks
from ode.data.dataset import PlantTrackStateDataset
from ode.overfit_test import OVERFIT_KINETIC_REG_WEIGHT, OVERFIT_LR, OVERFIT_Z0_REG_WEIGHT
from ode.rhs_families import param_near_bound_frac
from ode.train_baselines import select_discrete_tracks
from ode.train_multi import MultiTrackLightning
from ode.train_step8 import _install
from ode.training_loop import LINEAR_SIZE_WEIGHT, LOG_SIZE_EPS

REPO = Path(__file__).resolve().parent.parent
PARQUET = REPO / "metrics_with_features.parquet"
OUT = REPO / "logs" / "step8_diagnostics" / "early_ckpt_gates.csv"

ARMS = [
    ("T2", "step8_valmse_h1-s8-T2-zwiet-s0.ckpt", False, 0),
    ("T2", "step8_valmse_h1-s8-T2-zwiet-s1.ckpt", False, 1),
    ("T2", "step8_valmse_h1-s8-T2-zwiet-s2.ckpt", False, 2),
    ("T3", "step8_valmse_h1-s8-T3-zwiet-t0-s0.ckpt", True, 0),
    ("T3", "step8_valmse_h1-s8-T3-zwiet-t0-s1.ckpt", True, 1),
    ("T3", "step8_valmse_h1-s8-T3-zwiet-t0-s2.ckpt", True, 2),
]


def _build_module():
    return MultiTrackLightning(
        kinetic_reg_weight=OVERFIT_KINETIC_REG_WEIGHT,
        z0_reg_weight=OVERFIT_Z0_REG_WEIGHT,
        head_lr_mult=10.0,
        diversity_weight=0.0,
        geometry_size_loss="log",
        log_size_eps=LOG_SIZE_EPS,
        linear_size_weight=LINEAR_SIZE_WEIGHT,
        xy_loss_weight=0.0,
        size_loss_weight=1.0,
        phys_loss_weight=0.5,
        phys_dropout=0.2,
        latent_dim=64,
        n_context_frames=3,
        encoder="mlp",
        ode_hidden=256,
        normalize_z0=True,
        ode_growth_bias=True,
        ode_solver="dopri5",
        ode_rtol=1e-6,
        ode_atol=1e-6,
        lr=OVERFIT_LR,
        horizon_start_frac=1.0,
        horizon_ramp_start=0,
        horizon_ramp_end=0,
        affine_decoder=True,
        affine_scale_init=6.0,
        freeze_affine_epochs=50,
        hybrid_decoder=True,
        late_head_lr_scale=0.2,
        use_track_embed=True,
    )


def _datamodule(parquet: Path):
    train_ids, val_ids, _test_ids = select_discrete_tracks(
        parquet, max_tracks=100, species=None, min_observations=15,
        val_frac=0.15, test_frac=0.15,
    )
    dm = PlantTrackDataModule(
        parquet_path=parquet, batch_size=min(32, len(train_ids)),
        num_workers=0, sigma_mode="z_score",
    )
    dm.setup()
    dm._train_ds = PlantTrackStateDataset(
        dm.parquet_path, dm.transformer, track_ids=train_ids,
        valid_track_only=True, min_observations=15,
    )
    dm._val_ds = PlantTrackStateDataset(
        dm.parquet_path, dm.transformer, track_ids=val_ids,
        valid_track_only=True, min_observations=15,
    )
    dm.lock_setup()
    dm.val_dataloader = lambda: DataLoader(
        dm._val_ds, batch_size=len(val_ids), shuffle=False, num_workers=0,
        collate_fn=_collate_variable_length_tracks,
    )
    return dm, train_ids, val_ids


def _t_vec(batch):
    t_abs = batch["t_absolute"]
    lengths = batch.get("track_lengths")
    if lengths is not None and t_abs.dim() > 1:
        return t_abs[int(lengths.argmax().item())]
    return t_abs[0] if t_abs.dim() > 1 else t_abs


def _train_pass_gates(module, loader, device: torch.device) -> dict:
    ode = module.model.ode_func
    module.eval()
    dz_chunks: list[torch.Tensor] = []
    near_any = []
    mu_m, lam_m, nu_m, t0_m = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            t_vec = _t_vec(batch)
            out = module.model(
                batch["states_5d"], t_vec, return_aux=True,
                track_ids=batch.get("track_ids"),
            )
            pred, aux = out
            h0 = aux.z0
            t0 = torch.zeros((), device=h0.device, dtype=h0.dtype)
            dh = ode(t0, h0)
            dz_chunks.append(dh[:, :2].reshape(-1).detach().cpu())
            cache = getattr(ode, "_cache", None)
            if cache:
                near = param_near_bound_frac(cache)
                near_any.append(float(near.get("any", torch.tensor(1.0))))
                if "mu" in cache:
                    mu_m.append(float(cache["mu"].mean()))
                    lam_m.append(float(cache["lambda"].mean()))
                    nu_m.append(float(cache["nu"].mean()))
                    t0_m.append(float(cache["t0"].mean()))
    dz = torch.cat(dz_chunks) if dz_chunks else torch.zeros(1)
    return {
        "train_dz_dt_size_std": float(dz.std(unbiased=False)),
        "near_bound_any": max(near_any) if near_any else 1.0,
        "mu_mean": sum(mu_m) / len(mu_m) if mu_m else float("nan"),
        "lambda_mean": sum(lam_m) / len(lam_m) if lam_m else float("nan"),
        "nu_mean": sum(nu_m) / len(nu_m) if nu_m else float("nan"),
        "t0_mean": sum(t0_m) / len(t0_m) if t0_m else float("nan"),
    }


def _score_one(name: str, ckpt_name: str, use_t0: bool, seed: int, dm) -> dict:
    path = REPO / "checkpoints" / ckpt_name
    module = _build_module()
    _install(module, "zwietering", use_t0, True)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = module.load_state_dict(blob["state_dict"], strict=False)
    print(f"[rescore] {ckpt_name} missing={len(missing)} unexpected={len(unexpected)}")
    module.horizon_start_frac = 1.0
    module.horizon_ramp_start = 0
    module.horizon_ramp_end = 0
    module.eval()

    trainer = Trainer(
        accelerator="auto", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    trainer.validate(module, datamodule=dm, verbose=False)
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if torch.is_tensor(v) or isinstance(v, (int, float))}

    device = module.device
    train_loader = DataLoader(
        dm._train_ds, batch_size=min(32, len(dm._train_ds)),
        shuffle=False, num_workers=0, collate_fn=_collate_variable_length_tracks,
    )
    train_g = _train_pass_gates(module, train_loader, device)

    val_mse = metrics.get("val_track_mse_mean", float("nan"))
    accel = metrics.get("z_traj/acceleration_ratio", float("nan"))
    val_dz = metrics.get("val_dz_dt_size_std", float("nan"))
    near = train_g["near_bound_any"]
    train_dz = train_g["train_dz_dt_size_std"]
    gates = {
        "gate_near_bound": near == 0.0,
        "gate_train_dz": train_dz > 0.05,
        "gate_accel": accel > 1.0,
        "gate_val": val_mse < 0.40,
    }
    passed = all(gates.values())
    row = {
        "arm": name,
        "seed": seed,
        "ckpt": ckpt_name,
        "val_track_mse_mean": val_mse,
        "val_dz_dt_size_std": val_dz,
        "train_dz_dt_size_std": train_dz,
        "acceleration_ratio": accel,
        "near_bound_any": near,
        "mu_mean": train_g["mu_mean"],
        "lambda_mean": train_g["lambda_mean"],
        "nu_mean": train_g["nu_mean"],
        "t0_mean": train_g["t0_mean"],
        **{k: int(v) for k, v in gates.items()},
        "pass": int(passed),
    }
    print(f"[rescore] {name} s{seed} val={val_mse:.4f} dz={train_dz:.4f} "
          f"accel={accel:.3f} near={near:.3f} pass={passed}")
    return row


def main() -> None:
    parquet = PARQUET if PARQUET.is_file() else REPO / "Thesis" / "metrics_with_features.parquet"
    dm, train_ids, val_ids = _datamodule(parquet)
    print(f"[rescore] step8 split train={len(train_ids)} val={len(val_ids)}")
    rows = [_score_one(arm, ckpt, t0, seed, dm) for arm, ckpt, t0, seed in ARMS]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_pass = sum(r["pass"] for r in rows)
    print(f"[rescore] wrote {OUT}  pass={n_pass}/{len(rows)}")


if __name__ == "__main__":
    main()
