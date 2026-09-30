"""Path-length regularizer for the latent ODE (pathreg arm).

Does not edit ``hybrid_loss`` or the locked H1 trainer. The penalty is
added on top of the cached ``_last_aux.ht`` from the existing forward,
so it does not add a second ``odeint`` during training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback

from ode.train_multi import MultiTrackLightning


def lambda_tag(lam: float) -> str:
    text = f"{float(lam):.4g}".replace("-", "m").replace(".", "p")
    return text


def path_length_ht(
    ht: torch.Tensor,
    lengths: torch.Tensor | None,
    horizon_T: int,
    n_times: int | None = None,
) -> torch.Tensor:
    """Mean per-track polyline length of ``ht``.

    ``ht`` is ``(T, B, D)`` (odeint) or ``(B, T, D)``. Padding and
    post-horizon frames are excluded. Returns a scalar.
    """
    if ht.dim() != 3:
        raise ValueError(f"ht must be 3D, got {tuple(ht.shape)}")
    if n_times is not None:
        if ht.shape[0] == n_times:
            pass
        elif ht.shape[1] == n_times:
            ht = ht.permute(1, 0, 2)
    T, B, _D = ht.shape
    if T < 2:
        return ht.new_zeros(())
    cap = max(int(horizon_T), 2)
    T_use = min(T, cap)
    ht = ht[:T_use]
    seg = (ht[1:] - ht[:-1]).norm(dim=-1)
    t_idx = torch.arange(T_use - 1, device=ht.device)
    if lengths is None:
        valid = torch.ones(T_use - 1, B, dtype=torch.bool, device=ht.device)
    else:
        leng = lengths.to(device=ht.device).reshape(-1)
        if int(leng.numel()) != B:
            valid = torch.ones(T_use - 1, B, dtype=torch.bool, device=ht.device)
        else:
            leng = leng.clamp(max=T_use)
            valid = (t_idx.unsqueeze(1) + 1) < leng.unsqueeze(0)
    valid_f = valid.to(dtype=seg.dtype)
    per_track = (seg * valid_f).sum(dim=0)
    alive = valid.any(dim=0)
    if alive.any():
        return per_track[alive].mean()
    return seg.new_zeros(())


class PathRegLightning(MultiTrackLightning):
    """Locked H1 module plus an optional path-length penalty on ``ht``."""

    def __init__(self, pathreg_lambda: float = 0.0, **kwargs):
        kwargs.pop("pathreg_lambda", None)
        super().__init__(**kwargs)
        self.pathreg_lambda = float(pathreg_lambda)
        try:
            self.hparams["pathreg_lambda"] = self.pathreg_lambda
        except Exception:
            pass

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        aux = getattr(self, "_last_aux", None)
        ht = getattr(aux, "ht", None) if aux is not None else None
        bs = int(batch["states_5d"].shape[0]) if torch.is_tensor(batch.get("states_5d")) else 1
        if ht is None or float(self.pathreg_lambda) <= 0.0:
            self.log(
                "train_pathreg", torch.zeros((), device=self.device),
                on_step=True, on_epoch=True, batch_size=bs,
            )
            return loss
        lengths = batch.get("track_lengths")
        t_abs = batch.get("t_absolute")
        n_times = int(t_abs.shape[-1]) if torch.is_tensor(t_abs) and t_abs.dim() >= 1 else None
        T_full = int(n_times if n_times is not None else max(ht.shape[0], ht.shape[1]))
        try:
            frac = float(self._get_horizon_fraction())
        except Exception:
            frac = 1.0
        n_ctx = int(getattr(self, "_n_context_frames", 3))
        horizon_T = max(n_ctx + 1, int(T_full * frac))
        penalty = path_length_ht(ht, lengths, horizon_T, n_times=n_times)
        weighted = penalty * float(self.pathreg_lambda)
        self.log("train_pathreg", penalty, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_pathreg_weighted", weighted, on_step=True, on_epoch=True, batch_size=bs)
        return loss + weighted


def load_ode_module(path: Path, device: torch.device, strict: bool = False):
    """Load EMA ODE weights. Pathreg ckpts use ``PathRegLightning``; lock
    ckpts fall back to ``MultiTrackLightning``.
    """
    try:
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(str(path), map_location="cpu")
    hp = blob.get("hyper_parameters") or {}
    cls = PathRegLightning if "pathreg_lambda" in hp else MultiTrackLightning
    try:
        module = cls.load_from_checkpoint(str(path), map_location=device, strict=strict)
    except TypeError:
        module = MultiTrackLightning.load_from_checkpoint(
            str(path), map_location=device, strict=strict,
        )
    module.eval().to(device)
    module.horizon_start_frac = 1.0
    module.horizon_ramp_start = 0
    module.horizon_ramp_end = 0
    return module


def pairwise_z0_cosine(z0: torch.Tensor) -> dict[str, Any]:
    z = F.normalize(z0.float(), dim=-1, eps=1e-8)
    n = int(z.shape[0])
    if n < 2:
        return {
            "n": n, "mean_offdiag": float("nan"), "median_offdiag": float("nan"),
            "min_offdiag": float("nan"), "max_offdiag": float("nan"),
        }
    sim = z @ z.T
    eye = torch.eye(n, dtype=torch.bool, device=sim.device)
    off = sim.masked_select(~eye)
    return {
        "n": n,
        "mean_offdiag": float(off.mean().item()),
        "median_offdiag": float(off.median().item()),
        "min_offdiag": float(off.min().item()),
        "max_offdiag": float(off.max().item()),
    }


@torch.no_grad()
def collect_z0_and_dz(module, dataset, device: torch.device) -> dict[str, Any]:
    module.eval()
    z0s: list[torch.Tensor] = []
    dzs: list[torch.Tensor] = []
    ode = module.model.ode_func
    for sample in dataset:
        states = sample["states_5d"].to(device)
        t_abs = sample["t_absolute"].to(device)
        track_ids = sample.get("track_ids")
        if track_ids is None and "track_id" in sample:
            track_ids = sample["track_id"].reshape(1)
        if torch.is_tensor(track_ids):
            track_ids = track_ids.to(device)
        out = module.model(
            states.unsqueeze(0), t_abs, return_aux=True, track_ids=track_ids,
        )
        _pred, aux = out
        z0 = aux.z0
        t0 = torch.zeros((), device=z0.device, dtype=z0.dtype)
        dh = ode(t0, z0)
        z0s.append(z0[0].detach().cpu())
        dzs.append(dh[0].detach().cpu())
    z0 = torch.stack(z0s, dim=0)
    dz = torch.stack(dzs, dim=0)
    mag = dz.norm(dim=-1)
    size = dz[:, :2].reshape(-1)
    cosine = pairwise_z0_cosine(z0)
    return {
        "z0_cosine": cosine,
        "dz_dt_mean_norm": float(mag.mean().item()),
        "dz_dt_std_norm": float(mag.std(unbiased=False).item()) if mag.numel() > 1 else 0.0,
        "train_dz_dt_size_std": float(size.std(unbiased=False).item()) if size.numel() > 1 else 0.0,
        "n_tracks": int(z0.shape[0]),
        "z0": z0,
        "dz": dz,
    }


class PathRegDiagnosticsCallback(Callback):
    """Write z0 cosine and dz/dt stats at the end of training."""

    def __init__(self, out_dir: Path):
        super().__init__()
        self.out_dir = Path(out_dir)

    def on_fit_end(self, trainer, pl_module) -> None:
        dm = trainer.datamodule
        ds = getattr(dm, "_train_ds", None) if dm is not None else None
        if ds is None:
            print("[pathreg] diagnostics skipped: no train dataset")
            return
        device = pl_module.device
        try:
            report = collect_z0_and_dz(pl_module, ds, device)
        except Exception as exc:
            print(f"[pathreg] diagnostics failed: {exc}")
            return
        z0 = report.pop("z0")
        dz = report.pop("dz")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"z0": z0, "dz": dz}, self.out_dir / "z0_dz.pt")
        (self.out_dir / "diagnostics.json").write_text(json.dumps(report, indent=2))
        cos = report["z0_cosine"]["mean_offdiag"]
        print(
            f"[pathreg] z0_cosine_mean={cos:.6f} "
            f"dz_dt_size_std={report['train_dz_dt_size_std']:.6f} "
            f"dz_dt_mean_norm={report['dz_dt_mean_norm']:.6f}"
        )
        if getattr(trainer, "logger", None) is not None:
            try:
                trainer.logger.log_metrics({
                    "diag/z0_cosine_mean": float(cos),
                    "diag/train_dz_dt_size_std": float(report["train_dz_dt_size_std"]),
                    "diag/dz_dt_mean_norm": float(report["dz_dt_mean_norm"]),
                }, step=trainer.global_step)
            except Exception:
                pass


def write_provenance(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "PROVENANCE.md").write_text(_provenance_md(payload))
    (out_dir / "provenance.json").write_text(json.dumps(payload, indent=2, default=str))


def _provenance_md(payload: dict[str, Any]) -> str:
    lines = [
        "# pathreg arm provenance",
        "",
        "This directory is **not** the locked H1 reference. Do not write here",
        "from `h1_stab` or `h1_final_best` jobs.",
        "",
    ]
    for key, value in payload.items():
        lines.append(f"- **{key}**: `{value}`")
    lines.append("")
    return "\n".join(lines)
