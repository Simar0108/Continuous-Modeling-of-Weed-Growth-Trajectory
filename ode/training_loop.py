"""Base Lightning module and loss functions restored from Step-6 3.10 bytecode.

``hybrid_loss`` color_mask / pad_mask gating matches the 2026-09-24 pyc
(D-017a reconstruction). Do not change that physiology block.
"""

from __future__ import annotations

from typing import Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from ode.model import LatentODE, ModelAux, PHYSIOLOGY_INDEX

LOG_SIZE_EPS = 0.0001
LINEAR_SIZE_WEIGHT = 0.0


def hybrid_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    color_mask: torch.Tensor,
    size_loss: str = "log",
    log_eps: float = 0.0001,
    linear_size_weight: float = 0.0,
    t_weights: Optional[torch.Tensor] = None,
    xy_loss_weight: float = 0.0,
    size_loss_weight: float = 1.0,
    phys_loss_weight: float = 0.5,
    pad_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Reconstruction loss decomposed into geometry (x, y, σ) and physiology (Z).

    pad_mask
        Optional boolean mask (B, T): True = real frame, False = padding.
        When provided the loss is averaged over ONLY the real frames.

    Masked Z frames are excluded from phys_loss via ``color_mask & pad_mask``.
    """
    if pad_mask is not None:
        pm = pad_mask.float()
        weight = pm if t_weights is None else t_weights * pm
        n_real = weight.sum().clamp(min=1.0)
    else:
        weight = t_weights
        n_real = None

    def _wmean(err2d: torch.Tensor) -> torch.Tensor:
        """Weighted mean over (B, T[, C]) — handles None weight."""
        if weight is None:
            return err2d.mean()
        w = weight.unsqueeze(-1) if err2d.dim() == 3 else weight
        num = (err2d * w).sum()
        if err2d.dim() == 3:
            return num / (n_real * err2d.shape[-1])
        return num / (n_real * 1.0)

    if xy_loss_weight == 0.0:
        xy_loss = torch.zeros(1, device=pred.device, dtype=pred.dtype).squeeze()
    else:
        xy_err = (pred[:, :, :2] - target[:, :, :2]).pow(2)
        xy_loss = _wmean(xy_err)
    sp = pred[:, :, 2:4]
    st = target[:, :, 2:4]
    if size_loss == "log":
        log_p = torch.log(F.softplus(sp) + log_eps)
        log_t = torch.log(F.softplus(st) + log_eps)
        size_err = (log_p - log_t).pow(2)
        if linear_size_weight > 0.0:
            size_err = size_err + linear_size_weight * (sp - st).pow(2)
    else:
        size_err = (sp - st).pow(2)
    size_loss_val = _wmean(size_err)
    zp = pred[:, :, PHYSIOLOGY_INDEX]
    zt = target[:, :, PHYSIOLOGY_INDEX]
    phys_mask = color_mask & pad_mask if pad_mask is not None else color_mask
    if phys_mask.any():
        phys_err = (zp - zt).pow(2)[phys_mask]
        if t_weights is not None:
            phys_err = phys_err * t_weights[phys_mask]
        phys_loss = phys_err.mean()
    else:
        phys_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    geo_loss = xy_loss_weight * xy_loss + size_loss_weight * size_loss_val
    total = geo_loss + phys_loss_weight * phys_loss
    geo_diag = {"geo_xy_loss": xy_loss, "geo_size_loss": size_loss_val}
    return total, geo_loss, phys_loss, geo_diag


def monotonicity_loss(pred: torch.Tensor) -> torch.Tensor:
    """Penalise predicted shrinkage in σ_w and σ_h. Not applied in _step."""
    shrinkage = F.relu(pred[:, :-1, 2:4] - pred[:, 1:, 2:4])
    return shrinkage.mean()


def kinetic_regularization(model: LatentODE, aux: ModelAux) -> torch.Tensor:
    """Penalise fast ODE dynamics: E[||dz/dt(h0)||²]."""
    t_zero = torch.zeros(1, device=aux.z0.device, dtype=aux.z0.dtype)
    dz = model.ode_func(t_zero, aux.z0)
    return dz.pow(2).mean()


def z0_regularization(aux: ModelAux) -> torch.Tensor:
    """Penalise large initial latent states: E[||h0||²]."""
    return aux.z0.pow(2).mean()


def _js_vs_mean(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] < 2 or logits.shape[1] < 2:
        return logits.new_zeros(())
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    m = p.mean(dim=0).clamp_min(1e-08)
    log_m = m.log()
    kl_pm = (p * (log_p - log_m)).sum(dim=-1).mean()
    kl_mp = (m * (log_m - log_p)).sum(dim=-1).mean()
    return 0.5 * (kl_pm + kl_mp)


def head_diversity_loss(model: LatentODE, z0: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """Hinge on post-activation r / K track-std."""
    ode = model.ode_func
    if not hasattr(ode, "rate_net") or not hasattr(ode, "saturation_net"):
        return z0.new_zeros(())
    if z0.shape[0] < 2:
        return z0.new_zeros(())
    t_z = torch.zeros(z0.shape[0], 1, device=z0.device, dtype=z0.dtype)
    shared = ode.shared(torch.cat([z0, t_z], dim=-1))
    rate = ode._bound_rate(ode.rate_net(shared))
    sat = ode._bound_sat(ode.saturation_net(shared))
    r_std = rate.mean(dim=-1).std(unbiased=False)
    k_std = sat.mean(dim=-1).std(unbiased=False)
    return torch.stack([
        torch.relu(rate.new_tensor(tau) - r_std),
        torch.relu(sat.new_tensor(tau) - k_std),
    ]).mean()


def per_channel_mse_logs(
    pred: torch.Tensor,
    target: torch.Tensor,
    color_mask: torch.Tensor,
    prefix: str,
    pad_mask: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Per-dimension MSE dict for W&B logging."""
    names = ["x", "y", "sigma_w", "sigma_h", "Z"]
    out: dict[str, torch.Tensor] = {}
    for i, name in enumerate(names):
        err2 = (pred[:, :, i] - target[:, :, i]).pow(2)
        if i == PHYSIOLOGY_INDEX:
            valid = color_mask & pad_mask if pad_mask is not None else color_mask
        else:
            valid = pad_mask
        if valid is not None:
            err_vals = err2[valid]
        else:
            err_vals = err2.flatten()
        out[f"{prefix}_mse_{name}"] = (
            err_vals.mean() if len(err_vals) > 0
            else torch.tensor(0.0, device=pred.device)
        )
    return out


class LatentODELightning(L.LightningModule):
    """Base Lightning module for Latent ODE training."""

    def __init__(
        self,
        lr: float = 0.0005,
        kinetic_reg_weight: float = 0.0,
        z0_reg_weight: float = 0.01,
        geometry_size_loss: str = "log",
        log_size_eps: float = 0.0001,
        linear_size_weight: float = 0.0,
        xy_loss_weight: float = 0.0,
        size_loss_weight: float = 1.0,
        phys_loss_weight: float = 0.5,
        phys_dropout: float = 0.2,
        latent_dim: int = 64,
        n_context_frames: int = 3,
        encoder: str = "mlp",
        ode_hidden: int = 256,
        normalize_z0: bool = True,
        time_conditioned_decoder: bool = False,
        ode_growth_bias: bool = True,
        ode_solver: str = "dopri5",
        ode_rtol: float = 1e-06,
        ode_atol: float = 1e-06,
        time_weighted_loss: bool = False,
        ode_weight_decay: float = 0.0,
        monotonicity_weight: float = 0.0,
        curriculum_warmup_start: int = 50,
        curriculum_warmup_end: int = 100,
        horizon_start_frac: float = 1.0,
        horizon_ramp_start: int = 100,
        horizon_ramp_end: int = 250,
        affine_decoder: bool = True,
        affine_scale_init: float = 6.0,
        hybrid_decoder: bool = True,
        late_head_lr_scale: float = 0.2,
        freeze_affine_epochs: int = 50,
        ode_type: str = "stable_sigmoidal",
        phased_ode: bool = False,
        ode_t_mid: float = 0.5,
        curriculum_sigma_threshold: float = float("inf"),
        z0_reg_scale: float = 1.0,
        head_lr_mult: float = 1.0,
        diversity_weight: float = 0.0,
        use_track_embed: bool = False,
        track_embed_dim: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.kinetic_reg_weight = kinetic_reg_weight
        self.z0_reg_weight = z0_reg_weight
        self.z0_reg_scale = z0_reg_scale
        self.head_lr_mult = head_lr_mult
        self.diversity_weight = diversity_weight
        self.use_track_embed = use_track_embed
        self.track_embed_dim = track_embed_dim
        self.geometry_size_loss = geometry_size_loss
        self.log_size_eps = log_size_eps
        self.linear_size_weight = linear_size_weight
        self.xy_loss_weight = xy_loss_weight
        self.size_loss_weight = size_loss_weight
        self.phys_loss_weight = phys_loss_weight
        self.phys_dropout = phys_dropout
        self.time_weighted_loss = time_weighted_loss
        self.ode_weight_decay = ode_weight_decay
        self.monotonicity_weight = monotonicity_weight
        self.curriculum_warmup_start = curriculum_warmup_start
        self.curriculum_warmup_end = curriculum_warmup_end
        self.horizon_start_frac = horizon_start_frac
        self.horizon_ramp_start = horizon_ramp_start
        self.horizon_ramp_end = horizon_ramp_end
        self.late_head_lr_scale = late_head_lr_scale
        self._n_context_frames = n_context_frames
        self.freeze_affine_epochs = freeze_affine_epochs
        self.curriculum_sigma_threshold = curriculum_sigma_threshold
        self._sigma_criterion_met = curriculum_sigma_threshold == float("inf")
        self.model = LatentODE(
            latent_dim=latent_dim,
            n_context_frames=n_context_frames,
            encoder=encoder,
            ode_hidden=ode_hidden,
            normalize_z0=normalize_z0,
            ode_growth_bias=ode_growth_bias,
            time_conditioned_decoder=time_conditioned_decoder,
            solver=ode_solver,
            rtol=ode_rtol,
            atol=ode_atol,
            affine_decoder=affine_decoder,
            affine_scale_init=affine_scale_init,
            hybrid_decoder=hybrid_decoder,
            phys_dropout=phys_dropout,
            ode_type=ode_type,
            phased_ode=phased_ode,
            ode_t_mid=ode_t_mid,
            use_track_embed=use_track_embed,
            track_embed_dim=track_embed_dim,
        )
        self._last_aux = None
        self._last_pred = None

    def _get_monotonicity_weight(self) -> float:
        if not self._sigma_criterion_met:
            return 0.0
        epoch = self.current_epoch
        start = self.curriculum_warmup_start
        end = self.curriculum_warmup_end
        if epoch < start:
            return 0.0
        if epoch >= end:
            return self.monotonicity_weight
        frac = (epoch - start) / max(end - start, 1)
        return frac * self.monotonicity_weight

    def _get_horizon_fraction(self) -> float:
        if self.horizon_start_frac >= 1.0:
            return 1.0
        if not self._sigma_criterion_met:
            return self.horizon_start_frac
        epoch = self.current_epoch
        ramp_start = self.horizon_ramp_start
        ramp_end = self.horizon_ramp_end
        if epoch < ramp_start:
            return self.horizon_start_frac
        if epoch >= ramp_end:
            return 1.0
        frac = (epoch - ramp_start) / max(ramp_end - ramp_start, 1)
        return self.horizon_start_frac + frac * (1.0 - self.horizon_start_frac)

    def _step(self, batch: dict, prefix: str):
        states_5d = batch["states_5d"]
        t_absolute = batch["t_absolute"]
        color_mask = batch["color_mask"]
        if states_5d.dim() == 2:
            states_5d = states_5d.unsqueeze(0)
            t_absolute = t_absolute.unsqueeze(0)
            color_mask = color_mask.unsqueeze(0)
            if batch.get("track_ids") is not None and batch["track_ids"].dim() == 0:
                batch = dict(batch)
                batch["track_ids"] = batch["track_ids"].unsqueeze(0)
        batch_size = states_5d.size(0)
        track_lengths = batch.get("track_lengths", None)
        if track_lengths is not None and t_absolute.dim() > 1:
            t_abs = t_absolute[int(track_lengths.argmax().item())]
        elif t_absolute.dim() > 1:
            t_abs = t_absolute[0]
        else:
            t_abs = t_absolute
        pred, aux = self.model(
            states_5d, t_abs, return_aux=True, track_ids=batch.get("track_ids"),
        )
        pred = pred.permute(1, 0, 2)
        horizon_frac = self._get_horizon_fraction()
        t_full = states_5d.shape[1]
        horizon_t = max(self._n_context_frames + 1, int(t_full * horizon_frac))
        states_loss = states_5d[:, :horizon_t, :]
        pred_loss = pred[:, :horizon_t, :]
        mask_loss = color_mask[:, :horizon_t]
        if track_lengths is not None:
            idx = torch.arange(horizon_t, device=states_5d.device)
            pad_mask = (idx.unsqueeze(0) < track_lengths.unsqueeze(1))[:, :horizon_t]
        else:
            pad_mask = None
        t_weights = None
        if self.time_weighted_loss:
            t_rel = t_absolute - t_absolute[:, 0:1]
            t_max = t_rel.amax().clamp(min=1.0)
            tw_full = (t_rel / t_max + 1.0).to(pred.dtype)
            tw_full = tw_full / tw_full.mean()
            t_weights = tw_full[:, :horizon_t]
        total_loss, geo_loss, phys_loss, geo_diag = hybrid_loss(
            pred_loss, states_loss, mask_loss,
            size_loss=self.geometry_size_loss,
            log_eps=self.log_size_eps,
            linear_size_weight=self.linear_size_weight,
            t_weights=t_weights,
            xy_loss_weight=self.xy_loss_weight,
            size_loss_weight=self.size_loss_weight,
            phys_loss_weight=self.phys_loss_weight,
            pad_mask=pad_mask,
        )
        kin_loss = kinetic_regularization(self.model, aux)
        z0_loss = z0_regularization(aux)
        div_loss = head_diversity_loss(self.model, aux.z0)
        total_loss = (
            total_loss
            + self.kinetic_reg_weight * kin_loss
            + self.z0_reg_weight * self.z0_reg_scale * z0_loss
            + self.diversity_weight * div_loss
        )
        with torch.no_grad():
            ode = self.model.ode_func
            t_zero = torch.zeros(1, device=aux.z0.device, dtype=aux.z0.dtype)
            dz_at_z0 = ode(t_zero, aux.z0.detach())
            dz_size = dz_at_z0[:, :2]
            dz_size_mean = dz_size.mean()
            dz_size_std = dz_size.std()
            size_pred = pred[:, :, 2:4]
            t_full_pred = size_pred.shape[1]
            t_half = t_full_pred // 2
            if t_half > 1:
                growth_1st = (size_pred[:, t_half, :] - size_pred[:, 0, :]).mean()
                growth_2nd = (size_pred[:, -1, :] - size_pred[:, t_half, :]).mean()
                accel_ratio = growth_2nd / (growth_1st.abs() + 1e-06)
            else:
                accel_ratio = torch.tensor(float("nan"), device=pred.device)
            r_mean = torch.tensor(float("nan"), device=aux.z0.device)
            r_std = torch.tensor(float("nan"), device=aux.z0.device)
            k_mean = torch.tensor(float("nan"), device=aux.z0.device)
            logs_pre: dict = {}
            if hasattr(ode, "rate_net") and hasattr(ode, "saturation_net"):
                t_z = t_zero.reshape(1, 1).expand(aux.z0.shape[0], 1)
                shared_diag = ode.shared(torch.cat([aux.z0.detach(), t_z], dim=-1))
                rate_pre = ode.rate_net(shared_diag)
                sat_pre = ode.saturation_net(shared_diag)
                rate_bt = ode._bound_rate(rate_pre)
                sat_bt = ode._bound_sat(sat_pre)
                r_mean = rate_bt.mean()
                r_std = rate_bt.mean(dim=-1).std(unbiased=False)
                k_mean = sat_bt.mean()
                logs_pre = {
                    "head/rate_pre_mean": rate_pre.mean(),
                    "head/rate_pre_std": rate_pre.std(unbiased=False),
                    "head/sat_pre_mean": sat_pre.mean(),
                    "head/sat_pre_std": sat_pre.std(unbiased=False),
                    "head/satfrac": torch.stack([
                        (rate_pre.abs() > 5).float().mean(),
                        (sat_pre.abs() > 5).float().mean(),
                    ]).max(),
                }
        logs = {
            f"{prefix}_loss": total_loss.detach(),
            f"{prefix}_geo_loss": geo_loss.detach(),
            f"{prefix}_phys_loss": phys_loss.detach(),
            f"{prefix}_kinetic_reg": kin_loss.detach(),
            f"{prefix}_z0_reg": z0_loss.detach(),
            f"{prefix}_diversity": div_loss.detach(),
            f"{prefix}_horizon_frac": torch.tensor(horizon_frac, dtype=pred.dtype),
            f"{prefix}_dz_dt_size_mean": dz_size_mean.detach(),
            f"{prefix}_dz_dt_size_std": dz_size_std.detach(),
            "z_traj/acceleration_ratio": accel_ratio.detach(),
            "head/r_mean": r_mean.detach(),
            "head/K_mean": k_mean.detach(),
            "head/r_std": r_std.detach(),
        }
        logs.update(logs_pre)
        for key, value in geo_diag.items():
            logs[f"{prefix}_{key}"] = value.detach()
        logs.update(per_channel_mse_logs(
            pred_loss, states_loss, mask_loss, prefix, pad_mask=pad_mask,
        ))
        if batch_size > 1:
            with torch.no_grad():
                size_err = (pred_loss[:, :, 2:4] - states_loss[:, :, 2:4]).pow(2)
                if pad_mask is not None:
                    w = pad_mask.float()
                    denom = w.sum(dim=1).clamp(min=1.0)
                    track_mse = (size_err * w.unsqueeze(-1)).sum(dim=(1, 2)) / (denom * 2.0)
                else:
                    track_mse = size_err.mean(dim=(1, 2))
                logs[f"{prefix}_track_mse_mean"] = track_mse.mean()
                logs[f"{prefix}_track_mse_std"] = track_mse.std()
                logs[f"{prefix}_track_mse_max"] = track_mse.max()
        if getattr(self.model, "output_scale", None) is not None:
            logs["decoder/size_scale_0"] = self.model.output_scale[2].detach()
        self._last_aux = aux
        self._last_pred = pred
        return total_loss, logs, batch_size

    def on_train_epoch_start(self) -> None:
        if self.freeze_affine_epochs > 0 and self.current_epoch == self.freeze_affine_epochs:
            scale = getattr(self.model, "output_scale", None)
            bias = getattr(self.model, "output_bias", None)
            if scale is not None and not scale.requires_grad:
                scale.requires_grad_(True)
                if bias is not None:
                    bias.requires_grad_(True)
                print(
                    f"\n[Epoch {self.current_epoch}] Affine decoder unfrozen — "
                    "output_scale and output_bias are now learnable parameters.\n"
                )

    def on_after_backward(self) -> None:
        ode = self.model.ode_func
        if not hasattr(ode, "late_head") or not hasattr(ode, "early_head"):
            return

        def _grad_norm(mod: nn.Module) -> float:
            total = 0.0
            for p in mod.parameters():
                if p.grad is not None:
                    total += p.grad.detach().norm().item() ** 2
            return total ** 0.5

        late_gnorm = _grad_norm(ode.late_head)
        early_gnorm = _grad_norm(ode.early_head)
        ratio = late_gnorm / (early_gnorm + 1e-08)
        self.log(
            "grad/late_early_ratio", torch.tensor(ratio),
            on_step=True, on_epoch=False, prog_bar=False,
        )

    def training_step(self, batch, batch_idx):
        loss, logs, batch_size = self._step(batch, "train")
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs, batch_size = self._step(batch, "val")
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        if not self._sigma_criterion_met:
            sigma_w_mse = logs.get("val_mse_sigma_w", torch.tensor(float("inf")))
            sigma_h_mse = logs.get("val_mse_sigma_h", torch.tensor(float("inf")))
            val_sigma = float((sigma_w_mse + sigma_h_mse) / 2)
            if val_sigma < self.curriculum_sigma_threshold:
                self._sigma_criterion_met = True
        return loss

    def configure_optimizers(self):
        if not hasattr(self.model.ode_func, "late_head"):
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        late_ids = {id(p) for p in self.model.ode_func.late_head.parameters()}
        ode_ids = {id(p) for p in self.model.ode_func.parameters()}
        late_params = list(self.model.ode_func.late_head.parameters())
        ode_rest = [p for p in self.model.ode_func.parameters() if id(p) not in late_ids]
        other_params = [p for p in self.model.parameters() if id(p) not in ode_ids]
        return torch.optim.Adam(
            [
                {"params": other_params, "lr": self.lr},
                {"params": ode_rest, "lr": self.lr},
                {"params": late_params, "lr": self.lr * self.late_head_lr_scale},
            ],
            lr=self.lr,
        )
