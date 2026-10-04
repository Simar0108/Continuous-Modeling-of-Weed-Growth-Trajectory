"""Neural CDE arm: cubic Hermite control on [sigma_w, sigma_h, t].

Does not edit hybrid_loss, affine decoder init, or the locked H1 trainer.
The interpolator uses context-frame knots only (no future size labels).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ode.model import StiffODEMixin
from ode.train_multi import MultiTrackLightning

CONTROL_DIM = 3
N_CONTEXT = 3


def backward_diffs(knots: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Kidger-style backward-difference derivatives.

    knots: (K,) strictly increasing. values: (B, K, C).
    """
    dt = (knots[1:] - knots[:-1]).clamp(min=1e-8)
    dv = values[:, 1:] - values[:, :-1]
    slope = dv / dt.reshape(1, -1, 1)
    derivs = torch.zeros_like(values)
    derivs[:, 1:] = slope
    derivs[:, 0] = slope[:, 0]
    return derivs


def control_derivs(knots: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Backward differences, with size held (zero deriv) on the last interval."""
    derivs = backward_diffs(knots, values)
    if knots.shape[0] >= 2 and values.shape[-1] >= 2:
        derivs = derivs.clone()
        derivs[:, -2:, :2] = 0.0
    return derivs


def apply_identity_time_channel(
    x: torch.Tensor, dx: torch.Tensor, t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the last control channel equal to query time with derivative 1.

    After the last observation the size spline is held, so X' would be 0
    without a ticking time channel and the CDE would freeze (dz/dt = f·0).
    """
    t_s = t.reshape(()).to(dtype=x.dtype, device=x.device)
    x = x.clone()
    dx = dx.clone()
    x[:, -1] = t_s
    dx[:, -1] = x.new_ones(())
    return x, dx


def hermite_eval(
    knots: torch.Tensor,
    values: torch.Tensor,
    derivs: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cubic Hermite value and dt-derivative at scalar t.

    Returns x (B, C) and dx/dt (B, C). The time channel is then replaced
    by the identity map t ↦ (t, 1) so the tail cannot freeze.
    """
    k = int(knots.shape[0])
    t = t.reshape(())
    idx = int(torch.searchsorted(knots, t.detach()).clamp(min=1, max=k - 1).item()) - 1
    t0 = knots[idx]
    t1 = knots[idx + 1]
    dt = (t1 - t0).clamp(min=1e-8)
    s = ((t - t0) / dt).clamp(0.0, 1.0)
    y0 = values[:, idx]
    y1 = values[:, idx + 1]
    m0 = derivs[:, idx]
    m1 = derivs[:, idx + 1]
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    x = h00 * y0 + h10 * dt * m0 + h01 * y1 + h11 * dt * m1
    dh00 = (6.0 * s2 - 6.0 * s) / dt
    dh10 = 3.0 * s2 - 4.0 * s + 1.0
    dh01 = (-6.0 * s2 + 6.0 * s) / dt
    dh11 = 3.0 * s2 - 2.0 * s
    dx = dh00 * y0 + dh10 * m0 + dh01 * y1 + dh11 * m1
    return apply_identity_time_channel(x, dx, t)


def context_control(
    states: torch.Tensor,
    t: torch.Tensor,
    n_context: int = N_CONTEXT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (knots, values) from context frames: [sigma_w, sigma_h, t_norm]."""
    if states.dim() == 2:
        states = states.unsqueeze(0)
    if states.dim() != 3:
        raise ValueError(f"states must be (B,T,5), got {tuple(states.shape)}")
    if t.dim() != 1:
        t = t.reshape(-1)
    batch, n_times, _dim = states.shape
    n_ctx = max(2, min(int(n_context), int(n_times)))
    t_span = (t[-1] - t[0]).clamp(min=1e-3)
    t_norm = (t - t[0]) / t_span
    knots = t_norm[:n_ctx].contiguous()
    # Strictly increasing knots (duplicate timestamps from padding).
    bump = torch.arange(n_ctx, device=knots.device, dtype=knots.dtype) * 1e-5
    knots = knots + bump
    sigma = states[:, :n_ctx, 2:4]
    # Hold-size knot at t_norm=1 so the interpolant covers the full odeint
    # span. Size stays at the last context value; time continues to 1.
    if float(knots[-1].detach()) < 1.0 - 1e-6:
        knots = torch.cat([knots, knots.new_tensor([1.0])])
        sigma = torch.cat([sigma, sigma[:, -1:, :]], dim=1)
    t_ch = knots.reshape(1, -1, 1).expand(batch, int(knots.shape[0]), 1)
    values = torch.cat([sigma, t_ch], dim=-1)
    return knots, values


class NeuralCDEFunc(StiffODEMixin, nn.Module):
    """dz/dt = f_θ(z, X(t)) @ X'(t) with X = cubic Hermite([σ_w, σ_h, t])."""

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 3,
        control_dim: int = CONTROL_DIM,
        f_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.nfe = 0
        self.latent_dim = int(latent_dim)
        self.control_dim = int(control_dim)
        self.f_scale = float(f_scale)
        layers: list[nn.Module] = []
        in_dim = self.latent_dim + self.control_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.latent_dim * self.control_dim))
        self.net = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self._init_stiffness_params()
        self._knots: Optional[torch.Tensor] = None
        self._values: Optional[torch.Tensor] = None
        self._derivs: Optional[torch.Tensor] = None

    def set_control(self, knots: torch.Tensor, values: torch.Tensor) -> None:
        self._knots = knots
        self._values = values
        self._derivs = control_derivs(knots, values)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        if self._knots is None or self._values is None or self._derivs is None:
            return torch.zeros_like(h)
        x, dx = hermite_eval(self._knots, self._values, self._derivs, t)
        if x.shape[0] == 1 and h.shape[0] > 1:
            x = x.expand(h.shape[0], -1)
            dx = dx.expand(h.shape[0], -1)
        inp = torch.cat([h, x], dim=-1)
        mat = torch.tanh(self.net(inp)).view(h.shape[0], self.latent_dim, self.control_dim)
        dz = self.f_scale * torch.bmm(mat, dx.unsqueeze(-1)).squeeze(-1)
        dz_size = self._scale_slow_dynamics(dz[:, :2])
        rest = dz[:, 2:]
        if rest.numel():
            rest = self._bound_fast_dynamics(rest)
            dz = torch.cat([dz_size, rest], dim=-1)
        else:
            dz = dz_size
        return dz


def attach_ncde_control(model: nn.Module, n_context: int = N_CONTEXT) -> None:
    """Set the cubic-Hermite control path before each LatentODE.forward."""
    orig = model.forward
    encoder = getattr(model, "context_encoder", None)
    k = int(getattr(encoder, "n_context_frames", n_context)) if encoder is not None else n_context

    def wrapped(states, t, return_aux=False, track_ids=None):
        ode = model.ode_func
        if isinstance(ode, NeuralCDEFunc):
            ode.nfe = 0
            knots, values = context_control(states, t, n_context=k)
            ode.set_control(knots, values)
        return orig(states, t, return_aux=return_aux, track_ids=track_ids)

    model.forward = wrapped


class NCDELightning(MultiTrackLightning):
    """Locked H1 module with Neural CDE dynamics replacing StableSigmoidal."""

    def __init__(self, ncde: bool = True, **kwargs):
        kwargs.pop("ncde", None)
        super().__init__(**kwargs)
        self.ncde = True
        try:
            self.hparams["ncde"] = True
        except Exception:
            pass
        self._install_ncde()

    def _install_ncde(self) -> None:
        old = self.model.ode_func
        hidden = int(old.shared[0].out_features)
        n_layers = sum(1 for m in old.shared if isinstance(m, nn.Linear))
        n_layers = max(n_layers, 3)
        self.model.ode_func = NeuralCDEFunc(
            latent_dim=int(old.latent_dim),
            hidden_dim=hidden,
            n_layers=n_layers,
        )
        n_ctx = int(getattr(self.model.context_encoder, "n_context_frames", N_CONTEXT))
        attach_ncde_control(self.model, n_context=n_ctx)

    def configure_optimizers(self):
        """D-017a: two-group Adam. No late_head split — the CDE has none.

        Group 1 = encoder / decoder / affine (and non-net CDE params) at ``lr``.
        Group 2 = ``NeuralCDEFunc.net`` at ``lr``. Do not alias ``net`` as
        ``late_head`` (parameter-overlap risk across groups).
        """
        import torch

        ode = self.model.ode_func
        net_ids = {id(p) for p in ode.net.parameters()}
        rest = [p for p in self.model.parameters() if id(p) not in net_ids]
        groups = [
            {"params": rest, "lr": self.lr, "weight_decay": 0.0},
            {
                "params": list(ode.net.parameters()),
                "lr": self.lr,
                "weight_decay": getattr(self, "ode_weight_decay", 0.0),
            },
        ]
        groups = [g for g in groups if len(g["params"]) > 0]
        optimizer = torch.optim.Adam(groups, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
