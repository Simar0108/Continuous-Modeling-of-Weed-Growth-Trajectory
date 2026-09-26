"""Step 7 RHS families: Richards, time-varying rate, lag-off, constant Z."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def disable_lag_gate(ode: nn.Module) -> None:
    """Identity lag: σ((t − t_lag)/τ) → 1. Isolates the gate's 5× regression."""

    def _ones(t: torch.Tensor, batch: int) -> torch.Tensor:
        return torch.ones(batch, 1, device=t.device, dtype=t.dtype)

    ode._lag_gate = _ones
    if hasattr(ode, "shared_t_lag"):
        ode.shared_t_lag.requires_grad_(False)


class _HeadedODE(nn.Module):
    """Shared backbone + bounded r/K heads + residual Z channels."""

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 3,
        head_dim: int = 0,
    ) -> None:
        super().__init__()
        self.nfe = 0
        self.latent_dim = latent_dim
        self.r_max = 2.0
        self.k_min = 0.3
        self.k_max = 2.0
        if head_dim == 0:
            head_dim = max(hidden_dim // 4, 32)
        layers: list[nn.Module] = []
        in_dim = latent_dim + 1
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        self.shared = nn.Sequential(*layers)
        self.rate_net = nn.Sequential(
            nn.Linear(hidden_dim, head_dim), nn.SiLU(), nn.Linear(head_dim, 2),
        )
        self.saturation_net = nn.Sequential(
            nn.Linear(hidden_dim, head_dim), nn.SiLU(), nn.Linear(head_dim, 2),
        )
        n_residual = latent_dim - 2
        self.residual_head: Optional[nn.Module] = None
        if n_residual > 0:
            self.residual_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
                nn.Linear(hidden_dim // 2, n_residual),
            )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        self.early_head = self.rate_net
        self.late_head = self.saturation_net
        self.transition_center = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.transition_width = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        if hasattr(self, "_init_stiffness_params"):
            self._init_stiffness_params()

    def _bound_rate(self, pre: torch.Tensor) -> torch.Tensor:
        return self.r_max * torch.sigmoid(pre)

    def _bound_sat(self, pre: torch.Tensor) -> torch.Tensor:
        return self.k_min + (self.k_max - self.k_min) * torch.sigmoid(pre)

    def _shared(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        t_scalar = t.reshape(1, 1).expand(B, 1)
        return self.shared(torch.cat([h, t_scalar], dim=-1))


class RichardsODEFunc(_HeadedODE):
    """dz/dt = r · z_+ · (1 − (z_+/K)^ν), ν ∈ [0.25, 4]."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden = self.rate_net[0].in_features
        head_dim = self.rate_net[0].out_features
        self.nu_net = nn.Sequential(
            nn.Linear(hidden, head_dim), nn.SiLU(), nn.Linear(head_dim, 2),
        )
        for m in self.nu_net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def _bound_nu(self, pre: torch.Tensor) -> torch.Tensor:
        return 0.25 + 3.75 * torch.sigmoid(pre)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        shared = self._shared(t, h)
        rate = self._bound_rate(self.rate_net(shared))
        sat = self._bound_sat(self.saturation_net(shared))
        nu = self._bound_nu(self.nu_net(shared))
        z = F.softplus(h[:, :2]) + 1e-3
        frac = (z / sat.clamp(min=1e-3)).clamp(max=20.0)
        dh_size = rate * z * (1.0 - frac.pow(nu))
        if hasattr(self, "_scale_slow_dynamics"):
            dh_size = self._scale_slow_dynamics(dh_size)
        if self.residual_head is not None:
            dh_rest = self.residual_head(shared)
            if hasattr(self, "_bound_fast_dynamics"):
                dh_rest = self._bound_fast_dynamics(dh_rest)
            return torch.cat([dh_size, dh_rest], dim=-1)
        return dh_size


class ExpRateODEFunc(_HeadedODE):
    """r(t) = r0 · exp(β t), K fixed at 10 (carrying capacity out of window)."""

    def __init__(self, *args, k_fixed: float = 10.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k_fixed = k_fixed
        hidden = self.rate_net[0].in_features
        head_dim = self.rate_net[0].out_features
        self.beta_net = nn.Sequential(
            nn.Linear(hidden, head_dim), nn.SiLU(), nn.Linear(head_dim, 2),
        )
        for m in self.beta_net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def _bound_sat(self, pre: torch.Tensor) -> torch.Tensor:
        return pre.new_full(pre.shape, self.k_fixed)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        shared = self._shared(t, h)
        r0 = self._bound_rate(self.rate_net(shared))
        beta = 2.0 * torch.tanh(self.beta_net(shared))
        t_s = t.reshape(())
        rate = r0 * torch.exp(beta * t_s)
        sat = h.new_full((h.shape[0], 2), self.k_fixed)
        z = F.softplus(h[:, :2]) + 1e-3
        dh_size = rate * z * (1.0 - z / sat)
        if hasattr(self, "_scale_slow_dynamics"):
            dh_size = self._scale_slow_dynamics(dh_size)
        if self.residual_head is not None:
            dh_rest = self.residual_head(shared)
            if hasattr(self, "_bound_fast_dynamics"):
                dh_rest = self._bound_fast_dynamics(dh_rest)
            return torch.cat([dh_size, dh_rest], dim=-1)
        return dh_size


class PerTrackConstantZ(nn.Module):
    """Replace the Z residual with a per-track constant (measurement-noise model)."""

    def __init__(self, num_embeddings: int = 16384) -> None:
        super().__init__()
        self.table = nn.Embedding(num_embeddings, 1)
        nn.init.zeros_(self.table.weight)

    def forward(self, track_ids: torch.Tensor) -> torch.Tensor:
        idx = track_ids.long().reshape(-1) % self.table.num_embeddings
        return self.table(idx).squeeze(-1)


# Reporting ranges for the utilization gate (not hard clamps, except t0).
ZWIET_RANGES = {
    "mu": (1e-3, 8.0),
    "lambda": (1e-3, 0.6),
    "nu": (0.25, 10.0),
    "t0": (-0.3, 0.3),
}
K_FIXED = 10.0


class ZwieteringRichardsODEFunc(_HeadedODE):
    """Richards in Zwietering (μ, λ, ν) form. K=10 fixed. Softplus, no sigmoid ceiling.

    Closed-form (Zwietering 1990 modified Richards), y = size increment::

        m = (1+ν)^{1+1/ν}
        Q(t) = ν exp(1+ν) exp[(μ m / K)(λ − t + t0)]
        z(t) = K (1+Q)^{-1/ν}

    Autonomously, dz/dt = (μ m / ν) (z/K) (1 − (z/K)^ν).
    λ enters as an *initial-condition* parameter (not a multiplicative gate):
    z(0) is the closed form at the aligned clock. t0 ∈ [-0.3, 0.3] (tanh).
    λ and t0 alias as (λ+t0) in the IC; T2 vs T3 tests whether the extra
    degree of freedom still helps registration.
    """

    def __init__(self, *args, use_t0: bool = False, k_fixed: float = K_FIXED, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k_fixed = float(k_fixed)
        self.use_t0 = use_t0
        hidden = self.rate_net[0].in_features
        head_dim = self.rate_net[0].out_features
        self.mu_net = nn.Sequential(
            nn.Linear(hidden, head_dim), nn.SiLU(), nn.Linear(head_dim, 2),
        )
        self.lambda_net = nn.Sequential(
            nn.Linear(hidden, head_dim), nn.SiLU(), nn.Linear(head_dim, 2),
        )
        self.nu_net = nn.Sequential(
            nn.Linear(hidden, head_dim), nn.SiLU(), nn.Linear(head_dim, 2),
        )
        self.t0_net: Optional[nn.Module] = None
        if use_t0:
            self.t0_net = nn.Sequential(
                nn.Linear(hidden, head_dim), nn.SiLU(), nn.Linear(head_dim, 1),
            )
        for net in (self.mu_net, self.lambda_net, self.nu_net, self.t0_net):
            if net is None:
                continue
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.1)
                    nn.init.zeros_(m.bias)
        self.early_head = self.mu_net
        self.late_head = self.nu_net
        self._cache: dict[str, torch.Tensor] = {}

    def _params_from_shared(self, shared: torch.Tensor) -> dict[str, torch.Tensor]:
        mu = F.softplus(self.mu_net(shared)) + 1e-4
        lam = F.softplus(self.lambda_net(shared)) + 1e-4
        nu = 0.25 + F.softplus(self.nu_net(shared))
        if self.t0_net is not None:
            t0 = 0.3 * torch.tanh(self.t0_net(shared))
        else:
            t0 = shared.new_zeros(shared.shape[0], 1)
        return {"mu": mu, "lambda": lam, "nu": nu, "t0": t0}

    def heads_from_z0(self, z0: torch.Tensor) -> dict[str, torch.Tensor]:
        t0 = torch.zeros(z0.shape[0], 1, device=z0.device, dtype=z0.dtype)
        shared = self.shared(torch.cat([z0, t0], dim=-1))
        params = self._params_from_shared(shared)
        self._cache = {k: v.detach() for k, v in params.items()}
        self._cache_live = params
        return params

    def z0_from_params(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Zwietering closed form at window start (plant time = −t0)."""
        mu, lam, nu, t0 = params["mu"], params["lambda"], params["nu"], params["t0"]
        t_plant = -t0.expand_as(mu)
        return zwietering_closed(t_plant, mu, lam, nu, self.k_fixed)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        shared = self._shared(t, h)
        if getattr(self, "_cache_live", None) is not None:
            params = self._cache_live
        else:
            params = self._params_from_shared(shared)
        mu, nu = params["mu"], params["nu"]
        m = (1.0 + nu).pow(1.0 + 1.0 / nu.clamp(min=1e-3))
        z = F.softplus(h[:, :2]) + 1e-3
        frac = (z / self.k_fixed).clamp(max=20.0)
        dh_size = (mu * m / nu.clamp(min=1e-3)) * frac * (1.0 - frac.pow(nu))
        if hasattr(self, "_scale_slow_dynamics"):
            dh_size = self._scale_slow_dynamics(dh_size)
        if self.residual_head is not None:
            dh_rest = self.residual_head(shared)
            if hasattr(self, "_bound_fast_dynamics"):
                dh_rest = self._bound_fast_dynamics(dh_rest)
            return torch.cat([dh_size, dh_rest], dim=-1)
        return dh_size


def zwietering_closed(
    t: torch.Tensor,
    mu: torch.Tensor,
    lam: torch.Tensor,
    nu: torch.Tensor,
    k: float,
) -> torch.Tensor:
    m = (1.0 + nu).pow(1.0 + 1.0 / nu.clamp(min=1e-3))
    expo = (mu * m / k) * (lam - t)
    expo = expo.clamp(min=-40.0, max=40.0)
    q = nu * torch.exp(1.0 + nu) * torch.exp(expo)
    return k * (1.0 + q).clamp(min=1e-8).pow(-1.0 / nu.clamp(min=1e-3))


def param_near_bound_frac(
    params: dict[str, torch.Tensor],
    ranges: dict[str, tuple[float, float]] = ZWIET_RANGES,
    edge: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Fraction of values in the outer ``edge`` of each reporting range."""
    out: dict[str, torch.Tensor] = {}
    for name, (lo, hi) in ranges.items():
        if name not in params:
            continue
        x = params[name]
        span = hi - lo
        near = (x < lo + edge * span) | (x > hi - edge * span)
        out[name] = near.float().mean()
    if out:
        out["any"] = torch.stack(list(out.values())).max()
    return out


def attach_zwietering_initial_condition(model: nn.Module, ode: ZwieteringRichardsODEFunc) -> None:
    """Overwrite size channels of h0 with the Zwietering closed form at t=0."""
    orig = model.forward

    def wrapped(states, t, return_aux=False, track_ids=None):
        t_span = (t[-1] - t[0]).clamp(min=1e-3)
        t_norm = (t - t[0]) / t_span
        h0 = model.context_encoder(states, t_norm)
        if model.normalize_z0:
            h0 = h0 / h0.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        if getattr(model, "track_embedding", None) is not None and track_ids is not None:
            h0 = model.track_embedding(h0, track_ids)
        params = ode.heads_from_z0(h0)
        h0 = h0.clone()
        h0[:, :2] = ode.z0_from_params(params)

        from torchdiffeq import odeint
        from ode.model import ModelAux, _fourier_time_features

        ode.nfe = 0
        ht = odeint(
            ode, h0, t_norm,
            rtol=model.rtol, atol=model.atol, method=model.solver,
        )
        if model.time_conditioned_decoder:
            tf = _fourier_time_features(t_norm, model.n_fourier_freqs)
            T, B, D = ht.shape
            dec_in = torch.cat([ht, tf.unsqueeze(1).expand(T, B, -1)], dim=-1)
        else:
            dec_in = ht
        raw = model.decoder(dec_in, t_norm)
        if model.output_scale is not None:
            pred = raw * model.output_scale + model.output_bias
        else:
            pred = raw
        if return_aux:
            return pred, ModelAux(z0=h0, ht=ht)
        return pred

    model.forward = wrapped


def attach_constant_z(model: nn.Module) -> None:
    model.z_constant = PerTrackConstantZ()
    orig = model.forward

    def wrapped(states, t, return_aux=False, track_ids=None):
        out = orig(states, t, return_aux=return_aux, track_ids=track_ids)
        if track_ids is None:
            return out
        pred, aux = (out if return_aux else (out, None))
        z = model.z_constant(track_ids).to(pred.dtype)
        pred = pred.clone()
        pred[:, :, 4] = z.unsqueeze(0).expand(pred.shape[0], -1)
        return (pred, aux) if return_aux else pred

    model.forward = wrapped
