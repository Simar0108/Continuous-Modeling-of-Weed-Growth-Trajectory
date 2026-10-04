"""Neural ODE model restored from Step-6 3.10 bytecode (2026-09-24 16:46).

Affine decoder init stays frozen at construction (requires_grad=False).
``normalize_z0`` default is True. Default dynamics are StableSigmoidal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

STATE_DIM = 5
PHYSIOLOGY_INDEX = 4
SIZE_CHANNELS = slice(0, 2)
SIZE_CHANNEL_SLICE = slice(0, 2)
PHYS_CHANNEL_START = 2
LAG_T_MAX = 0.4
DEFAULT_SIGMA_TIMESCALE = 0.35
DEFAULT_Z_DERIV_CAP = 3.0


def _fourier_time_features(t: torch.Tensor, n_freqs: int = 4) -> torch.Tensor:
    """Encode scalar time t into Fourier features. t : (T,) → (T, 2*n_freqs)."""
    freqs = 2 ** torch.arange(n_freqs, dtype=t.dtype, device=t.device)
    angles = t.unsqueeze(-1) * freqs * math.pi
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


@dataclass
class ModelAux:
    z0: torch.Tensor
    ht: Optional[torch.Tensor] = None


class ContextEncoderMLP(nn.Module):
    def __init__(
        self,
        state_dim: int = 5,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        n_context_frames: int = 3,
    ) -> None:
        super().__init__()
        self.n_context_frames = n_context_frames
        self.state_dim = state_dim
        flat_dim = n_context_frames * (state_dim + 1)
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, states: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        batch = states.shape[0]
        k_avail = min(self.n_context_frames, states.shape[1])
        k = self.n_context_frames
        ctx_states = states[:, :k_avail, :]
        t_ctx = timestamps[:k_avail]
        t_rel = t_ctx - t_ctx[0]
        t_span = t_rel[-1].clamp(min=1.0)
        t_rel_norm = t_rel / t_span
        t_exp = t_rel_norm.unsqueeze(0).unsqueeze(-1).expand(batch, k_avail, 1).contiguous()
        framed = torch.cat([ctx_states, t_exp], dim=-1)
        if k_avail < k:
            pad = torch.zeros(
                batch, k - k_avail, self.state_dim + 1,
                device=states.device, dtype=states.dtype,
            )
            framed = torch.cat([framed, pad], dim=1)
        return self.net(framed.reshape(batch, -1))


class ContextEncoderGRU(nn.Module):
    def __init__(
        self,
        state_dim: int = 5,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        n_context_frames: int = 3,
    ) -> None:
        super().__init__()
        self.n_context_frames = n_context_frames
        self.frame_proj = nn.Sequential(nn.Linear(state_dim + 1, hidden_dim), nn.Tanh())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, states: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        batch = states.shape[0]
        k = min(self.n_context_frames, states.shape[1])
        t_ctx = timestamps[:k]
        t_rel = t_ctx - t_ctx[0]
        t_norm = t_rel / t_rel[-1].clamp(min=1.0)
        t_exp = t_norm.unsqueeze(0).unsqueeze(-1).expand(batch, k, 1).contiguous()
        inp = self.frame_proj(torch.cat([states[:, :k, :], t_exp], dim=-1))
        _, h_n = self.gru(inp)
        return self.output_proj(h_n.squeeze(0))


class TrackEmbedding(nn.Module):
    """H1-only symmetry breaker: per-track table added to z0."""

    def __init__(self, latent_dim: int, embed_dim: int = 16, num_embeddings: int = 16384) -> None:
        super().__init__()
        self.table = nn.Embedding(num_embeddings, embed_dim)
        nn.init.normal_(self.table.weight, mean=0.0, std=0.1)
        self.proj = nn.Linear(embed_dim, latent_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight, gain=0.1)

    def forward(self, z0: torch.Tensor, track_ids: torch.Tensor) -> torch.Tensor:
        idx = track_ids.long().reshape(-1) % self.table.num_embeddings
        return z0 + self.proj(self.table(idx))


class GrowthDecoder(nn.Module):
    def __init__(self, dec_in: int, hidden_dim: int, state_dim: int = 5) -> None:
        super().__init__()
        self.hidden = nn.Linear(dec_in, hidden_dim)
        self.final = nn.Linear(hidden_dim, state_dim)
        nn.init.xavier_normal_(self.final.weight, gain=2.0)
        nn.init.zeros_(self.final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(F.silu(self.hidden(x)))


class StiffODEMixin:
    """Timescale separation + Lipschitz bounding for slow/fast ODE channels."""

    def _init_stiffness_params(
        self, *, sigma_timescale: float = DEFAULT_SIGMA_TIMESCALE, z_deriv_cap: float = DEFAULT_Z_DERIV_CAP,
    ) -> None:
        self.log_sigma_scale = nn.Parameter(torch.tensor(math.log(max(sigma_timescale, 0.001))))
        self.register_buffer("z_deriv_cap", torch.tensor(float(z_deriv_cap)))

    def _scale_slow_dynamics(self, dh_size: torch.Tensor) -> torch.Tensor:
        return dh_size * torch.exp(self.log_sigma_scale).clamp(max=1.0)

    def _bound_fast_dynamics(self, dh_rest: torch.Tensor) -> torch.Tensor:
        cap = self.z_deriv_cap.clamp(min=0.25)
        return cap * torch.tanh(dh_rest / cap)


class ODEFunc(StiffODEMixin, nn.Module):
    """Phase-dependent early/late heads (legacy)."""

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 3,
        ode_growth_bias: bool = True,
        phased: bool = True,
        t_mid_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.nfe = 0
        self.latent_dim = latent_dim
        shared_layers: list[nn.Module] = []
        in_dim = latent_dim + 1
        for _ in range(n_layers):
            shared_layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        self.shared = nn.Sequential(*shared_layers)
        self.early_head = nn.Linear(hidden_dim, latent_dim)
        self.late_head = nn.Linear(hidden_dim, latent_dim)
        self.transition_center = nn.Parameter(torch.tensor(float(t_mid_init)))
        self.transition_width = nn.Parameter(torch.tensor(0.1))
        self.size_growth_bias = nn.Parameter(torch.zeros(2)) if ode_growth_bias else None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        self._init_stiffness_params()

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        t_exp = t.reshape(1, 1).expand(h.shape[0], 1)
        shared = self.shared(torch.cat([h, t_exp], dim=-1))
        gate = torch.sigmoid(
            (t - self.transition_center) / self.transition_width.abs().clamp(min=0.001)
        )
        out = (1.0 - gate) * self.early_head(shared) + gate * self.late_head(shared)
        out_size = out[:, :2]
        if self.size_growth_bias is not None:
            out_size = out_size + F.softplus(self.size_growth_bias)
        dh_size = self._scale_slow_dynamics(out_size)
        if out.shape[1] > 2:
            dh_rest = self._bound_fast_dynamics(out[:, 2:])
            return torch.cat([dh_size, dh_rest], dim=-1)
        return dh_size


class SigmoidalODEFunc(StiffODEMixin, nn.Module):
    """Legacy Verhulst: dz/dt = r · z · (1 − z/K)."""

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 2,
        head_dim: int = 0,
        ode_growth_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.nfe = 0
        self.latent_dim = latent_dim
        if head_dim == 0:
            head_dim = max(hidden_dim // 4, 32)
        shared_layers: list[nn.Module] = []
        in_dim = latent_dim + 1
        for _ in range(n_layers):
            shared_layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        self.shared = nn.Sequential(*shared_layers)
        self.state_proj = nn.Linear(latent_dim, 2)
        self.time_proj = nn.Linear(1, 2)
        self.r_head = nn.Sequential(nn.Linear(2, head_dim), nn.SiLU(), nn.Linear(head_dim, 2))
        self.k_head = nn.Sequential(nn.Linear(2, head_dim), nn.SiLU(), nn.Linear(head_dim, 2))
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
        self.early_head = self.r_head
        self.late_head = self.k_head
        self.transition_center = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.transition_width = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self._init_stiffness_params()

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        batch = h.shape[0]
        t_scalar = t.reshape(1, 1).expand(batch, 1)
        interaction = self.state_proj(h) * self.time_proj(t_scalar)
        rate = F.softplus(self.r_head(interaction))
        sat = F.softplus(self.k_head(interaction)) + 0.001
        h_size = h[:, :2]
        dh_size = self._scale_slow_dynamics(rate * h_size * (1.0 - h_size / sat))
        shared = self.shared(torch.cat([h, t_scalar], dim=-1))
        if self.residual_head is not None:
            dh_rest = self._bound_fast_dynamics(self.residual_head(shared))
            return torch.cat([dh_size, dh_rest], dim=-1)
        return dh_size


class StableSigmoidalODEFunc(StiffODEMixin, nn.Module):
    """Default H1 RHS: dz/dt = rate · σ(sat − z). late_head aliases saturation_net."""

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 2,
        head_dim: int = 0,
        ode_growth_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.nfe = 0
        self.latent_dim = latent_dim
        self.r_max = 2.0
        self.k_min = 0.3
        self.k_max = 2.0
        if head_dim == 0:
            head_dim = max(hidden_dim // 4, 32)
        shared_layers: list[nn.Module] = []
        in_dim = latent_dim + 1
        for _ in range(n_layers):
            shared_layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        self.shared = nn.Sequential(*shared_layers)
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
        self.shared_t_lag = nn.Parameter(torch.tensor(0.0))
        self.shared_tau = nn.Parameter(torch.tensor(0.05))
        self._t_lag = None
        self._tau = None
        self._init_stiffness_params()

    def _bound_rate(self, pre: torch.Tensor) -> torch.Tensor:
        return self.r_max * torch.sigmoid(pre)

    def _bound_sat(self, pre: torch.Tensor) -> torch.Tensor:
        return self.k_min + (self.k_max - self.k_min) * torch.sigmoid(pre)

    def set_lag(self, t_lag: torch.Tensor, tau: torch.Tensor) -> None:
        self._t_lag = t_lag
        self._tau = tau

    def _lag_gate(self, t: torch.Tensor, batch: int) -> torch.Tensor:
        t_s = t.reshape(())
        if self._t_lag is not None and self._tau is not None:
            t_lag = self._t_lag.to(device=t.device, dtype=t.dtype)
            tau = self._tau.to(device=t.device, dtype=t.dtype).clamp(min=0.001)
            return torch.sigmoid((t_s - t_lag) / tau).unsqueeze(-1)
        t_lag = self.shared_t_lag.to(device=t.device, dtype=t.dtype)
        tau = self.shared_tau.to(device=t.device, dtype=t.dtype).clamp(min=0.001)
        gate = torch.sigmoid((t_s - t_lag) / tau)
        return gate.expand(batch).unsqueeze(-1)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        batch = h.shape[0]
        t_scalar = t.reshape(1, 1).expand(batch, 1)
        shared = self.shared(torch.cat([h, t_scalar], dim=-1))
        rate = self._bound_rate(self.rate_net(shared))
        sat = self._bound_sat(self.saturation_net(shared))
        h_size = h[:, :2]
        lag = self._lag_gate(t, batch)
        dh_size = self._scale_slow_dynamics(rate * torch.sigmoid(sat - h_size) * lag)
        if self.residual_head is not None:
            dh_rest = self._bound_fast_dynamics(self.residual_head(shared))
            return torch.cat([dh_size, dh_rest], dim=-1)
        return dh_size


class AffineDecoder(nn.Module):
    def __init__(self, dec_in: int, hidden_dim: int, state_dim: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dec_in, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        nn.init.xavier_normal_(self.net[-1].weight, gain=2.0)
        nn.init.zeros_(self.net[-1].bias)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.net[-1]:
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net(x)


class HybridDecoder(nn.Module):
    def __init__(
        self, dec_in: int, hidden_dim: int, state_dim: int = 5, phys_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        n_size = state_dim - 1
        n_phys = 1
        self.size_head = nn.Sequential(
            nn.Linear(dec_in, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, n_size),
        )
        phys_in = dec_in + 1
        phys_hidden = max(hidden_dim // 2, 32)
        phys_layers: list[nn.Module] = [nn.Linear(phys_in, phys_hidden), nn.SiLU()]
        if phys_dropout > 0:
            phys_layers.append(nn.Dropout(p=phys_dropout))
        phys_layers += [nn.Linear(phys_hidden, phys_hidden), nn.SiLU()]
        if phys_dropout > 0:
            phys_layers.append(nn.Dropout(p=phys_dropout))
        phys_layers.append(nn.Linear(phys_hidden, n_phys))
        self.phys_head = nn.Sequential(*phys_layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.size_head[-1] or m is self.phys_head[-1]:
                    nn.init.xavier_normal_(m.weight, gain=2.0)
                else:
                    nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        size_out = self.size_head(x)
        if t is None:
            t_exp = x.new_zeros(*x.shape[:-1], 1)
        elif x.dim() == 3:
            steps, batch, _ = x.shape
            t_exp = t.reshape(steps, 1, 1).expand(steps, batch, 1)
        else:
            t_exp = t.reshape(1, 1).expand(x.shape[0], 1)
        phys_out = self.phys_head(torch.cat([x, t_exp], dim=-1))
        return torch.cat([size_out, phys_out], dim=-1)


class LatentODE(nn.Module):
    def __init__(
        self,
        state_dim: int = 5,
        latent_dim: int = 32,
        ode_hidden: int = 256,
        n_ode_layers: int = 3,
        n_context_frames: int = 3,
        encoder: str = "mlp",
        solver: str = "dopri5",
        rtol: float = 1e-06,
        atol: float = 1e-06,
        normalize_z0: bool = True,
        ode_growth_bias: bool = True,
        time_conditioned_decoder: bool = False,
        n_fourier_freqs: int = 4,
        affine_decoder: bool = True,
        affine_scale_init: float = 6.0,
        hybrid_decoder: bool = True,
        phys_dropout: float = 0.2,
        ode_type: str = "stable_sigmoidal",
        ode_t_mid: float = 0.5,
        phased_ode: bool = False,
        use_track_embed: bool = False,
        track_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.normalize_z0 = normalize_z0
        self.time_conditioned_decoder = time_conditioned_decoder
        self.n_fourier_freqs = n_fourier_freqs
        enc_cls = ContextEncoderMLP if encoder == "mlp" else ContextEncoderGRU
        self.context_encoder = enc_cls(
            state_dim=state_dim, latent_dim=latent_dim,
            hidden_dim=ode_hidden, n_context_frames=n_context_frames,
        )
        ode_kind = "phased" if phased_ode else ode_type
        self.use_track_embed = use_track_embed
        self.track_embedding = (
            TrackEmbedding(latent_dim, embed_dim=track_embed_dim) if use_track_embed else None
        )
        if ode_kind == "stable_sigmoidal":
            self.ode_func = StableSigmoidalODEFunc(
                latent_dim=latent_dim, hidden_dim=ode_hidden,
                n_layers=n_ode_layers, ode_growth_bias=ode_growth_bias,
            )
        elif ode_kind == "sigmoidal":
            self.ode_func = SigmoidalODEFunc(
                latent_dim=latent_dim, hidden_dim=ode_hidden,
                n_layers=n_ode_layers, ode_growth_bias=ode_growth_bias,
            )
        else:
            self.ode_func = ODEFunc(
                latent_dim=latent_dim, hidden_dim=ode_hidden, n_layers=n_ode_layers,
                ode_growth_bias=ode_growth_bias, phased=True, t_mid_init=ode_t_mid,
            )
        dec_in = latent_dim + 2 * n_fourier_freqs if time_conditioned_decoder else latent_dim
        self._hybrid_decoder = hybrid_decoder
        if hybrid_decoder:
            self.decoder = HybridDecoder(dec_in, ode_hidden, state_dim, phys_dropout=phys_dropout)
        else:
            self.decoder = AffineDecoder(dec_in, ode_hidden, state_dim)
        if affine_decoder:
            self.output_scale = nn.Parameter(
                torch.full((state_dim,), affine_scale_init), requires_grad=False,
            )
            self.output_bias = nn.Parameter(torch.zeros(state_dim), requires_grad=False)
        else:
            self.output_scale = None
            self.output_bias = None

    def forward(
        self,
        states: torch.Tensor,
        t: torch.Tensor,
        return_aux: bool = False,
        track_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ModelAux]:
        t_span = (t[-1] - t[0]).clamp(min=0.001)
        t_norm = (t - t[0]) / t_span
        h0 = self.context_encoder(states, t_norm)
        if self.normalize_z0:
            h0 = h0 / h0.norm(dim=-1, keepdim=True).clamp(min=1e-08)
        if self.track_embedding is not None:
            if track_ids is None:
                raise ValueError("use_track_embed=True requires track_ids in forward")
            h0 = self.track_embedding(h0, track_ids)
        lag_head = getattr(self, "lag_head", None)
        if lag_head is not None:
            t_lag, tau = lag_head(h0)
            self.ode_func.set_lag(t_lag, tau)
        self.ode_func.nfe = 0
        ht = odeint(
            self.ode_func, h0, t_norm,
            rtol=self.rtol, atol=self.atol, method=self.solver,
        )
        if self.time_conditioned_decoder:
            tf = _fourier_time_features(t_norm, self.n_fourier_freqs)
            steps, batch, _ = ht.shape
            tf_exp = tf.unsqueeze(1).expand(steps, batch, -1)
            dec_in = torch.cat([ht, tf_exp], dim=-1)
        else:
            dec_in = ht
        raw = self.decoder(dec_in, t_norm)
        pred = raw * self.output_scale + self.output_bias if self.output_scale is not None else raw
        if return_aux:
            return pred, ModelAux(z0=h0, ht=ht)
        return pred
