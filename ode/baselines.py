"""
Discrete sequence baselines for plant growth forecasting.

All baselines:
  - condition on the first K frames via the same ContextEncoderMLP used by the ODE
  - predict only [sigma_w, sigma_h, Z]
  - support teacher-forced training and autoregressive rollout
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from .model import ContextEncoderMLP, STATE_DIM

TARGET_DIM = 3
TARGET_SLICE = slice(2, 5)  # [sigma_w, sigma_h, Z] within the 5-D state


def _ensure_batched(
    states_5d: torch.Tensor,
    t_absolute: torch.Tensor,
    track_lengths: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Canonicalise inputs to batched tensors."""
    if states_5d.dim() == 2:
        states_5d = states_5d.unsqueeze(0)
    if t_absolute.dim() == 1:
        t_absolute = t_absolute.unsqueeze(0)
    if track_lengths is None:
        track_lengths = torch.full(
            (states_5d.shape[0],),
            states_5d.shape[1],
            dtype=torch.long,
            device=states_5d.device,
        )
    return states_5d, t_absolute, track_lengths


def _normalise_timestamps(
    t_absolute: torch.Tensor,
    track_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalise each track's timestamps independently to [0, 1]."""
    if t_absolute.dim() == 1:
        t_absolute = t_absolute.unsqueeze(0)
    start = t_absolute[:, :1]
    if track_lengths is None:
        end = t_absolute[:, -1:]
    else:
        last_idx = (track_lengths - 1).clamp(min=0)
        end = t_absolute[
            torch.arange(t_absolute.shape[0], device=t_absolute.device),
            last_idx,
        ].unsqueeze(1)
    span = (end - start).clamp(min=1e-3)
    return (t_absolute - start) / span


def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
    mask = torch.ones((length, length), device=device, dtype=torch.bool)
    return torch.triu(mask, diagonal=1)


class GrowthBaselineBase(nn.Module, ABC):
    """Shared utilities for all discrete baselines."""

    def __init__(
        self,
        *,
        n_context_frames: int = 3,
        context_hidden_dim: int = 256,
        context_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_context_frames = n_context_frames
        self.context_encoder = ContextEncoderMLP(
            state_dim=STATE_DIM,
            latent_dim=context_dim,
            hidden_dim=context_hidden_dim,
            n_context_frames=n_context_frames,
        )

    @staticmethod
    def target_channels(states_5d: torch.Tensor) -> torch.Tensor:
        return states_5d[..., TARGET_SLICE]

    def encode_context(
        self,
        states_5d: torch.Tensor,
        t_absolute: torch.Tensor,
        track_lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode each track's first K frames with the shared ODE context encoder.

        Returns:
          context      : (B, context_dim)
          t_norm       : (B, T)
          track_lengths: (B,)
        """
        states_5d, t_absolute, track_lengths = _ensure_batched(
            states_5d, t_absolute, track_lengths
        )
        t_norm = _normalise_timestamps(t_absolute, track_lengths)

        ctx: list[torch.Tensor] = []
        for b in range(states_5d.shape[0]):
            length_b = int(track_lengths[b].item())
            states_b = states_5d[b : b + 1, :length_b, :]
            t_b = t_norm[b, :length_b]
            ctx.append(self.context_encoder(states_b, t_b).squeeze(0))
        return torch.stack(ctx, dim=0), t_norm, track_lengths

    def _future_inputs(
        self,
        states_5d: torch.Tensor,
        t_absolute: torch.Tensor,
        track_lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build autoregressive input tensors for the post-context horizon.

        Returns:
          prev_targets  : (B, H, 3) ground-truth previous states for teacher forcing
          delta_t       : (B, H, 1) normalised step sizes
          target_t      : (B, H, 1) normalised timestamps for the prediction targets
          future_lengths: (B,)
          context       : (B, context_dim)
        """
        context, t_norm, track_lengths = self.encode_context(
            states_5d, t_absolute, track_lengths
        )
        targets = self.target_channels(states_5d)
        k = self.n_context_frames
        if targets.shape[1] <= k:
            horizon = targets.shape[1] * 0
            empty = targets[:, :horizon, :]
            return (
                empty,
                empty[..., :1],
                empty[..., :1],
                (track_lengths - k).clamp(min=0),
                context,
            )

        prev_targets = targets[:, k - 1 : -1, :]
        delta_t = (t_norm[:, k:] - t_norm[:, k - 1 : -1]).unsqueeze(-1)
        target_t = t_norm[:, k:].unsqueeze(-1)
        future_lengths = (track_lengths - k).clamp(min=0)
        return prev_targets, delta_t, target_t, future_lengths, context

    @abstractmethod
    def teacher_forcing_rollout(
        self,
        states_5d: torch.Tensor,
        t_absolute: torch.Tensor,
        track_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict the post-context horizon using ground-truth previous inputs."""

    @abstractmethod
    def autoregressive_rollout(
        self,
        states_5d: torch.Tensor,
        t_absolute: torch.Tensor,
        track_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict the post-context horizon using the model's own previous outputs."""

    def forward(
        self,
        states_5d: torch.Tensor,
        t_absolute: torch.Tensor,
        *,
        teacher_forcing: bool = True,
        track_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if teacher_forcing:
            return self.teacher_forcing_rollout(states_5d, t_absolute, track_lengths)
        return self.autoregressive_rollout(states_5d, t_absolute, track_lengths)


class _RecurrentGrowthModelBase(GrowthBaselineBase):
    """Shared recurrent rollout logic for the LSTM and GRU baselines."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        num_layers: int = 2,
        n_context_frames: int = 3,
        context_hidden_dim: int = 256,
    ) -> None:
        super().__init__(
            n_context_frames=n_context_frames,
            context_hidden_dim=context_hidden_dim,
            context_dim=hidden_dim,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_proj = nn.Sequential(
            nn.Linear(TARGET_DIM + 1, hidden_dim),
            nn.SiLU(),
        )
        self.output_proj = nn.Linear(hidden_dim, TARGET_DIM)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    @abstractmethod
    def _initial_state(self, context: torch.Tensor):
        ...

    @abstractmethod
    def _rnn_forward(self, x: torch.Tensor, state):
        ...

    def teacher_forcing_rollout(
        self,
        states_5d: torch.Tensor,
        t_absolute: torch.Tensor,
        track_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prev_targets, delta_t, _, _, context = self._future_inputs(
            states_5d, t_absolute, track_lengths
        )
        if prev_targets.shape[1] == 0:
            return prev_targets

        x = self.input_proj(torch.cat([prev_targets, delta_t], dim=-1))
        out, _ = self._rnn_forward(x, self._initial_state(context))
        return self.output_proj(out)

    def autoregressive_rollout(
        self,
        states_5d: torch.Tensor,
        t_absolute: torch.Tensor,
        track_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prev_targets, delta_t, _, future_lengths, context = self._future_inputs(
            states_5d, t_absolute, track_lengths
        )
        if prev_targets.shape[1] == 0:
            return prev_targets

        state = self._initial_state(context)
        prev = self.target_channels(states_5d)[:, self.n_context_frames - 1, :]
        preds: list[torch.Tensor] = []

        for step in range(prev_targets.shape[1]):
            x_t = torch.cat([prev, delta_t[:, step, :]], dim=-1).unsqueeze(1)
            x_t = self.input_proj(x_t)
            out, state = self._rnn_forward(x_t, state)
            pred_t = self.output_proj(out.squeeze(1))
            preds.append(pred_t)

            active = step < future_lengths
            prev = torch.where(active.unsqueeze(-1), pred_t, prev)

        return torch.stack(preds, dim=1)


class LSTMGrowthModel(_RecurrentGrowthModelBase):
    """2-layer LSTM baseline with 64 hidden units."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        num_layers: int = 2,
        n_context_frames: int = 3,
        context_hidden_dim: int = 256,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            n_context_frames=n_context_frames,
            context_hidden_dim=context_hidden_dim,
        )
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.h0_proj = nn.Linear(hidden_dim, num_layers * hidden_dim)
        self.c0_proj = nn.Linear(hidden_dim, num_layers * hidden_dim)
        self._init_weights()

    def _initial_state(self, context: torch.Tensor):
        batch_size = context.shape[0]
        h0 = self.h0_proj(context).view(batch_size, self.num_layers, self.hidden_dim)
        c0 = self.c0_proj(context).view(batch_size, self.num_layers, self.hidden_dim)
        return h0.transpose(0, 1).contiguous(), c0.transpose(0, 1).contiguous()

    def _rnn_forward(self, x: torch.Tensor, state):
        return self.rnn(x, state)


class GRUGrowthModel(_RecurrentGrowthModelBase):
    """2-layer GRU baseline with 64 hidden units."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        num_layers: int = 2,
        n_context_frames: int = 3,
        context_hidden_dim: int = 256,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            n_context_frames=n_context_frames,
            context_hidden_dim=context_hidden_dim,
        )
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.h0_proj = nn.Linear(hidden_dim, num_layers * hidden_dim)
        self._init_weights()

    def _initial_state(self, context: torch.Tensor):
        batch_size = context.shape[0]
        h0 = self.h0_proj(context).view(batch_size, self.num_layers, self.hidden_dim)
        return h0.transpose(0, 1).contiguous()

    def _rnn_forward(self, x: torch.Tensor, state):
        return self.rnn(x, state)


class TransformerGrowthModel(GrowthBaselineBase):
    """2-layer causal Transformer baseline with continuous-time token features."""

    def __init__(
        self,
        *,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_context_frames: int = 3,
        context_hidden_dim: int = 256,
    ) -> None:
        super().__init__(
            n_context_frames=n_context_frames,
            context_hidden_dim=context_hidden_dim,
            context_dim=d_model,
        )
        self.d_model = d_model
        self.input_proj = nn.Sequential(
            nn.Linear(TARGET_DIM + 2, d_model),  # prev state, delta_t, target timestamp
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, TARGET_DIM)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def teacher_forcing_rollout(
        self,
        states_5d: torch.Tensor,
        t_absolute: torch.Tensor,
        track_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prev_targets, delta_t, target_t, future_lengths, context = self._future_inputs(
            states_5d, t_absolute, track_lengths
        )
        if prev_targets.shape[1] == 0:
            return prev_targets

        tokens = torch.cat([prev_targets, delta_t, target_t], dim=-1)
        x = self.input_proj(tokens) + context.unsqueeze(1)
        length = x.shape[1]
        steps = torch.arange(length, device=x.device).unsqueeze(0)
        padding_mask = steps >= future_lengths.unsqueeze(1)
        out = self.encoder(
            x,
            mask=_causal_mask(length, x.device),
            src_key_padding_mask=padding_mask,
        )
        return self.output_proj(out)

    def autoregressive_rollout(
        self,
        states_5d: torch.Tensor,
        t_absolute: torch.Tensor,
        track_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, delta_t, target_t, future_lengths, context = self._future_inputs(
            states_5d, t_absolute, track_lengths
        )
        if delta_t.shape[1] == 0:
            return self.target_channels(states_5d)[:, :0, :]

        prev = self.target_channels(states_5d)[:, self.n_context_frames - 1, :]
        token_history: list[torch.Tensor] = []
        preds: list[torch.Tensor] = []

        for step in range(delta_t.shape[1]):
            token_t = torch.cat([prev, delta_t[:, step, :], target_t[:, step, :]], dim=-1)
            token_history.append(token_t)
            token_seq = torch.stack(token_history, dim=1)

            x = self.input_proj(token_seq) + context.unsqueeze(1)
            out = self.encoder(x, mask=_causal_mask(x.shape[1], x.device))
            pred_t = self.output_proj(out[:, -1, :])
            preds.append(pred_t)

            active = step < future_lengths
            prev = torch.where(active.unsqueeze(-1), pred_t, prev)

        return torch.stack(preds, dim=1)


def build_baseline_model(
    model_type: str,
    *,
    n_context_frames: int = 3,
) -> GrowthBaselineBase:
    model_type = model_type.lower()
    if model_type == "lstm":
        return LSTMGrowthModel(n_context_frames=n_context_frames)
    if model_type == "gru":
        return GRUGrowthModel(n_context_frames=n_context_frames)
    if model_type == "transformer":
        return TransformerGrowthModel(n_context_frames=n_context_frames)
    raise ValueError(f"Unknown model_type={model_type!r}; expected lstm, gru, or transformer.")
