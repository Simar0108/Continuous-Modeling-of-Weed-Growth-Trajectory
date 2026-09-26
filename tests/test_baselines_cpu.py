"""CPU forward-pass tests for LSTM / GRU / Transformer H1 baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ode.baselines import build_baseline_model


def test_build_each_baseline():
    for name in ("lstm", "gru", "transformer"):
        model = build_baseline_model(name)
        assert sum(p.numel() for p in model.parameters()) > 0


def test_teacher_force_and_ar_shapes():
    torch.manual_seed(0)
    b, t, d = 3, 12, 5
    states = torch.randn(b, t, d)
    t_abs = torch.linspace(0.0, 100.0, t).unsqueeze(0).expand(b, t)
    lengths = torch.tensor([12, 10, 8])
    horizon = t - 3
    for name in ("lstm", "gru", "transformer"):
        model = build_baseline_model(name)
        model.eval()
        with torch.inference_mode():
            tf = model.teacher_forcing_rollout(states, t_abs, lengths)
            ar = model.autoregressive_rollout(states, t_abs, lengths)
        assert tf.shape == (b, horizon, 3), name
        assert ar.shape == (b, horizon, 3), name
        assert torch.isfinite(tf).all(), name
        assert torch.isfinite(ar).all(), name
