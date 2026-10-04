"""Oracle check: restored hybrid_loss matches archived Step-6 bytecode."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
ARCHIVE = REPO / "research" / "archive" / "step6_bytecode" / "training_loop.cpython-310.pyc"


def _load_archived_hybrid_loss():
    if not ARCHIVE.is_file():
        return None
    spec = importlib.util.spec_from_file_location("ode._archived_training_loop", ARCHIVE)
    mod = importlib.util.module_from_spec(spec)
    # Archived pyc imports `from model import ...`; put restored model on path.
    sys.modules.setdefault("model", importlib.import_module("ode.model"))
    spec.loader.exec_module(mod)
    return mod.hybrid_loss


def test_hybrid_loss_matches_archived_pyc():
    from ode.training_loop import hybrid_loss

    archived = _load_archived_hybrid_loss()
    if archived is None:
        return
    torch.manual_seed(0)
    pred = torch.randn(3, 8, 5)
    target = torch.randn(3, 8, 5)
    color = torch.zeros(3, 8, dtype=torch.bool)
    color[:, 3:] = True
    pad = torch.ones(3, 8, dtype=torch.bool)
    pad[:, -2:] = False
    a = hybrid_loss(pred, target, color, pad_mask=pad, xy_loss_weight=0.0)
    b = archived(pred, target, color, pad_mask=pad, xy_loss_weight=0.0)
    for x, y in zip(a[:3], b[:3]):
        assert torch.allclose(x, y, atol=1e-6, rtol=1e-5)
    color_none = torch.zeros(3, 8, dtype=torch.bool)
    a2 = hybrid_loss(pred, target, color_none, pad_mask=pad)
    b2 = archived(pred, target, color_none, pad_mask=pad)
    assert torch.allclose(a2[2], b2[2], atol=1e-6)
    assert float(a2[2]) == 0.0
