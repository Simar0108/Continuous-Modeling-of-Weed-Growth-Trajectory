"""CPU checks for path-length masking. No GPU."""

from ode._pyc_bootstrap import bootstrap

bootstrap()

import torch

from ode.pathreg import path_length_ht, pairwise_z0_cosine, lambda_tag


def test_lambda_tag():
    assert lambda_tag(0.01) == "0p01"
    assert lambda_tag(0.1) == "0p1"
    assert lambda_tag(1.0) == "1"


def test_path_length_zero_when_constant():
    ht = torch.ones(8, 3, 4)
    lengths = torch.tensor([8, 8, 8])
    val = path_length_ht(ht, lengths, horizon_T=8, n_times=8)
    assert float(val) < 1e-6


def test_path_length_masks_padding():
    ht = torch.zeros(6, 2, 3)
    ht[1, 0, 0] = 1.0
    ht[2, 0, 0] = 2.0
    lengths = torch.tensor([3, 1])
    val = path_length_ht(ht, lengths, horizon_T=6, n_times=6)
    # track 0: two unit segments; track 1 has no valid segment
    assert abs(float(val) - 2.0) < 1e-5


def test_z0_cosine_identical():
    z = torch.ones(5, 8)
    stats = pairwise_z0_cosine(z)
    assert stats["mean_offdiag"] > 0.999
