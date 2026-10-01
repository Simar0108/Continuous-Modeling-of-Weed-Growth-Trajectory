"""CPU checks for cubic Hermite control. No GPU."""

from ode._pyc_bootstrap import bootstrap

bootstrap()

import torch

from ode.ncde import NeuralCDEFunc, backward_diffs, context_control, hermite_eval


def test_constant_path_zero_derivative():
    knots = torch.tensor([0.0, 0.5, 1.0])
    values = torch.ones(2, 3, 3)
    derivs = backward_diffs(knots, values)
    x, dx = hermite_eval(knots, values, derivs, torch.tensor(0.3))
    assert torch.allclose(x, torch.ones_like(x), atol=1e-5)
    assert float(dx.abs().max()) < 1e-5


def test_linear_path_recovered():
    knots = torch.tensor([0.0, 0.5, 1.0])
    t_knots = knots.reshape(1, 3, 1).expand(1, 3, 1)
    values = torch.cat([t_knots, t_knots, t_knots], dim=-1)
    derivs = backward_diffs(knots, values)
    t = torch.tensor(0.25)
    x, dx = hermite_eval(knots, values, derivs, t)
    assert torch.allclose(x, torch.full((1, 3), 0.25), atol=1e-4)
    assert torch.allclose(dx, torch.ones(1, 3), atol=1e-3)


def test_context_control_no_future_sizes():
    t = torch.linspace(0, 10, 8)
    states = torch.zeros(1, 8, 5)
    states[0, :, 2] = torch.arange(8).float()
    states[0, :, 3] = 2.0 * torch.arange(8).float()
    knots, values = context_control(states, t, n_context=3)
    assert knots.shape == (3,)
    assert values.shape == (1, 3, 3)
    assert torch.allclose(values[0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.allclose(values[0, :, 1], torch.tensor([0.0, 2.0, 4.0]))


def test_cde_zero_init_small_dz():
    ode = NeuralCDEFunc(latent_dim=8, hidden_dim=16, n_layers=2)
    t = torch.linspace(0, 1, 5)
    states = torch.randn(2, 5, 5)
    knots, values = context_control(states, t, n_context=3)
    ode.set_control(knots, values)
    z = torch.randn(2, 8)
    dz = ode(torch.tensor(0.1), z)
    assert dz.shape == z.shape
    assert float(dz.abs().max()) < 1e-5
