"""CPU checks for cubic Hermite control. No GPU."""

from ode._pyc_bootstrap import bootstrap

bootstrap()

import torch

from ode.ncde import (
    NeuralCDEFunc,
    attach_ncde_control,
    backward_diffs,
    context_control,
    control_derivs,
    hermite_eval,
)


def test_constant_path_zero_derivative():
    knots = torch.tensor([0.0, 0.5, 1.0])
    values = torch.ones(2, 3, 3)
    derivs = backward_diffs(knots, values)
    x, dx = hermite_eval(knots, values, derivs, torch.tensor(0.3))
    assert torch.allclose(x[:, :2], torch.ones_like(x[:, :2]), atol=1e-5)
    assert float(dx[:, :2].abs().max()) < 1e-5
    # Time channel is identity, not a frozen spline.
    assert abs(float(x[0, 2]) - 0.3) < 1e-5
    assert abs(float(dx[0, 2]) - 1.0) < 1e-5


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
    # 3 context knots + terminal hold at t_norm=1
    assert knots.shape[0] == 4
    assert values.shape == (1, 4, 3)
    assert torch.allclose(values[0, :3, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.allclose(values[0, :3, 1], torch.tensor([0.0, 2.0, 4.0]))
    assert abs(float(knots[-1]) - 1.0) < 1e-6
    assert torch.allclose(values[0, -1, :2], values[0, 2, :2])


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


def test_time_channel_prevents_frozen_tail():
    """After the last context frame, size is held but time still ticks."""
    t = torch.linspace(0, 10, 20)
    states = torch.zeros(1, 20, 5)
    states[0, :3, 2] = torch.tensor([0.0, 1.0, 2.0])
    states[0, :3, 3] = torch.tensor([0.0, 2.0, 4.0])
    knots, values = context_control(states, t, n_context=3)
    derivs = control_derivs(knots, values)
    # t_norm=0.9 is well past the three context knots.
    x, dx = hermite_eval(knots, values, derivs, torch.tensor(0.9))
    assert abs(float(x[0, 2]) - 0.9) < 1e-5
    assert abs(float(dx[0, 2]) - 1.0) < 1e-4
    # Size hold: last context sigma, near-zero derivative.
    assert torch.allclose(x[0, :2], values[0, 2, :2], atol=0.05)
    assert float(dx[0, :2].abs().max()) < 0.05
    # Without a time channel, ||X'|| would be ~0 and the CDE would freeze.
    sigma_only = float(dx[0, :2].norm())
    assert float(dx[0].norm()) > sigma_only + 0.5


def test_nfe_resets_per_forward():
    class _M:
        def __init__(self):
            self.ode_func = NeuralCDEFunc(latent_dim=4, hidden_dim=8, n_layers=2)
            self.context_encoder = type("E", (), {"n_context_frames": 3})()

        def forward(self, states, t, return_aux=False, track_ids=None):
            t0 = t[0] if t.dim() else t
            return self.ode_func(t0, torch.zeros(states.shape[0], 4))

    model = _M()
    attach_ncde_control(model, n_context=3)
    t = torch.linspace(0, 1, 5)
    states = torch.randn(1, 5, 5)
    model.forward(states, t)
    first = int(model.ode_func.nfe)
    model.forward(states, t)
    second = int(model.ode_func.nfe)
    assert first >= 1
    assert second == first


def test_nfe_budget_pauses_after_sustain():
    from ode.callbacks_h1 import NFEBudgetCallback

    class _Mod:
        def __init__(self):
            self.model = type("M", (), {"ode_func": type("O", (), {"nfe": 200})()})()
            self.device = torch.device("cpu")

        def log(self, *args, **kwargs):
            return None

    class _Tr:
        current_epoch = 0
        sanity_checking = False
        should_stop = False

    callback = NFEBudgetCallback(limit=150, sustain_epochs=3)
    module = _Mod()
    trainer = _Tr()
    for epoch in range(3):
        trainer.current_epoch = epoch
        callback.on_train_batch_end(trainer, module, None, None, 0)
        callback.on_validation_epoch_end(trainer, module)
        callback.on_train_epoch_end(trainer, module)
    assert callback.paused
    assert trainer.should_stop
