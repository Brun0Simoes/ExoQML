from __future__ import annotations

import torch

from exoqml.training.model import HAS_PENNYLANE, TransitHybridQMLNet, TransitResidualQMLNet


def test_qml_model_forward_backward() -> None:
    if not HAS_PENNYLANE:
        return

    model = TransitHybridQMLNet(
        scalar_dim=4,
        base_channels=8,
        dropout=0.0,
        n_qubits=4,
        n_q_layers=1,
        q_device="default.qubit",
    )
    global_view = torch.randn(2, 1, 401)
    local_view = torch.randn(2, 1, 121)
    scalar = torch.randn(2, 4)
    logits = model(global_view, local_view, scalar)
    loss = logits.pow(2).mean()
    loss.backward()
    assert logits.shape == (2, 1)


def test_residual_qml_model_forward_components() -> None:
    if not HAS_PENNYLANE:
        return

    model = TransitResidualQMLNet(
        scalar_dim=4,
        base_channels=8,
        dropout=0.0,
        n_qubits=4,
        n_q_layers=1,
        q_device="default.qubit",
        residual_alpha_init=0.2,
    )
    global_view = torch.randn(2, 1, 401)
    local_view = torch.randn(2, 1, 121)
    scalar = torch.randn(2, 4)
    logits, classical_logits, residual_logits, residual_alpha = model.forward_with_components(global_view, local_view, scalar)
    loss = logits.pow(2).mean() + classical_logits.pow(2).mean() + residual_logits.pow(2).mean()
    loss.backward()
    assert logits.shape == (2, 1)
    assert classical_logits.shape == (2, 1)
    assert residual_logits.shape == (2, 1)
    assert 0.0 < float(residual_alpha.detach().cpu().item()) < 1.0
