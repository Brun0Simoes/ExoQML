from __future__ import annotations

from pathlib import Path

import torch

from exoqml.checkpoints import ALLOWED_ARCHITECTURES, CheckpointValidationError, load_checkpoint, validate_checkpoint


def test_validate_checkpoint_accepts_minimal_inference_checkpoint() -> None:
    payload = {
        "architecture": "transit-multiview-tce",
        "threshold": 0.65,
        "state_dict": {"layer.weight": torch.zeros(2, 2)},
    }

    checkpoint = validate_checkpoint(payload, allowed_architectures=ALLOWED_ARCHITECTURES)
    assert checkpoint["architecture"] == "transit-multiview-tce"


def test_validate_checkpoint_rejects_unknown_architecture() -> None:
    payload = {
        "architecture": "unknown-model",
        "state_dict": {"layer.weight": torch.zeros(2, 2)},
    }

    try:
        validate_checkpoint(payload, allowed_architectures=ALLOWED_ARCHITECTURES)
    except CheckpointValidationError as exc:
        assert "Unsupported checkpoint architecture" in str(exc)
    else:
        raise AssertionError("Expected CheckpointValidationError for unsupported architecture")


def test_load_checkpoint_uses_safe_weights_only(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "architecture": "transit-cnn-1d",
            "threshold": 0.5,
            "state_dict": {"conv.weight": torch.ones(1, 1, 3)},
        },
        path,
    )

    checkpoint = load_checkpoint(path, map_location="cpu", allowed_architectures=ALLOWED_ARCHITECTURES)
    assert checkpoint["architecture"] == "transit-cnn-1d"
