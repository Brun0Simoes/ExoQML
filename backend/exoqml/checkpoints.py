from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None


ALLOWED_ARCHITECTURES = {
    "transit-cnn-1d",
    "transit-multiview-tce",
    "transit-hybrid-qml-tce",
    "transit-residual-qml-tce",
}


class CheckpointValidationError(RuntimeError):
    pass


def _is_tensor_mapping(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not value:
        return False
    for key, tensor in value.items():
        if not isinstance(key, str):
            return False
        if not HAS_TORCH or not isinstance(tensor, torch.Tensor):
            return False
    return True


def _coerce_checkpoint_dict(payload: object) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if _is_tensor_mapping(payload):
        return {"state_dict": dict(payload)}
    raise CheckpointValidationError("Checkpoint payload must be a dict or a state_dict mapping.")


def validate_checkpoint(
    payload: object,
    *,
    allowed_architectures: set[str] | None = None,
    require_state_dict: bool = True,
    require_optimizer: bool = False,
) -> dict[str, Any]:
    checkpoint = _coerce_checkpoint_dict(payload)

    if require_state_dict:
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not _is_tensor_mapping(state_dict):
            raise CheckpointValidationError("Checkpoint state_dict is missing or invalid.")

    architecture = checkpoint.get("architecture")
    if architecture is not None:
        if not isinstance(architecture, str):
            raise CheckpointValidationError("Checkpoint architecture must be a string.")
        if allowed_architectures is not None and architecture not in allowed_architectures:
            raise CheckpointValidationError(f"Unsupported checkpoint architecture: {architecture}")

    threshold = checkpoint.get("threshold")
    if threshold is not None:
        try:
            float(threshold)
        except Exception as exc:
            raise CheckpointValidationError("Checkpoint threshold must be numeric.") from exc

    if require_optimizer and not isinstance(checkpoint.get("optimizer"), dict):
        raise CheckpointValidationError("Training checkpoint is missing optimizer state.")

    return checkpoint


def load_checkpoint(
    path: Path,
    *,
    map_location: str,
    allowed_architectures: set[str] | None = None,
    require_state_dict: bool = True,
    require_optimizer: bool = False,
) -> dict[str, Any]:
    if not HAS_TORCH:
        raise CheckpointValidationError("Torch is unavailable in this environment.")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    return validate_checkpoint(
        payload,
        allowed_architectures=allowed_architectures,
        require_state_dict=require_state_dict,
        require_optimizer=require_optimizer,
    )
