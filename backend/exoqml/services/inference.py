from __future__ import annotations

from pathlib import Path

import numpy as np

from exoqml.config import Settings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as nnf

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None
    nn = None
    nnf = None


if HAS_TORCH:
    class TransitCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(1, 16, kernel_size=9, padding=4)
            self.conv2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
            self.conv3 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(32, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = nnf.relu(self.conv1(x))
            x = nnf.relu(self.conv2(x))
            x = nnf.relu(self.conv3(x))
            x = self.pool(x).squeeze(-1)
            return self.head(x)


def _select_device(settings: Settings) -> str:
    if settings.device == "cpu":
        return "cpu"
    if settings.device == "cuda":
        if HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _safe_sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, x))
    return float(1.0 / (1.0 + np.exp(-x)))


def _heuristic_inference(
    flux: np.ndarray,
    bls_peaks: list[dict[str, float]],
    experimental_qml: bool,
) -> dict:
    depth = float(max(0.0, 1.0 - np.min(flux)))
    variability = float(np.std(flux))
    bls_power = float(bls_peaks[0]["power"]) if bls_peaks else 0.0
    logit = (8.0 * depth) + (0.25 * bls_power) - (15.0 * variability) - 1.2
    probability = _safe_sigmoid(logit)
    relevance = np.clip(1.0 - flux, 0.0, None)
    relevance = relevance / (np.max(relevance) + 1e-12)

    warnings = ["Torch unavailable; using deterministic heuristic model."]
    if experimental_qml:
        warnings.append("QML mode requested but no QML head is configured in this environment.")

    return {
        "probability": probability,
        "label": "transit_candidate" if probability >= 0.5 else "non_transit",
        "model_name": "heuristic-depth-bls",
        "model_version": "v1",
        "relevance": relevance.tolist(),
        "warnings": warnings,
        "device": "cpu",
    }


def _torch_inference(
    flux: np.ndarray,
    settings: Settings,
    experimental_qml: bool,
) -> dict:
    device = _select_device(settings)
    model = TransitCNN().to(device)
    warnings: list[str] = []

    if settings.model_path:
        model_path = Path(settings.model_path)
        if model_path.exists():
            state = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            elif isinstance(state, dict):
                model.load_state_dict(state)
            else:
                warnings.append("Unsupported model file structure; using random bootstrap weights.")
            model_version = model_path.name
        else:
            warnings.append(f"Model file not found at {model_path}; using bootstrap weights.")
            model_version = "bootstrap-untrained-v1"
    else:
        warnings.append("No model path configured; using bootstrap weights.")
        model_version = "bootstrap-untrained-v1"

    model.eval()

    input_signal = np.expand_dims(np.expand_dims(1.0 - flux, axis=0), axis=0).astype(np.float32)
    x = torch.tensor(input_signal, device=device, requires_grad=True)

    activations = {}
    gradients = {}

    def save_activation(_, __, output):
        activations["value"] = output

    def save_gradient(_, grad_input, grad_output):  # noqa: ARG001
        gradients["value"] = grad_output[0]

    handle_forward = model.conv3.register_forward_hook(save_activation)
    handle_backward = model.conv3.register_full_backward_hook(save_gradient)

    logit = model(x)
    probability = float(torch.sigmoid(logit).squeeze().detach().cpu().item())

    model.zero_grad(set_to_none=True)
    logit.backward(torch.ones_like(logit))

    handle_forward.remove()
    handle_backward.remove()

    if "value" in activations and "value" in gradients:
        acts = activations["value"]
        grads = gradients["value"]
        weights = grads.mean(dim=2, keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = nnf.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False)
        relevance = cam.squeeze().detach().cpu().numpy()
        relevance = relevance / (np.max(relevance) + 1e-12)
    else:
        relevance = np.clip(1.0 - flux, 0.0, None)
        relevance = relevance / (np.max(relevance) + 1e-12)
        warnings.append("Grad-CAM fallback used because hooks did not capture tensors.")

    if experimental_qml:
        warnings.append("QML mode requested; feature flag is accepted but QML head is not wired yet.")

    return {
        "probability": probability,
        "label": "transit_candidate" if probability >= 0.5 else "non_transit",
        "model_name": "transit-cnn-1d",
        "model_version": model_version,
        "relevance": relevance.astype(float).tolist(),
        "warnings": warnings,
        "device": device,
    }


def run_inference(
    flux: np.ndarray,
    bls_peaks: list[dict[str, float]],
    settings: Settings,
    experimental_qml: bool,
) -> dict:
    if HAS_TORCH:
        try:
            return _torch_inference(flux=flux, settings=settings, experimental_qml=experimental_qml)
        except Exception as exc:
            heuristic = _heuristic_inference(flux, bls_peaks=bls_peaks, experimental_qml=experimental_qml)
            heuristic["warnings"].append(f"Torch inference failed ({exc.__class__.__name__}); fallback applied.")
            return heuristic

    return _heuristic_inference(flux, bls_peaks=bls_peaks, experimental_qml=experimental_qml)
