from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from exoqml.config import Settings
from exoqml.training.model import TransitMultiViewNet
from exoqml.transit_features import (
    GLOBAL_VIEW_BINS,
    LOCAL_VIEW_BINS,
    SCALAR_FEATURE_DIM,
    build_scalar_features,
    build_tce_views,
    estimate_epoch,
    phase_fold,
)

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


def _normalize_relevance(flux: np.ndarray) -> np.ndarray:
    relevance = np.clip(1.0 - flux, 0.0, None).astype(np.float64)
    scale = float(np.max(relevance)) if relevance.size > 0 else 0.0
    if scale > 0.0:
        relevance /= scale
    return relevance.astype(float)


def _heuristic_inference(
    flux: np.ndarray,
    bls_peaks: list[dict[str, float]],
    experimental_qml: bool,
) -> dict[str, Any]:
    depth = float(max(0.0, 1.0 - np.min(flux)))
    variability = float(np.std(flux))
    bls_power = float(bls_peaks[0]["power"]) if bls_peaks else 0.0
    logit = (8.0 * depth) + (0.25 * bls_power) - (15.0 * variability) - 1.2
    probability = _safe_sigmoid(logit)
    warnings = ["Torch unavailable; using deterministic heuristic model."]
    if experimental_qml:
        warnings.append("QML mode requested but no QML head is configured in this environment.")

    return {
        "probability": probability,
        "label": "transit_candidate" if probability >= 0.5 else "non_transit",
        "model_name": "heuristic-depth-bls",
        "model_version": "v1",
        "relevance": _normalize_relevance(flux).tolist(),
        "warnings": warnings,
        "device": "cpu",
    }


def _sanitize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            clean[key[len("_orig_mod.") :]] = value
        else:
            clean[key] = value
    return clean


def _estimate_duration_hours(time: np.ndarray, flux: np.ndarray, period: float, epoch: float) -> float:
    if time.size < 16 or period <= 0.0:
        return max(1.5, period * 24.0 * 0.03)

    phase = phase_fold(time=time.astype(np.float64), period=period, epoch=epoch)
    order = np.argsort(phase)
    phase_sorted = phase[order]
    flux_sorted = flux[order]
    bins = np.linspace(-0.5, 0.5, 257, dtype=np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    idx = np.digitize(phase_sorted, bins) - 1
    idx = np.clip(idx, 0, centers.size - 1)
    binned = np.full(centers.shape[0], np.nan, dtype=np.float64)

    for i in range(centers.size):
        values = flux_sorted[idx == i]
        if values.size > 0:
            binned[i] = float(np.median(values))

    valid = np.isfinite(binned)
    if not np.any(valid):
        return max(1.5, period * 24.0 * 0.03)

    filled = np.interp(centers, centers[valid], binned[valid])
    baseline = float(np.median(filled))
    depth = float(max(0.0, baseline - float(np.min(filled))))
    if depth <= 1e-6:
        return max(1.5, period * 24.0 * 0.03)

    threshold = baseline - (0.5 * depth)
    in_transit = filled <= threshold
    if not np.any(in_transit):
        return max(1.5, period * 24.0 * 0.03)

    center = int(np.argmin(filled))
    left = center
    right = center
    while left > 0 and in_transit[left - 1]:
        left -= 1
    while right + 1 < in_transit.size and in_transit[right + 1]:
        right += 1

    duration_phase = max(float(centers[right] - centers[left]), 1.0 / centers.size)
    duration_hours = duration_phase * period * 24.0
    return float(np.clip(duration_hours, 0.5, period * 24.0 * 0.25))


def _load_model_state(model_path: Path, device: str) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    if not model_path.exists():
        warnings.append(f"Model file not found at {model_path}; using bootstrap weights.")
        return None, warnings

    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        return state, warnings

    warnings.append("Unsupported model file structure; using bootstrap weights.")
    return None, warnings


def _legacy_torch_inference(
    flux: np.ndarray,
    device: str,
    checkpoint: dict[str, Any] | None,
    model_version: str,
    experimental_qml: bool,
    warnings: list[str],
) -> dict[str, Any]:
    model = TransitCNN().to(device)
    threshold = 0.5

    if checkpoint is not None:
        state_dict = checkpoint.get("state_dict", checkpoint)
        if isinstance(state_dict, dict):
            model.load_state_dict(_sanitize_state_dict(state_dict))
        threshold = float(checkpoint.get("threshold", 0.5))

    model.eval()
    input_signal = np.expand_dims(np.expand_dims(1.0 - flux, axis=0), axis=0).astype(np.float32)
    x = torch.tensor(input_signal, device=device, requires_grad=True)

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def save_activation(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations["value"] = output

    def save_gradient(_: nn.Module, __: tuple[torch.Tensor | None, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
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
        scale = float(np.max(relevance)) if relevance.size > 0 else 0.0
        if scale > 0.0:
            relevance = relevance / scale
    else:
        relevance = _normalize_relevance(flux)
        warnings.append("Grad-CAM fallback used because hooks did not capture tensors.")

    if experimental_qml:
        warnings.append("QML mode requested; feature flag is accepted but QML head is not wired yet.")

    return {
        "probability": probability,
        "label": "transit_candidate" if probability >= threshold else "non_transit",
        "model_name": "transit-cnn-1d",
        "model_version": model_version,
        "relevance": np.asarray(relevance, dtype=float).tolist(),
        "warnings": warnings,
        "device": device,
    }


def _multiview_torch_inference(
    time: np.ndarray,
    flux: np.ndarray,
    bls_peaks: list[dict[str, float]],
    device: str,
    checkpoint: dict[str, Any],
    model_version: str,
    experimental_qml: bool,
    warnings: list[str],
) -> dict[str, Any]:
    global_bins = int(checkpoint.get("global_bins", GLOBAL_VIEW_BINS))
    local_bins = int(checkpoint.get("local_bins", LOCAL_VIEW_BINS))
    scalar_dim = int(checkpoint.get("scalar_dim", SCALAR_FEATURE_DIM))
    threshold = float(checkpoint.get("threshold", 0.5))

    model = TransitMultiViewNet(scalar_dim=scalar_dim, base_channels=32, dropout=0.2).to(device)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint missing state_dict for multiview inference")
    model.load_state_dict(_sanitize_state_dict(state_dict))
    model.eval()

    if bls_peaks:
        period = float(bls_peaks[0]["period"])
        depth_ppm = max(0.0, float(bls_peaks[0].get("depth", 0.0))) * 1_000_000.0
        model_snr = max(0.0, float(bls_peaks[0].get("power", 0.0)))
    else:
        time_span = max(float(time.max()) - float(time.min()), 1.0)
        period = max(0.5, min(20.0, time_span / 10.0))
        depth_ppm = max(0.0, 1.0 - float(np.min(flux))) * 1_000_000.0
        model_snr = 0.0
        warnings.append("BLS peaks unavailable; using fallback TCE estimates for inference.")

    epoch = estimate_epoch(time=time.astype(np.float64), flux=flux.astype(np.float64), period=period)
    duration_hours = _estimate_duration_hours(
        time=time.astype(np.float64),
        flux=flux.astype(np.float64),
        period=period,
        epoch=epoch,
    )

    global_view, local_view = build_tce_views(
        time=time.astype(np.float64),
        flux=flux.astype(np.float64),
        period=period,
        epoch=epoch,
        duration_hours=duration_hours,
        global_bins=global_bins,
        local_bins=local_bins,
    )
    scalar_features = build_scalar_features(
        period=period,
        duration_hours=duration_hours,
        depth_ppm=depth_ppm,
        model_snr=model_snr,
    )

    global_tensor = torch.tensor(global_view[None, None, :], dtype=torch.float32, device=device)
    local_tensor = torch.tensor(local_view[None, None, :], dtype=torch.float32, device=device)
    scalar_tensor = torch.tensor(scalar_features[None, :], dtype=torch.float32, device=device)

    with torch.no_grad():
        logit = model(global_tensor, local_tensor, scalar_tensor)
        probability = float(torch.sigmoid(logit).squeeze().detach().cpu().item())

    if experimental_qml:
        warnings.append("QML mode requested; feature flag is accepted but QML head is not wired yet.")

    warnings.append("Inference uses online-estimated TCE parameters from BLS and folded flux.")
    return {
        "probability": probability,
        "label": "transit_candidate" if probability >= threshold else "non_transit",
        "model_name": "transit-multiview-tce",
        "model_version": model_version,
        "relevance": _normalize_relevance(flux).tolist(),
        "warnings": warnings,
        "device": device,
    }


def _torch_inference(
    time: np.ndarray,
    flux: np.ndarray,
    bls_peaks: list[dict[str, float]],
    settings: Settings,
    experimental_qml: bool,
) -> dict[str, Any]:
    device = _select_device(settings)
    warnings: list[str] = []

    checkpoint: dict[str, Any] | None = None
    model_version = "bootstrap-untrained-v1"
    architecture = "transit-cnn-1d"

    if settings.model_path:
        model_path = Path(settings.model_path)
        checkpoint, load_warnings = _load_model_state(model_path, device=device)
        warnings.extend(load_warnings)
        if checkpoint is not None:
            model_version = model_path.name
            architecture = str(checkpoint.get("architecture", "transit-cnn-1d"))
    else:
        warnings.append("No model path configured; using bootstrap weights.")

    if checkpoint is not None and architecture == "transit-multiview-tce":
        return _multiview_torch_inference(
            time=time,
            flux=flux,
            bls_peaks=bls_peaks,
            device=device,
            checkpoint=checkpoint,
            model_version=model_version,
            experimental_qml=experimental_qml,
            warnings=warnings,
        )

    return _legacy_torch_inference(
        flux=flux,
        device=device,
        checkpoint=checkpoint,
        model_version=model_version,
        experimental_qml=experimental_qml,
        warnings=warnings,
    )


def run_inference(
    time: np.ndarray,
    flux: np.ndarray,
    bls_peaks: list[dict[str, float]],
    settings: Settings,
    experimental_qml: bool,
) -> dict[str, Any]:
    if HAS_TORCH:
        try:
            return _torch_inference(
                time=time,
                flux=flux,
                bls_peaks=bls_peaks,
                settings=settings,
                experimental_qml=experimental_qml,
            )
        except Exception as exc:
            heuristic = _heuristic_inference(flux, bls_peaks=bls_peaks, experimental_qml=experimental_qml)
            heuristic["warnings"].append(f"Torch inference failed ({exc.__class__.__name__}); fallback applied.")
            return heuristic

    return _heuristic_inference(flux, bls_peaks=bls_peaks, experimental_qml=experimental_qml)
