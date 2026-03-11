from __future__ import annotations

import math

import numpy as np

GLOBAL_VIEW_BINS = 401
LOCAL_VIEW_BINS = 121
SCALAR_FEATURE_DIM = 4


def phase_fold(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    phase = np.mod(time - epoch + (0.5 * period), period) / max(period, 1e-9)
    return phase - 0.5


def estimate_epoch(time: np.ndarray, flux: np.ndarray, period: float) -> float:
    if time.size == 0:
        return 0.0
    phase = phase_fold(time, period=period, epoch=float(time.min()))
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
    if np.isfinite(binned).any():
        min_phase = float(centers[int(np.nanargmin(binned))])
        return float(time.min() + ((min_phase + 0.5) * period))
    return float(time[int(np.argmin(flux))])


def _median_bin_view(
    phase: np.ndarray,
    flux: np.ndarray,
    left: float,
    right: float,
    bins: int,
    fill_value: float = 1.0,
) -> np.ndarray:
    edges = np.linspace(left, right, bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(phase, edges) - 1
    idx = np.clip(idx, 0, bins - 1)

    binned = np.full(bins, np.nan, dtype=np.float64)
    for i in range(bins):
        values = flux[idx == i]
        if values.size > 0:
            binned[i] = float(np.median(values))

    valid = np.isfinite(binned)
    if not np.any(valid):
        return np.full(bins, fill_value, dtype=np.float32)
    if np.sum(valid) == 1:
        return np.full(bins, float(binned[valid][0]), dtype=np.float32)

    filled = np.interp(centers, centers[valid], binned[valid]).astype(np.float32)
    return filled


def build_tce_views(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration_hours: float,
    global_bins: int = GLOBAL_VIEW_BINS,
    local_bins: int = LOCAL_VIEW_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    phase = phase_fold(time=time, period=period, epoch=epoch)
    duration_phase = max((duration_hours / 24.0) / max(period, 1e-9), 1e-4)
    local_half_width = min(0.5, max(duration_phase * 4.0, 0.03))

    global_flux = _median_bin_view(phase=phase, flux=flux, left=-0.5, right=0.5, bins=global_bins)
    local_flux = _median_bin_view(
        phase=phase,
        flux=flux,
        left=-local_half_width,
        right=local_half_width,
        bins=local_bins,
    )

    global_view = np.clip(1.0 - global_flux, -0.2, 0.2).astype(np.float32)
    local_view = np.clip(1.0 - local_flux, -0.2, 0.2).astype(np.float32)
    return global_view, local_view


def build_scalar_features(period: float, duration_hours: float, depth_ppm: float, model_snr: float) -> np.ndarray:
    period_feature = math.log1p(max(period, 0.0)) / math.log1p(400.0)
    duration_feature = math.log1p(max(duration_hours, 0.0)) / math.log1p(24.0)
    depth_feature = math.log1p(max(depth_ppm, 0.0)) / math.log1p(1e5)
    snr_feature = math.log1p(max(model_snr, 0.0)) / math.log1p(100.0)
    return np.array(
        [period_feature, duration_feature, depth_feature, snr_feature],
        dtype=np.float32,
    )
