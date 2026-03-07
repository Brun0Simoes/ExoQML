from __future__ import annotations

import numpy as np


def _remove_outliers(flux: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    median = np.median(flux)
    mad = np.median(np.abs(flux - median)) + 1e-12
    z_score = 0.6745 * (flux - median) / mad
    return np.abs(z_score) < sigma


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def preprocess_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    max_points: int = 2048,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    if time.size != flux.size:
        raise ValueError("time and flux must have the same length")
    if time.size < 32:
        raise ValueError("insufficient points for preprocessing")

    mask = np.isfinite(time) & np.isfinite(flux)
    time = time[mask]
    flux = flux[mask]
    n_after_nans = int(time.size)

    if time.size < 32:
        raise ValueError("insufficient finite points after NaN removal")

    outlier_mask = _remove_outliers(flux, sigma=6.0)
    time = time[outlier_mask]
    flux = flux[outlier_mask]
    n_after_outliers = int(time.size)

    if time.size < 32:
        raise ValueError("insufficient points after outlier removal")

    sort_idx = np.argsort(time)
    time = time[sort_idx]
    flux = flux[sort_idx]

    median_flux = np.median(flux) + 1e-12
    flux_norm = flux / median_flux

    window = max(11, min(401, (len(flux_norm) // 40) * 2 + 1))
    trend = _moving_average(flux_norm, window=window)
    trend = np.clip(trend, 1e-6, None)
    flattened = flux_norm / trend
    flattened = np.clip(flattened, 0.9, 1.1)

    if len(flattened) > max_points:
        new_time = np.linspace(time.min(), time.max(), max_points)
        new_flux = np.interp(new_time, time, flattened)
    else:
        new_time = time
        new_flux = flattened

    params: dict[str, float | int | str] = {
        "n_input": int(mask.size),
        "n_after_nans": n_after_nans,
        "n_after_outliers": n_after_outliers,
        "window": int(window),
        "max_points": int(max_points),
        "normalization": "median",
        "detrending": "moving_average",
    }
    return new_time.astype(float), new_flux.astype(float), params
