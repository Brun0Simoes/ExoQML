from __future__ import annotations

import numpy as np


def run_bls_baseline(
    time: np.ndarray,
    flux: np.ndarray,
    min_period: float = 0.5,
    max_period: float = 20.0,
    n_periods: int = 180,
    n_bins: int = 64,
) -> tuple[float | None, list[dict[str, float]]]:
    if time.size < 128 or flux.size < 128:
        return None, []

    periods = np.linspace(min_period, max_period, n_periods, dtype=float)
    scores: list[dict[str, float]] = []

    for period in periods:
        phase = np.mod(time, period) / period
        order = np.argsort(phase)
        phase_sorted = phase[order]
        flux_sorted = flux[order]

        bins = np.linspace(0.0, 1.0, n_bins + 1, dtype=float)
        idx = np.digitize(phase_sorted, bins) - 1
        idx = np.clip(idx, 0, n_bins - 1)

        binned = np.full(n_bins, np.nan, dtype=float)
        for i in range(n_bins):
            values = flux_sorted[idx == i]
            if values.size > 0:
                binned[i] = float(np.median(values))

        valid = binned[np.isfinite(binned)]
        if valid.size < max(10, n_bins // 2):
            continue

        median_flux = float(np.median(valid))
        dip_flux = float(np.min(valid))
        depth = max(0.0, median_flux - dip_flux)
        noise = float(np.std(valid) + 1e-9)
        power = depth / noise
        n_transits = max(1.0, (float(time.max()) - float(time.min())) / float(period))
        score = power * np.sqrt(n_transits)

        scores.append(
            {
                "period": float(period),
                "power": float(power),
                "depth": float(depth),
                "score": float(score),
            }
        )

    if not scores:
        return None, []

    top_raw = sorted(scores, key=lambda x: x["score"], reverse=True)[:5]
    top = [{"period": item["period"], "power": item["power"], "depth": item["depth"]} for item in top_raw]
    return top[0]["period"], top
