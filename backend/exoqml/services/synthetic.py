from __future__ import annotations

import hashlib

import numpy as np


def synthetic_lightcurve(seed_key: str, n_points: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)

    time = np.linspace(0.0, 27.0, n_points, dtype=float)
    flux = np.ones_like(time)
    flux += rng.normal(0.0, 0.0008, size=n_points)

    period = 2.0 + rng.random() * 8.0
    depth = 0.003 + rng.random() * 0.01
    width = 0.05 + rng.random() * 0.03

    phase = np.mod(time, period) / period
    phase_dist = np.minimum(phase, 1.0 - phase)
    dips = np.exp(-0.5 * (phase_dist / width) ** 2)
    flux -= depth * dips

    return time, flux
