import numpy as np

from exoqml.services.bls import run_bls_baseline


def test_bls_returns_peak_for_periodic_dip() -> None:
    n_points = 3000
    time = np.linspace(0.0, 30.0, n_points)
    flux = np.ones_like(time)

    period_true = 3.0
    phase = np.mod(time, period_true) / period_true
    dips = np.exp(-0.5 * (np.minimum(phase, 1 - phase) / 0.03) ** 2)
    flux -= 0.01 * dips

    best_period, peaks = run_bls_baseline(time, flux)
    assert best_period is not None
    assert len(peaks) > 0
    assert 1.0 <= best_period <= 5.0
