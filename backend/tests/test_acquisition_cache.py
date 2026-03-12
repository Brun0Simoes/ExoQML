from __future__ import annotations

import numpy as np

import exoqml.services.acquisition as acquisition
from exoqml.config import Settings
from exoqml.services.acquisition import AcquisitionOutput, _save_cached_output, fetch_lightcurve
from exoqml.services.identifier import ResolvedTarget


def test_fetch_lightcurve_uses_local_cache(monkeypatch, tmp_path) -> None:
    settings = Settings(cache_dir=str(tmp_path), allow_synthetic_fallback=False)
    target = ResolvedTarget(target_id="10000490", target_type="kic", query="KIC 10000490")

    live_calls = {"count": 0}

    def fake_live(*args, **kwargs):
        live_calls["count"] += 1
        return AcquisitionOutput(
            time=np.array([1.0, 2.0, 3.0], dtype=float),
            flux=np.array([1.0, 0.99, 1.01], dtype=float),
            mission="Kepler Quarter 02",
            data_source="MAST/STScI via lightkurve",
            sector_or_quarter="2",
            ra=286.5560,
            dec=46.9573,
            warnings=[],
        )

    monkeypatch.setattr(acquisition, "HAS_LIGHTKURVE", True)
    monkeypatch.setattr(acquisition, "_from_lightkurve", fake_live)

    first = fetch_lightcurve(target=target, settings=settings)
    second = fetch_lightcurve(target=target, settings=settings)

    assert live_calls["count"] == 1
    assert first.mission == "Kepler Quarter 02"
    assert first.ra == 286.5560
    assert first.dec == 46.9573
    assert second.mission == "Kepler Quarter 02"
    assert second.ra == 286.5560
    assert second.dec == 46.9573
    assert "Loaded cached light curve from local cache." in second.warnings


def test_fetch_lightcurve_refreshes_coordinates_for_old_cache(monkeypatch, tmp_path) -> None:
    settings = Settings(cache_dir=str(tmp_path), allow_synthetic_fallback=False)
    target = ResolvedTarget(target_id="10000490", target_type="kic", query="KIC 10000490")

    _save_cached_output(
        target=target,
        settings=settings,
        output=AcquisitionOutput(
            time=np.array([1.0, 2.0, 3.0], dtype=float),
            flux=np.array([1.0, 0.99, 1.01], dtype=float),
            mission="Kepler Quarter 02",
            data_source="MAST/STScI via lightkurve",
            sector_or_quarter="2",
            ra=None,
            dec=None,
            warnings=[],
        ),
    )

    monkeypatch.setattr(acquisition, "_resolve_coordinates", lambda target: (286.5560, 46.9573))

    cached = fetch_lightcurve(target=target, settings=settings)

    assert cached.ra == 286.5560
    assert cached.dec == 46.9573
