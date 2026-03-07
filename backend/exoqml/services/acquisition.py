from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from exoqml.config import Settings
from exoqml.services.identifier import ResolvedTarget
from exoqml.services.synthetic import synthetic_lightcurve

try:
    import lightkurve as lk

    HAS_LIGHTKURVE = True
except Exception:
    HAS_LIGHTKURVE = False
    lk = None


@dataclass(slots=True)
class AcquisitionOutput:
    time: np.ndarray
    flux: np.ndarray
    mission: str
    data_source: str
    sector_or_quarter: str | None
    warnings: list[str]


def _extract_sector(row: object) -> str | None:
    if row is None:
        return None
    for key in ("sequence_number", "quarter", "campaign"):
        try:
            value = row.get(key)  # type: ignore[attr-defined]
            if value is not None:
                return str(value)
        except Exception:
            continue
    return None


def _from_lightkurve(target: ResolvedTarget, settings: Settings) -> AcquisitionOutput:
    warnings: list[str] = []
    cache_dir = Path(settings.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    search = lk.search_lightcurve(target.query)
    if len(search) == 0 and target.target_type == "tic":
        tesscut = lk.search_tesscut(target.query)
        if len(tesscut) > 0:
            tpf = tesscut.download(cutout_size=11, download_dir=str(cache_dir))
            lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
            time = np.asarray(lc.time.value, dtype=float)
            flux = np.asarray(lc.flux.value, dtype=float)
            return AcquisitionOutput(
                time=time,
                flux=flux,
                mission="TESS",
                data_source="MAST/STScI TESScut via lightkurve",
                sector_or_quarter=None,
                warnings=warnings,
            )

    if len(search) == 0:
        raise RuntimeError("No compatible light curve products found")

    collection = search.download_all(download_dir=str(cache_dir))
    if collection is None or len(collection) == 0:
        raise RuntimeError("Download returned no light curves")

    stitched = collection.stitch()
    time = np.asarray(stitched.time.value, dtype=float)
    flux = np.asarray(stitched.flux.value, dtype=float)

    row = None
    try:
        row = search.table[0]
    except Exception:
        row = None

    mission = "unknown"
    if row is not None:
        try:
            if "mission" in row.colnames:  # type: ignore[attr-defined]
                mission = str(row["mission"])
        except Exception:
            try:
                mission = str(row.get("mission", "unknown"))  # type: ignore[attr-defined]
            except Exception:
                mission = "unknown"
    sector_or_quarter = _extract_sector(row)

    return AcquisitionOutput(
        time=time,
        flux=flux,
        mission=mission,
        data_source="MAST/STScI via lightkurve",
        sector_or_quarter=sector_or_quarter,
        warnings=warnings,
    )


def fetch_lightcurve(target: ResolvedTarget, settings: Settings) -> AcquisitionOutput:
    warnings: list[str] = []

    if HAS_LIGHTKURVE:
        try:
            return _from_lightkurve(target, settings)
        except Exception as exc:
            warnings.append(f"Live acquisition failed: {exc.__class__.__name__}: {exc}")
            if not settings.allow_synthetic_fallback:
                raise
    else:
        warnings.append("lightkurve is not installed; synthetic fallback used.")
        if not settings.allow_synthetic_fallback:
            raise RuntimeError("lightkurve unavailable and synthetic fallback disabled")

    time, flux = synthetic_lightcurve(seed_key=target.query)
    return AcquisitionOutput(
        time=time,
        flux=flux,
        mission="synthetic",
        data_source="local synthetic generator",
        sector_or_quarter=None,
        warnings=warnings + ["This result is synthetic and for demo/dev only."],
    )
