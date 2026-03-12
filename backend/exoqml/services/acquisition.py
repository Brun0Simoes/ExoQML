from __future__ import annotations

import json
from hashlib import sha1
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

try:
    from astroquery.simbad import Simbad

    HAS_SIMBAD = True
except Exception:
    HAS_SIMBAD = False
    Simbad = None


@dataclass(slots=True)
class AcquisitionOutput:
    time: np.ndarray
    flux: np.ndarray
    mission: str
    data_source: str
    sector_or_quarter: str | None
    ra: float | None
    dec: float | None
    warnings: list[str]


def _cache_base_dir(settings: Settings) -> Path:
    return Path(settings.cache_dir) / "lightcurves"


def _cache_paths(target: ResolvedTarget, settings: Settings) -> tuple[Path, Path]:
    raw = f"{target.target_type}|{target.target_id}|{target.query.strip().lower()}"
    digest = sha1(raw.encode("utf-8")).hexdigest()
    base = _cache_base_dir(settings) / digest[:2] / digest
    return base.with_suffix(".npz"), base.with_suffix(".json")


def _load_cached_output(target: ResolvedTarget, settings: Settings) -> AcquisitionOutput | None:
    npz_path, meta_path = _cache_paths(target, settings)
    if not npz_path.exists() or not meta_path.exists():
        return None

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    with np.load(npz_path, allow_pickle=False) as payload:
        time = payload["time"].astype(float)
        flux = payload["flux"].astype(float)

    warnings = list(meta.get("warnings", []))
    warnings.append("Loaded cached light curve from local cache.")
    return AcquisitionOutput(
        time=time,
        flux=flux,
        mission=str(meta.get("mission", "unknown")),
        data_source=str(meta.get("data_source", "cache")),
        sector_or_quarter=meta.get("sector_or_quarter"),
        ra=float(meta["ra"]) if meta.get("ra") is not None else None,
        dec=float(meta["dec"]) if meta.get("dec") is not None else None,
        warnings=warnings,
    )


def _save_cached_output(target: ResolvedTarget, settings: Settings, output: AcquisitionOutput) -> None:
    if output.mission.lower() == "synthetic":
        return

    npz_path, meta_path = _cache_paths(target, settings)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    temp_npz = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(temp_npz, time=output.time.astype(np.float32), flux=output.flux.astype(np.float32))
    temp_npz.replace(npz_path)

    meta = {
        "mission": output.mission,
        "data_source": output.data_source,
        "sector_or_quarter": output.sector_or_quarter,
        "ra": output.ra,
        "dec": output.dec,
        "warnings": output.warnings,
    }
    temp_meta = meta_path.with_suffix(".tmp.json")
    temp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_meta.replace(meta_path)


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


def _extract_coordinate(row: object, keys: tuple[str, ...]) -> float | None:
    if row is None:
        return None
    for key in keys:
        try:
            if hasattr(row, "colnames") and key in row.colnames:  # type: ignore[attr-defined]
                value = row[key]
            else:
                value = row.get(key)  # type: ignore[attr-defined]
        except Exception:
            continue
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _resolve_coordinates_from_search(target: ResolvedTarget) -> tuple[float | None, float | None]:
    if not HAS_LIGHTKURVE:
        return None, None
    try:
        search = lk.search_lightcurve(target.query)
        if len(search) > 0:
            row = search.table[0]
            ra = _extract_coordinate(row, ("s_ra", "ra"))
            dec = _extract_coordinate(row, ("s_dec", "dec"))
            if ra is not None and dec is not None:
                return ra, dec
    except Exception:
        pass

    if target.target_type == "tic":
        try:
            tesscut = lk.search_tesscut(target.query)
            if len(tesscut) > 0:
                row = tesscut.table[0]
                ra = _extract_coordinate(row, ("s_ra", "ra"))
                dec = _extract_coordinate(row, ("s_dec", "dec"))
                if ra is not None and dec is not None:
                    return ra, dec
        except Exception:
            pass
    return None, None


def _resolve_coordinates(target: ResolvedTarget) -> tuple[float | None, float | None]:
    ra, dec = _resolve_coordinates_from_search(target)
    if ra is not None and dec is not None:
        return ra, dec

    if HAS_SIMBAD:
        try:
            result = Simbad.query_object(target.query)
            if result is not None and len(result) > 0:
                return float(result["ra"][0]), float(result["dec"][0])
        except Exception:
            pass
    return None, None


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
            row = None
            try:
                row = tesscut.table[0]
            except Exception:
                row = None
            return AcquisitionOutput(
                time=time,
                flux=flux,
                mission="TESS",
                data_source="MAST/STScI TESScut via lightkurve",
                sector_or_quarter=None,
                ra=_extract_coordinate(row, ("s_ra", "ra")),
                dec=_extract_coordinate(row, ("s_dec", "dec")),
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
    ra = _extract_coordinate(row, ("s_ra", "ra"))
    dec = _extract_coordinate(row, ("s_dec", "dec"))

    return AcquisitionOutput(
        time=time,
        flux=flux,
        mission=mission,
        data_source="MAST/STScI via lightkurve",
        sector_or_quarter=sector_or_quarter,
        ra=ra,
        dec=dec,
        warnings=warnings,
    )


def fetch_lightcurve(target: ResolvedTarget, settings: Settings) -> AcquisitionOutput:
    cached = _load_cached_output(target, settings)
    if cached is not None:
        if cached.ra is None or cached.dec is None:
            ra, dec = _resolve_coordinates(target)
            if ra is not None and dec is not None:
                cached = AcquisitionOutput(
                    time=cached.time,
                    flux=cached.flux,
                    mission=cached.mission,
                    data_source=cached.data_source,
                    sector_or_quarter=cached.sector_or_quarter,
                    ra=ra,
                    dec=dec,
                    warnings=cached.warnings,
                )
                _save_cached_output(target, settings, cached)
        return cached

    warnings: list[str] = []

    if HAS_LIGHTKURVE:
        try:
            output = _from_lightkurve(target, settings)
            _save_cached_output(target, settings, output)
            return output
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
        ra=None,
        dec=None,
        warnings=warnings + ["This result is synthetic and for demo/dev only."],
    )
