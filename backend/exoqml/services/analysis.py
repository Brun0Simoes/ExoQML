from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session

from exoqml.config import Settings
from exoqml.models import AnalysisLog
from exoqml.schemas import AnalysisResponse, AnalyzeRequest, BLSPeak, Provenance, SeriesPoint
from exoqml.services.acquisition import fetch_lightcurve
from exoqml.services.bls import run_bls_baseline
from exoqml.services.identifier import resolve_target
from exoqml.services.inference import run_inference
from exoqml.services.preprocess import preprocess_lightcurve


def _points(time: np.ndarray, values: np.ndarray) -> list[SeriesPoint]:
    return [SeriesPoint(x=float(t), y=float(v)) for t, v in zip(time.tolist(), values.tolist(), strict=True)]


def run_analysis(db: Session, settings: Settings, request: AnalyzeRequest) -> AnalysisResponse:
    target = resolve_target(request.target_id, request.target_type)
    acquisition = fetch_lightcurve(target=target, settings=settings)
    proc_time, proc_flux, params = preprocess_lightcurve(
        time=acquisition.time,
        flux=acquisition.flux,
        max_points=settings.max_points,
    )

    bls_period, bls_peaks_raw = run_bls_baseline(proc_time, proc_flux)
    inference = run_inference(
        flux=proc_flux,
        bls_peaks=bls_peaks_raw,
        settings=settings,
        experimental_qml=request.experimental_qml,
    )

    warnings = acquisition.warnings + inference["warnings"]
    created_at = datetime.now(timezone.utc)
    provenance = Provenance(
        mission=acquisition.mission,
        data_source=acquisition.data_source,
        sector_or_quarter=acquisition.sector_or_quarter,
        analysis_timestamp=created_at,
    )

    response = AnalysisResponse(
        id=0,
        status="success",
        target_id=target.target_id,
        target_type=target.target_type,
        prediction_label=inference["label"],
        prediction_score=float(inference["probability"]),
        bls_period=float(bls_period) if bls_period is not None else None,
        model_name=inference["model_name"],
        model_version=inference["model_version"],
        warnings=warnings,
        preprocess_params=params,
        provenance=provenance,
        lightcurve_points=_points(proc_time, proc_flux),
        xai_points=_points(proc_time, np.asarray(inference["relevance"], dtype=float)),
        bls_peaks=[BLSPeak(**peak) for peak in bls_peaks_raw],
    )

    payload = response.model_dump(mode="json")
    payload["inference_device"] = inference.get("device", "cpu")

    row = AnalysisLog(
        target_id=response.target_id,
        target_type=response.target_type,
        mission=response.provenance.mission,
        data_source=response.provenance.data_source,
        model_name=response.model_name,
        model_version=response.model_version,
        prediction_label=response.prediction_label,
        prediction_score=response.prediction_score,
        bls_period=response.bls_period,
        status=response.status,
        payload_json=json.dumps(payload, ensure_ascii=False),
        created_at=created_at,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    return response.model_copy(update={"id": row.id})
