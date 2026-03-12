from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session

from exoqml.config import Settings
from exoqml.errors import AnalysisError
from exoqml.logging_utils import get_logger
from exoqml.models import AnalysisLog
from exoqml.schemas import (
    AnalysisResponse,
    AnalyzeRequest,
    BLSPeak,
    ExperimentalComparison,
    ExperimentalInferenceSummary,
    Provenance,
    SeriesPoint,
    SkyCoordinates,
)
from exoqml.services.acquisition import fetch_lightcurve
from exoqml.services.bls import run_bls_baseline
from exoqml.services.identifier import resolve_target
from exoqml.services.inference import run_inference_comparison
from exoqml.services.preprocess import preprocess_lightcurve

logger = get_logger(__name__)


def _points(time: np.ndarray, values: np.ndarray) -> list[SeriesPoint]:
    return [SeriesPoint(x=float(t), y=float(v)) for t, v in zip(time.tolist(), values.tolist(), strict=True)]


def run_analysis(db: Session, settings: Settings, request: AnalyzeRequest) -> AnalysisResponse:
    logger.info(
        "analysis started",
        extra={
            "event": "analysis_started",
            "target_id": request.target_id,
            "target_type": request.target_type or "auto",
            "experimental_qml": request.experimental_qml,
        },
    )

    try:
        target = resolve_target(request.target_id, request.target_type)
    except ValueError as exc:
        raise AnalysisError(
            code="invalid_target",
            message="O identificador informado não pôde ser interpretado.",
            suggestion="Use um TIC, KIC ou nome suportado. Para TIC ou KIC, envie os dígitos do alvo.",
            status_code=400,
            stage="target_validation",
            cause=str(exc),
        ) from exc

    try:
        acquisition = fetch_lightcurve(target=target, settings=settings)
    except Exception as exc:
        raise AnalysisError(
            code="acquisition_failed",
            message="Não foi possível obter uma curva de luz utilizável para esse alvo.",
            suggestion="Tente outro identificador, aguarde a fonte externa ou use um alvo conhecido suportado.",
            status_code=502,
            stage="acquisition",
            cause=str(exc),
        ) from exc

    try:
        proc_time, proc_flux, params = preprocess_lightcurve(
            time=acquisition.time,
            flux=acquisition.flux,
            max_points=settings.max_points,
        )
    except ValueError as exc:
        raise AnalysisError(
            code="preprocess_failed",
            message="A curva foi encontrada, mas não ficou utilizável após o preprocessamento.",
            suggestion="Teste outro alvo ou tente novamente se a fonte externa tiver retornado dados incompletos.",
            status_code=422,
            stage="preprocess",
            cause=str(exc),
        ) from exc

    logger.info(
        "preprocess completed",
        extra={
            "event": "preprocess_completed",
            "target_id": target.target_id,
            "target_type": target.target_type,
            "mission": acquisition.mission,
            "points_input": int(acquisition.time.size),
            "points_output": int(proc_time.size),
        },
    )

    bls_period, bls_peaks_raw = run_bls_baseline(proc_time, proc_flux)
    try:
        inference_bundle = run_inference_comparison(
            time=proc_time,
            flux=proc_flux,
            bls_peaks=bls_peaks_raw,
            settings=settings,
            experimental_qml=request.experimental_qml,
        )
    except Exception as exc:
        raise AnalysisError(
            code="inference_failed",
            message="O pipeline conseguiu preparar a curva, mas falhou na inferência.",
            suggestion="Tente novamente. Se o erro persistir, revise o checkpoint configurado do modelo.",
            status_code=502,
            stage="inference",
            cause=str(exc),
        ) from exc

    inference = inference_bundle["primary"]
    comparison_payload = inference_bundle["comparison"]
    warnings = acquisition.warnings + inference["warnings"]
    created_at = datetime.now(timezone.utc)
    provenance = Provenance(
        mission=acquisition.mission,
        data_source=acquisition.data_source,
        sector_or_quarter=acquisition.sector_or_quarter,
        analysis_timestamp=created_at,
        sky_coordinates=SkyCoordinates(ra=acquisition.ra, dec=acquisition.dec)
        if acquisition.ra is not None and acquisition.dec is not None
        else None,
    )
    experimental_comparison = ExperimentalComparison(
        requested=bool(comparison_payload["requested"]),
        available=bool(comparison_payload["available"]),
        activated=bool(comparison_payload.get("activated", False)),
        activation_reason=comparison_payload.get("activation_reason"),
        selected_mode=comparison_payload.get("selected_mode"),
        ambiguity_lower=comparison_payload.get("ambiguity_lower"),
        ambiguity_upper=comparison_payload.get("ambiguity_upper"),
        score_delta=comparison_payload.get("score_delta"),
        absolute_score_delta=comparison_payload.get("absolute_score_delta"),
        classical=ExperimentalInferenceSummary(**comparison_payload["classical"])
        if comparison_payload.get("classical")
        else None,
        qml=ExperimentalInferenceSummary(**comparison_payload["qml"]) if comparison_payload.get("qml") else None,
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
        experimental_comparison=experimental_comparison,
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
    try:
        db.add(row)
        db.commit()
        db.refresh(row)
    except Exception as exc:
        db.rollback()
        raise AnalysisError(
            code="history_persist_failed",
            message="A análise foi concluída, mas o histórico não pôde ser gravado.",
            suggestion="Repita a análise depois de verificar a configuração do banco local.",
            status_code=500,
            stage="persistence",
            cause=str(exc),
        ) from exc

    logger.info(
        "analysis completed",
        extra={
            "event": "analysis_completed",
            "analysis_id": row.id,
            "target_id": response.target_id,
            "target_type": response.target_type,
            "mission": response.provenance.mission,
            "prediction_label": response.prediction_label,
            "prediction_score": response.prediction_score,
            "bls_period": response.bls_period,
            "model_name": response.model_name,
            "model_version": response.model_version,
        },
    )

    return response.model_copy(update={"id": row.id})
