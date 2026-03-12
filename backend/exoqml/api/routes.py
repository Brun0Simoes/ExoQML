from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from exoqml.config import Settings, get_settings
from exoqml.db import get_db
from exoqml.errors import AnalysisError
from exoqml.logging_utils import get_logger
from exoqml.models import AnalysisLog
from exoqml.schemas import AnalysisHistoryItem, AnalysisResponse, AnalyzeRequest, TargetCatalogItem
from exoqml.services.analysis import run_analysis
from exoqml.services.target_catalog import load_target_catalog

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.app_env,
    }


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(
    request: AnalyzeRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AnalysisResponse:
    try:
        return run_analysis(db=db, settings=settings, request=request)
    except AnalysisError as exc:
        logger.warning(
            "analysis request failed",
            extra=exc.log_context(target_id=request.target_id, target_type=request.target_type or "auto"),
        )
        raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc
    except ValueError as exc:
        logger.warning(
            "analysis request rejected",
            extra={
                "event": "analysis_failed",
                "error_code": "invalid_request",
                "error_stage": "request",
                "error_message": str(exc),
                "target_id": request.target_id,
                "target_type": request.target_type or "auto",
            },
        )
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_request",
                "message": str(exc),
                "stage": "request",
                "suggestion": "Revise os campos enviados e tente novamente.",
            },
        ) from exc
    except Exception as exc:
        logger.exception(
            "analysis request crashed",
            extra={
                "event": "analysis_failed",
                "error_code": "analysis_failed",
                "error_stage": "analysis",
                "target_id": request.target_id,
                "target_type": request.target_type or "auto",
            },
        )
        raise HTTPException(
            status_code=502,
            detail={
                "code": "analysis_failed",
                "message": "A análise falhou antes de produzir um resultado utilizável.",
                "stage": "analysis",
                "suggestion": "Tente novamente em alguns minutos. Se o erro persistir, teste outro alvo.",
            },
        ) from exc


@router.get("/targets/catalog", response_model=list[TargetCatalogItem])
def target_catalog(
    search: str = Query(default="", max_length=128),
    limit: int = Query(default=10000, ge=1, le=20000),
) -> list[TargetCatalogItem]:
    query = search.strip().lower()
    items = load_target_catalog()
    if query:
        filtered = [
            item
            for item in items
            if query in str(item["query"]).lower()
            or query in str(item["display_name"]).lower()
            or query in str(item["summary"]).lower()
            or query in str(item["mission"]).lower()
        ]
    else:
        filtered = items
    return [TargetCatalogItem.model_validate(item) for item in filtered[:limit]]


@router.get("/history", response_model=list[AnalysisHistoryItem])
def history(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[AnalysisHistoryItem]:
    rows = db.execute(select(AnalysisLog).order_by(desc(AnalysisLog.created_at)).limit(limit)).scalars().all()
    return [
        AnalysisHistoryItem(
            id=row.id,
            target_id=row.target_id,
            target_type=row.target_type,  # type: ignore[arg-type]
            mission=row.mission,
            prediction_label=row.prediction_label,
            prediction_score=row.prediction_score,
            bls_period=row.bls_period,
            status=row.status,
            created_at=row.created_at,
        )
        for row in rows
    ]


@router.get("/history/{analysis_id}", response_model=AnalysisResponse)
def history_item(analysis_id: int, db: Session = Depends(get_db)) -> AnalysisResponse:
    row = db.get(AnalysisLog, analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail="analysis not found")
    payload = json.loads(row.payload_json)
    return AnalysisResponse.model_validate(payload)


@router.get("/history/{analysis_id}/export")
def export_analysis(
    analysis_id: int,
    format: str = Query(default="json", pattern="^(json|csv)$"),  # noqa: A002
    db: Session = Depends(get_db),
) -> Response:
    row = db.get(AnalysisLog, analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail="analysis not found")

    payload = json.loads(row.payload_json)
    if format == "json":
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        return Response(content=content, media_type="application/json")

    flat = {
        "id": row.id,
        "target_id": row.target_id,
        "target_type": row.target_type,
        "mission": row.mission,
        "prediction_label": row.prediction_label,
        "prediction_score": row.prediction_score,
        "bls_period": row.bls_period,
        "model_name": row.model_name,
        "model_version": row.model_version,
        "status": row.status,
        "created_at": row.created_at.isoformat(),
        "warnings": " | ".join(payload.get("warnings", [])),
    }
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(flat.keys()))
    writer.writeheader()
    writer.writerow(flat)
    return Response(content=out.getvalue(), media_type="text/csv")
