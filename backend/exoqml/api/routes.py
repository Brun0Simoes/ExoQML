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
from exoqml.models import AnalysisLog
from exoqml.schemas import AnalysisHistoryItem, AnalysisResponse, AnalyzeRequest
from exoqml.services.analysis import run_analysis

router = APIRouter()


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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"analysis failed: {exc}") from exc


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
