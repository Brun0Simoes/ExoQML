from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

TargetType = Literal["tic", "kic", "name"]


class AnalyzeRequest(BaseModel):
    target_id: str = Field(min_length=1, max_length=128)
    target_type: TargetType | None = None
    experimental_qml: bool = False


class SeriesPoint(BaseModel):
    x: float
    y: float


class BLSPeak(BaseModel):
    period: float
    power: float
    depth: float


class Provenance(BaseModel):
    mission: str
    data_source: str
    sector_or_quarter: str | None = None
    analysis_timestamp: datetime


class AnalysisResponse(BaseModel):
    id: int
    status: str
    target_id: str
    target_type: TargetType
    prediction_label: str
    prediction_score: float
    bls_period: float | None
    model_name: str
    model_version: str
    warnings: list[str]
    preprocess_params: dict[str, float | int | str]
    provenance: Provenance
    lightcurve_points: list[SeriesPoint]
    xai_points: list[SeriesPoint]
    bls_peaks: list[BLSPeak]


class AnalysisHistoryItem(BaseModel):
    id: int
    target_id: str
    target_type: TargetType
    mission: str
    prediction_label: str
    prediction_score: float
    bls_period: float | None
    status: str
    created_at: datetime
