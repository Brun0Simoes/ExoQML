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


class SkyCoordinates(BaseModel):
    ra: float
    dec: float


class Provenance(BaseModel):
    mission: str
    data_source: str
    sector_or_quarter: str | None = None
    analysis_timestamp: datetime
    sky_coordinates: SkyCoordinates | None = None


class ExperimentalInferenceSummary(BaseModel):
    mode: Literal["classical", "qml"]
    prediction_label: str
    prediction_score: float
    model_name: str
    model_version: str
    score_delta_vs_classical: float | None = None


class ExperimentalComparison(BaseModel):
    requested: bool
    available: bool
    activated: bool = False
    activation_reason: str | None = None
    selected_mode: Literal["classical", "qml"] | None = None
    ambiguity_lower: float | None = None
    ambiguity_upper: float | None = None
    score_delta: float | None = None
    absolute_score_delta: float | None = None
    classical: ExperimentalInferenceSummary | None = None
    qml: ExperimentalInferenceSummary | None = None


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
    experimental_comparison: ExperimentalComparison | None = None


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


class TargetCatalogItem(BaseModel):
    query: str
    target_id: str
    target_type: TargetType
    display_name: str
    mission: str
    source: str
    summary: str
    tce_count: int
    positive_tce_count: int
    sky_coordinates: SkyCoordinates | None = None
