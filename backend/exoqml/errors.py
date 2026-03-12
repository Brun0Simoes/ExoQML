from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AnalysisError(Exception):
    code: str
    message: str
    suggestion: str | None = None
    status_code: int = 400
    stage: str = "analysis"
    cause: str | None = None

    def __str__(self) -> str:
        return self.message

    def to_detail(self) -> dict[str, str]:
        detail = {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
        }
        if self.suggestion:
            detail["suggestion"] = self.suggestion
        return detail

    def log_context(self, **extra: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "event": "analysis_failed",
            "error_code": self.code,
            "error_stage": self.stage,
            "error_message": self.message,
        }
        if self.suggestion:
            payload["suggestion"] = self.suggestion
        if self.cause:
            payload["cause"] = self.cause
        payload.update(extra)
        return payload
