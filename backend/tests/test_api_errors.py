from __future__ import annotations

from fastapi.testclient import TestClient

import exoqml.api.routes as routes
from exoqml.errors import AnalysisError
from exoqml.main import app


client = TestClient(app)


def test_analyze_returns_structured_analysis_error(monkeypatch) -> None:
    def fake_run_analysis(*args, **kwargs):
        raise AnalysisError(
            code="acquisition_failed",
            message="Não foi possível obter uma curva de luz utilizável para esse alvo.",
            suggestion="Tente outro identificador, aguarde a fonte externa ou use um alvo de teste.",
            status_code=502,
            stage="acquisition",
        )

    monkeypatch.setattr(routes, "run_analysis", fake_run_analysis)
    response = client.post(
        "/api/v1/analyze",
        json={"target_id": "TIC 25155310", "experimental_qml": False},
    )

    assert response.status_code == 502
    payload = response.json()
    assert payload["detail"]["code"] == "acquisition_failed"
    assert payload["detail"]["stage"] == "acquisition"
    assert "curva de luz" in payload["detail"]["message"]
    assert "Tente outro identificador" in payload["detail"]["suggestion"]
