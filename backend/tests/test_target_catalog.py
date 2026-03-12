from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from exoqml.main import app
from exoqml.services import target_catalog

client = TestClient(app)


def test_load_target_catalog_aggregates_ready_entries(monkeypatch, tmp_path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "star_id,label,split,status,npz_path,num_points,error,updated_at\n"
        "100,1,train,ready,dummy.npz,4096,,2026-03-12T00:00:00+00:00\n"
        "101,0,train,error,dummy.npz,4096,error,2026-03-12T00:00:00+00:00\n",
        encoding="utf-8",
    )
    tce_catalog_path = tmp_path / "tce_catalog.csv"
    tce_catalog_path.write_text(
        "tce_id,star_id,label,label_name,split,period,duration_hours,epoch,depth_ppm,model_snr\n"
        "100_1,100,1,PC,train,3.5,2.0,1.0,100.0,22.0\n"
        "100_2,100,0,AFP,train,6.0,2.0,1.0,40.0,10.0\n"
        "101_1,101,0,NTP,train,4.0,2.0,1.0,40.0,9.0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(target_catalog, "READY_MANIFEST_PATH", Path(manifest_path))
    monkeypatch.setattr(target_catalog, "TCE_CATALOG_PATH", Path(tce_catalog_path))
    target_catalog.load_target_catalog.cache_clear()

    items = target_catalog.load_target_catalog()
    dataset_items = [item for item in items if item["source"] == "local_dataset"]

    assert len(dataset_items) == 1
    assert dataset_items[0]["query"] == "KIC 100"
    assert dataset_items[0]["positive_tce_count"] == 1
    assert dataset_items[0]["tce_count"] == 2
    assert "PC" in str(dataset_items[0]["summary"])
    target_catalog.load_target_catalog.cache_clear()


def test_target_catalog_endpoint_returns_items() -> None:
    target_catalog.load_target_catalog.cache_clear()
    response = client.get("/api/v1/targets/catalog?limit=5")

    assert response.status_code == 200
    payload = response.json()
    assert payload
    assert "query" in payload[0]
    assert "summary" in payload[0]
