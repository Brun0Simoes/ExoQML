from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np

from exoqml.config import Settings
from exoqml.services.acquisition import fetch_lightcurve
from exoqml.services.bls import run_bls_baseline
from exoqml.services.identifier import resolve_target
from exoqml.services.inference import run_inference, run_inference_comparison
from exoqml.services.preprocess import preprocess_lightcurve


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def round_float(value: float) -> float:
    return round(float(value), 4)


def main() -> None:
    settings = Settings(_env_file=str(BACKEND_ROOT / ".env"), device="cpu")
    target = resolve_target("KIC 10000490", "kic")

    acquisition = fetch_lightcurve(target=target, settings=settings)
    proc_time, proc_flux, _ = preprocess_lightcurve(acquisition.time, acquisition.flux, max_points=settings.max_points)
    _, bls_peaks = run_bls_baseline(proc_time, proc_flux)

    classical_times: list[float] = []
    qml_direct_times: list[float] = []
    qml_compare_times: list[float] = []
    classical_scores: list[float] = []
    qml_direct_scores: list[float] = []
    qml_compare_scores: list[float] = []
    qml_selected_modes: list[str] = []

    classical_settings = settings.model_copy(update={"enable_qml": False})
    qml_direct_settings = settings.model_copy(update={"model_path": settings.qml_model_path})

    for _ in range(5):
        t0 = time.perf_counter()
        classical = run_inference(
            time=proc_time,
            flux=proc_flux,
            bls_peaks=bls_peaks,
            settings=classical_settings,
            experimental_qml=False,
        )
        classical_times.append(time.perf_counter() - t0)
        classical_scores.append(float(classical["probability"]))

        t1 = time.perf_counter()
        qml_direct = run_inference(
            time=proc_time,
            flux=proc_flux,
            bls_peaks=bls_peaks,
            settings=qml_direct_settings,
            experimental_qml=True,
        )
        qml_direct_times.append(time.perf_counter() - t1)
        qml_direct_scores.append(float(qml_direct["probability"]))

        t2 = time.perf_counter()
        qml_bundle = run_inference_comparison(
            time=proc_time,
            flux=proc_flux,
            bls_peaks=bls_peaks,
            settings=settings,
            experimental_qml=True,
        )
        qml_compare_times.append(time.perf_counter() - t2)
        qml_compare_scores.append(float(qml_bundle["primary"]["probability"]))
        qml_selected_modes.append(str(qml_bundle["comparison"]["selected_mode"]))

    report = {
        "generated_at": now_iso(),
        "target_id": "KIC 10000490",
        "target_type": "kic",
        "classical_model_path": settings.model_path,
        "qml_model_path": settings.qml_model_path,
        "classical": {
            "median_inference_sec": round_float(statistics.median(classical_times)),
            "min_inference_sec": round_float(min(classical_times)),
            "max_inference_sec": round_float(max(classical_times)),
            "median_score": round_float(statistics.median(classical_scores)),
        },
        "qml_direct": {
            "median_inference_sec": round_float(statistics.median(qml_direct_times)),
            "min_inference_sec": round_float(min(qml_direct_times)),
            "max_inference_sec": round_float(max(qml_direct_times)),
            "median_score": round_float(statistics.median(qml_direct_scores)),
        },
        "qml_second_stage": {
            "median_inference_sec": round_float(statistics.median(qml_compare_times)),
            "min_inference_sec": round_float(min(qml_compare_times)),
            "max_inference_sec": round_float(max(qml_compare_times)),
            "median_score": round_float(statistics.median(qml_compare_scores)),
            "selected_modes": qml_selected_modes,
        },
    }

    output_root = BACKEND_ROOT / "data" / "benchmarks"
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"qml_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    latest_path = output_root / "latest_qml_compare.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
