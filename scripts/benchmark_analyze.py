from __future__ import annotations

import json
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from exoqml.config import Settings
from exoqml.services.acquisition import fetch_lightcurve
from exoqml.services.bls import run_bls_baseline
from exoqml.services.identifier import resolve_target
from exoqml.services.inference import run_inference
from exoqml.services.preprocess import preprocess_lightcurve


@dataclass(slots=True)
class TimingResult:
    acquisition_sec: float
    preprocess_sec: float
    bls_sec: float
    inference_sec: float
    total_sec: float
    points_input: int
    points_output: int
    mission: str
    prediction_label: str
    prediction_score: float
    warnings: list[str]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def round_float(value: float) -> float:
    return round(float(value), 4)


def summarize_runs(runs: list[TimingResult]) -> dict[str, float | int]:
    return {
        "count": len(runs),
        "median_total_sec": round_float(statistics.median(item.total_sec for item in runs)),
        "median_acquisition_sec": round_float(statistics.median(item.acquisition_sec for item in runs)),
        "median_preprocess_sec": round_float(statistics.median(item.preprocess_sec for item in runs)),
        "median_bls_sec": round_float(statistics.median(item.bls_sec for item in runs)),
        "median_inference_sec": round_float(statistics.median(item.inference_sec for item in runs)),
        "max_total_sec": round_float(max(item.total_sec for item in runs)),
        "min_total_sec": round_float(min(item.total_sec for item in runs)),
    }


def build_settings(cache_dir: Path, *, device: str = "cpu", allow_synthetic_fallback: bool = False) -> Settings:
    return Settings(
        _env_file=str(BACKEND_ROOT / ".env"),
        cache_dir=str(cache_dir),
        device=device,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )


def run_full_pipeline(settings: Settings, *, target_id: str, target_type: str) -> TimingResult:
    target = resolve_target(target_id, target_type)  # type: ignore[arg-type]

    t0 = time.perf_counter()
    acquisition = fetch_lightcurve(target=target, settings=settings)
    t1 = time.perf_counter()
    proc_time, proc_flux, _ = preprocess_lightcurve(
        time=acquisition.time,
        flux=acquisition.flux,
        max_points=settings.max_points,
    )
    t2 = time.perf_counter()
    _, bls_peaks = run_bls_baseline(proc_time, proc_flux)
    t3 = time.perf_counter()
    inference = run_inference(
        time=proc_time,
        flux=proc_flux,
        bls_peaks=bls_peaks,
        settings=settings,
        experimental_qml=False,
    )
    t4 = time.perf_counter()

    return TimingResult(
        acquisition_sec=t1 - t0,
        preprocess_sec=t2 - t1,
        bls_sec=t3 - t2,
        inference_sec=t4 - t3,
        total_sec=t4 - t0,
        points_input=int(acquisition.time.size),
        points_output=int(proc_time.size),
        mission=acquisition.mission,
        prediction_label=str(inference["label"]),
        prediction_score=float(inference["probability"]),
        warnings=[*acquisition.warnings, *inference["warnings"]],
    )


def run_compute_only(settings: Settings, *, target_id: str, target_type: str, repeats: int) -> list[TimingResult]:
    target = resolve_target(target_id, target_type)  # type: ignore[arg-type]
    acquisition = fetch_lightcurve(target=target, settings=settings)
    results: list[TimingResult] = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        proc_time, proc_flux, _ = preprocess_lightcurve(
            time=acquisition.time,
            flux=acquisition.flux,
            max_points=settings.max_points,
        )
        t1 = time.perf_counter()
        _, bls_peaks = run_bls_baseline(proc_time, proc_flux)
        t2 = time.perf_counter()
        inference = run_inference(
            time=proc_time,
            flux=proc_flux,
            bls_peaks=bls_peaks,
            settings=settings,
            experimental_qml=False,
        )
        t3 = time.perf_counter()

        results.append(
            TimingResult(
                acquisition_sec=0.0,
                preprocess_sec=t1 - t0,
                bls_sec=t2 - t1,
                inference_sec=t3 - t2,
                total_sec=t3 - t0,
                points_input=int(acquisition.time.size),
                points_output=int(proc_time.size),
                mission=acquisition.mission,
                prediction_label=str(inference["label"]),
                prediction_score=float(inference["probability"]),
                warnings=[*acquisition.warnings, *inference["warnings"]],
            )
        )
    return results


def main() -> None:
    benchmark_root = BACKEND_ROOT / "data" / "benchmarks"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    warm_cache = benchmark_root / "warm_cache"
    cold_root = benchmark_root / "cold_runs"
    shutil.rmtree(cold_root, ignore_errors=True)
    warm_cache.mkdir(parents=True, exist_ok=True)
    cold_root.mkdir(parents=True, exist_ok=True)

    target_id = "KIC 10000490"
    target_type = "kic"

    warm_settings = build_settings(warm_cache, device="cpu", allow_synthetic_fallback=False)

    warmup = run_full_pipeline(warm_settings, target_id=target_id, target_type=target_type)
    warm_runs = [
        run_full_pipeline(warm_settings, target_id=target_id, target_type=target_type)
        for _ in range(3)
    ]

    cold_runs: list[TimingResult] = []
    for idx in range(2):
        cold_cache = cold_root / f"cold_{idx + 1}"
        cold_cache.mkdir(parents=True, exist_ok=True)
        settings = build_settings(cold_cache, device="cpu", allow_synthetic_fallback=False)
        cold_runs.append(run_full_pipeline(settings, target_id=target_id, target_type=target_type))

    compute_runs = run_compute_only(warm_settings, target_id=target_id, target_type=target_type, repeats=5)

    report = {
        "generated_at": now_iso(),
        "target_id": target_id,
        "target_type": target_type,
        "device": "cpu",
        "model_path": warm_settings.model_path,
        "scenarios": {
            "warmup_full_pipeline": asdict(warmup),
            "warm_full_pipeline": {
                "summary": summarize_runs(warm_runs),
                "runs": [asdict(item) for item in warm_runs],
            },
            "cold_full_pipeline": {
                "summary": summarize_runs(cold_runs),
                "runs": [asdict(item) for item in cold_runs],
            },
            "compute_only": {
                "summary": summarize_runs(compute_runs),
                "runs": [asdict(item) for item in compute_runs],
            },
        },
    }

    output_path = benchmark_root / f"analysis_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_path = benchmark_root / "latest_analysis_benchmark.json"
    latest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output_path)
    print(json.dumps(report["scenarios"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
