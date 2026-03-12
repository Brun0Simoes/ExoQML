from __future__ import annotations

import json
import subprocess
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_qml_residual.py"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Grid search for residual QML ambiguity band and residual alpha.")
    parser.add_argument("--run-root", type=Path, default=BACKEND_ROOT / "data" / "qml_residual_tuning")
    parser.add_argument("--dataset-root", type=Path, default=BACKEND_ROOT / "data" / "train_max")
    parser.add_argument(
        "--classical-checkpoint",
        type=Path,
        default=BACKEND_ROOT / "data" / "train_max" / "runs" / "20260311_053035" / "best_model_calibrated.pt",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.10, 0.15, 0.20])
    parser.add_argument(
        "--bands",
        nargs="+",
        default=["0.25:0.75", "0.30:0.70", "0.35:0.65"],
        help="Pairs lower:upper for the ambiguity band.",
    )
    return parser


def parse_band(raw: str) -> tuple[float, float]:
    lower_raw, upper_raw = raw.split(":")
    lower = float(lower_raw)
    upper = float(upper_raw)
    if not (0.0 <= lower <= upper <= 1.0):
        raise ValueError(f"Invalid band {raw}")
    return lower, upper


def score_key(candidate: dict[str, Any]) -> tuple[float, float, float]:
    metrics = candidate["metrics"]
    return (
        float(metrics["f1"]),
        float(metrics["pr_auc"]),
        float(metrics["roc_auc"]),
    )


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    args.run_root.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []

    for band_raw in args.bands:
        lower, upper = parse_band(band_raw)
        for alpha in args.alphas:
            existing_runs = {path.name for path in (args.run_root / "runs").glob("*")} if (args.run_root / "runs").exists() else set()
            cmd = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--dataset-root",
                str(args.dataset_root),
                "--classical-checkpoint",
                str(args.classical_checkpoint),
                "--run-root",
                str(args.run_root),
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--ambiguity-lower",
                str(lower),
                "--ambiguity-upper",
                str(upper),
                "--residual-alpha-init",
                str(alpha),
            ]
            print(f"[tune] lower={lower:.2f} upper={upper:.2f} alpha={alpha:.2f}")
            subprocess.run(cmd, check=True)
            run_dirs = sorted((args.run_root / "runs").glob("*"))
            new_run_dirs = [path for path in run_dirs if path.name not in existing_runs]
            if not new_run_dirs:
                raise RuntimeError("Could not identify new tuning run directory.")
            run_dir = new_run_dirs[-1]
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            metrics = report["training"]["test_metrics_threshold_best_on_val"]
            candidate = {
                "run_dir": str(run_dir),
                "ambiguity_lower": lower,
                "ambiguity_upper": upper,
                "residual_alpha_init": alpha,
                "metrics": metrics,
                "threshold": report["training"]["final_threshold"],
                "focus_tces": report["dataset"]["focus_tces"],
                "val_gate_samples": report["dataset"]["val_gate_samples"],
                "test_gate_samples": report["dataset"]["test_gate_samples"],
                "checkpoint": report["training"]["best_calibrated_checkpoint"],
            }
            candidates.append(candidate)

    best = max(candidates, key=score_key)
    summary = {
        "generated_at": now_iso(),
        "status": "completed",
        "search": {
            "device": args.device,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "alphas": args.alphas,
            "bands": args.bands,
        },
        "best": best,
        "candidates": sorted(candidates, key=score_key, reverse=True),
    }
    latest_path = args.run_root / "latest_tuning_summary.json"
    latest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(latest_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
