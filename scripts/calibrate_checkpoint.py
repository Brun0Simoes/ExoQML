from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from exoqml.checkpoints import ALLOWED_ARCHITECTURES, load_checkpoint
from exoqml.training.max_train import (
    TCEDataset,
    create_dataloader,
    forward_epoch,
    load_manifest,
    load_star_series_cache,
    load_tce_catalog,
    build_star_series_signature,
)
from exoqml.training.metrics import best_f1_threshold, binary_metrics
from exoqml.training.model import TransitHybridQMLNet, TransitMultiViewNet


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def safe_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def calibrate_scores(val_scores: np.ndarray, val_y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float, float]:
    classifier = LogisticRegression(solver="lbfgs")
    classifier.fit(safe_logit(val_scores).reshape(-1, 1), val_y.astype(np.int64))
    coef = float(classifier.coef_[0][0])
    intercept = float(classifier.intercept_[0])
    calibrated = safe_sigmoid((coef * safe_logit(scores)) + intercept)
    return calibrated.astype(np.float64), coef, intercept


def score_split(
    rows,
    ready_manifest_rows,
    checkpoint,
    *,
    device: str,
    batch_size: int,
    num_workers: int,
):
    star_ids = sorted({row.star_id for row in rows})
    star_cache_rows = {star_id: ready_manifest_rows[star_id] for star_id in star_ids}
    star_series = load_star_series_cache(star_cache_rows)
    star_signatures = {star_id: build_star_series_signature(row) for star_id, row in star_cache_rows.items()}
    global_bins = int(checkpoint.get("global_bins", 401))
    local_bins = int(checkpoint.get("local_bins", 121))
    dataset_root = Path(ready_manifest_rows[star_ids[0]].npz_path).resolve().parents[1]
    folded_cache_dir = dataset_root / "folded_cache" / f"g{global_bins}_l{local_bins}"
    dataset = TCEDataset(
        rows,
        star_series=star_series,
        global_bins=global_bins,
        local_bins=local_bins,
        folded_cache_dir=folded_cache_dir if folded_cache_dir.exists() else None,
        star_signatures=star_signatures,
    )
    loader = create_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, device=device)
    architecture = str(checkpoint.get("architecture", "transit-multiview-tce"))
    if architecture == "transit-hybrid-qml-tce":
        model = TransitHybridQMLNet(
            scalar_dim=int(checkpoint.get("scalar_dim", 4)),
            base_channels=32,
            dropout=0.2,
            n_qubits=int(checkpoint.get("qml_qubits", 4)),
            n_q_layers=int(checkpoint.get("qml_layers", 2)),
            q_device=str(checkpoint.get("qml_device", "default.qubit")),
        ).to(device)
    else:
        model = TransitMultiViewNet(
            scalar_dim=int(checkpoint.get("scalar_dim", 4)),
            base_channels=32,
            dropout=0.2,
        ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    _, y_true, y_score = forward_epoch(
        loader,
        model,
        criterion,
        optimizer=None,
        device=device,
        use_amp=device == "cuda",
        scaler=scaler,
    )
    return y_true.astype(np.float64), y_score.astype(np.float64)


def main() -> None:
    parser = ArgumentParser(description="Fit Platt scaling on the current best checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=BACKEND_ROOT / "data" / "train_max")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BACKEND_ROOT / "data" / "train_max" / "runs" / "20260311_053035" / "best_model.pt",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    checkpoint = load_checkpoint(
        args.checkpoint,
        map_location=args.device,
        allowed_architectures=ALLOWED_ARCHITECTURES,
        require_state_dict=True,
    )
    manifest = load_manifest(args.dataset_root / "manifest.csv")
    tce_rows = load_tce_catalog(args.dataset_root / "tce_catalog.csv")
    ready_manifest_rows = {
        star_id: row
        for star_id, row in manifest.items()
        if row.status == "ready" and row.npz_path and Path(row.npz_path).exists()
    }
    val_rows = [row for row in tce_rows if row.split == "val" and row.star_id in ready_manifest_rows]
    test_rows = [row for row in tce_rows if row.split == "test" and row.star_id in ready_manifest_rows]

    val_y, val_scores = score_split(
        val_rows,
        ready_manifest_rows,
        checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_y, test_scores = score_split(
        test_rows,
        ready_manifest_rows,
        checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    calibrated_val, coef, intercept = calibrate_scores(val_scores, val_y, val_scores)
    calibrated_test, _, _ = calibrate_scores(val_scores, val_y, test_scores)
    raw_threshold = float(checkpoint.get("threshold", 0.5))
    calibrated_threshold, calibrated_val_best_f1 = best_f1_threshold(val_y, calibrated_val)

    raw_val_metrics = binary_metrics(val_y, val_scores, threshold=raw_threshold)
    calibrated_val_metrics = binary_metrics(val_y, calibrated_val, threshold=calibrated_threshold)
    raw_test_metrics = binary_metrics(test_y, test_scores, threshold=raw_threshold)
    calibrated_test_metrics = binary_metrics(test_y, calibrated_test, threshold=calibrated_threshold)

    calibrated_checkpoint = dict(checkpoint)
    calibrated_checkpoint["raw_threshold"] = raw_threshold
    calibrated_checkpoint["threshold"] = float(calibrated_threshold)
    calibrated_checkpoint["calibration"] = {
        "kind": "platt",
        "coef": coef,
        "intercept": intercept,
        "fitted_on": "val",
        "generated_at": now_iso(),
    }

    output_checkpoint = args.checkpoint.with_name(f"{args.checkpoint.stem}_calibrated{args.checkpoint.suffix}")
    torch.save(calibrated_checkpoint, output_checkpoint)

    report = {
        "generated_at": now_iso(),
        "checkpoint_input": str(args.checkpoint),
        "checkpoint_output": str(output_checkpoint),
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "calibration": calibrated_checkpoint["calibration"],
        "raw_threshold": raw_threshold,
        "calibrated_threshold": float(calibrated_threshold),
        "calibrated_val_best_f1": float(calibrated_val_best_f1),
        "val_metrics_before": raw_val_metrics,
        "val_metrics_after": calibrated_val_metrics,
        "test_metrics_before": raw_test_metrics,
        "test_metrics_after": calibrated_test_metrics,
    }
    report_path = output_checkpoint.with_name(f"{output_checkpoint.stem}_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
