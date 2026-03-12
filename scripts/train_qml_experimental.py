from __future__ import annotations

import json
import math
import sys
import time
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from exoqml.checkpoints import ALLOWED_ARCHITECTURES, load_checkpoint
from exoqml.training.max_train import (
    TCEDataset,
    build_star_series_signature,
    create_dataloader,
    forward_epoch,
    load_manifest,
    load_star_series_cache,
    load_tce_catalog,
)
from exoqml.training.metrics import best_f1_threshold, binary_metrics
from exoqml.training.model import TransitHybridQMLNet


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Train experimental PennyLane hybrid head on cached ExoQML features.")
    parser.add_argument("--dataset-root", type=Path, default=BACKEND_ROOT / "data" / "train_max")
    parser.add_argument(
        "--classical-checkpoint",
        type=Path,
        default=BACKEND_ROOT / "data" / "train_max" / "runs" / "20260311_053035" / "best_model_calibrated.pt",
    )
    parser.add_argument("--run-root", type=Path, default=BACKEND_ROOT / "data" / "qml_experiment")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--qml-qubits", type=int, default=4)
    parser.add_argument("--qml-layers", type=int, default=2)
    parser.add_argument("--qml-device", type=str, default="default.qubit")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    return parser


def build_datasets(dataset_root: Path):
    manifest = load_manifest(dataset_root / "manifest.csv")
    tce_rows = load_tce_catalog(dataset_root / "tce_catalog.csv")
    ready_manifest_rows = {
        star_id: row
        for star_id, row in manifest.items()
        if row.status == "ready" and row.npz_path and Path(row.npz_path).exists()
    }
    train_rows = [row for row in tce_rows if row.split == "train" and row.star_id in ready_manifest_rows]
    val_rows = [row for row in tce_rows if row.split == "val" and row.star_id in ready_manifest_rows]
    test_rows = [row for row in tce_rows if row.split == "test" and row.star_id in ready_manifest_rows]

    star_ids = sorted({row.star_id for row in (train_rows + val_rows + test_rows)})
    star_cache_rows = {star_id: ready_manifest_rows[star_id] for star_id in star_ids}
    star_series = load_star_series_cache(star_cache_rows)
    star_signatures = {star_id: build_star_series_signature(row) for star_id, row in star_cache_rows.items()}
    return train_rows, val_rows, test_rows, ready_manifest_rows, star_series, star_signatures


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    device = args.device
    checkpoint = load_checkpoint(
        args.classical_checkpoint,
        map_location=device,
        allowed_architectures=ALLOWED_ARCHITECTURES,
        require_state_dict=True,
    )

    train_rows, val_rows, test_rows, ready_manifest_rows, star_series, star_signatures = build_datasets(
        args.dataset_root
    )
    global_bins = int(checkpoint.get("global_bins", 401))
    local_bins = int(checkpoint.get("local_bins", 121))
    scalar_dim = int(checkpoint.get("scalar_dim", 4))
    folded_cache_dir = args.dataset_root / "folded_cache" / f"g{global_bins}_l{local_bins}"

    train_ds = TCEDataset(
        train_rows,
        star_series=star_series,
        global_bins=global_bins,
        local_bins=local_bins,
        folded_cache_dir=folded_cache_dir if folded_cache_dir.exists() else None,
        star_signatures=star_signatures,
    )
    val_ds = TCEDataset(
        val_rows,
        star_series=star_series,
        global_bins=global_bins,
        local_bins=local_bins,
        folded_cache_dir=folded_cache_dir if folded_cache_dir.exists() else None,
        star_signatures=star_signatures,
    )
    test_ds = TCEDataset(
        test_rows,
        star_series=star_series,
        global_bins=global_bins,
        local_bins=local_bins,
        folded_cache_dir=folded_cache_dir if folded_cache_dir.exists() else None,
        star_signatures=star_signatures,
    )

    model = TransitHybridQMLNet(
        scalar_dim=scalar_dim,
        base_channels=args.base_channels,
        dropout=args.dropout,
        n_qubits=args.qml_qubits,
        n_q_layers=args.qml_layers,
        q_device=args.qml_device,
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if name.startswith("global_encoder") or name.startswith("local_encoder") or name.startswith("scalar_head"):
                param.requires_grad = False

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    y_train_labels = np.asarray([row.label for row in train_rows], dtype=np.float32)
    pos = float(np.sum(y_train_labels == 1.0))
    neg = float(np.sum(y_train_labels == 0.0))
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    run_dir = args.run_root / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = run_dir / "best_model.pt"
    latest_checkpoint_path = run_dir / "latest_model.pt"

    history: list[dict[str, Any]] = []
    best_key = -1.0
    best_epoch = 0
    best_threshold = 0.5
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loader = create_dataloader(
            train_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            device=device,
        )
        val_loader = create_dataloader(
            val_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            device=device,
        )
        train_loss, y_train, score_train = forward_epoch(
            train_loader,
            model,
            criterion,
            optimizer=optimizer,
            device=device,
            use_amp=False,
            scaler=scaler,
        )
        val_loss, y_val, score_val = forward_epoch(
            val_loader,
            model,
            criterion,
            optimizer=None,
            device=device,
            use_amp=False,
            scaler=scaler,
        )
        thr, val_best_f1 = best_f1_threshold(y_val, score_val)
        val_metrics = binary_metrics(y_val, score_val, threshold=thr)
        train_metrics = binary_metrics(y_train, score_train, threshold=0.5)

        epoch_payload = {
            "epoch": epoch,
            "seconds": round(time.time() - epoch_start, 2),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_f1": float(train_metrics["f1"]),
            "val_f1_best": float(val_best_f1),
            "val_best_threshold": float(thr),
            "val_pr_auc": float(val_metrics["pr_auc"]),
            "val_roc_auc": float(val_metrics["roc_auc"]),
            "val_recall_best": float(val_metrics["recall"]),
            "val_precision_best": float(val_metrics["precision"]),
            "val_brier": float(val_metrics["brier"]),
            "val_ece": float(val_metrics["ece"]),
        }
        history.append(epoch_payload)
        print(
            f"[qml epoch {epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_pr_auc={val_metrics['pr_auc']:.4f} val_f1={val_best_f1:.4f} thr={thr:.2f}"
        )

        metric_key = float(val_metrics["pr_auc"]) if not math.isnan(float(val_metrics["pr_auc"])) else float(val_best_f1)
        payload = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "threshold": float(thr),
            "architecture": "transit-hybrid-qml-tce",
            "global_bins": global_bins,
            "local_bins": local_bins,
            "scalar_dim": scalar_dim,
            "qml_qubits": int(args.qml_qubits),
            "qml_layers": int(args.qml_layers),
            "qml_device": str(args.qml_device),
            "source_checkpoint": str(args.classical_checkpoint),
            "history": history,
            "freeze_backbone": bool(args.freeze_backbone),
            "missing_from_init": missing,
            "unexpected_from_init": unexpected,
        }
        atomic_torch_save(latest_checkpoint_path, payload)

        if metric_key > best_key:
            best_key = metric_key
            best_epoch = epoch
            best_threshold = float(thr)
            patience_counter = 0
            atomic_torch_save(best_checkpoint_path, payload)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    best_checkpoint = load_checkpoint(
        best_checkpoint_path,
        map_location=device,
        allowed_architectures=ALLOWED_ARCHITECTURES,
        require_state_dict=True,
    )
    model.load_state_dict(best_checkpoint["state_dict"])
    test_loader = create_dataloader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        device=device,
    )
    test_loss, y_test, score_test = forward_epoch(
        test_loader,
        model,
        criterion,
        optimizer=None,
        device=device,
        use_amp=False,
        scaler=scaler,
    )
    test_metrics = binary_metrics(y_test, score_test, threshold=best_threshold)
    test_best_thr, test_best_f1 = best_f1_threshold(y_test, score_test)
    test_metrics_best = binary_metrics(y_test, score_test, threshold=test_best_thr)

    report = {
        "generated_at": now_iso(),
        "status": "completed",
        "config": {
            "dataset_root": str(args.dataset_root),
            "classical_checkpoint": str(args.classical_checkpoint),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": device,
            "qml_qubits": args.qml_qubits,
            "qml_layers": args.qml_layers,
            "qml_device": args.qml_device,
            "freeze_backbone": bool(args.freeze_backbone),
        },
        "dataset": {
            "train_tces": len(train_rows),
            "val_tces": len(val_rows),
            "test_tces": len(test_rows),
        },
        "training": {
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            "test_loss": float(test_loss),
            "test_metrics_threshold_best_on_val": test_metrics,
            "test_metrics_best_on_test": test_metrics_best,
            "test_best_threshold": float(test_best_thr),
            "test_best_f1": float(test_best_f1),
            "history": history,
            "best_checkpoint": str(best_checkpoint_path),
            "latest_checkpoint": str(latest_checkpoint_path),
        },
    }
    write_json(run_dir / "report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
