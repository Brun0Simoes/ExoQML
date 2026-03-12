from __future__ import annotations

import json
import math
import sys
import time
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    build_star_series_signature,
    create_dataloader,
    forward_epoch,
    load_manifest,
    load_star_series_cache,
    load_tce_catalog,
)
from exoqml.training.metrics import best_f1_threshold, binary_metrics
from exoqml.training.model import TransitMultiViewNet, TransitResidualQMLNet


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


def safe_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def safe_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def apply_platt(probabilities: np.ndarray, calibration: dict[str, float] | None) -> np.ndarray:
    if not calibration:
        return probabilities.astype(np.float64)
    coef = float(calibration["coef"])
    intercept = float(calibration["intercept"])
    return safe_sigmoid((coef * safe_logit(probabilities)) + intercept)


def fit_platt(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float] | None:
    if probabilities.size < 32:
        return None
    unique = np.unique(labels.astype(np.int64))
    if unique.size < 2:
        return None
    classifier = LogisticRegression(solver="lbfgs", max_iter=2000)
    classifier.fit(safe_logit(probabilities).reshape(-1, 1), labels.astype(np.int64))
    return {
        "coef": float(classifier.coef_[0][0]),
        "intercept": float(classifier.intercept_[0]),
    }


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Train residual QML second stage on ambiguous ExoQML samples.")
    parser.add_argument("--dataset-root", type=Path, default=BACKEND_ROOT / "data" / "train_max")
    parser.add_argument(
        "--classical-checkpoint",
        type=Path,
        default=BACKEND_ROOT / "data" / "train_max" / "runs" / "20260311_053035" / "best_model_calibrated.pt",
    )
    parser.add_argument("--run-root", type=Path, default=BACKEND_ROOT / "data" / "qml_residual")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--qml-qubits", type=int, default=4)
    parser.add_argument("--qml-layers", type=int, default=2)
    parser.add_argument("--qml-device", type=str, default="default.qubit")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--residual-alpha-init", type=float, default=0.15)
    parser.add_argument("--ambiguity-lower", type=float, default=0.30)
    parser.add_argument("--ambiguity-upper", type=float, default=0.70)
    parser.add_argument("--min-focus-samples", type=int, default=512)
    parser.add_argument("--include-classical-errors", action=BooleanOptionalAction, default=True)
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


def build_dataset(rows, star_series, global_bins: int, local_bins: int, folded_cache_dir: Path | None, star_signatures):
    return TCEDataset(
        rows,
        star_series=star_series,
        global_bins=global_bins,
        local_bins=local_bins,
        folded_cache_dir=folded_cache_dir if folded_cache_dir and folded_cache_dir.exists() else None,
        star_signatures=star_signatures,
    )


def score_dataset(dataset: TCEDataset, model: nn.Module, *, batch_size: int, num_workers: int, device: str):
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    loader = create_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, device=device)
    _, y_true, y_score = forward_epoch(
        loader,
        model,
        criterion,
        optimizer=None,
        device=device,
        use_amp=False,
        scaler=scaler,
    )
    return y_true.astype(np.float64), y_score.astype(np.float64)


def compose_scores(
    classical_scores: np.ndarray,
    residual_scores: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (classical_scores >= lower) & (classical_scores <= upper)
    composed = classical_scores.astype(np.float64).copy()
    composed[mask] = residual_scores[mask]
    return composed, mask


def select_focus_indices(
    classical_scores: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
    lower: float,
    upper: float,
    min_focus_samples: int,
    include_classical_errors: bool,
) -> np.ndarray:
    focus = (classical_scores >= lower) & (classical_scores <= upper)
    if include_classical_errors:
        predictions = (classical_scores >= threshold).astype(np.float64)
        focus |= predictions != labels
    if int(np.sum(focus)) < min_focus_samples or np.unique(labels[focus].astype(np.int64)).size < 2:
        focus = (classical_scores >= lower) & (classical_scores <= upper)
    if int(np.sum(focus)) < min_focus_samples or np.unique(labels[focus].astype(np.int64)).size < 2:
        focus = np.ones_like(labels, dtype=bool)
    return focus


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    device = args.device

    classical_checkpoint = load_checkpoint(
        args.classical_checkpoint,
        map_location=device,
        allowed_architectures=ALLOWED_ARCHITECTURES,
        require_state_dict=True,
    )
    if str(classical_checkpoint.get("architecture", "")) != "transit-multiview-tce":
        raise RuntimeError("Residual QML second stage expects a multiview classical checkpoint as the source model.")

    train_rows, val_rows, test_rows, ready_manifest_rows, star_series, star_signatures = build_datasets(args.dataset_root)
    global_bins = int(classical_checkpoint.get("global_bins", 401))
    local_bins = int(classical_checkpoint.get("local_bins", 121))
    scalar_dim = int(classical_checkpoint.get("scalar_dim", 4))
    folded_cache_dir = args.dataset_root / "folded_cache" / f"g{global_bins}_l{local_bins}"

    full_train_ds = build_dataset(train_rows, star_series, global_bins, local_bins, folded_cache_dir, star_signatures)
    val_ds = build_dataset(val_rows, star_series, global_bins, local_bins, folded_cache_dir, star_signatures)
    test_ds = build_dataset(test_rows, star_series, global_bins, local_bins, folded_cache_dir, star_signatures)

    classical_model = TransitMultiViewNet(scalar_dim=scalar_dim, base_channels=32, dropout=0.2).to(device)
    classical_model.load_state_dict(classical_checkpoint["state_dict"])
    classical_model.eval()

    y_train, classical_train_scores = score_dataset(
        full_train_ds,
        classical_model,
        batch_size=max(64, args.batch_size),
        num_workers=args.num_workers,
        device=device,
    )
    y_val, classical_val_scores = score_dataset(
        val_ds,
        classical_model,
        batch_size=max(64, args.batch_size),
        num_workers=args.num_workers,
        device=device,
    )
    y_test, classical_test_scores = score_dataset(
        test_ds,
        classical_model,
        batch_size=max(64, args.batch_size),
        num_workers=args.num_workers,
        device=device,
    )

    classical_train_scores = apply_platt(classical_train_scores, classical_checkpoint.get("calibration"))
    classical_val_scores = apply_platt(classical_val_scores, classical_checkpoint.get("calibration"))
    classical_test_scores = apply_platt(classical_test_scores, classical_checkpoint.get("calibration"))
    classical_threshold = float(classical_checkpoint.get("threshold", 0.5))

    focus_mask = select_focus_indices(
        classical_train_scores,
        y_train,
        threshold=classical_threshold,
        lower=float(args.ambiguity_lower),
        upper=float(args.ambiguity_upper),
        min_focus_samples=int(args.min_focus_samples),
        include_classical_errors=bool(args.include_classical_errors),
    )
    focus_rows = [row for index, row in enumerate(train_rows) if bool(focus_mask[index])]
    focus_ds = build_dataset(focus_rows, star_series, global_bins, local_bins, folded_cache_dir, star_signatures)

    model = TransitResidualQMLNet(
        scalar_dim=scalar_dim,
        base_channels=args.base_channels,
        dropout=args.dropout,
        n_qubits=args.qml_qubits,
        n_q_layers=args.qml_layers,
        q_device=args.qml_device,
        residual_alpha_init=args.residual_alpha_init,
    ).to(device)
    missing, unexpected = model.load_state_dict(classical_checkpoint["state_dict"], strict=False)

    for name, param in model.named_parameters():
        if (
            name.startswith("global_encoder")
            or name.startswith("local_encoder")
            or name.startswith("scalar_head")
            or name.startswith("head")
        ):
            param.requires_grad = False

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    focus_labels = np.asarray([row.label for row in focus_rows], dtype=np.float32)
    pos = float(np.sum(focus_labels == 1.0))
    neg = float(np.sum(focus_labels == 0.0))
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    run_dir = args.run_root / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = run_dir / "best_model.pt"
    calibrated_checkpoint_path = run_dir / "best_model_calibrated.pt"
    latest_checkpoint_path = run_dir / "latest_model.pt"

    history: list[dict[str, Any]] = []
    best_key = -1.0
    best_epoch = 0
    best_threshold = classical_threshold
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loader = create_dataloader(
            focus_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            device=device,
        )
        val_loader = create_dataloader(
            val_ds,
            batch_size=max(64, args.batch_size),
            num_workers=args.num_workers,
            shuffle=False,
            device=device,
        )

        train_loss, focus_y, focus_scores = forward_epoch(
            train_loader,
            model,
            criterion,
            optimizer=optimizer,
            device=device,
            use_amp=False,
            scaler=scaler,
        )
        val_loss, _, residual_val_scores = forward_epoch(
            val_loader,
            model,
            criterion,
            optimizer=None,
            device=device,
            use_amp=False,
            scaler=scaler,
        )
        composed_val_scores, val_gate_mask = compose_scores(
            classical_val_scores,
            residual_val_scores,
            lower=float(args.ambiguity_lower),
            upper=float(args.ambiguity_upper),
        )
        thr, val_best_f1 = best_f1_threshold(y_val, composed_val_scores)
        val_metrics = binary_metrics(y_val, composed_val_scores, threshold=thr)
        focus_metrics = binary_metrics(focus_y, focus_scores, threshold=0.5)

        epoch_payload = {
            "epoch": epoch,
            "seconds": round(time.time() - epoch_start, 2),
            "train_loss_focus": float(train_loss),
            "val_loss_residual_raw": float(val_loss),
            "focus_f1": float(focus_metrics["f1"]),
            "val_f1_best": float(val_best_f1),
            "val_best_threshold": float(thr),
            "val_pr_auc": float(val_metrics["pr_auc"]),
            "val_roc_auc": float(val_metrics["roc_auc"]),
            "val_gate_samples": int(np.sum(val_gate_mask)),
            "residual_alpha": float(torch.sigmoid(model.residual_gate_logit).detach().cpu().item()),
        }
        history.append(epoch_payload)
        print(
            f"[residual qml epoch {epoch:02d}] "
            f"focus_loss={train_loss:.4f} val_pr_auc={val_metrics['pr_auc']:.4f} "
            f"val_f1={val_best_f1:.4f} gate={int(np.sum(val_gate_mask))}"
        )

        metric_key = float(val_metrics["pr_auc"]) if not math.isnan(float(val_metrics["pr_auc"])) else float(val_best_f1)
        checkpoint_payload = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "threshold": float(thr),
            "architecture": "transit-residual-qml-tce",
            "global_bins": global_bins,
            "local_bins": local_bins,
            "scalar_dim": scalar_dim,
            "qml_qubits": int(args.qml_qubits),
            "qml_layers": int(args.qml_layers),
            "qml_device": str(args.qml_device),
            "source_checkpoint": str(args.classical_checkpoint),
            "ambiguity_lower": float(args.ambiguity_lower),
            "ambiguity_upper": float(args.ambiguity_upper),
            "residual_alpha_init": float(args.residual_alpha_init),
            "focus_samples": len(focus_rows),
            "history": history,
            "missing_from_init": missing,
            "unexpected_from_init": unexpected,
        }
        atomic_torch_save(latest_checkpoint_path, checkpoint_payload)

        if metric_key > best_key:
            best_key = metric_key
            best_epoch = epoch
            best_threshold = float(thr)
            patience_counter = 0
            atomic_torch_save(best_checkpoint_path, checkpoint_payload)
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

    _, residual_val_scores = score_dataset(
        val_ds,
        model,
        batch_size=max(64, args.batch_size),
        num_workers=args.num_workers,
        device=device,
    )
    _, residual_test_scores = score_dataset(
        test_ds,
        model,
        batch_size=max(64, args.batch_size),
        num_workers=args.num_workers,
        device=device,
    )

    _, val_gate_mask = compose_scores(
        classical_val_scores,
        residual_val_scores,
        lower=float(args.ambiguity_lower),
        upper=float(args.ambiguity_upper),
    )
    _, test_gate_mask = compose_scores(
        classical_test_scores,
        residual_test_scores,
        lower=float(args.ambiguity_lower),
        upper=float(args.ambiguity_upper),
    )

    calibration = fit_platt(residual_val_scores[val_gate_mask], y_val[val_gate_mask])
    calibrated_residual_val = residual_val_scores.astype(np.float64)
    calibrated_residual_test = residual_test_scores.astype(np.float64)
    if calibration:
        calibrated_residual_val[val_gate_mask] = apply_platt(residual_val_scores[val_gate_mask], calibration)
        calibrated_residual_test[test_gate_mask] = apply_platt(residual_test_scores[test_gate_mask], calibration)

    final_val_scores, _ = compose_scores(
        classical_val_scores,
        calibrated_residual_val,
        lower=float(args.ambiguity_lower),
        upper=float(args.ambiguity_upper),
    )
    final_test_scores, _ = compose_scores(
        classical_test_scores,
        calibrated_residual_test,
        lower=float(args.ambiguity_lower),
        upper=float(args.ambiguity_upper),
    )
    final_threshold, final_val_best_f1 = best_f1_threshold(y_val, final_val_scores)
    final_val_metrics = binary_metrics(y_val, final_val_scores, threshold=final_threshold)
    final_test_metrics = binary_metrics(y_test, final_test_scores, threshold=final_threshold)
    final_test_best_threshold, final_test_best_f1 = best_f1_threshold(y_test, final_test_scores)
    final_test_metrics_best = binary_metrics(y_test, final_test_scores, threshold=final_test_best_threshold)
    classical_test_metrics = binary_metrics(y_test, classical_test_scores, threshold=classical_threshold)

    final_checkpoint_payload = dict(best_checkpoint)
    final_checkpoint_payload["threshold"] = float(final_threshold)
    final_checkpoint_payload["raw_threshold"] = float(best_threshold)
    final_checkpoint_payload["calibration"] = {
        "kind": "platt",
        "coef": float(calibration["coef"]),
        "intercept": float(calibration["intercept"]),
        "fitted_on": "val_ambiguity_band",
        "generated_at": now_iso(),
    } if calibration else None
    final_checkpoint_payload["ambiguity_lower"] = float(args.ambiguity_lower)
    final_checkpoint_payload["ambiguity_upper"] = float(args.ambiguity_upper)
    final_checkpoint_payload["source_checkpoint"] = str(args.classical_checkpoint)
    final_checkpoint_payload["focus_samples"] = len(focus_rows)
    final_checkpoint_payload["focus_positive"] = int(np.sum(focus_labels == 1.0))
    final_checkpoint_payload["focus_negative"] = int(np.sum(focus_labels == 0.0))
    atomic_torch_save(calibrated_checkpoint_path, final_checkpoint_payload)

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
            "residual_alpha_init": float(args.residual_alpha_init),
            "ambiguity_lower": float(args.ambiguity_lower),
            "ambiguity_upper": float(args.ambiguity_upper),
        },
        "dataset": {
            "train_tces": len(train_rows),
            "val_tces": len(val_rows),
            "test_tces": len(test_rows),
            "focus_tces": len(focus_rows),
            "focus_positive": int(np.sum(focus_labels == 1.0)),
            "focus_negative": int(np.sum(focus_labels == 0.0)),
            "val_gate_samples": int(np.sum(val_gate_mask)),
            "test_gate_samples": int(np.sum(test_gate_mask)),
        },
        "classical_baseline": {
            "threshold": float(classical_threshold),
            "test_metrics": classical_test_metrics,
        },
        "training": {
            "best_epoch": best_epoch,
            "best_threshold_raw": best_threshold,
            "final_threshold": float(final_threshold),
            "final_val_best_f1": float(final_val_best_f1),
            "test_metrics_threshold_best_on_val": final_test_metrics,
            "test_metrics_best_on_test": final_test_metrics_best,
            "test_best_threshold": float(final_test_best_threshold),
            "test_best_f1": float(final_test_best_f1),
            "history": history,
            "best_checkpoint": str(best_checkpoint_path),
            "best_calibrated_checkpoint": str(calibrated_checkpoint_path),
            "latest_checkpoint": str(latest_checkpoint_path),
            "calibration": final_checkpoint_payload.get("calibration"),
            "residual_alpha_final": float(torch.sigmoid(model.residual_gate_logit).detach().cpu().item()),
        },
    }
    write_json(run_dir / "report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
