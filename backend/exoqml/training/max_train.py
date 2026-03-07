from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from exoqml.services.preprocess import preprocess_lightcurve
from exoqml.training.archive import fetch_dr24_tce_labels
from exoqml.training.hardware import detect_hardware, recommended_num_workers
from exoqml.training.metrics import best_f1_threshold, binary_metrics
from exoqml.training.model import TransitResNet1D

try:
    import lightkurve as lk

    HAS_LIGHTKURVE = True
except Exception:
    HAS_LIGHTKURVE = False
    lk = None


@dataclass(slots=True)
class ManifestRow:
    star_id: str
    label: int
    split: str
    status: str
    npz_path: str
    num_points: int
    error: str
    updated_at: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExoQML aggressive training pipeline.")
    parser.add_argument("--dataset-root", type=Path, default=Path("./data/train_max"))
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--disk-utilization", type=float, default=0.90)
    parser.add_argument("--reserve-free-gb", type=float, default=60.0)
    parser.add_argument("--max-stars", type=int, default=0, help="0 means all available stars")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=0, help="0 means auto")
    parser.add_argument("--num-workers", type=int, default=0, help="0 means auto")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-train-samples", type=int, default=512)
    parser.add_argument("--min-val-samples", type=int, default=32)
    parser.add_argument("--min-test-samples", type=int, default=32)
    parser.add_argument("--enable-compile", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Disable automatic resume from previous interrupted run")
    parser.add_argument("--download-timeout-sec", type=int, default=180)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_manifest(path: Path) -> dict[str, ManifestRow]:
    if not path.exists():
        return {}
    rows: dict[str, ManifestRow] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = ManifestRow(
                star_id=str(raw["star_id"]),
                label=int(raw["label"]),
                split=str(raw["split"]),
                status=str(raw["status"]),
                npz_path=str(raw["npz_path"]),
                num_points=int(raw.get("num_points") or 0),
                error=str(raw.get("error") or ""),
                updated_at=str(raw.get("updated_at") or ""),
            )
            rows[row.star_id] = row
    return rows


def save_manifest(path: Path, rows: dict[str, ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "star_id",
                "label",
                "split",
                "status",
                "npz_path",
                "num_points",
                "error",
                "updated_at",
            ],
        )
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda r: int(r.star_id)):
            writer.writerow(asdict(row))


def resolve_run_dir(dataset_root: Path, resume_enabled: bool) -> tuple[Path, Path]:
    pointer = dataset_root / "current_run.json"
    if resume_enabled and pointer.exists():
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            run_dir = Path(payload["run_dir"])
            status = str(payload.get("status", "running"))
            if status != "completed" and run_dir.exists():
                return run_dir.resolve(), pointer
        except Exception:
            pass

    run_dir = (dataset_root / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        pointer,
        {
            "run_dir": str(run_dir),
            "status": "running",
            "updated_at": now_iso(),
        },
    )
    return run_dir, pointer


def aggregate_star_labels(rows: list[dict[str, str | int]]) -> tuple[dict[str, int], dict[str, int]]:
    classes_by_star: dict[str, set[str]] = {}
    raw_counts = {"PC": 0, "AFP": 0, "NTP": 0}

    for row in rows:
        star_id = str(int(row["kepid"]))
        label = str(row["av_training_set"]).upper()
        if label not in raw_counts:
            continue
        raw_counts[label] += 1
        classes_by_star.setdefault(star_id, set()).add(label)

    labels: dict[str, int] = {}
    for star_id, classes in classes_by_star.items():
        labels[star_id] = 1 if "PC" in classes else 0

    summary = {
        "raw_pc": raw_counts["PC"],
        "raw_afp": raw_counts["AFP"],
        "raw_ntp": raw_counts["NTP"],
        "stars_total": len(labels),
        "stars_positive": int(sum(labels.values())),
        "stars_negative": int(len(labels) - sum(labels.values())),
    }
    return labels, summary


def split_by_star(star_labels: dict[str, int], seed: int, train_ratio: float, val_ratio: float) -> dict[str, str]:
    stars = np.array(sorted(star_labels.keys(), key=int))
    rng = np.random.default_rng(seed)
    rng.shuffle(stars)

    n = len(stars)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_train = max(1, min(n - 2, n_train))
    n_val = max(1, min(n - n_train - 1, n_val))
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train = max(1, n_train - 1)

    result: dict[str, str] = {}
    for star in stars[:n_train]:
        result[str(star)] = "train"
    for star in stars[n_train : n_train + n_val]:
        result[str(star)] = "val"
    for star in stars[n_train + n_val :]:
        result[str(star)] = "test"
    return result


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def select_device(preferred: str) -> str:
    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("CUDA requested but no GPU is available to torch")
    return "cuda" if torch.cuda.is_available() else "cpu"


def disk_budget_bytes(dataset_root: Path, reserve_free_gb: float, disk_utilization: float) -> int:
    usage = shutil.disk_usage(dataset_root)
    reserve_bytes = int(reserve_free_gb * (1024**3))
    by_reserve = max(0, usage.free - reserve_bytes)
    by_util = max(0, int(usage.free * disk_utilization))
    return min(by_reserve, by_util)


def process_star(
    star_id: str,
    label: int,
    split: str,
    raw_dir: Path,
    processed_dir: Path,
    max_points: int,
    _timeout_seconds: int,
) -> ManifestRow:
    npz_path = processed_dir / f"kic_{star_id}.npz"

    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=False) as payload:
                points = int(payload["flux"].shape[0])
            return ManifestRow(
                star_id=star_id,
                label=label,
                split=split,
                status="ready",
                npz_path=str(npz_path),
                num_points=points,
                error="",
                updated_at=now_iso(),
            )
        except Exception:
            npz_path.unlink(missing_ok=True)

    query = f"KIC {star_id}"
    try:
        search = lk.search_lightcurve(query, mission="Kepler")
        if len(search) == 0:
            return ManifestRow(
                star_id=star_id,
                label=label,
                split=split,
                status="no_data",
                npz_path="",
                num_points=0,
                error="No Kepler light curves found",
                updated_at=now_iso(),
            )

        collection = search.download_all(download_dir=str(raw_dir), quality_bitmask="default")
        if collection is None or len(collection) == 0:
            return ManifestRow(
                star_id=star_id,
                label=label,
                split=split,
                status="download_empty",
                npz_path="",
                num_points=0,
                error="Download returned empty collection",
                updated_at=now_iso(),
            )

        stitched = collection.stitch()
        time_arr = np.asarray(stitched.time.value, dtype=float)
        flux_arr = np.asarray(stitched.flux.value, dtype=float)
        proc_time, proc_flux, _ = preprocess_lightcurve(time_arr, flux_arr, max_points=max_points)
        np.savez_compressed(
            npz_path,
            time=proc_time.astype(np.float32),
            flux=proc_flux.astype(np.float32),
            label=np.array([label], dtype=np.int8),
        )
        return ManifestRow(
            star_id=star_id,
            label=label,
            split=split,
            status="ready",
            npz_path=str(npz_path),
            num_points=int(proc_flux.shape[0]),
            error="",
            updated_at=now_iso(),
        )
    except Exception as exc:
        return ManifestRow(
            star_id=star_id,
            label=label,
            split=split,
            status="error",
            npz_path="",
            num_points=0,
            error=f"{exc.__class__.__name__}: {exc}",
            updated_at=now_iso(),
        )


class NPZDataset(Dataset):
    def __init__(self, rows: list[ManifestRow], preload: bool = True) -> None:
        self.rows = rows
        self.preload = preload
        self.cached_x: list[np.ndarray] | None = None
        self.cached_y: list[np.ndarray] | None = None

        if preload:
            x_cache: list[np.ndarray] = []
            y_cache: list[np.ndarray] = []
            for row in rows:
                with np.load(row.npz_path, allow_pickle=False) as payload:
                    flux = payload["flux"].astype(np.float32)
                    x = (1.0 - flux).astype(np.float32)
                    y = np.array([row.label], dtype=np.float32)
                x_cache.append(x)
                y_cache.append(y)
            self.cached_x = x_cache
            self.cached_y = y_cache

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cached_x is not None and self.cached_y is not None:
            x = self.cached_x[index]
            y = self.cached_y[index]
        else:
            row = self.rows[index]
            with np.load(row.npz_path, allow_pickle=False) as payload:
                flux = payload["flux"].astype(np.float32)
                x = (1.0 - flux).astype(np.float32)
                y = np.array([row.label], dtype=np.float32)

        return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y)


def auto_batch_size(model: nn.Module, seq_len: int, device: str) -> int:
    if device == "cpu":
        return 256

    criterion = nn.BCEWithLogitsLoss()
    trial = 64
    best = 32
    model.train()

    while trial <= 2048:
        try:
            x = torch.randn(trial, 1, seq_len, device=device)
            y = torch.randint(0, 2, (trial, 1), dtype=torch.float32, device=device)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            best = trial
            trial *= 2
            del x, y, out, loss
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                break
            raise

    return max(16, int(best * 0.75))


def create_dataloader(
    dataset: NPZDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: str,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(**kwargs)


def forward_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    losses: list[float] = []
    all_targets: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if is_train and optimizer is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        targets = y.detach().cpu().numpy().reshape(-1)
        all_scores.append(probs)
        all_targets.append(targets)
        losses.append(float(loss.detach().cpu().item()))

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.array([], dtype=np.float32)
    y_score = np.concatenate(all_scores, axis=0) if all_scores else np.array([], dtype=np.float32)
    return mean_loss, y_true, y_score


def train(
    train_rows: list[ManifestRow],
    val_rows: list[ManifestRow],
    test_rows: list[ManifestRow],
    args: argparse.Namespace,
    run_dir: Path,
    device: str,
    num_workers: int,
    resume_enabled: bool,
    state_path: Path,
) -> dict[str, Any]:
    train_ds = NPZDataset(train_rows, preload=True)
    val_ds = NPZDataset(val_rows, preload=True)
    test_ds = NPZDataset(test_rows, preload=True)

    model = TransitResNet1D(base_channels=64, dropout=0.2).to(device)
    seq_len = train_ds[0][0].shape[-1]

    if args.batch_size > 0:
        batch_size = args.batch_size
    else:
        batch_size = auto_batch_size(model, seq_len=seq_len, device=device)

    train_loader = create_dataloader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, device=device)
    val_loader = create_dataloader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, device=device)
    test_loader = create_dataloader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, device=device)

    positives = sum(row.label for row in train_rows)
    negatives = len(train_rows) - positives
    pos_weight_value = float(negatives / max(1, positives))
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_key = -1.0
    best_epoch = 0
    best_threshold = 0.5
    patience_counter = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1

    best_checkpoint = run_dir / "best_model.pt"
    latest_checkpoint = run_dir / "latest_model.pt"

    if resume_enabled and latest_checkpoint.exists():
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint and use_amp:
            scaler.load_state_dict(checkpoint["scaler"])
        history = list(checkpoint.get("history", []))
        best_key = float(checkpoint.get("best_key", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_threshold = float(checkpoint.get("best_threshold", 0.5))
        patience_counter = int(checkpoint.get("patience_counter", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resuming training from epoch {start_epoch}")

    if args.enable_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[assignment]
        except Exception:
            pass

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_loss, y_train, score_train = forward_epoch(
            train_loader,
            model,
            criterion,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
        )
        scheduler.step()
        val_loss, y_val, score_val = forward_epoch(
            val_loader,
            model,
            criterion,
            optimizer=None,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
        )

        train_metrics = binary_metrics(y_train, score_train, threshold=0.5)
        val_metrics_05 = binary_metrics(y_val, score_val, threshold=0.5)
        thr, best_f1 = best_f1_threshold(y_val, score_val)
        val_metrics_best = binary_metrics(y_val, score_val, threshold=thr)

        pr_auc = val_metrics_best["pr_auc"]
        metric_key = pr_auc if not math.isnan(pr_auc) else val_metrics_best["f1"]

        epoch_payload = {
            "epoch": epoch,
            "seconds": round(time.time() - epoch_start, 2),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_f1": float(train_metrics["f1"]),
            "val_f1_thr05": float(val_metrics_05["f1"]),
            "val_f1_best": float(best_f1),
            "val_best_threshold": float(thr),
            "val_pr_auc": float(val_metrics_best["pr_auc"]),
            "val_roc_auc": float(val_metrics_best["roc_auc"]),
            "val_recall_best": float(val_metrics_best["recall"]),
            "val_precision_best": float(val_metrics_best["precision"]),
        }
        history.append(epoch_payload)
        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_pr_auc={val_metrics_best['pr_auc']:.4f} val_f1={best_f1:.4f} thr={thr:.2f}"
        )

        if metric_key > best_key:
            best_key = metric_key
            best_epoch = epoch
            best_threshold = float(thr)
            patience_counter = 0
            atomic_torch_save(
                best_checkpoint,
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "threshold": best_threshold,
                    "val_metrics": val_metrics_best,
                },
            )
        else:
            patience_counter += 1

        atomic_torch_save(
            latest_checkpoint,
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "threshold": float(thr),
                "val_pr_auc": float(val_metrics_best["pr_auc"]),
                "val_f1": float(best_f1),
                "history": history,
                "best_key": best_key,
                "best_epoch": best_epoch,
                "best_threshold": best_threshold,
                "patience_counter": patience_counter,
            },
        )

        atomic_write_json(
            state_path,
            {
                "phase": "training",
                "run_dir": str(run_dir),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "updated_at": now_iso(),
            },
        )

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])

    test_loss, y_test, score_test = forward_epoch(
        test_loader,
        model,
        criterion,
        optimizer=None,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
    )
    test_metrics = binary_metrics(y_test, score_test, threshold=best_threshold)
    test_threshold, test_best_f1 = best_f1_threshold(y_test, score_test)
    test_metrics_best = binary_metrics(y_test, score_test, threshold=test_threshold)

    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seq_len": int(seq_len),
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "test_loss": float(test_loss),
        "test_metrics_threshold_best_on_val": test_metrics,
        "test_metrics_best_on_test": test_metrics_best,
        "test_best_threshold": float(test_threshold),
        "test_best_f1": float(test_best_f1),
        "history": history,
        "best_checkpoint": str(best_checkpoint),
        "latest_checkpoint": str(latest_checkpoint),
    }


def main() -> None:
    args = parse_args()
    if not HAS_LIGHTKURVE:
        raise RuntimeError("lightkurve is required for training. Install backend with [science] extras.")

    resume_enabled = not args.no_resume
    seed_everything(args.seed)

    dataset_root = args.dataset_root.resolve()
    raw_dir = dataset_root / "raw"
    processed_dir = dataset_root / "processed"
    manifest_path = dataset_root / "manifest.csv"
    split_path = dataset_root / "split.json"
    run_dir, pointer_path = resolve_run_dir(dataset_root, resume_enabled=resume_enabled)
    state_path = run_dir / "state.json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    atomic_write_json(
        state_path,
        {
            "phase": "initializing",
            "run_dir": str(run_dir),
            "updated_at": now_iso(),
        },
    )

    hardware = detect_hardware(dataset_root)
    device = select_device(args.device)
    num_workers = args.num_workers if args.num_workers > 0 else recommended_num_workers(hardware.cpu_cores_logical)

    budget_bytes = disk_budget_bytes(dataset_root, args.reserve_free_gb, args.disk_utilization)
    if budget_bytes <= 0:
        raise RuntimeError("No disk budget available with current reserve settings")

    rows = fetch_dr24_tce_labels()
    star_labels, label_summary = aggregate_star_labels(rows)
    split_map = split_by_star(star_labels, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    write_json(split_path, {"generated_at": now_iso(), "split_map": split_map})

    manifest = load_manifest(manifest_path)
    candidates = sorted(star_labels.keys(), key=int)
    if args.max_stars > 0:
        candidates = candidates[: args.max_stars]

    print(f"Detected device={device} workers={num_workers}")
    print(f"Disk free={hardware.disk_free_gb:.1f} GB | budget for this run={budget_bytes / (1024**3):.1f} GB")
    print(
        f"Catalog stars={label_summary['stars_total']} "
        f"(pos={label_summary['stars_positive']} neg={label_summary['stars_negative']})"
    )
    print(f"Target stars this run={len(candidates)}")

    downloaded = 0
    skipped = 0
    failures = 0
    for idx, star_id in enumerate(candidates, start=1):
        label = star_labels[star_id]
        split = split_map[star_id]

        existing = manifest.get(star_id)
        if existing and existing.status == "ready" and existing.npz_path and Path(existing.npz_path).exists():
            skipped += 1
            continue

        if idx % 10 == 0:
            used = directory_size(dataset_root)
            if used >= budget_bytes:
                print("Disk budget reached; stopping data acquisition.")
                break

        row = process_star(
            star_id=star_id,
            label=label,
            split=split,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            max_points=args.max_points,
            _timeout_seconds=args.download_timeout_sec,
        )
        manifest[star_id] = row
        if row.status == "ready":
            downloaded += 1
        else:
            failures += 1

        save_manifest(manifest_path, manifest)
        atomic_write_json(
            state_path,
            {
                "phase": "ingestion",
                "run_dir": str(run_dir),
                "progress_index": idx,
                "candidates_total": len(candidates),
                "downloaded": downloaded,
                "skipped": skipped,
                "failures": failures,
                "updated_at": now_iso(),
            },
        )
        if idx % 20 == 0:
            print(f"[ingest] {idx}/{len(candidates)} ready={downloaded} fail={failures} skipped={skipped}")

    save_manifest(manifest_path, manifest)

    ready_rows = [row for row in manifest.values() if row.status == "ready" and row.npz_path and Path(row.npz_path).exists()]
    train_rows = [row for row in ready_rows if row.split == "train"]
    val_rows = [row for row in ready_rows if row.split == "val"]
    test_rows = [row for row in ready_rows if row.split == "test"]

    if (
        len(train_rows) < args.min_train_samples
        or len(val_rows) < args.min_val_samples
        or len(test_rows) < args.min_test_samples
    ):
        report = {
            "status": "insufficient_data",
            "generated_at": now_iso(),
            "hardware": hardware.as_dict(),
            "device": device,
            "dataset": {
                "ready_total": len(ready_rows),
                "train": len(train_rows),
                "val": len(val_rows),
                "test": len(test_rows),
                "downloaded_now": downloaded,
                "skipped_ready": skipped,
                "failures": failures,
            },
            "message": (
                "Insufficient data to start a robust train run. "
                "Re-run to continue ingesting more stars or lower "
                "--min-train-samples/--min-val-samples/--min-test-samples."
            ),
        }
        write_json(run_dir / "report.json", report)
        atomic_write_json(
            pointer_path,
            {
                "run_dir": str(run_dir),
                "status": "paused_insufficient_data",
                "updated_at": now_iso(),
            },
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print(f"Starting training with samples: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    train_report = train(
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        args=args,
        run_dir=run_dir,
        device=device,
        num_workers=num_workers,
        resume_enabled=resume_enabled,
        state_path=state_path,
    )

    report = {
        "status": "completed",
        "generated_at": now_iso(),
        "hardware": hardware.as_dict(),
        "device": device,
        "config": vars(args),
        "label_summary": label_summary,
        "dataset": {
            "ready_total": len(ready_rows),
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "downloaded_now": downloaded,
            "skipped_ready": skipped,
            "failures": failures,
            "dataset_size_gb": directory_size(dataset_root) / (1024**3),
        },
        "training": train_report,
    }
    write_json(run_dir / "report.json", report)
    atomic_write_json(
        state_path,
        {
            "phase": "completed",
            "run_dir": str(run_dir),
            "updated_at": now_iso(),
        },
    )
    atomic_write_json(
        pointer_path,
        {
            "run_dir": str(run_dir),
            "status": "completed",
            "updated_at": now_iso(),
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
