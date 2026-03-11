from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader, Dataset

from exoqml.transit_features import (
    GLOBAL_VIEW_BINS,
    LOCAL_VIEW_BINS,
    SCALAR_FEATURE_DIM,
    build_scalar_features,
    build_tce_views,
)
from exoqml.services.preprocess import preprocess_lightcurve
from exoqml.training.archive import fetch_dr24_tce_catalog
from exoqml.training.hardware import detect_hardware, recommended_num_workers
from exoqml.training.metrics import best_f1_threshold, binary_metrics
from exoqml.training.model import TransitMultiViewNet

warnings.filterwarnings("ignore", message="Warning: the tpfmodel submodule is not available", category=UserWarning)
warnings.filterwarnings("ignore", message="`torch.cuda.amp.GradScaler\\(args\\.\\.\\.\\)` is deprecated", category=FutureWarning)


def configure_runtime_environment() -> dict[str, str]:
    backend_root = Path(__file__).resolve().parents[2]
    runtime_root = backend_root / "data" / "runtime_cache"
    env_dirs = {
        "runtime_root": str(runtime_root),
        "home": str(runtime_root / "home"),
        "tmp": str(runtime_root / "tmp"),
        "xdg_cache": str(runtime_root / "xdg_cache"),
        "xdg_config": str(runtime_root / "xdg_config"),
        "astropy_cache": str(runtime_root / "astropy_cache"),
        "astropy_config": str(runtime_root / "astropy_config"),
        "lightkurve_cache": str(runtime_root / "lightkurve_cache"),
        "matplotlib_config": str(runtime_root / "mplconfig"),
        "torch_home": str(runtime_root / "torch_home"),
        "pooch_home": str(runtime_root / "pooch"),
        "joblib_tmp": str(runtime_root / "joblib_tmp"),
    }

    for path in env_dirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    os.environ["TMP"] = env_dirs["tmp"]
    os.environ["TEMP"] = env_dirs["tmp"]
    os.environ["HOME"] = env_dirs["home"]
    os.environ["USERPROFILE"] = env_dirs["home"]
    if os.name != "nt":
        os.environ["XDG_CACHE_HOME"] = env_dirs["xdg_cache"]
        os.environ["XDG_CONFIG_HOME"] = env_dirs["xdg_config"]
    else:
        os.environ.pop("XDG_CACHE_HOME", None)
        os.environ.pop("XDG_CONFIG_HOME", None)

    os.environ["ASTROPY_CACHE_DIR"] = env_dirs["astropy_cache"]
    os.environ["ASTROPY_CONFIG_DIR"] = env_dirs["astropy_config"]
    os.environ["LIGHTKURVE_CACHE_DIR"] = env_dirs["lightkurve_cache"]
    os.environ["MPLCONFIGDIR"] = env_dirs["matplotlib_config"]
    os.environ["TORCH_HOME"] = env_dirs["torch_home"]
    os.environ["POOCH_HOME"] = env_dirs["pooch_home"]
    os.environ["JOBLIB_TEMP_FOLDER"] = env_dirs["joblib_tmp"]
    return env_dirs


RUNTIME_ENV_DIRS = configure_runtime_environment()

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


@dataclass(slots=True)
class TCERow:
    tce_id: str
    star_id: str
    label: int
    label_name: str
    split: str
    period: float
    duration_hours: float
    epoch: float
    depth_ppm: float
    model_snr: float


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExoQML aggressive training pipeline.")
    parser.add_argument("--dataset-root", type=Path, default=Path("./data/train_max"))
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--disk-utilization", type=float, default=0.95)
    parser.add_argument("--reserve-free-gb", type=float, default=40.0)
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
    parser.add_argument("--loss", type=str, choices=["bce", "focal"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--global-view-bins", type=int, default=GLOBAL_VIEW_BINS)
    parser.add_argument("--local-view-bins", type=int, default=LOCAL_VIEW_BINS)
    parser.add_argument("--enable-compile", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Disable automatic resume from previous interrupted run")
    parser.add_argument("--skip-ingestion", action="store_true", help="Train only from already ready samples in manifest.csv")
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


def load_tce_catalog(path: Path) -> list[TCERow]:
    if not path.exists():
        return []
    rows: list[TCERow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(
                TCERow(
                    tce_id=str(raw["tce_id"]),
                    star_id=str(raw["star_id"]),
                    label=int(raw["label"]),
                    label_name=str(raw["label_name"]),
                    split=str(raw["split"]),
                    period=float(raw["period"]),
                    duration_hours=float(raw["duration_hours"]),
                    epoch=float(raw["epoch"]),
                    depth_ppm=float(raw["depth_ppm"]),
                    model_snr=float(raw["model_snr"]),
                )
            )
    return rows


def save_tce_catalog(path: Path, rows: list[TCERow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tce_id",
                "star_id",
                "label",
                "label_name",
                "split",
                "period",
                "duration_hours",
                "epoch",
                "depth_ppm",
                "model_snr",
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (int(item.star_id), item.tce_id)):
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
    rng = np.random.default_rng(seed)

    def assign_group(stars: list[str]) -> dict[str, str]:
        shuffled = np.array(sorted(stars, key=int))
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n == 0:
            return {}
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))
        elif n == 2:
            n_train = 1
            n_val = 0
        else:
            n_train = 1
            n_val = 0
        result_local: dict[str, str] = {}
        for star in shuffled[:n_train]:
            result_local[str(star)] = "train"
        for star in shuffled[n_train : n_train + n_val]:
            result_local[str(star)] = "val"
        for star in shuffled[n_train + n_val :]:
            result_local[str(star)] = "test"
        return result_local

    positive_stars = [star for star, label in star_labels.items() if label == 1]
    negative_stars = [star for star, label in star_labels.items() if label == 0]
    result: dict[str, str] = {}
    result.update(assign_group(positive_stars))
    result.update(assign_group(negative_stars))
    return result


def build_tce_catalog(
    raw_rows: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[TCERow], dict[str, int | float]]:
    star_labels: dict[str, int] = {}
    for row in raw_rows:
        star_id = str(int(row["kepid"]))
        label_name = str(row["av_training_set"]).upper()
        current = star_labels.get(star_id, 0)
        star_labels[star_id] = 1 if label_name == "PC" else current

    split_map = split_by_star(star_labels, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
    rows: list[TCERow] = []
    raw_pc = 0
    raw_afp = 0
    raw_ntp = 0

    for row in raw_rows:
        star_id = str(int(row["kepid"]))
        tce_num = int(row["tce_plnt_num"])
        label_name = str(row["av_training_set"]).upper()
        if label_name == "PC":
            raw_pc += 1
        elif label_name == "AFP":
            raw_afp += 1
        elif label_name == "NTP":
            raw_ntp += 1

        rows.append(
            TCERow(
                tce_id=f"{star_id}_{tce_num}",
                star_id=star_id,
                label=1 if label_name == "PC" else 0,
                label_name=label_name,
                split=split_map[star_id],
                period=float(row["tce_period"]),
                duration_hours=float(row["tce_duration"]),
                epoch=float(row["tce_time0bk"]),
                depth_ppm=float(row["tce_depth"]),
                model_snr=float(row["tce_model_snr"]),
            )
        )

    summary: dict[str, int | float] = {
        "raw_pc": raw_pc,
        "raw_afp": raw_afp,
        "raw_ntp": raw_ntp,
        "tces_total": len(rows),
        "tces_positive": int(sum(item.label for item in rows)),
        "tces_negative": int(len(rows) - sum(item.label for item in rows)),
        "stars_total": len(star_labels),
        "stars_positive": int(sum(star_labels.values())),
        "stars_negative": int(len(star_labels) - sum(star_labels.values())),
    }
    return rows, summary


def summarize_tce_rows(rows: list[TCERow]) -> dict[str, int]:
    return {
        "raw_pc": sum(1 for row in rows if row.label_name == "PC"),
        "raw_afp": sum(1 for row in rows if row.label_name == "AFP"),
        "raw_ntp": sum(1 for row in rows if row.label_name == "NTP"),
        "tces_total": len(rows),
        "tces_positive": int(sum(row.label for row in rows)),
        "tces_negative": int(len(rows) - sum(row.label for row in rows)),
        "stars_total": len({row.star_id for row in rows}),
        "stars_positive": len({row.star_id for row in rows if row.label == 1}),
        "stars_negative": len({row.star_id for row in rows if row.label == 0}),
    }


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


def should_retry_lightkurve_error(exc: Exception) -> bool:
    text = str(exc).lower()
    retry_tokens = (
        "not recognized as a supported data product",
        "this file may be corrupt",
        "file may have been truncated",
        "error in reading data product",
        "connection reset",
        "timed out",
    )
    return any(token in text for token in retry_tokens)


def cleanup_star_download_cache(raw_dir: Path, star_id: str) -> int:
    mast_dir = raw_dir / "mastDownload"
    if not mast_dir.exists():
        return 0

    normalized_id = f"{int(star_id):09d}"
    patterns = (
        f"*kplr{normalized_id}*",
        f"*{star_id}*",
    )
    removed = 0
    for mission_dir in (mast_dir / "HLSP", mast_dir / "Kepler"):
        if not mission_dir.exists():
            continue
        for pattern in patterns:
            for path in mission_dir.glob(pattern):
                try:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
                    removed += 1
                except Exception:
                    continue
    return removed


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
        # Prefer standard Kepler long-cadence products and avoid IRIS HLSP,
        # which often yields unsupported/corrupted FITS for lightkurve.
        search = lk.search_lightcurve(query, mission="Kepler", author="Kepler", exptime=1800)
        if len(search) == 0:
            search = lk.search_lightcurve(query, mission="Kepler", author="Kepler")
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

        collection = None
        download_error: Exception | None = None
        for attempt in range(2):
            try:
                collection = search.download_all(download_dir=str(raw_dir), quality_bitmask="default")
                download_error = None
                break
            except Exception as exc:
                download_error = exc
                if attempt == 0 and should_retry_lightkurve_error(exc):
                    cleanup_star_download_cache(raw_dir=raw_dir, star_id=star_id)
                    continue
                raise

        if download_error is not None:
            raise download_error
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


def load_star_series_cache(rows: dict[str, ManifestRow]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for star_id, row in rows.items():
        with np.load(row.npz_path, allow_pickle=False) as payload:
            time_arr = payload["time"].astype(np.float32)
            flux_arr = payload["flux"].astype(np.float32)
        cache[star_id] = (time_arr, flux_arr)
    return cache


class TCEDataset(Dataset):
    def __init__(
        self,
        rows: list[TCERow],
        star_series: dict[str, tuple[np.ndarray, np.ndarray]],
        global_bins: int,
        local_bins: int,
        preload: bool = True,
    ) -> None:
        self.rows = rows
        self.star_series = star_series
        self.global_bins = global_bins
        self.local_bins = local_bins
        self.preload = preload
        self.cached_global: list[np.ndarray] | None = None
        self.cached_local: list[np.ndarray] | None = None
        self.cached_scalar: list[np.ndarray] | None = None
        self.cached_y: list[np.ndarray] | None = None

        if preload:
            global_cache: list[np.ndarray] = []
            local_cache: list[np.ndarray] = []
            scalar_cache: list[np.ndarray] = []
            y_cache: list[np.ndarray] = []
            for row in rows:
                global_view, local_view, scalar_features, y = self._build_sample(row)
                global_cache.append(global_view)
                local_cache.append(local_view)
                scalar_cache.append(scalar_features)
                y_cache.append(y)
            self.cached_global = global_cache
            self.cached_local = local_cache
            self.cached_scalar = scalar_cache
            self.cached_y = y_cache

    def __len__(self) -> int:
        return len(self.rows)

    def _build_sample(self, row: TCERow) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        time_arr, flux_arr = self.star_series[row.star_id]
        global_view, local_view = build_tce_views(
            time=time_arr.astype(np.float64),
            flux=flux_arr.astype(np.float64),
            period=row.period,
            epoch=row.epoch,
            duration_hours=row.duration_hours,
            global_bins=self.global_bins,
            local_bins=self.local_bins,
        )
        scalar_features = build_scalar_features(
            period=row.period,
            duration_hours=row.duration_hours,
            depth_ppm=row.depth_ppm,
            model_snr=row.model_snr,
        )
        y = np.array([row.label], dtype=np.float32)
        return global_view, local_view, scalar_features, y

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.cached_global is not None
            and self.cached_local is not None
            and self.cached_scalar is not None
            and self.cached_y is not None
        ):
            global_view = self.cached_global[index]
            local_view = self.cached_local[index]
            scalar_features = self.cached_scalar[index]
            y = self.cached_y[index]
        else:
            global_view, local_view, scalar_features, y = self._build_sample(self.rows[index])

        global_tensor = torch.as_tensor(global_view, dtype=torch.float32).unsqueeze(0).contiguous().clone()
        local_tensor = torch.as_tensor(local_view, dtype=torch.float32).unsqueeze(0).contiguous().clone()
        scalar_tensor = torch.as_tensor(scalar_features, dtype=torch.float32).contiguous().clone()
        y_tensor = torch.as_tensor(y, dtype=torch.float32).contiguous().clone()
        return global_tensor, local_tensor, scalar_tensor, y_tensor


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.tensor([1.0]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nnf.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal = torch.pow(1.0 - pt, self.gamma)
        return (focal * bce).mean()


def auto_batch_size(model: nn.Module, global_len: int, local_len: int, scalar_dim: int, device: str) -> int:
    if device == "cpu":
        return 256

    criterion = nn.BCEWithLogitsLoss()
    trial = 64
    best = 32
    model.train()

    while trial <= 2048:
        try:
            global_view = torch.randn(trial, 1, global_len, device=device)
            local_view = torch.randn(trial, 1, local_len, device=device)
            scalar_features = torch.randn(trial, scalar_dim, device=device)
            y = torch.randint(0, 2, (trial, 1), dtype=torch.float32, device=device)
            out = model(global_view, local_view, scalar_features)
            loss = criterion(out, y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            best = trial
            trial *= 2
            del global_view, local_view, scalar_features, y, out, loss
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                break
            raise

    return max(16, int(best * 0.75))


def safe_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    global_items = [item[0].contiguous().clone() for item in batch]
    local_items = [item[1].contiguous().clone() for item in batch]
    scalar_items = [item[2].contiguous().clone() for item in batch]
    y_items = [item[3].contiguous().clone() for item in batch]
    return (
        torch.stack(global_items, dim=0),
        torch.stack(local_items, dim=0),
        torch.stack(scalar_items, dim=0),
        torch.stack(y_items, dim=0),
    )


def create_dataloader(
    dataset: TCEDataset,
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
        "collate_fn": safe_collate,
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

    for batch_index, (global_view, local_view, scalar_features, y) in enumerate(loader):
        global_view = global_view.to(device, non_blocking=True)
        local_view = local_view.to(device, non_blocking=True)
        scalar_features = scalar_features.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            logits = model(global_view, local_view, scalar_features)
            loss = criterion(logits, y)

        if is_train and optimizer is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if device == "cuda" and batch_index == 0:
                allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                reserved_gb = torch.cuda.memory_reserved() / (1024**3)
                print(f"[gpu] allocated_gb={allocated_gb:.3f} reserved_gb={reserved_gb:.3f}")

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
    train_rows: list[TCERow],
    val_rows: list[TCERow],
    test_rows: list[TCERow],
    ready_manifest_rows: dict[str, ManifestRow],
    args: argparse.Namespace,
    run_dir: Path,
    device: str,
    num_workers: int,
    resume_enabled: bool,
    state_path: Path,
) -> dict[str, Any]:
    star_ids = sorted({row.star_id for row in (train_rows + val_rows + test_rows)})
    star_cache_rows = {star_id: ready_manifest_rows[star_id] for star_id in star_ids}
    star_series = load_star_series_cache(star_cache_rows)

    global_bins = int(args.global_view_bins)
    local_bins = int(args.local_view_bins)
    train_ds = TCEDataset(train_rows, star_series=star_series, global_bins=global_bins, local_bins=local_bins, preload=True)
    val_ds = TCEDataset(val_rows, star_series=star_series, global_bins=global_bins, local_bins=local_bins, preload=True)
    test_ds = TCEDataset(test_rows, star_series=star_series, global_bins=global_bins, local_bins=local_bins, preload=True)
    model = TransitMultiViewNet(scalar_dim=SCALAR_FEATURE_DIM, base_channels=32, dropout=0.2).to(device)

    if args.batch_size > 0:
        batch_size = args.batch_size
    else:
        batch_size = auto_batch_size(
            model,
            global_len=global_bins,
            local_len=local_bins,
            scalar_dim=SCALAR_FEATURE_DIM,
            device=device,
        )

    train_loader = create_dataloader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, device=device)
    val_loader = create_dataloader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, device=device)
    test_loader = create_dataloader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, device=device)

    positives = sum(row.label for row in train_rows)
    negatives = len(train_rows) - positives
    pos_weight_value = float(negatives / max(1, positives))
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    if args.loss == "focal":
        criterion: nn.Module = FocalBCEWithLogitsLoss(gamma=float(args.focal_gamma), pos_weight=pos_weight)
    else:
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
                    "architecture": "transit-multiview-tce",
                    "global_bins": global_bins,
                    "local_bins": local_bins,
                    "scalar_dim": SCALAR_FEATURE_DIM,
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
                "architecture": "transit-multiview-tce",
                "global_bins": global_bins,
                "local_bins": local_bins,
                "scalar_dim": SCALAR_FEATURE_DIM,
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
        "global_bins": global_bins,
        "local_bins": local_bins,
        "scalar_dim": SCALAR_FEATURE_DIM,
        "loss_name": args.loss,
        "pos_weight": pos_weight_value,
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
    catalog_path = dataset_root / "tce_catalog.csv"
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

    manifest = load_manifest(manifest_path)
    tce_rows = load_tce_catalog(catalog_path)
    if not tce_rows:
        raw_tce_rows = fetch_dr24_tce_catalog()
        tce_rows, label_summary = build_tce_catalog(
            raw_tce_rows,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        save_tce_catalog(catalog_path, tce_rows)
        write_json(
            split_path,
            {
                "generated_at": now_iso(),
                "split_map": {row.star_id: row.split for row in tce_rows},
            },
        )
    else:
        label_summary = summarize_tce_rows(tce_rows)

    if args.max_stars > 0:
        allowed_stars = set(sorted({row.star_id for row in tce_rows}, key=int)[: args.max_stars])
        tce_rows = [row for row in tce_rows if row.star_id in allowed_stars]

    target_star_ids = sorted({row.star_id for row in tce_rows}, key=int)
    target_tce_count = len(tce_rows)
    downloaded = 0
    skipped = 0
    failures = 0

    if args.skip_ingestion:
        print(f"Detected device={device} workers={num_workers}")
        print(f"Disk free={hardware.disk_free_gb:.1f} GB | budget for this run={budget_bytes / (1024**3):.1f} GB")
        print("Skipping ingestion and using cached star curves from manifest.csv")
        print(
            f"Catalog TCEs={label_summary['tces_total']} "
            f"(pos={label_summary['tces_positive']} neg={label_summary['tces_negative']})"
        )
        print(f"Target stars this run={len(target_star_ids)} | target TCEs={target_tce_count}")
        print(f"Runtime cache root={RUNTIME_ENV_DIRS['runtime_root']}")
    else:
        print(f"Detected device={device} workers={num_workers}")
        print(f"Disk free={hardware.disk_free_gb:.1f} GB | budget for this run={budget_bytes / (1024**3):.1f} GB")
        print(
            f"Catalog TCEs={label_summary['tces_total']} "
            f"(pos={label_summary['tces_positive']} neg={label_summary['tces_negative']})"
        )
        print(f"Target stars this run={len(target_star_ids)} | target TCEs={target_tce_count}")
        print(f"Runtime cache root={RUNTIME_ENV_DIRS['runtime_root']}")

        for idx, star_id in enumerate(target_star_ids, start=1):
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
                label=0,
                split="cache",
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
            ready_total = sum(1 for item in manifest.values() if item.status == "ready")
            atomic_write_json(
                state_path,
                {
                    "phase": "ingestion",
                    "run_dir": str(run_dir),
                    "progress_index": idx,
                    "candidates_total": len(target_star_ids),
                    "ready_total": ready_total,
                    "downloaded": downloaded,
                    "skipped": skipped,
                    "failures": failures,
                    "updated_at": now_iso(),
                },
            )
            if idx % 20 == 0:
                print(
                    f"[ingest] {idx}/{len(target_star_ids)} "
                    f"new_ready={downloaded} ready_total={ready_total} "
                    f"reused_ready={skipped} fail={failures}"
                )

        save_manifest(manifest_path, manifest)

    ready_manifest_rows = {
        star_id: row
        for star_id, row in manifest.items()
        if row.status == "ready" and row.npz_path and Path(row.npz_path).exists()
    }
    ready_rows = [row for row in tce_rows if row.star_id in ready_manifest_rows]
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
            "runtime_env_dirs": RUNTIME_ENV_DIRS,
            "dataset": {
                "ready_tces": len(ready_rows),
                "ready_stars": len(ready_manifest_rows),
                "train_tces": len(train_rows),
                "val_tces": len(val_rows),
                "test_tces": len(test_rows),
                "downloaded_stars_now": downloaded,
                "reused_stars": skipped,
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
        ready_manifest_rows=ready_manifest_rows,
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
        "runtime_env_dirs": RUNTIME_ENV_DIRS,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "label_summary": label_summary,
        "dataset": {
            "ready_tces": len(ready_rows),
            "ready_stars": len(ready_manifest_rows),
            "train_tces": len(train_rows),
            "val_tces": len(val_rows),
            "test_tces": len(test_rows),
            "downloaded_stars_now": downloaded,
            "reused_stars": skipped,
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
