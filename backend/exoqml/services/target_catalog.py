from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from exoqml.config import BACKEND_ROOT

READY_MANIFEST_PATH = BACKEND_ROOT / "data" / "train_max" / "manifest.csv"
TCE_CATALOG_PATH = BACKEND_ROOT / "data" / "train_max" / "tce_catalog.csv"

GUIDED_TARGETS: list[dict[str, object]] = [
    {
        "query": "KIC 10000490",
        "target_id": "10000490",
        "target_type": "kic",
        "display_name": "KIC 10000490",
        "mission": "Kepler",
        "source": "guide",
        "summary": "Exemplo numerico do catalogo Kepler para quem quer testar um KIC conhecido.",
        "tce_count": 0,
        "positive_tce_count": 0,
        "sky_coordinates": {"ra": 286.5560, "dec": 46.9573},
    },
    {
        "query": "TIC 25155310",
        "target_id": "25155310",
        "target_type": "tic",
        "display_name": "TIC 25155310",
        "mission": "TESS",
        "source": "guide",
        "summary": "Exemplo TESS no hemisferio sul para quem prefere um alvo pronto.",
        "tce_count": 0,
        "positive_tce_count": 0,
        "sky_coordinates": {"ra": 63.3739, "dec": -69.2268},
    },
    {
        "query": "Kepler-10",
        "target_id": "Kepler-10",
        "target_type": "name",
        "display_name": "Kepler-10",
        "mission": "Kepler",
        "source": "guide",
        "summary": "Nome conhecido para quem nao quer comecar por um codigo de catalogo.",
        "tce_count": 0,
        "positive_tce_count": 0,
        "sky_coordinates": {"ra": 285.6794, "dec": 50.2413},
    },
]


def _ready_star_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ready_ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("status") or "").strip().lower() != "ready":
                continue
            star_id = (row.get("star_id") or "").strip()
            if star_id:
                ready_ids.add(star_id)
    return ready_ids


def _summarize_star(entry: dict[str, object]) -> str:
    positive_tce_count = int(entry["positive_tce_count"])
    tce_count = int(entry["tce_count"])
    best_label = str(entry["best_label"])
    best_period = entry["best_period"]
    best_snr = entry["best_snr"]

    if positive_tce_count > 0:
        summary = f"{positive_tce_count} TCE(s) com rotulo PC no conjunto local."
    else:
        summary = f"{tce_count} TCE(s) locais sem rotulo PC; melhor rotulo local: {best_label}."

    details: list[str] = []
    if isinstance(best_period, float):
        details.append(f"periodo de referencia {best_period:.3f} d")
    if isinstance(best_snr, float):
        details.append(f"SNR max {best_snr:.1f}")
    if details:
        summary = f"{summary} {'; '.join(details)}."
    return summary


def _load_dataset_catalog(manifest_path: Path, tce_catalog_path: Path) -> list[dict[str, object]]:
    ready_ids = _ready_star_ids(manifest_path)
    if not ready_ids or not tce_catalog_path.exists():
        return []

    aggregated: dict[str, dict[str, object]] = {}
    with tce_catalog_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            star_id = (row.get("star_id") or "").strip()
            if not star_id or star_id not in ready_ids:
                continue

            label_name = (row.get("label_name") or "unknown").strip().upper() or "unknown"
            period_raw = (row.get("period") or "").strip()
            snr_raw = (row.get("model_snr") or "").strip()
            period = float(period_raw) if period_raw else None
            snr = float(snr_raw) if snr_raw else None

            star_entry = aggregated.setdefault(
                star_id,
                {
                    "query": f"KIC {star_id}",
                    "target_id": star_id,
                    "target_type": "kic",
                    "display_name": f"KIC {star_id}",
                    "mission": "Kepler",
                    "source": "local_dataset",
                    "tce_count": 0,
                    "positive_tce_count": 0,
                    "best_label": label_name,
                    "best_period": period,
                    "best_snr": snr,
                    "sky_coordinates": None,
                },
            )
            star_entry["tce_count"] = int(star_entry["tce_count"]) + 1
            if label_name == "PC":
                star_entry["positive_tce_count"] = int(star_entry["positive_tce_count"]) + 1
            if snr is not None and (
                star_entry["best_snr"] is None or float(snr) > float(star_entry["best_snr"])
            ):
                star_entry["best_snr"] = snr
                star_entry["best_period"] = period
                star_entry["best_label"] = label_name

    catalog: list[dict[str, object]] = []
    for star_id, entry in aggregated.items():
        item = dict(entry)
        item["summary"] = _summarize_star(entry)
        item["sort_key"] = (0 if int(entry["positive_tce_count"]) > 0 else 1, -float(entry["best_snr"] or 0.0), int(star_id))
        catalog.append(item)
    catalog.sort(key=lambda item: item["sort_key"])
    for item in catalog:
        item.pop("sort_key", None)
        item.pop("best_label", None)
        item.pop("best_period", None)
        item.pop("best_snr", None)
    return catalog


@lru_cache(maxsize=1)
def load_target_catalog() -> list[dict[str, object]]:
    seen_queries: set[str] = set()
    items: list[dict[str, object]] = []

    for item in GUIDED_TARGETS:
        query = str(item["query"]).strip().lower()
        if query in seen_queries:
            continue
        seen_queries.add(query)
        items.append(dict(item))

    for item in _load_dataset_catalog(READY_MANIFEST_PATH, TCE_CATALOG_PATH):
        query = str(item["query"]).strip().lower()
        if query in seen_queries:
            continue
        seen_queries.add(query)
        items.append(item)

    return items
