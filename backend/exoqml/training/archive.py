from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any

EXOPLANET_ARCHIVE_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={query}&format=json"


def tap_query(sql: str, timeout: int = 120, retries: int = 3, sleep_seconds: float = 1.5) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    url = EXOPLANET_ARCHIVE_TAP.format(query=urllib.parse.quote(sql, safe=""))
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            data = json.loads(payload)
            if isinstance(data, list):
                return data
            raise RuntimeError("Unexpected TAP response shape")
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)
    raise RuntimeError(f"TAP query failed after {retries} attempts: {last_error}") from last_error


def fetch_dr24_tce_labels() -> list[dict[str, str | int]]:
    sql = """
        select kepid, av_training_set
        from q1_q17_dr24_tce
        where av_training_set in ('PC','AFP','NTP')
    """
    rows = tap_query(sql)
    result: list[dict[str, str | int]] = []
    for row in rows:
        try:
            kepid = int(row["kepid"])
            label = str(row["av_training_set"]).strip().upper()
        except Exception:
            continue
        if label in {"PC", "AFP", "NTP"}:
            result.append({"kepid": kepid, "av_training_set": label})
    return result


def fetch_dr24_tce_catalog() -> list[dict[str, Any]]:
    sql = """
        select
            kepid,
            tce_plnt_num,
            av_training_set,
            tce_period,
            tce_duration,
            tce_time0bk,
            tce_depth,
            tce_model_snr
        from q1_q17_dr24_tce
        where av_training_set in ('PC','AFP','NTP')
    """
    rows = tap_query(sql)
    result: list[dict[str, Any]] = []
    for row in rows:
        try:
            kepid = int(row["kepid"])
            tce_num = int(row["tce_plnt_num"])
            label = str(row["av_training_set"]).strip().upper()
            period = float(row["tce_period"])
            duration = float(row["tce_duration"])
            epoch = float(row["tce_time0bk"])
            depth = float(row["tce_depth"]) if row["tce_depth"] is not None else 0.0
            model_snr = float(row["tce_model_snr"]) if row["tce_model_snr"] is not None else 0.0
        except Exception:
            continue

        if label not in {"PC", "AFP", "NTP"}:
            continue
        if period <= 0.0 or duration <= 0.0:
            continue

        result.append(
            {
                "kepid": kepid,
                "tce_plnt_num": tce_num,
                "av_training_set": label,
                "tce_period": period,
                "tce_duration": duration,
                "tce_time0bk": epoch,
                "tce_depth": depth,
                "tce_model_snr": model_snr,
            }
        )
    return result
