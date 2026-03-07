from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

TargetType = Literal["tic", "kic", "name"]


@dataclass(slots=True)
class ResolvedTarget:
    target_id: str
    target_type: TargetType
    query: str


def _extract_digits(raw: str) -> str:
    match = re.search(r"(\d+)", raw)
    return match.group(1) if match else ""


def resolve_target(target_id: str, target_type: TargetType | None = None) -> ResolvedTarget:
    cleaned = target_id.strip()
    if not cleaned:
        raise ValueError("target_id is empty")

    upper = cleaned.upper()
    if target_type is not None:
        if target_type in {"tic", "kic"}:
            digits = _extract_digits(upper)
            if not digits:
                raise ValueError(f"{target_type.upper()} id must contain digits")
            return ResolvedTarget(target_id=digits, target_type=target_type, query=f"{target_type.upper()} {digits}")
        return ResolvedTarget(target_id=cleaned, target_type="name", query=cleaned)

    if upper.startswith("TIC"):
        digits = _extract_digits(upper)
        if not digits:
            raise ValueError("TIC id must contain digits")
        return ResolvedTarget(target_id=digits, target_type="tic", query=f"TIC {digits}")

    if upper.startswith("KIC"):
        digits = _extract_digits(upper)
        if not digits:
            raise ValueError("KIC id must contain digits")
        return ResolvedTarget(target_id=digits, target_type="kic", query=f"KIC {digits}")

    if upper.isdigit():
        return ResolvedTarget(target_id=upper, target_type="tic", query=f"TIC {upper}")

    return ResolvedTarget(target_id=cleaned, target_type="name", query=cleaned)
