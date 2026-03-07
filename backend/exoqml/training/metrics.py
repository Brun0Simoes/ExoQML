from __future__ import annotations

import math

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def roc_auc_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    y_sorted = y_true[order]
    ranks = np.arange(1, len(y_sorted) + 1, dtype=np.float64)
    sum_ranks_pos = np.sum(ranks[y_sorted == 1])
    auc = (sum_ranks_pos - (pos * (pos + 1) / 2.0)) / float(pos * neg)
    return float(auc)


def pr_auc_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    pos = int(np.sum(y_true == 1))
    if pos == 0:
        return float("nan")

    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos

    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    auc = np.trapezoid(precision, recall)
    return float(auc)


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = y_true.astype(np.int64)
    y_pred = (y_score >= threshold).astype(np.int64)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, len(y_true))
    specificity = _safe_div(tn, tn + fp)

    roc_auc = roc_auc_score_binary(y_true, y_score)
    pr_auc = pr_auc_score_binary(y_true, y_score)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "roc_auc": float(roc_auc) if not math.isnan(roc_auc) else float("nan"),
        "pr_auc": float(pr_auc) if not math.isnan(pr_auc) else float("nan"),
    }


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 37)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        metrics = binary_metrics(y_true, y_score, threshold=float(thr))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thr = float(thr)
    return best_thr, float(best_f1)
