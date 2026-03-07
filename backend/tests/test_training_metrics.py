import numpy as np

from exoqml.training.metrics import best_f1_threshold, binary_metrics


def test_binary_metrics_range() -> None:
    y_true = np.array([1, 0, 1, 0, 1, 0], dtype=np.int64)
    y_score = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3], dtype=np.float64)
    metrics = binary_metrics(y_true, y_score, threshold=0.5)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["roc_auc"] > 0.9
    assert metrics["pr_auc"] > 0.9


def test_best_f1_threshold_valid() -> None:
    y_true = np.array([1, 1, 0, 0], dtype=np.int64)
    y_score = np.array([0.8, 0.7, 0.6, 0.1], dtype=np.float64)
    threshold, f1 = best_f1_threshold(y_true, y_score)
    assert 0.0 < threshold < 1.0
    assert 0.0 <= f1 <= 1.0
