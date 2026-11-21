import numpy as np
import pandas as pd

from ewt_final.pipeline import classification_metrics, choose_cost_threshold


def test_metrics_and_cost_threshold_are_deterministic():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.20, 0.40, 0.60, 0.75, 0.90])
    threshold = choose_cost_threshold(y, p, false_negative_cost=5, false_positive_cost=1)
    metrics = classification_metrics(y, p, threshold, pd.date_range("2024-01-01", periods=6, freq="W-MON"))
    assert 0 <= threshold <= 1
    assert metrics["average_precision"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["fn"] == 0
