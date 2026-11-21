"""Pruebas de humo del analizador; no requieren reentrenar modelos."""
from pathlib import Path
import importlib.util
import pandas as pd

MODULE = Path(__file__).with_name("analyze_ewt_results.py")
spec = importlib.util.spec_from_file_location("analyzer", MODULE)
analyzer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
import sys
sys.modules[spec.name] = analyzer
spec.loader.exec_module(analyzer)


def test_normalize_code():
    assert analyzer.normalize_code(8001) == "08001"
    assert analyzer.normalize_code("08758") == "08758"


def test_feature_family():
    assert analyzer.feature_family("ch_log_gsd") == "ch"
    assert analyzer.feature_family("clim_temperature_c_lag0") == "clim"
    assert analyzer.feature_family("unknown") == "other"


def test_summary_table_minimal(tmp_path):
    data = analyzer.RunData(root=tmp_path)
    data.results = pd.DataFrame([{
        "task": "zone_future", "municipality_code": "08001", "municipality": "Barranquilla",
        "winner_algorithm": "logistic", "winner_feature_set": "epi", "test_n": 10,
        "test_positives": 7, "test_prevalence": .7, "test_average_precision": .8,
        "test_roc_auc": .75, "test_brier": .18, "test_recall": .9,
        "test_specificity": .5, "test_precision": .8, "test_f1": .85,
        "test_threshold": .3, "test_tn": 1, "test_fp": 2, "test_fn": 1, "test_tp": 6,
    }])
    out = analyzer.summary_table(data)
    assert len(out) == 1
    assert out.iloc[0]["municipality_code"] == "08001"
