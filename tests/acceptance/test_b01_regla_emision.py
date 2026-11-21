"""Regression of rule B01 on the saved predictions of the frozen recipe (análisis complementario, 2026-11-20).

Control figures declared before the integration of the complementary analysis; they are reproduced, never tuned.
Needs the channel product (--channel-parquet) and the versioned diag_serie.csv of the test run and of the
out-of-year development diagnosis.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

from src.forecasting.config import ROOT, load_config
from src.forecasting.dataset import load_channel_product
from src.forecasting.guard import RecipeAccess, load_frozen_recipe

sys.path.insert(0, str(ROOT))

TEST_SERIES = ROOT / "reports/forecasting/evidence/prueba_20261112/diag_serie.csv"
DEV_SERIES = ROOT / "reports/forecasting/evidence/diagnostico_desarrollo_sin_clima_20261112/diag_serie.csv"


@pytest.fixture(scope="module")
def results(request):
    option = request.config.getoption("--channel-parquet")
    if option is None:
        pytest.skip("needs --channel-parquet")
    from scripts.analisis_b01 import _read_series, evaluate_rules

    config = load_config()
    access = RecipeAccess.from_config(config)
    recipe = load_frozen_recipe(access.recipe_path, channel_version=access.channel_version)
    observed = load_channel_product(Path(option), expected_md5=config.datos.version_dvc_canal).to_pandas()
    out = {}
    for period, path in (("prueba", TEST_SERIES), ("desarrollo", DEV_SERIES)):
        out[period] = evaluate_rules(_read_series(path), observed, tau=float(recipe["decision"]["tau"]),
                                     k=config.decision.k, window=config.decision.W, alpha=config.comparacion.alfa,
                                     access=access, period=period)
    return out


def _global(table: pd.DataFrame, system: str) -> dict:
    return table[(table["nivel"] == "global") & (table["sistema"] == system)].iloc[0].to_dict()


@pytest.mark.parametrize("period, system, onsets, entries, n_alerts, false_alarms, weeks", [
    ("prueba", "regla_actual", (4, 10), (13, 41), 26, 18, None),
    ("prueba", "B01", (9, 10), (24, 41), 38, 26, (192, 936)),
    ("prueba", "canal", (8, 10), (19, 41), 39, 30, None),
    ("desarrollo", "B01", (10, 10), (38, 44), 59, 37, (244, 1248)),
])
def test_event_counts(results, period, system, onsets, entries, n_alerts, false_alarms, weeks):
    row = _global(results[period][0], system)
    assert (row["inicios_avisados"], row["inicios_evaluables"]) == onsets
    assert (row["entradas_avisadas"], row["entradas_evaluables"]) == entries
    assert row["alertas"] == n_alerts and row["falsas_alarmas"] == false_alarms
    if weeks is not None:
        assert (row["semanas_bajo_alerta"], row["semanas"]) == weeks


@pytest.mark.parametrize("system, contract, first", [("regla_actual", 4.5, 7.5), ("B01", 1.0, 3.0),
                                                      ("canal", 1.5, 2.0)])
def test_test_period_leads(results, system, contract, first):
    row = _global(results["prueba"][0], system)
    assert row["anticipacion_mediana_contrato"] == contract
    assert row["anticipacion_mediana_primera"] == first


def test_test_period_discordant_onsets_and_composition(results):
    _, _, comparisons, composition = results["prueba"]
    assert (comparisons["B01_vs_regla_actual"]["discrepantes"], comparisons["B01_vs_regla_actual"]["a_favor"]) == (5, 5)
    assert (comparisons["B01_vs_canal"]["discrepantes"], comparisons["B01_vs_canal"]["a_favor"]) == (1, 1)
    assert (composition["emisiones_B01_por_zona"]["alerta"], composition["emisiones_B01"]) == (268, 272)
    assert (composition["semanas_zona_alerta_descartadas_por_p"], composition["semanas_zona_alerta"]) == (196, 464)
    assert round(100 * composition["fraccion_descartada"], 1) == 42.2
