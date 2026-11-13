"""Unit tests of reports.py (U7.5) with synthetic data; the formulas of the plan are frozen in
tests/acceptance/test_f7c_formulas.py and the identities with the evidence in test_f7c_identidades_real.py."""

from datetime import date, timedelta
import math

import numpy as np
import pandas as pd
import pytest

from src.forecasting import reports
from src.forecasting.config import load_config
from src.forecasting.guard import RecipeAccess
from src.epidemiology.validation import EpidemiologyValidationError as ValidationError

CODE = "08001"
F = date(2019, 11, 11)


@pytest.fixture
def access(tmp_path):
    config = load_config()
    return RecipeAccess(test_years=tuple(config.validacion.prueba), channel_version=config.datos.version_dvc_canal,
                        recipe_path=tmp_path / "sin_receta.yaml")


def _weeks(n: int) -> list[date]:
    return [date(2020, 1, 6) + timedelta(weeks=i) for i in range(n)]


def test_constant_p_is_one_interval_without_residual(access):
    y = np.array([0, 1, 1, 0, 1])
    result = reports.murphy_decomposition(y, np.full(5, 0.4), _weeks(5), access=access)
    assert result["intervalos"] == 1 and result["resolucion"] == 0.0 and abs(result["residuo"]) < 1e-15
    assert result["confiabilidad"] == pytest.approx((0.4 - 0.6) ** 2)


def test_bins_cover_every_row_once(access):
    rng = np.random.default_rng(1)
    p, y = rng.uniform(size=137), rng.integers(0, 2, size=137)
    bins = reports.reliability_bins(y, p, _weeks(137), access=access)
    assert bins["n"].sum() == 137 and len(bins) == 10
    assert (bins["p_min"].to_numpy()[1:] > bins["p_max"].to_numpy()[:-1]).all()


def test_classification_rejects_an_invalid_tau(access):
    with pytest.raises(ValidationError):
        reports.classification_at_tau(np.array([0, 1]), np.array([0.2, 0.3]), _weeks(2), tau=1.5, access=access)


def test_paired_skill_needs_one_probability_per_row(access):
    rows = pd.DataFrame({"municipality_code": CODE, "target_week": _weeks(4), "y": [0, 1, 0, 1]})
    with pytest.raises(ValidationError):
        reports.paired_skill(rows, np.full(3, 0.5), np.full(4, 0.5), clip=1e-15, access=access)


def _period(n: int = 52, zones=None):
    zones = zones or {}
    rows = pd.DataFrame([(CODE, F + timedelta(weeks=i), F + timedelta(weeks=i + 8), zones.get(i, "seguridad"))
                         for i in range(n)], columns=["municipality_code", "t", "target_week", "zona_t"])
    observed = pd.DataFrame([(CODE, F + timedelta(weeks=i), 0, True) for i in range(-30, n + 20)],
                            columns=["municipality_code", "week_start", "target_u1e", "channel_eligible"])
    return rows, observed


def test_an_alert_reaching_the_last_emission_is_truncated(access):
    rows, observed = _period()
    positive = np.zeros(52, dtype=bool)
    positive[48:] = True
    table = reports.alert_durations(rows, positive, observed, k=2, window=12, access=access)
    row = table[table["nivel"] == "global"].iloc[0]
    assert (row["alertas"], row["alertas_truncadas"], row["duracion_max"]) == (1, 1, 3.0)


def test_no_alert_gives_nan_durations_and_ratio(access):
    rows, observed = _period()
    durations = reports.alert_durations(rows, np.zeros(52, dtype=bool), observed, k=2, window=12, access=access)
    assert math.isnan(durations.iloc[0]["duracion_mediana"]) and durations.iloc[0]["semanas_bajo_alerta"] == 0
    events = reports.event_report(rows, np.zeros(52, dtype=bool), observed, k=2, window=12, access=access)
    assert math.isnan(events.iloc[0]["razon_alertas"]) and math.isnan(events.iloc[0]["compuesto"])


def test_amoc_curve_has_one_row_per_tau_and_the_channel(access):
    rows, observed = _period(zones={10: "alerta", 11: "alerta"})
    curve = reports.amoc_curve(rows, np.linspace(0, 1, 52), observed, tau_grid=[0.25, 0.5], k=2, window=12,
                               access=access)
    assert curve["sistema"].tolist() == ["modelo", "modelo", "canal"]
    assert math.isnan(curve["tau"].iloc[-1]) and curve["falsas_alarmas"].iloc[-1] == 1


def test_development_vs_test_puts_the_channel_instead_of_ablation_events():
    def tables(periodo):
        skill = pd.DataFrame([{"periodo": periodo, "rotulo": "r", "sistema": s, "referencia": r, "bss_filas": 0.1}
                              for s, r in reports._SKILL_PAIRS])
        events = pd.DataFrame([{"periodo": periodo, "rotulo": "r", "sistema": s, "nivel": "global", "grupo": "todos",
                                "inicios_evaluables": 10, "inicios_avisados": n, "falsas_alarmas": 3,
                                "municipio_anios": 18, "falsas_por_municipio_anio": 1 / 6}
                               for s, n in (("modelo", 4), ("canal", 8))])
        return {"habilidad": skill, "eventos_razon": events}

    table = reports.development_vs_test(tables("desarrollo"), tables("prueba"))
    assert len(table) == 2 * (len(reports._SKILL_PAIRS) + 1)
    ablation = table[table["sistema"] == "ablacion"]
    assert ablation["inicios_avisados"].isna().all()
    assert table.loc[table["sistema"] == "canal", "inicios_avisados"].tolist() == [8, 8]
    assert table.loc[table["sistema"] == "modelo", "inicios_avisados"].tolist() == [4] * 6


def test_relative_paths(tmp_path):
    assert reports._relative(reports.ROOT / "reports" / "x.csv") == "reports/x.csv"
    assert reports._relative(tmp_path / "y.png").endswith("/y.png")
