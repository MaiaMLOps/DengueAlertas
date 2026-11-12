"""F7c acceptance with synthetic data: formulas of the derivable reports (unit U7.5; contrato §7.2, §7.3, §7.4 and
§8.4; especificación §6.F7). Plan approved by Jesús on 2026-11-12, with these declared conventions:

D1  Reliability intervals: pd.qcut(p, 10, labels=False, duplicates="drop") on the p of each system over the rows of the
    set being decomposed (the same call as artifacts._calibration, the diagram of the performance card). Intervals are
    right-closed; equal p always fall in the same interval; coinciding quantile edges merge intervals, so there may be
    fewer than 10 (reported). By year, the deciles are recomputed within the rows of that year. RELIABILITY_BINS = 10 is a
    module constant (part of the definition of §7.2, like events.ONSET_MIN_RUN).
D2  Murphy (1973) with the reliability intervals: REL = (1/N) Σ_k n_k (p̄_k − ō_k)², RES = (1/N) Σ_k n_k (ō_k − ō)²,
    UNC = ō(1 − ō); residuo = BS − (REL − RES + UNC) = WBV − 2·WBC, with WBV = (1/N) Σ_k Σ_{i∈k} (p_i − p̄_k)² and
    WBC = (1/N) Σ_k Σ_{i∈k} (p_i − p̄_k)(y_i − ō_k) (Stephenson, Coelho and Jolliffe, 2008, Weather and Forecasting,
    23(4), 752–757).
D3  Classification at tau: a positive emission is p >= tau. Sensitivity, specificity and PPV are NaN when their
    denominator is 0; F1 = 2TP/(2TP + FP + FN), NaN only when TP + FP + FN = 0; MCC is NaN when any factor of its
    denominator is 0 (not scikit-learn's 0); the F1 reference is 2π/(1 + π) (0 when π = 0).
D4  Alert ratio of events._summarize (frozen in F4): evaluable onsets warned by at least one judgeable non-continuity
    alert / judgeable non-continuity alerts. It counts onsets, so one alert may count for two onsets and the ratio may
    exceed 1; reported alongside, for information, alertas_que_avisan (judgeable non-continuity alerts with at least one
    evaluable onset in [t_c + 1, t_c + W]), alertas_con_varios_inicios (two or more) and proporcion_falsas_alarmas
    (false alarms / judgeable alerts). Composite = 0.5·detection + 0.5·ratio.
    Descriptive, outside the contract: the duration of a non-continuity alert is the number of emissions from t_c to its
    end, both included (run length − k + 1, the alert state of diag_serie); median, p75 (numpy.percentile, linear) and
    maximum; weeks under alert = emissions from t_c to the end of every alert.
D5  Paired skill per (municipality, calendar year of the target week): bss and log-loss improvement over the rows
    together and as the mean of the cell improvements; a win is a strictly smaller loss.
Every function that receives rows goes through the test-period barrier first.

API frozen here (`src.forecasting.reports`):
- RELIABILITY_BINS = 10
- reliability_bins(y, p, target_weeks, *, access) -> DataFrame (intervalo, n, p_media, frecuencia_observada, p_min,
  p_max)
- murphy_decomposition(y, p, target_weeks, *, access) -> dict (filas, intervalos, brier, confiabilidad, resolucion,
  incertidumbre, varianza_intra, covarianza_intra, residuo)
- classification_at_tau(y, p, target_weeks, *, tau, access) -> dict (filas, positivas, prevalencia,
  emisiones_positivas, vp, fp, fn, vn, sensibilidad, especificidad, vpp, f1, f1_referencia, mcc)
- calibration_by_year(rows, probabilities, *, access) -> DataFrame: one row per year (text) and "todos", with filas,
  positivas, prevalencia, p_medio_<system> and calibracion_<system> = p_medio − prevalencia
- paired_skill(rows, p_option, p_reference, *, clip, access) -> dict (casillas, ganadas_brier, ganadas_logloss,
  bss_filas, bss_casillas, mejora_logloss_filas, mejora_logloss_casillas)
- event_report(rows, positive, observed, *, k, window, access) -> DataFrame with nivel ("global" | "municipio"), grupo
  and the event columns of §7.4
- alert_durations(rows, positive, observed, *, k, window, access) -> DataFrame with nivel, grupo and the duration
  columns
- amoc_curve(rows, p, observed, *, tau_grid, k, window, access) -> DataFrame (one row per tau, sistema "modelo", plus
  one "canal" row with tau NaN)
"""

from datetime import date, timedelta
import math

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.acceptance

H = 8
CODE = "08001"
F = date(2019, 11, 11)          # first emission of the synthetic event period (as in test_f4_events.py)
N_EMISSIONS = 52
LAST = 59                       # index of the last target week, 2020-12-28


def _reports():
    from src.forecasting import reports

    return reports


def _config():
    from src.forecasting.config import load_config

    return load_config()


def _access(tmp_path):
    from src.forecasting.guard import RecipeAccess

    config = _config()
    return RecipeAccess(test_years=tuple(config.validacion.prueba), channel_version=config.datos.version_dvc_canal,
                        recipe_path=tmp_path / "receta_inexistente_f7c.yaml")


def _weeks(n: int, first: date = date(2020, 1, 6)) -> list[date]:
    return [first + timedelta(weeks=i) for i in range(n)]


# --------------------------------------------------------------------------- S1–S4: reliability and Murphy
S1_P = 0.025 + 0.05 * np.arange(20)
S1_Y = np.tile([0, 1], 10)
S2_P = np.repeat(0.05 + 0.1 * np.arange(10), 2)
S2_Y = np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1])
S3_P = np.array([0.1] * 6 + list(0.2 + 0.05 * np.arange(14)))
S3_Y = np.array([0, 0, 0, 1, 0, 0] + [0, 1] * 7)


def _check_identity(result):
    assert result["brier"] == pytest.approx(
        result["confiabilidad"] - result["resolucion"] + result["incertidumbre"] + result["residuo"], abs=1e-12)
    assert result["residuo"] == pytest.approx(result["varianza_intra"] - 2 * result["covarianza_intra"], abs=1e-12)


def test_s1_murphy_with_variation_inside_the_intervals(tmp_path):
    reports = _reports()
    assert reports.RELIABILITY_BINS == 10
    result = reports.murphy_decomposition(S1_Y, S1_P, _weeks(20), access=_access(tmp_path))
    assert (result["filas"], result["intervalos"]) == (20, 10)
    assert result["brier"] == pytest.approx(0.308125, abs=1e-12)
    assert result["confiabilidad"] == pytest.approx(0.0825, abs=1e-12)
    assert result["resolucion"] == pytest.approx(0.0, abs=1e-12)
    assert result["incertidumbre"] == pytest.approx(0.25, abs=1e-12)
    assert result["varianza_intra"] == pytest.approx(0.000625, abs=1e-12)
    assert result["covarianza_intra"] == pytest.approx(0.0125, abs=1e-12)
    assert result["residuo"] == pytest.approx(-0.024375, abs=1e-12)
    _check_identity(result)
    bins = reports.reliability_bins(S1_Y, S1_P, _weeks(20), access=_access(tmp_path))
    assert bins["n"].tolist() == [2] * 10 and bins["intervalo"].tolist() == list(range(10))
    assert bins["frecuencia_observada"].tolist() == pytest.approx([0.5] * 10)
    assert bins["p_media"].tolist() == pytest.approx(list(0.05 + 0.1 * np.arange(10)))


def test_s2_constant_p_inside_each_interval_leaves_no_residual(tmp_path):
    result = _reports().murphy_decomposition(S2_Y, S2_P, _weeks(20), access=_access(tmp_path))
    assert result["intervalos"] == 10
    assert abs(result["residuo"]) < 1e-12
    assert result["confiabilidad"] == pytest.approx(0.0525, abs=1e-12)
    assert result["resolucion"] == pytest.approx(0.1025, abs=1e-12)
    assert result["incertidumbre"] == pytest.approx(0.2275, abs=1e-12)
    assert result["brier"] == pytest.approx(0.1775, abs=1e-12)
    _check_identity(result)


def test_s3_ties_stay_together_and_merge_intervals(tmp_path):
    reports = _reports()
    bins = reports.reliability_bins(S3_Y, S3_P, _weeks(20), access=_access(tmp_path))
    assert bins["n"].tolist() == [6, 2, 2, 2, 2, 2, 2, 2]
    assert bins.loc[0, "p_min"] == bins.loc[0, "p_max"] == 0.1
    result = reports.murphy_decomposition(S3_Y, S3_P, _weeks(20), access=_access(tmp_path))
    assert result["intervalos"] == 8
    assert result["varianza_intra"] == pytest.approx(0.0004375, abs=1e-12)
    assert result["covarianza_intra"] == pytest.approx(0.00875, abs=1e-12)
    assert result["residuo"] == pytest.approx(-0.0170625, abs=1e-12)
    _check_identity(result)


def test_s4_intervals_are_those_of_the_performance_card(tmp_path):
    from src.forecasting.artifacts import _calibration

    for y, p in ((S1_Y, S1_P), (S2_Y, S2_P), (S3_Y, S3_P)):
        bins = _reports().reliability_bins(y, p, _weeks(len(y)), access=_access(tmp_path))
        card = _calibration(y.astype(float), p)
        assert bins["n"].tolist() == [entry["n"] for entry in card]
        assert bins["p_media"].tolist() == pytest.approx([entry["probabilidad_media"] for entry in card], abs=1e-15)
        assert bins["frecuencia_observada"].tolist() == pytest.approx(
            [entry["frecuencia_observada"] for entry in card], abs=1e-15)


# --------------------------------------------------------------------------- S5–S7: classification at tau
S5_Y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
S5_P = np.array([0.9, 0.6, 0.02, 0.5, 0.04, 0.01, 0.03, 0.02, 0.01, 0.6])


def test_s5_classification_at_tau(tmp_path):
    result = _reports().classification_at_tau(S5_Y, S5_P, _weeks(10), tau=0.05, access=_access(tmp_path))
    assert (result["vp"], result["fp"], result["fn"], result["vn"]) == (2, 2, 1, 5)
    assert (result["filas"], result["positivas"], result["emisiones_positivas"]) == (10, 3, 4)
    assert result["prevalencia"] == pytest.approx(0.3)
    assert result["sensibilidad"] == pytest.approx(2 / 3)
    assert result["especificidad"] == pytest.approx(5 / 7)
    assert result["vpp"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(4 / 7)
    assert result["mcc"] == pytest.approx(8 / math.sqrt(504))
    assert result["mcc"] == pytest.approx(0.356348, abs=1e-6)
    assert result["f1_referencia"] == pytest.approx(0.6 / 1.3)


def test_s6_undefined_denominators_are_nan(tmp_path):
    reports, access = _reports(), _access(tmp_path)
    none = reports.classification_at_tau(np.zeros(6, dtype=int), np.full(6, 0.01), _weeks(6), tau=0.05, access=access)
    assert math.isnan(none["sensibilidad"]) and none["especificidad"] == 1.0 and math.isnan(none["vpp"])
    assert math.isnan(none["f1"]) and math.isnan(none["mcc"]) and none["f1_referencia"] == 0.0
    silent = reports.classification_at_tau(np.array([1, 1, 0, 0]), np.full(4, 0.01), _weeks(4), tau=0.05,
                                           access=access)
    assert silent["sensibilidad"] == 0.0 and math.isnan(silent["vpp"])
    assert silent["f1"] == 0.0 and math.isnan(silent["mcc"])


def test_s7_p_equal_to_tau_is_a_positive_emission(tmp_path):
    result = _reports().classification_at_tau(np.array([1, 0]), np.array([0.05, 0.05]), _weeks(2), tau=0.05,
                                              access=_access(tmp_path))
    assert (result["emisiones_positivas"], result["vp"], result["fp"]) == (2, 1, 1)


# --------------------------------------------------------------------------- S8–S10: events (synthetic period of F4)
def _week(i: int) -> date:
    return F + timedelta(weeks=i)


def _rows(zones: dict | None = None) -> pd.DataFrame:
    zones = zones or {}
    data = [(CODE, _week(i), _week(i + H), zones.get(i, "seguridad")) for i in range(N_EMISSIONS)]
    return pd.DataFrame(data, columns=["municipality_code", "t", "target_week", "zona_t"])


def _observed(ones=()) -> pd.DataFrame:
    data = [(CODE, _week(i), int(i in set(ones)), True) for i in range(-30, LAST + 11)]
    return pd.DataFrame(data, columns=["municipality_code", "week_start", "target_u1e", "channel_eligible"])


def _positive(*runs: tuple[int, int]) -> np.ndarray:
    """Positive emissions on the closed index ranges of each run."""
    positive = np.zeros(N_EMISSIONS, dtype=bool)
    for first, last in runs:
        positive[first:last + 1] = True
    return positive


def _global(frame: pd.DataFrame) -> pd.Series:
    rows = frame[frame["nivel"] == "global"]
    assert len(rows) == 1
    return rows.iloc[0]


def test_s8_alert_ratio_composite_and_false_alarm_breakdown(tmp_path):
    config = _config()
    report = _reports().event_report(_rows(), _positive((9, 10), (29, 30)), _observed({15, 16, 17}),
                                     k=config.decision.k, window=config.decision.W, access=_access(tmp_path))
    row = _global(report)
    assert (row["inicios_evaluables"], row["inicios_avisados"], row["alertas_juzgables"]) == (1, 1, 2)
    assert (row["inicios_avisados_por_juzgable"], row["alertas_que_avisan"], row["alertas_con_varios_inicios"]) == (
        1, 1, 0)
    assert row["deteccion"] == 1.0 and row["razon_alertas"] == 0.5 and row["compuesto"] == 0.75
    assert (row["falsas_alarmas"], row["falsas_seguidas_epidemia"], row["falsas_no_seguidas"]) == (1, 0, 1)
    assert row["proporcion_falsas_alarmas"] == 0.5
    assert row["municipio_anios"] == 1 and row["falsas_por_municipio_anio"] == 1.0
    municipality = report[report["nivel"] == "municipio"]
    assert municipality["grupo"].tolist() == [CODE]


def test_s9_one_alert_can_count_for_two_onsets(tmp_path):
    config = _config()
    report = _reports().event_report(_rows(), _positive((9, 10)), _observed({15, 16, 17, 20, 21, 22}),
                                     k=config.decision.k, window=config.decision.W, access=_access(tmp_path))
    row = _global(report)
    assert (row["inicios_evaluables"], row["inicios_avisados"], row["alertas_juzgables"]) == (2, 2, 1)
    assert (row["alertas_que_avisan"], row["alertas_con_varios_inicios"]) == (1, 1)
    assert row["razon_alertas"] == 2.0 and row["compuesto"] == 1.5 and row["falsas_alarmas"] == 0


def test_s10_alert_durations_and_weeks_under_alert(tmp_path):
    config = _config()
    rows = _rows(zones={41: "epidemia"})
    positive = _positive((5, 6), (15, 17), (25, 31), (40, 43))
    durations = _reports().alert_durations(rows, positive, _observed(), k=config.decision.k,
                                           window=config.decision.W, access=_access(tmp_path))
    row = _global(durations)
    assert (row["alertas"], row["alertas_no_continuidad"], row["alertas_truncadas"]) == (4, 3, 0)
    assert (row["duracion_mediana"], row["duracion_p75"], row["duracion_max"]) == (2.0, 4.0, 6.0)
    assert (row["semanas_emision"], row["semanas_bajo_alerta"], row["semanas_bajo_alerta_no_continuidad"]) == (
        52, 12, 9)
    assert row["fraccion_bajo_alerta"] == pytest.approx(12 / 52)


# --------------------------------------------------------------------------- S11: paired skill (and calibration)
def _skill_rows():
    """Two cells of 20 rows (08001 in 2020 and 2021), y = 0, p in {0, 1}: Brier = wrong rows / 20."""
    weeks = _weeks(20, date(2020, 1, 6)) + _weeks(20, date(2021, 1, 4))
    rows = pd.DataFrame({"municipality_code": CODE, "target_week": weeks, "y": 0})

    def wrong(first: int, second: int) -> np.ndarray:
        return np.array([1.0] * first + [0.0] * (20 - first) + [1.0] * second + [0.0] * (20 - second))

    return rows, wrong(2, 6), wrong(4, 5)     # option: 0.10 and 0.30; reference: 0.20 and 0.25


def test_s11_cell_mean_and_pooled_rows_can_disagree(tmp_path):
    rows, option, reference = _skill_rows()
    result = _reports().paired_skill(rows, option, reference, clip=_config().metricas.recorte_log_loss,
                                     access=_access(tmp_path))
    assert (result["casillas"], result["ganadas_brier"], result["ganadas_logloss"]) == (2, 1, 1)
    assert result["bss_casillas"] == pytest.approx(0.15, abs=1e-12)
    assert result["bss_filas"] == pytest.approx(1 / 9, abs=1e-12)
    assert result["mejora_logloss_casillas"] == pytest.approx(0.15, rel=1e-9)
    assert result["mejora_logloss_filas"] == pytest.approx(1 / 9, rel=1e-9)


def test_s11_calibration_in_the_large_by_year(tmp_path):
    rows, option, reference = _skill_rows()
    rows = rows.assign(y=[1] * 5 + [0] * 15 + [0] * 20)
    table = _reports().calibration_by_year(rows, {"modelo": option, "B2": reference}, access=_access(tmp_path))
    table = table.set_index("anio")
    assert list(table.index) == ["2020", "2021", "todos"]
    assert table.loc["2020", "prevalencia"] == 0.25 and table.loc["2021", "prevalencia"] == 0.0
    assert table.loc["2020", "calibracion_modelo"] == pytest.approx(0.10 - 0.25)
    assert table.loc["2021", "calibracion_B2"] == pytest.approx(0.25)
    assert table.loc["todos", "p_medio_modelo"] == pytest.approx(0.20)
    assert table.loc["todos", "calibracion_modelo"] == pytest.approx(0.20 - 0.125)


# --------------------------------------------------------------------------- S12: test-period barrier
def test_s12_every_function_with_rows_needs_the_recipe_on_test_weeks(tmp_path):
    from src.forecasting.guard import FrozenRecipeError

    reports, config, access = _reports(), _config(), _access(tmp_path)
    weeks = _weeks(20, date(2023, 1, 2))
    rows = pd.DataFrame({"municipality_code": CODE, "target_week": weeks, "y": S1_Y})
    calls = [
        lambda: reports.reliability_bins(S1_Y, S1_P, weeks, access=access),
        lambda: reports.murphy_decomposition(S1_Y, S1_P, weeks, access=access),
        lambda: reports.classification_at_tau(S1_Y, S1_P, weeks, tau=0.05, access=access),
        lambda: reports.calibration_by_year(rows, {"modelo": S1_P}, access=access),
        lambda: reports.paired_skill(rows, S1_P, S2_P, clip=config.metricas.recorte_log_loss, access=access),
    ]
    period = _rows()
    period["t"] = [t + timedelta(weeks=165) for t in period["t"]]                 # emissions from 2023
    period["target_week"] = [t + timedelta(weeks=165) for t in period["target_week"]]
    observed = pd.DataFrame([(CODE, F + timedelta(weeks=i), 0, True) for i in range(100, 240)],
                            columns=["municipality_code", "week_start", "target_u1e", "channel_eligible"])
    positive = _positive((9, 10))
    calls += [
        lambda: reports.event_report(period, positive, observed, k=2, window=12, access=access),
        lambda: reports.alert_durations(period, positive, observed, k=2, window=12, access=access),
        lambda: reports.amoc_curve(period, np.full(N_EMISSIONS, 0.5), observed, tau_grid=[0.05], k=2, window=12,
                                   access=access),
    ]
    for call in calls:
        with pytest.raises(FrozenRecipeError):
            call()
