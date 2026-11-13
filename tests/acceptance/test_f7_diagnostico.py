"""F7 acceptance with synthetic inputs: the six inputs of the integrated diagnostic analysis (contrato §8.8 and
§10.22, D9.1; especificación §6.F7). Approved by Jesús on 2026-11-12 (U7.1).

Scenario (one municipality, cells Y and Y + 1 with Y = 2019; s = first Monday of Y + 1):
- observed eligible series from the first Monday of Y - 1 to the last Monday of Y + 1; target_u1e = 1 (epidemic
  zone) in the first 26 Mondays of Y + 1, so Y is quiet and Y + 1 epidemic, with one onset and one entry in s;
- zone "alerta" at s - 7 and s - 6 weeks (two channel emissions: channel alert confirmed at s - 6, lead 6);
- rows: every target week of Y and Y + 1, t = target - 8 weeks, zona_t = observed zone at t, y = target_u1e of
  the target week;
- predictions (constant per cell): model 0.01 in Y and 0.6 in Y + 1 (alert confirmed at s - 7, lead 7, so
  delta_w = 1 with tau = 0.5); B2 0.02 in Y and 0.5 in Y + 1; B1 0.3 in both.
Hand values: in Y (no positive) Brier model 0.0001 and B2 0.0004 (BSS 0.75); in Y + 1 (26 of 52 positive)
Brier model 0.26 and B2 0.25 (BSS -0.04). The mean of the cell BSS is 0.355 > 0, but the BSS over the rows
together is 1 - 0.13005 / 0.1252 < 0: the two aggregations of §8.8 can disagree in sign.

API frozen here (`src.forecasting.diagnostics`):
- `diagnostic_inputs(rows, predictions, observed, *, tau, k, window, clip, access, rotulo) -> Diagnostics` with
  the DataFrames casillas, resumenes, transicion, eventos, alertas and serie (columns below). predictions is
  long, with sistema in {"modelo", "B2", "B1"}, and must hold exactly the keys (municipality_code, t) of rows
  for each system. The test-period barrier is checked first with the target weeks of rows. rotulo is
  "usado_para_elegir" for development rows and "prueba" for test rows; anything else fails.
- Differences are option minus reference (negative = the model is better); mejora_* and bss_* are 1 - L/L_ref;
  a win is a strictly smaller loss (a tie is not a win).
"""

from datetime import date, timedelta
import math

import numpy as np
import pandas as pd
import pytest

from src.epidemiology.config import EpidemiologyValidationError

pytestmark = pytest.mark.acceptance

WEEK = timedelta(weeks=1)
CODE = "00001"
TAU, K, W, CLIP = 0.5, 2, 12, 1e-15

CASILLAS = [
    "municipality_code", "anio", "regimen", "rotulo", "candidato", "filas", "positivas", "prevalencia",
    "p_medio_modelo", "p_medio_b2", "p_medio_b1", "calibracion_modelo", "calibracion_b2", "calibracion_b1",
    "logloss_modelo", "logloss_b2", "logloss_b1", "brier_modelo", "brier_b2", "brier_b1",
    "dif_logloss_b2", "dif_logloss_b1", "dif_brier_b2", "dif_brier_b1",
    "mejora_logloss_b2", "mejora_logloss_b1", "bss_b2", "bss_b1",
    "gana_logloss_b2", "gana_brier_b2", "gana_logloss_b1", "gana_brier_b1"]
RESUMENES = [
    "nivel", "grupo", "rotulo", "casillas", "filas", "positivas",
    "victorias_logloss_b2", "victorias_brier_b2", "victorias_logloss_b1", "victorias_brier_b1",
    "mejora_logloss_b2_casillas", "mejora_logloss_b2_filas", "bss_b2_casillas", "bss_b2_filas",
    "mejora_logloss_b1_casillas", "mejora_logloss_b1_filas", "bss_b1_casillas", "bss_b1_filas",
    "calibracion_modelo_casillas", "calibracion_modelo_filas", "calibracion_b2_casillas", "calibracion_b2_filas"]
TRANSICION = ["anio", "estrato", "rotulo", "filas", "positivas", "logloss_modelo", "logloss_b2", "brier_modelo",
              "brier_b2", "mejora_logloss_b2", "bss_b2", "ap_modelo", "ap_b2"]
EVENTOS = ["municipality_code", "tipo", "semana", "anio", "regimen", "rotulo", "evaluable", "censura",
           "avisado_modelo", "avisado_canal", "confirmacion_modelo", "confirmacion_canal",
           "anticipacion_modelo", "anticipacion_canal", "delta_w"]
ALERTAS = ["sistema", "municipality_code", "inicio", "confirmacion", "fin", "continuidad", "juzgable",
           "falsa_alarma", "seguida_de_epidemia", "anio", "regimen", "rotulo"]
SERIE = ["municipality_code", "t", "target_week", "anio", "regimen", "rotulo", "casos_objetivo", "central_objetivo",
         "u1e_objetivo", "zona_t", "y", "p_modelo", "p_b2", "p_b1", "emision_modelo", "emision_canal",
         "alerta_modelo", "alerta_canal"]


def _mondays(year: int) -> list[date]:
    first = date(year, 1, 1)
    first += timedelta(days=(7 - first.weekday()) % 7)
    weeks = []
    while first.year == year:
        weeks.append(first)
        first += WEEK
    return weeks


def _scenario(year: int):
    s = _mondays(year + 1)[0]
    weeks = _mondays(year - 1) + _mondays(year) + _mondays(year + 1)
    exceed = set(_mondays(year + 1)[:26])
    alert = {s - 7 * WEEK, s - 6 * WEEK}
    zone = {week: "epidemia" if week in exceed else "alerta" if week in alert else "seguridad" for week in weeks}
    observed = pd.DataFrame({
        "municipality_code": CODE, "week_start": weeks,
        "target_u1e": np.array([1 if week in exceed else 0 for week in weeks], dtype="int8"),
        "channel_eligible": True, "channel_zone": [zone[week] for week in weeks],
        "confirmed_records_screening": [10 if week in exceed else 2 for week in weeks],
        "epico_central": 3.0, "epico_upper": 8.0})
    targets = _mondays(year) + _mondays(year + 1)
    rows = pd.DataFrame({"municipality_code": CODE, "t": [w - 8 * WEEK for w in targets], "target_week": targets})
    rows["zona_t"] = [zone[t] for t in rows["t"]]
    rows["y"] = [1 if w in exceed else 0 for w in targets]
    rows.index = pd.RangeIndex(100, 100 + len(rows))
    second = np.array([w.year == year + 1 for w in targets])
    p = {"modelo": np.where(second, 0.6, 0.01), "B2": np.where(second, 0.5, 0.02), "B1": np.full(len(rows), 0.3)}
    predictions = pd.concat([rows[["municipality_code", "t", "target_week", "y"]].assign(
        sistema=system, candidato="cand" if system == "modelo" else system, p=values) for system, values in p.items()],
        ignore_index=True)
    return rows, predictions, observed, s


def _access(tmp_path):
    from src.forecasting.guard import RecipeAccess

    return RecipeAccess(test_years=(2023, 2025), channel_version="0" * 32,
                        recipe_path=tmp_path / "receta_inexistente.yaml")


def _run(tmp_path, year=2019, tau=TAU, rotulo="usado_para_elegir", **changes):
    from src.forecasting.diagnostics import diagnostic_inputs

    rows, predictions, observed, s = _scenario(year)
    rows, predictions = changes.get("rows", rows), changes.get("predictions", predictions)
    return diagnostic_inputs(rows, predictions, observed, tau=tau, k=K, window=W, clip=CLIP,
                             access=_access(tmp_path), rotulo=rotulo), s


def _ll(y_rate, p):
    return -(y_rate * math.log(p) + (1 - y_rate) * math.log(1 - p))


def test_columns_of_the_six_inputs(tmp_path):
    diag, _ = _run(tmp_path)
    assert list(diag.casillas.columns) == CASILLAS
    assert list(diag.resumenes.columns) == RESUMENES
    assert list(diag.transicion.columns) == TRANSICION
    assert list(diag.eventos.columns) == EVENTOS
    assert list(diag.alertas.columns) == ALERTAS
    assert list(diag.serie.columns) == SERIE


def test_cells_by_hand(tmp_path):
    diag, _ = _run(tmp_path)
    cells = diag.casillas.set_index("anio")
    assert list(cells.index) == [2019, 2020]
    first, second = cells.loc[2019], cells.loc[2020]
    assert (first["regimen"], second["regimen"]) == ("tranquila", "epidemica")
    assert (first["rotulo"], first["candidato"]) == ("usado_para_elegir", "cand")
    assert (first["filas"], first["positivas"], second["filas"], second["positivas"]) == (52, 0, 52, 26)
    assert first["prevalencia"] == 0 and second["prevalencia"] == 0.5
    assert first["p_medio_modelo"] == pytest.approx(0.01, rel=1e-12)
    assert first["calibracion_modelo"] == pytest.approx(0.01, rel=1e-12)
    assert second["calibracion_modelo"] == pytest.approx(0.1, rel=1e-12)
    assert second["calibracion_b2"] == pytest.approx(0.0, abs=1e-12)
    assert second["calibracion_b1"] == pytest.approx(-0.2, rel=1e-12)

    assert first["logloss_modelo"] == pytest.approx(-math.log(0.99), rel=1e-12)
    assert first["logloss_b2"] == pytest.approx(-math.log(0.98), rel=1e-12)
    assert first["logloss_b1"] == pytest.approx(-math.log(0.7), rel=1e-12)
    assert second["logloss_modelo"] == pytest.approx(_ll(0.5, 0.6), rel=1e-12)
    assert second["logloss_b2"] == pytest.approx(math.log(2), rel=1e-12)
    assert first["brier_modelo"] == pytest.approx(0.0001, rel=1e-12)
    assert first["brier_b2"] == pytest.approx(0.0004, rel=1e-12)
    assert second["brier_modelo"] == pytest.approx(0.26, rel=1e-12)
    assert second["brier_b2"] == pytest.approx(0.25, rel=1e-12)
    assert second["brier_b1"] == pytest.approx(0.29, rel=1e-12)

    assert first["dif_logloss_b2"] == pytest.approx(math.log(0.98) - math.log(0.99), rel=1e-12)
    assert second["dif_brier_b2"] == pytest.approx(0.01, rel=1e-9)
    assert second["dif_brier_b1"] == pytest.approx(-0.03, rel=1e-9)
    assert first["mejora_logloss_b2"] == pytest.approx(1 - math.log(0.99) / math.log(0.98), rel=1e-12)
    assert first["bss_b2"] == pytest.approx(0.75, rel=1e-12)
    assert second["bss_b2"] == pytest.approx(-0.04, rel=1e-9)
    assert bool(first["gana_logloss_b2"]) and bool(first["gana_brier_b2"])
    assert not bool(second["gana_logloss_b2"]) and not bool(second["gana_brier_b2"])
    assert bool(second["gana_logloss_b1"]) and bool(second["gana_brier_b1"])


def test_a_tie_is_not_a_win(tmp_path):
    rows, predictions, _, _ = _scenario(2019)
    b2 = predictions.loc[predictions["sistema"] == "B2", "p"].to_numpy()
    predictions.loc[predictions["sistema"] == "modelo", "p"] = b2
    diag, _ = _run(tmp_path, predictions=predictions)
    assert (diag.casillas["dif_brier_b2"] == 0).all() and (diag.casillas["dif_logloss_b2"] == 0).all()
    assert not diag.casillas["gana_brier_b2"].any() and not diag.casillas["gana_logloss_b2"].any()


def test_two_aggregations_can_disagree_in_sign(tmp_path):
    diag, _ = _run(tmp_path)
    summaries = diag.resumenes
    assert sorted(set(summaries["nivel"])) == ["anio", "global", "municipio", "regimen"]
    total = summaries[summaries["nivel"] == "global"].iloc[0]
    assert (total["grupo"], total["casillas"], total["filas"], total["positivas"]) == ("todos", 2, 104, 26)
    assert total["bss_b2_casillas"] == pytest.approx(0.355, rel=1e-9)
    assert total["bss_b2_filas"] == pytest.approx(1 - 0.13005 / 0.1252, rel=1e-9)
    assert total["bss_b2_casillas"] > 0 > total["bss_b2_filas"]
    assert (total["victorias_brier_b2"], total["victorias_brier_b1"]) == (1, 2)
    assert total["calibracion_modelo_filas"] == pytest.approx((0.01 + 0.6) / 2 - 0.25, rel=1e-12)
    by_regime = summaries[summaries["nivel"] == "regimen"].set_index("grupo")
    assert sorted(by_regime.index) == ["epidemica", "tranquila"]
    assert by_regime.loc["tranquila", "bss_b2_casillas"] == pytest.approx(0.75, rel=1e-12)
    assert by_regime.loc["tranquila", "bss_b2_filas"] == pytest.approx(0.75, rel=1e-12)


def test_transition_strata(tmp_path):
    diag, s = _run(tmp_path)
    strata = diag.transicion.set_index(["anio", "estrato"])
    assert list(strata.index) == [("2019", "fuera"), ("2019", "dentro"), ("2020", "fuera"), ("2020", "dentro"),
                                  ("todos", "fuera"), ("todos", "dentro")]
    # Inside = zona_t "epidemia": emissions t in the 26 exceeding weeks of 2020 whose target is still in 2020.
    assert strata.loc[("2020", "dentro"), "filas"] == 26 and strata.loc[("2020", "dentro"), "positivas"] == 18
    assert strata.loc[("2019", "dentro"), "filas"] == 0 and math.isnan(strata.loc[("2019", "dentro"), "bss_b2"])
    assert strata.loc[("2019", "fuera"), "positivas"] == 0 and math.isnan(strata.loc[("2019", "fuera"), "ap_modelo"])
    assert strata.loc[("todos", "fuera"), "filas"] + strata.loc[("todos", "dentro"), "filas"] == 104


def test_events_and_delta_w(tmp_path):
    diag, s = _run(tmp_path)
    onsets = diag.eventos[diag.eventos["tipo"] == "inicio"]
    assert len(onsets) == 1
    onset = onsets.iloc[0]
    assert (onset["semana"], onset["anio"], onset["regimen"]) == (s, s.year, "epidemica")
    assert bool(onset["evaluable"]) and bool(onset["avisado_modelo"]) and bool(onset["avisado_canal"])
    assert (onset["confirmacion_modelo"], onset["confirmacion_canal"]) == (s - 7 * WEEK, s - 6 * WEEK)
    assert (onset["anticipacion_modelo"], onset["anticipacion_canal"], onset["delta_w"]) == (7, 6, 1)
    assert list(diag.eventos.loc[diag.eventos["tipo"] == "entrada", "semana"]) == [s]


def test_delta_w_is_nan_when_only_one_system_warns(tmp_path):
    diag, _ = _run(tmp_path, tau=0.7)   # the model never reaches tau
    onset = diag.eventos[diag.eventos["tipo"] == "inicio"].iloc[0]
    assert not bool(onset["avisado_modelo"]) and bool(onset["avisado_canal"])
    assert math.isnan(onset["delta_w"])


def test_alerts_of_both_systems(tmp_path):
    diag, s = _run(tmp_path)
    alerts = diag.alertas.set_index("sistema")
    assert sorted(alerts.index) == ["canal", "modelo"]
    model, channel = alerts.loc["modelo"], alerts.loc["canal"]
    assert (model["inicio"], model["confirmacion"]) == (s - 8 * WEEK, s - 7 * WEEK)
    assert (channel["inicio"], channel["confirmacion"], channel["fin"]) == (s - 7 * WEEK, s - 6 * WEEK, s - 6 * WEEK)
    for alert in (model, channel):
        assert not bool(alert["continuidad"]) and bool(alert["juzgable"]) and not bool(alert["falsa_alarma"])
        assert bool(alert["seguida_de_epidemia"]) and alert["anio"] == s.year and alert["regimen"] == "epidemica"


def test_weekly_series(tmp_path):
    diag, s = _run(tmp_path)
    serie = diag.serie
    assert len(serie) == 104 and not serie.duplicated(["municipality_code", "t"]).any()
    assert list(serie["t"]) == sorted(serie["t"])
    by_t = serie.set_index("t")
    assert by_t.loc[s - 8 * WEEK, "casos_objetivo"] == 10 and by_t.loc[s - 9 * WEEK, "casos_objetivo"] == 2
    assert (serie["central_objetivo"] == 3.0).all() and (serie["u1e_objetivo"] == 8.0).all()
    assert list(serie.loc[serie["alerta_canal"], "t"]) == [s - 6 * WEEK]
    model_alert = serie.loc[serie["alerta_modelo"], "t"]
    assert model_alert.min() == s - 7 * WEEK and model_alert.max() == serie["t"].max()
    assert int(serie["emision_canal"].sum()) == 2


def test_rotulo_must_match_the_period(tmp_path):
    with pytest.raises(EpidemiologyValidationError):
        _run(tmp_path, rotulo="prueba")
    with pytest.raises(EpidemiologyValidationError):
        _run(tmp_path, rotulo="desarrollo")


def test_predictions_must_cover_exactly_the_rows(tmp_path):
    rows, predictions, _, _ = _scenario(2019)
    missing = predictions.drop(predictions.index[(predictions["sistema"] == "B1")][:1])
    with pytest.raises(EpidemiologyValidationError):
        _run(tmp_path, predictions=missing)


def test_test_period_requires_the_frozen_recipe(tmp_path):
    from src.forecasting.guard import FrozenRecipeError

    with pytest.raises(FrozenRecipeError):
        _run(tmp_path, year=2023, rotulo="prueba")
