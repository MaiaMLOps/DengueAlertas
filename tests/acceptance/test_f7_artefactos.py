"""F7 acceptance with synthetic inputs: precomputed artifacts for the API (contrato §6.1, §8.4, §8.8, §9.4 and
§10.18; api_salida_d10.md §2, §6 and §7; especificación §6.F7). Approved by Jesús on 2026-11-12 (U7.1).

Rules:
- history: only rolling-origin predictions (origen "desarrollo_origen_movil" or "prueba_origen_movil"); the final
  model never enters it. Development shows the recipe configuration (D2, «usado para elegir»). The alert state is
  computed per period (D4): the first emission of a period never inherits a run from the previous one.
- alert state of an emission (api §2): "sin_alerta" if p < tau; "senal_en_observacion" if p >= tau but the run of
  consecutive emissions >= tau is shorter than k; otherwise "alerta_confirmada", or "continuidad_epidemica" when
  the observed zone at that emission is "epidemia". The emissions in an alert state are exactly the weeks from t_c
  to the end of the alerts of `events.alerts`.
- performance card (api §7): only the test period; levels in the API spelling (mejora_consistente,
  tendencia_favorable, sin_mejora); the global conclusion of each weekly comparison is the lower of the Brier and
  log-loss levels (contrato §8.4); per municipality there is no statistical conclusion but the descriptive level
  of §8.8: "favorable" if the global metric is favourable and the model wins more than half of the cells,
  "mixto" if only one holds, "desfavorable" if none; the lower of Brier and log-loss is adopted.
- final model: trained with all the rows of the municipality (target year >= 2012) and the recipe.

API frozen here (`src.forecasting.artifacts`):
- `alert_states(p, zones, *, tau, k) -> list[str]` for one municipality's chronological emissions;
- `history_table(rows, predictions, observed, *, origen, tau, k, window, access) -> History(filas, eventos)`;
- `descriptive_level(favorable, wins, n) -> str` and `lower_descriptive(first, second) -> str`;
- `performance_card(rows, predictions, observed, *, config, tau, access) -> dict` with keys version_esquema,
  global and municipios (one card per municipality);
- `train_final_models(examples, recipe, *, config) -> dict[str, FinalModel]` with fields model, municipality_code,
  familia, candidato, hiperparametros, filas, ultima_emision and ultima_objetivo.
"""

from datetime import date, timedelta
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from src.epidemiology.config import EpidemiologyValidationError

pytestmark = pytest.mark.acceptance

WEEK = timedelta(weeks=1)
TAU, K, W = 0.5, 2, 12
HISTORY_COLUMNS = ["municipality_code", "semana_emision", "semana_objetivo", "periodo", "origen", "probabilidad",
                   "b2_estado_canal", "b1_climatologia_estacional", "estado_alerta", "casos_objetivo",
                   "central_objetivo", "u1e_objetivo", "zona_objetivo", "observado_objetivo"]
EVENT_COLUMNS = ["municipality_code", "periodo", "tipo_evento", "sistema", "semana", "semana_alerta",
                 "anticipacion_semanas"]
EVENT_TYPES = {"inicio_avisado", "inicio_no_avisado", "falsa_alarma", "alerta_continuidad", "inicio_censurado",
               "alerta_censurada"}
API_LEVELS = {"mejora_consistente", "tendencia_favorable", "sin_mejora"}
_RANK = {"sin_mejora": 0, "tendencia_favorable": 1, "mejora_consistente": 2}


# --------------------------------------------------------------------------- synthetic period
def _mondays(year: int) -> list[date]:
    first = date(year, 1, 1)
    first += timedelta(days=(7 - first.weekday()) % 7)
    weeks = []
    while first.year == year:
        weeks.append(first)
        first += WEEK
    return weeks


def _period(years, codes_with_onset, codes_without, seed=7):
    """Rows, long predictions (modelo, sin_clima, B2, B1) and observed series. A municipality with onset has
    target_u1e = 1 in the first 10 Mondays of years[1], zone "alerta" 7 and 6 weeks before; the model emits 0.6
    from 8 weeks before the onset to the end of the epidemic weeks and random values below 0.5 elsewhere."""
    rng = np.random.default_rng(seed)
    weeks = [w for year in range(years[0] - 1, years[-1] + 1) for w in _mondays(year)]
    targets = [w for year in years for w in _mondays(year)]
    observed, rows, predictions = [], [], []
    for code in list(codes_with_onset) + list(codes_without):
        s = _mondays(years[1])[0]
        exceed = set(_mondays(years[1])[:10]) if code in codes_with_onset else set()
        alert = {s - 7 * WEEK, s - 6 * WEEK} if code in codes_with_onset else set()
        zone = {w: "epidemia" if w in exceed else "alerta" if w in alert else "seguridad" for w in weeks}
        observed.append(pd.DataFrame({
            "municipality_code": code, "week_start": weeks,
            "target_u1e": np.array([1 if w in exceed else 0 for w in weeks], dtype="int8"),
            "channel_eligible": True, "channel_zone": [zone[w] for w in weeks],
            "confirmed_records_screening": [10 if w in exceed else 2 for w in weeks],
            "epico_central": 3.0, "epico_upper": 8.0}))
        own = pd.DataFrame({"municipality_code": code, "t": [w - 8 * WEEK for w in targets], "target_week": targets})
        own["zona_t"] = [zone[t] for t in own["t"]]
        own["y"] = [1 if w in exceed else 0 for w in targets]
        rows.append(own)
        high = np.array([code in codes_with_onset and s - 8 * WEEK <= t <= s + 9 * WEEK for t in own["t"]])
        p = {"modelo": np.where(high, 0.6, rng.uniform(0.01, 0.45, len(own))),
             "sin_clima": rng.uniform(0.01, 0.6, len(own)), "B2": rng.uniform(0.05, 0.5, len(own)),
             "B1": rng.uniform(0.05, 0.5, len(own))}
        for system, values in p.items():
            predictions.append(own[["municipality_code", "t", "target_week", "y"]].assign(
                sistema=system, candidato=system, p=values))
    rows = pd.concat(rows, ignore_index=True)
    rows.index = pd.RangeIndex(500, 500 + len(rows))
    return rows, pd.concat(predictions, ignore_index=True), pd.concat(observed, ignore_index=True)


def _config():
    from src.forecasting.config import load_config

    return load_config()


def _access(tmp_path, *, with_recipe):
    from src.forecasting.guard import RecipeAccess, recipe_hash

    config = _config()
    path = tmp_path / "receta.yaml"
    if with_recipe:
        recipe = {"municipios": {}, "procedencia": {"commit_codigo": "a" * 40, "commit_contrato": "b" * 40,
                                                    "version_dvc_canal": config.datos.version_dvc_canal},
                  "aprobacion": {"nombre": "Receta sintética", "fecha": "2026-11-12"}}
        recipe["hash_sha256"] = recipe_hash(recipe)
        path.write_text(yaml.safe_dump(recipe, allow_unicode=True), encoding="utf-8")
    return RecipeAccess.from_config(config, recipe_path=path)


# --------------------------------------------------------------------------- alert states
def test_alert_states_follow_api_section_2():
    from src.forecasting.artifacts import alert_states

    p = [0.1, 0.6, 0.2, 0.6, 0.7, 0.8, 0.9, 0.1]
    zones = ["seguridad"] * 5 + ["epidemia", "alerta", "seguridad"]
    assert alert_states(p, zones, tau=TAU, k=K) == [
        "sin_alerta", "senal_en_observacion", "sin_alerta", "senal_en_observacion", "alerta_confirmada",
        "continuidad_epidemica", "alerta_confirmada", "sin_alerta"]


def test_history_alert_states_match_the_events_module(tmp_path):
    from src.forecasting.artifacts import history_table
    from src.forecasting.events import alerts, model_positive

    rows, predictions, observed = _period((2019, 2020), ["00001"], ["00002"])
    history = history_table(rows, predictions, observed, origen="desarrollo_origen_movil", tau=TAU, k=K, window=W,
                            access=_access(tmp_path, with_recipe=False))
    filas = history.filas
    assert list(filas.columns) == HISTORY_COLUMNS
    assert len(filas) == len(rows) and set(filas["periodo"]) == {"desarrollo"}
    assert set(filas["origen"]) == {"desarrollo_origen_movil"}
    for code in ("00001", "00002"):
        own_rows = rows[rows["municipality_code"] == code].sort_values("t")
        own_p = predictions[(predictions["sistema"] == "modelo") & (predictions["municipality_code"] == code)]
        own_p = own_p.set_index("t").loc[list(own_rows["t"]), "p"].to_numpy()
        found = alerts(list(own_rows["t"]), model_positive(own_p, tau=TAU), list(own_rows["zona_t"]), k=K)
        expected = {t for alert in found for t in own_rows["t"] if alert.confirmation <= t <= alert.end}
        own = filas[filas["municipality_code"] == code].set_index("semana_emision")
        alerted = set(own.index[own["estado_alerta"].isin(["alerta_confirmada", "continuidad_epidemica"])])
        assert alerted == expected
        for alert in found:
            state = own.loc[alert.confirmation, "estado_alerta"]
            assert state == ("continuidad_epidemica" if alert.continuity else "alerta_confirmada")
        assert own["probabilidad"].tolist() == own_p.tolist()


def test_history_events_use_the_api_types(tmp_path):
    from src.forecasting.artifacts import history_table

    rows, predictions, observed = _period((2019, 2020), ["00001"], ["00002"])
    history = history_table(rows, predictions, observed, origen="desarrollo_origen_movil", tau=TAU, k=K, window=W,
                            access=_access(tmp_path, with_recipe=False))
    events = history.eventos
    assert list(events.columns) == EVENT_COLUMNS
    assert set(events["tipo_evento"]) <= EVENT_TYPES and set(events["sistema"]) <= {"modelo", "canal"}
    s = _mondays(2020)[0]
    warned = events[(events["tipo_evento"] == "inicio_avisado") & (events["semana"] == s)].set_index("sistema")
    assert warned.loc["modelo", "semana_alerta"] == s - 7 * WEEK and warned.loc["modelo", "anticipacion_semanas"] == 7
    assert warned.loc["canal", "semana_alerta"] == s - 6 * WEEK and warned.loc["canal", "anticipacion_semanas"] == 6


def test_history_state_is_computed_per_period(tmp_path):
    """D4: a run of emissions >= tau at the end of development does not carry into the first test emission."""
    from src.forecasting.artifacts import history_table

    rows, predictions, observed = _period((2023, 2024, 2025), [], ["00002"])
    first_t = rows["t"].min()
    predictions.loc[(predictions["sistema"] == "modelo") & (predictions["t"] == first_t), "p"] = 0.9
    history = history_table(rows, predictions, observed, origen="prueba_origen_movil", tau=TAU, k=K, window=W,
                            access=_access(tmp_path, with_recipe=True))
    first = history.filas.sort_values("semana_emision").iloc[0]
    assert first["estado_alerta"] == "senal_en_observacion" and first["periodo"] == "prueba"


@pytest.mark.parametrize("origen", ["modelo_final", "carga_usuario", "desarrollo"])
def test_history_rejects_other_origins(tmp_path, origen):
    from src.forecasting.artifacts import history_table

    rows, predictions, observed = _period((2019, 2020), ["00001"], [])
    with pytest.raises(EpidemiologyValidationError):
        history_table(rows, predictions, observed, origen=origen, tau=TAU, k=K, window=W,
                      access=_access(tmp_path, with_recipe=False))


def test_history_rejects_final_model_predictions(tmp_path):
    from src.forecasting.artifacts import history_table

    rows, predictions, observed = _period((2019, 2020), ["00001"], [])
    predictions["origen"] = "desarrollo_origen_movil"
    predictions.loc[predictions.index[:3], "origen"] = "modelo_final"
    with pytest.raises(EpidemiologyValidationError):
        history_table(rows, predictions, observed, origen="desarrollo_origen_movil", tau=TAU, k=K, window=W,
                      access=_access(tmp_path, with_recipe=False))


def test_history_origin_must_match_the_period(tmp_path):
    from src.forecasting.artifacts import history_table

    rows, predictions, observed = _period((2019, 2020), ["00001"], [])
    with pytest.raises(EpidemiologyValidationError):
        history_table(rows, predictions, observed, origen="prueba_origen_movil", tau=TAU, k=K, window=W,
                      access=_access(tmp_path, with_recipe=False))


# --------------------------------------------------------------------------- descriptive level (§8.8)
@pytest.mark.parametrize("favorable, wins, expected", [
    (True, 2, "favorable"), (True, 3, "favorable"), (True, 1, "mixto"), (False, 2, "mixto"),
    (False, 1, "desfavorable"), (False, 0, "desfavorable")])
def test_descriptive_level(favorable, wins, expected):
    from src.forecasting.artifacts import descriptive_level

    assert descriptive_level(favorable, wins, 3) == expected


def test_lower_descriptive():
    from src.forecasting.artifacts import lower_descriptive

    assert lower_descriptive("favorable", "mixto") == "mixto"
    assert lower_descriptive("desfavorable", "favorable") == "desfavorable"
    assert lower_descriptive("mixto", "mixto") == "mixto"


# --------------------------------------------------------------------------- performance card
@pytest.fixture
def card(tmp_path):
    from src.forecasting.artifacts import performance_card

    rows, predictions, observed = _period((2023, 2024, 2025), ["00001"], ["00002"])
    return performance_card(rows, predictions, observed, config=_config(), tau=TAU,
                            access=_access(tmp_path, with_recipe=True))


def test_card_without_recipe_is_refused(tmp_path):
    from src.forecasting.artifacts import performance_card
    from src.forecasting.guard import FrozenRecipeError

    rows, predictions, observed = _period((2023, 2024, 2025), ["00001"], [])
    with pytest.raises(FrozenRecipeError):
        performance_card(rows, predictions, observed, config=_config(), tau=TAU,
                         access=_access(tmp_path, with_recipe=False))


def test_card_has_the_fields_of_api_section_7(card):
    assert set(card) >= {"version_esquema", "global", "municipios"}
    assert sorted(card["municipios"]) == ["00001", "00002"]
    for item in [card["global"], *card["municipios"].values()]:
        assert {"estimacion", "eventos", "calibracion", "curva_amoc", "casillas", "notas"} <= set(item)
        estimation = item["estimacion"]
        assert {"bss_vs_b2", "bss_vs_b1", "mejora_logloss_vs_b2", "mejora_logloss_vs_b1"} <= set(estimation)
        for comparison in ("vs_b2", "vs_b1", "ablacion_clima"):
            assert {"nivel_conclusion", "casillas_ganadas", "casillas_total"} <= set(estimation[comparison]["conclusion"])
        events = item["eventos"]
        assert {"inicios_evaluables", "avisados_modelo", "avisados_canal", "mcnemar",
                "falsas_alarmas_por_municipio_anio", "anticipacion_mediana_semanas"} <= set(events)
        assert {"discrepantes", "a_favor_modelo", "nivel_conclusion"} <= set(events["mcnemar"])
        assert set(events["falsas_alarmas_por_municipio_anio"]) == {"modelo", "canal"}
        assert set(events["anticipacion_mediana_semanas"]) == {"modelo", "canal"}
        assert all({"probabilidad_media", "frecuencia_observada", "n"} <= set(b) for b in item["calibracion"])
        assert len(item["curva_amoc"]) == 19
        assert all({"tau", "deteccion", "falsas_alarmas_por_municipio_anio", "anticipacion_mediana_semanas"}
                   <= set(point) for point in item["curva_amoc"])
        assert all({"municipio", "anio", "brier_modelo", "brier_b2", "brier_b1", "logloss_modelo", "logloss_b2",
                    "gana_vs_b2"} <= set(cell) for cell in item["casillas"])
    assert len(card["global"]["casillas"]) == 6
    assert all(len(item["casillas"]) == 3 for item in card["municipios"].values())
    assert sum(b["n"] for b in card["global"]["calibracion"]) == 2 * sum(len(_mondays(y)) for y in (2023, 2024, 2025))


def test_global_levels_use_the_api_spelling_and_the_lower_level(card):
    estimation = card["global"]["estimacion"]
    for comparison in ("vs_b2", "vs_b1", "ablacion_clima"):
        conclusion = estimation[comparison]["conclusion"]
        assert {conclusion["nivel_conclusion"], conclusion["nivel_brier"], conclusion["nivel_logloss"]} <= API_LEVELS
        lower = min(conclusion["nivel_brier"], conclusion["nivel_logloss"], key=_RANK.get)
        assert conclusion["nivel_conclusion"] == lower
        assert conclusion["casillas_total"] == 6
    assert card["global"]["eventos"]["mcnemar"]["nivel_conclusion"] in API_LEVELS


def test_municipality_cards_use_the_descriptive_level(card):
    from src.forecasting.artifacts import descriptive_level, lower_descriptive

    for code, item in card["municipios"].items():
        estimation = item["estimacion"]
        conclusion = estimation["vs_b2"]["conclusion"]
        assert conclusion["nivel_conclusion"] is None
        cells = item["casillas"]
        wins_brier = sum(cell["brier_modelo"] < cell["brier_b2"] for cell in cells)
        wins_logloss = sum(cell["logloss_modelo"] < cell["logloss_b2"] for cell in cells)
        expected = lower_descriptive(descriptive_level(estimation["bss_vs_b2"] > 0, wins_brier, 3),
                                     descriptive_level(estimation["mejora_logloss_vs_b2"] > 0, wins_logloss, 3))
        assert conclusion["nivel_descriptivo"] == expected


def test_card_events_and_the_note_for_a_municipality_without_onsets(card):
    events = card["global"]["eventos"]
    assert events["inicios_evaluables"] == 1 and events["avisados_modelo"] == 1 and events["avisados_canal"] == 1
    assert card["municipios"]["00002"]["eventos"]["inicios_evaluables"] == 0
    assert any("00002" in note and "No hubo inicios de brote" in note for note in card["global"]["notas"])
    assert not any("00001" in note for note in card["global"]["notas"])


# --------------------------------------------------------------------------- final model
def test_final_models_use_all_rows_and_the_recipe():
    from src.forecasting.artifacts import train_final_models
    from src.forecasting.dataset import FEATURES
    from src.forecasting.models import fit_model, predict_positive
    from src.epidemiology.channel_epico import ZONES

    rng = np.random.default_rng(3)
    t = [date(2012, 3, 19) + i * WEEK for i in range(711)]
    examples = pd.DataFrame({"municipality_code": "00001", "t": t, "target_week": [w + 8 * WEEK for w in t]})
    for name in FEATURES:
        if name != "zona_t":
            examples[name] = rng.normal(size=len(examples))
    examples["zona_t"] = rng.choice(list(ZONES), size=len(examples))
    examples["y"] = (rng.random(len(examples)) < 0.3).astype("int8")
    spec = {"familia": "logistica", "candidato": "logistica_03", "hiperparametros": {"C": 0.3, "max_iter": 5000}}
    final = train_final_models(examples, {"municipios": {"00001": spec}}, config=_config())["00001"]
    assert (final.filas, final.ultima_emision, final.ultima_objetivo) == (711, max(t), max(t) + 8 * WEEK)
    assert (final.familia, final.candidato, final.hiperparametros) == ("logistica", "logistica_03",
                                                                       {"C": 0.3, "max_iter": 5000})
    expected = predict_positive(fit_model("logistica", spec["hiperparametros"], examples), examples)
    assert np.array_equal(predict_positive(final.model, examples), expected)
