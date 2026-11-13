"""F7b acceptance with synthetic inputs: performance card and final models with the recipe of D7.2 (contrato §4.4,
§6.1, §6.6, §8.4, §8.8, §9.4 and §10.18; api_salida_d10.md §7; especificación §6.F7; DV-02). Replaces the five tests
of test_f7_artefactos.py superseded by DV-02 (four of the card, one of the final model); the history, alert-state and
descriptive-level tests of that file stay in force. Approved by Jesús on 2026-11-12 (U7.4): same claims as F7, with
the ablation system named "ablacion" and the feature set read from the recipe, plus the changes marked NEW.

Rules:
- performance card (api §7): only the test period; levels in the API spelling (mejora_consistente,
  tendencia_favorable, sin_mejora); the global conclusion of each weekly comparison (vs_b2, vs_b1, ablacion_clima,
  the last one against the system "ablacion") is the lower of the Brier and log-loss levels (contrato §8.4); per
  municipality there is no statistical conclusion but the descriptive level of §8.8, computed from the favourable
  global metric of the municipality (favorable_brier, favorable_logloss: its Brier or log-loss over its rows lower
  than the reference's, i.e. BSS > 0) and its wins of 3 cells, the lower of Brier and log-loss adopted (NEW: the two
  booleans are reported in each municipal conclusion); a municipality without evaluable onsets gets the fixed note;
- NEW: the global card uses the same figures as evaluate.score_summary (the resumen_prueba.json of the single run):
  BSS and log-loss improvements, wins per comparison and events;
- final model: per municipality, the family, hyperparameters and feature set of the recipe (municipios[code]),
  trained with all its rows (target year >= 2012); NEW: FinalModel has conjunto_variables, and a recipe without it
  (the F6 format) fails;
- NEW: write_final_models(models, folder, *, recipe_hash) writes <code>.joblib and manifiesto.json (receta_hash and,
  per municipality, familia, candidato, conjunto_variables, hiperparametros, filas, ultima_emision and
  ultima_objetivo as ISO dates); a saved model predicts exactly like the fitted one; nothing is overwritten.

API frozen here (`src.forecasting.artifacts`): performance_card(rows, predictions, observed, *, config, tau, access)
with predictions of the systems modelo, ablacion, B2 and B1; train_final_models(examples, recipe, *, config) ->
dict[str, FinalModel]; write_final_models(models, folder, *, recipe_hash) -> Path of the manifest.
"""

from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.epidemiology.config import EpidemiologyValidationError

pytestmark = pytest.mark.acceptance

WEEK = timedelta(weeks=1)
TAU, K, W = 0.5, 2, 12
API_LEVELS = {"mejora_consistente", "tendencia_favorable", "sin_mejora"}
_RANK = {"sin_mejora": 0, "tendencia_favorable": 1, "mejora_consistente": 2}
_SPACED = {"mejora consistente": "mejora_consistente", "tendencia favorable": "tendencia_favorable",
           "sin mejora": "sin_mejora"}
SYSTEMS = ("modelo", "ablacion", "B2", "B1")


def _mondays(year: int) -> list[date]:
    first = date(year, 1, 1)
    first += timedelta(days=(7 - first.weekday()) % 7)
    weeks = []
    while first.year == year:
        weeks.append(first)
        first += WEEK
    return weeks


def _period(years, codes_with_onset, codes_without, seed=7):
    """Rows, long predictions (modelo, ablacion, B2, B1) and observed series (as in test_f7_artefactos.py)."""
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
             "ablacion": rng.uniform(0.01, 0.6, len(own)), "B2": rng.uniform(0.05, 0.5, len(own)),
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


def _access(tmp_path):
    from src.forecasting.guard import RecipeAccess, recipe_hash

    config = _config()
    path = tmp_path / "receta.yaml"
    recipe = {"municipios": {}, "procedencia": {"commit_codigo": "a" * 40, "commit_contrato": "b" * 40,
                                                "version_dvc_canal": config.datos.version_dvc_canal},
              "aprobacion": {"nombre": "Receta sintética", "fecha": "2026-11-12"}}
    recipe["hash_sha256"] = recipe_hash(recipe)
    path.write_text(yaml.safe_dump(recipe, allow_unicode=True), encoding="utf-8")
    return RecipeAccess.from_config(config, recipe_path=path)


_CARD = {}


@pytest.fixture
def card(tmp_path):
    from src.forecasting.artifacts import performance_card
    from src.forecasting.evaluate import score_summary

    if "card" not in _CARD:
        rows, predictions, observed = _period((2023, 2024, 2025), ["00001"], ["00002"])
        access = _access(tmp_path)
        result = performance_card(rows, predictions, observed, config=_config(), tau=TAU, access=access)
        _, summary = score_summary(rows, predictions, observed, config=_config(), tau=TAU, access=access)
        _CARD["card"] = (result, summary)
    return _CARD["card"]


# --------------------------------------------------------------------------- performance card
def test_card_has_the_fields_of_api_section_7(card):
    card, _ = card
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
    assert card["global"]["tau"] == TAU and card["global"]["periodo"] == "prueba"


def test_global_levels_use_the_api_spelling_and_the_lower_level(card):
    card, _ = card
    estimation = card["global"]["estimacion"]
    for comparison in ("vs_b2", "vs_b1", "ablacion_clima"):
        conclusion = estimation[comparison]["conclusion"]
        assert {conclusion["nivel_conclusion"], conclusion["nivel_brier"], conclusion["nivel_logloss"]} <= API_LEVELS
        assert conclusion["nivel_conclusion"] == min(conclusion["nivel_brier"], conclusion["nivel_logloss"],
                                                     key=_RANK.get)
        assert conclusion["casillas_total"] == 6
    assert card["global"]["eventos"]["mcnemar"]["nivel_conclusion"] in API_LEVELS


def test_global_card_uses_the_figures_of_score_summary(card):
    card, summary = card
    estimation = card["global"]["estimacion"]
    for key in ("bss_vs_b2", "bss_vs_b1", "mejora_logloss_vs_b2", "mejora_logloss_vs_b1"):
        assert estimation[key] == pytest.approx(summary["estimacion"][key], rel=1e-12)
    for comparison in ("vs_b2", "vs_b1", "ablacion_clima"):
        conclusion, reference = estimation[comparison]["conclusion"], summary["conclusiones"][comparison]
        assert conclusion["nivel_conclusion"] == _SPACED[reference["nivel"]]
        assert conclusion["casillas_ganadas"] == reference["casillas_ganadas_brier"]
        assert conclusion["casillas_ganadas_logloss"] == reference["casillas_ganadas_logloss"]
    events, reference = card["global"]["eventos"], summary["eventos"]
    assert (events["inicios_evaluables"], events["avisados_modelo"], events["avisados_canal"]) == (
        reference["inicios_evaluables"], reference["avisados_modelo"], reference["avisados_canal"])
    assert events["mcnemar"]["nivel_conclusion"] == _SPACED[reference["mcnemar"]["nivel"]]


def test_municipality_cards_use_the_descriptive_level(card):
    from src.forecasting.artifacts import descriptive_level, lower_descriptive

    card, _ = card
    for item in card["municipios"].values():
        estimation = item["estimacion"]
        for comparison, reference in (("vs_b2", "b2"), ("vs_b1", "b1"), ("ablacion_clima", "ablacion")):
            conclusion = estimation[comparison]["conclusion"]
            assert conclusion["nivel_conclusion"] is None
            cells = item["casillas"]
            wins_brier = sum(cell["brier_modelo"] < cell[f"brier_{reference}"] for cell in cells)
            wins_logloss = sum(cell["logloss_modelo"] < cell[f"logloss_{reference}"] for cell in cells)
            assert conclusion["casillas_ganadas"] == wins_brier
            assert conclusion["casillas_ganadas_logloss"] == wins_logloss
            assert conclusion["nivel_descriptivo"] == lower_descriptive(
                descriptive_level(conclusion["favorable_brier"], wins_brier, 3),
                descriptive_level(conclusion["favorable_logloss"], wins_logloss, 3))


def test_card_events_and_the_note_for_a_municipality_without_onsets(card):
    card, _ = card
    events = card["global"]["eventos"]
    assert events["inicios_evaluables"] == 1 and events["avisados_modelo"] == 1 and events["avisados_canal"] == 1
    assert card["municipios"]["00002"]["eventos"]["inicios_evaluables"] == 0
    assert any("00002" in note and "No hubo inicios de brote" in note for note in card["global"]["notas"])
    assert not any("00001" in note for note in card["global"]["notas"])


def test_designed_card_adopts_the_lower_level_and_compares_with_the_ablation(tmp_path):
    """Designed predictions: the model beats B2 and B1 in the 6 cells by Brier, but one confident miss (p = 0 on a
    positive row of 00001-2024) loses that cell by log-loss: Brier 6 of 6 (mejora_consistente, threshold 6),
    log-loss 5 of 6 (tendencia_favorable) -> tendencia_favorable. The ablation is almost perfect: 0 of 6 wins."""
    from src.forecasting.artifacts import performance_card
    from src.forecasting.evaluate import score_summary

    rows, predictions, observed = _period((2023, 2024, 2025), ["00001"], ["00002"])
    y = predictions["y"].to_numpy()
    values = {"B2": np.full(len(y), 0.3), "B1": np.full(len(y), 0.4), "ablacion": y * 0.98 + 0.01,
              "modelo": y * 0.9 + (1 - y) * 0.05}
    for system, p in values.items():
        mask = (predictions["sistema"] == system).to_numpy()
        predictions.loc[mask, "p"] = p[mask]
    miss = predictions.index[(predictions["sistema"] == "modelo") & (predictions["y"] == 1)][0]
    predictions.loc[miss, "p"] = 0.0
    access = _access(tmp_path)
    card = performance_card(rows, predictions, observed, config=_config(), tau=TAU, access=access)
    _, summary = score_summary(rows, predictions, observed, config=_config(), tau=TAU, access=access)
    vs_b2 = card["global"]["estimacion"]["vs_b2"]["conclusion"]
    assert (vs_b2["nivel_brier"], vs_b2["nivel_logloss"], vs_b2["nivel_conclusion"]) == (
        "mejora_consistente", "tendencia_favorable", "tendencia_favorable")
    assert (vs_b2["casillas_ganadas"], vs_b2["casillas_ganadas_logloss"]) == (6, 5)
    ablation = card["global"]["estimacion"]["ablacion_clima"]["conclusion"]
    assert ablation["casillas_ganadas"] == summary["conclusiones"]["ablacion_clima"]["casillas_ganadas_brier"] == 0
    assert card["global"]["estimacion"]["vs_b1"]["conclusion"]["casillas_ganadas"] == 6


def test_card_needs_the_four_systems(tmp_path):
    from src.forecasting.artifacts import performance_card

    rows, predictions, observed = _period((2023, 2024, 2025), ["00001"], [])
    predictions = predictions[predictions["sistema"] != "ablacion"]
    with pytest.raises(EpidemiologyValidationError):
        performance_card(rows, predictions, observed, config=_config(), tau=TAU, access=_access(tmp_path))


# --------------------------------------------------------------------------- final models
def _examples():
    from src.epidemiology.channel_epico import ZONES
    from src.forecasting.dataset import FEATURES

    rng = np.random.default_rng(3)
    t = [date(2012, 3, 19) + i * WEEK for i in range(711)]
    examples = pd.DataFrame({"municipality_code": "00001", "t": t, "target_week": [w + 8 * WEEK for w in t]})
    for name in FEATURES:
        if name != "zona_t":
            examples[name] = rng.normal(size=len(examples))
    examples["zona_t"] = rng.choice(list(ZONES), size=len(examples))
    examples["y"] = (rng.random(len(examples)) < 0.3).astype("int8")
    return examples, t


SPEC = {"familia": "logistica", "candidato": "logistica_05", "hiperparametros": {"C": 3.0, "max_iter": 5000},
        "conjunto_variables": "sin_clima"}


def test_final_models_use_all_rows_and_the_feature_set_of_the_recipe():
    from src.forecasting.artifacts import train_final_models
    from src.forecasting.dataset import FEATURE_SETS
    from src.forecasting.models import fit_model, predict_positive

    examples, t = _examples()
    final = train_final_models(examples, {"municipios": {"00001": SPEC}}, config=_config())["00001"]
    assert (final.filas, final.ultima_emision, final.ultima_objetivo) == (711, max(t), max(t) + 8 * WEEK)
    assert (final.familia, final.candidato, final.conjunto_variables, final.hiperparametros) == (
        "logistica", "logistica_05", "sin_clima", {"C": 3.0, "max_iter": 5000})
    expected = predict_positive(fit_model("logistica", SPEC["hiperparametros"], examples,
                                          features=FEATURE_SETS["sin_clima"]), examples)
    assert np.array_equal(predict_positive(final.model, examples), expected)


def test_final_models_need_the_feature_set():
    from src.forecasting.artifacts import train_final_models

    examples, _ = _examples()
    old = {key: value for key, value in SPEC.items() if key != "conjunto_variables"}
    with pytest.raises(EpidemiologyValidationError):
        train_final_models(examples, {"municipios": {"00001": old}}, config=_config())


def test_final_models_are_written_with_their_manifest(tmp_path):
    import joblib

    from src.forecasting.artifacts import train_final_models, write_final_models
    from src.forecasting.models import predict_positive

    examples, t = _examples()
    models = train_final_models(examples, {"municipios": {"00001": SPEC}}, config=_config())
    manifest_path = write_final_models(models, tmp_path / "modelos_finales", recipe_hash="f" * 64)
    assert manifest_path == tmp_path / "modelos_finales" / "manifiesto.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["receta_hash"] == "f" * 64
    assert manifest["modelos"] == {"00001": {
        "familia": "logistica", "candidato": "logistica_05", "conjunto_variables": "sin_clima",
        "hiperparametros": {"C": 3.0, "max_iter": 5000}, "filas": 711, "ultima_emision": max(t).isoformat(),
        "ultima_objetivo": (max(t) + 8 * WEEK).isoformat()}}
    loaded = joblib.load(tmp_path / "modelos_finales" / "00001.joblib")
    assert np.array_equal(predict_positive(loaded, examples), predict_positive(models["00001"].model, examples))
    with pytest.raises(EpidemiologyValidationError):
        write_final_models(models, tmp_path / "modelos_finales", recipe_hash="f" * 64)
