"""F7b acceptance with synthetic inputs: the single test run with the recipe of D7.2 (contrato §4.4, §6.1 with D7.1,
§6.6, §7.7; especificación §4 «Barrera de prueba: detalle», §6.F7 and §6.F6b; DV-02). Replaces
test_f7_evaluacion.py, superseded by DV-02. Approved by Jesús on 2026-11-12 (U7.3): same claims as F7, except that
the ablation is read from the recipe.

Rules (unchanged from F7 unless stated):
- the test-period barrier is checked before anything else: without a valid frozen recipe, FrozenRecipeError and the
  log is not created;
- NEW (D7.2): the recipe names, per municipality, the family, hyperparameters and feature set of the model
  (municipios[code].conjunto_variables) and of the ablation (ablacion[code]): same family, the OTHER feature set.
  A recipe without them (the F6 format), or with an inconsistent ablation, fails before the log is written;
- the log (JSON Lines, one entry per execution) is written when the run starts, after the barrier and the recipe
  checks: a failed attempt also counts. A second execution needs an explicit reason, which is recorded; without it
  the run fails before writing anything. Outputs are never overwritten. NEW: the entry has no ablacion_hash, because
  the ablation is part of the recipe and of its hash;
- one trial per (municipality, test year) with `splits.yearly_trials` (training rows with t + h <= t0, D7.1); in each
  trial the model and the ablation are refitted with their family, hyperparameters and feature set
  (dataset.FEATURE_SETS), and B2/B1 with the same training rows. All four systems predict exactly the same rows.

API frozen here (`src.forecasting.evaluate`):
- SYSTEMS = ("modelo", "ablacion", "B2", "B1"), PREDICTIONS_FILENAME = "predicciones_prueba.parquet",
  TRIALS_FILENAME = "ensayos_prueba.parquet", DEFAULT_LOG_PATH = <repo>/reports/forecasting/bitacora_prueba.jsonl;
- `run_test(examples, *, config, access, log_path, output_dir, reason=None) -> TestRun` with the DataFrames
  predicciones (municipality_code, t, target_week, anio, y, zona_t, sistema, candidato, conjunto_variables, p) and
  ensayos, and the log entry `entrada`. conjunto_variables is null for B2 and B1.
"""

from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.epidemiology.channel_epico import ZONES
from src.epidemiology.config import EpidemiologyValidationError

pytestmark = pytest.mark.acceptance

ROOT = Path(__file__).resolve().parents[2]
WEEK = timedelta(weeks=1)
CODES = ("00001", "00002")
SYSTEMS = ("modelo", "ablacion", "B2", "B1")
LOG_KEYS = {"ejecucion", "fecha_hora", "commit", "receta_hash", "version_dvc_canal", "razon", "salida"}
PREDICTION_COLUMNS = ["municipality_code", "t", "target_week", "anio", "y", "zona_t", "sistema", "candidato",
                      "conjunto_variables", "p"]
# 00001 as the D7.2 recipe (model without climate, ablation with it); 00002 the other way round, to prove that the
# feature set is read per municipality and per system.
MODEL = {"00001": ("logistica_05", {"C": 3.0, "max_iter": 5000}, "sin_clima"),
         "00002": ("logistica_03", {"C": 0.3, "max_iter": 5000}, "completo")}
ABLATION = {"00001": ("logistica_03", {"C": 0.3, "max_iter": 5000}, "completo"),
            "00002": ("logistica_02", {"C": 0.1, "max_iter": 5000}, "sin_clima")}


def _config():
    from src.forecasting.config import load_config

    return load_config()


def _examples():
    from src.forecasting.dataset import FEATURES

    rng = np.random.default_rng(20261112)
    frames = []
    for code in CODES:
        t = [date(2012, 3, 19) + i * WEEK for i in range(711)]
        frame = pd.DataFrame({"municipality_code": code, "t": t, "target_week": [w + 8 * WEEK for w in t]})
        for name in FEATURES:
            if name != "zona_t":
                frame[name] = rng.normal(size=len(frame))
        frame["zona_t"] = rng.choice(list(ZONES), size=len(frame), p=[0.5, 0.3, 0.2])
        logit = -1.0 + 1.5 * (frame["zona_t"] == "epidemia") + 0.8 * frame["posicion_4s"] + 0.5 * frame["anom_prec_m1"]
        frame["y"] = (rng.random(len(frame)) < 1 / (1 + np.exp(-logit))).astype("int8")
        frames.append(frame)
    examples = pd.concat(frames, ignore_index=True)
    examples.index = pd.RangeIndex(1000, 1000 + len(examples))
    return examples


def _recipe(config, **changes):
    from src.forecasting.dataset import FEATURE_SETS
    from src.forecasting.guard import recipe_hash

    municipios = {code: {"familia": "logistica", "candidato": MODEL[code][0], "hiperparametros": MODEL[code][1],
                         "conjunto_variables": MODEL[code][2]} for code in CODES}
    ablacion = {code: {"familia": "logistica", "conjunto_variables": ABLATION[code][2],
                       "features": list(FEATURE_SETS[ABLATION[code][2]]), "candidato": ABLATION[code][0],
                       "hiperparametros": ABLATION[code][1]} for code in CODES}
    recipe = {"familia_comun": "logistica", "municipios": municipios, "ablacion": ablacion,
              "decision": {"tau": 0.05, "k": 2, "W": 12},
              "procedencia": {"commit_codigo": "a" * 40, "commit_contrato": "b" * 40,
                              "version_dvc_canal": config.datos.version_dvc_canal},
              "aprobacion": {"nombre": "Receta sintética", "fecha": "2026-11-12"}}
    for key, value in changes.items():
        if value is None:
            recipe.pop(key)
        else:
            recipe[key] = value
    recipe["hash_sha256"] = recipe_hash(recipe)
    return recipe


def _write_recipe(path, config, **changes):
    recipe = _recipe(config, **changes)
    path.write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return recipe


def _run(tmp_path, examples, *, output="salida", reason=None, recipe=True, **changes):
    from src.forecasting.evaluate import run_test
    from src.forecasting.guard import RecipeAccess

    config = _config()
    recipe_path = tmp_path / "receta.yaml"
    if recipe and not recipe_path.exists():
        _write_recipe(recipe_path, config, **changes)
    return run_test(examples, config=config, access=RecipeAccess.from_config(config, recipe_path=recipe_path),
                    log_path=tmp_path / "bitacora.jsonl", output_dir=tmp_path / output, reason=reason)


def _log(tmp_path):
    return [json.loads(line) for line in (tmp_path / "bitacora.jsonl").read_text(encoding="utf-8").splitlines()]


_CACHE = {}


@pytest.fixture
def first_run(tmp_path_factory):
    if "run" not in _CACHE:
        folder = tmp_path_factory.mktemp("f7b_evaluacion")
        examples = _examples()
        _CACHE["run"] = (folder, examples, _run(folder, examples))
    return _CACHE["run"]


def test_constants():
    from src.forecasting import evaluate

    assert evaluate.SYSTEMS == SYSTEMS
    assert evaluate.PREDICTIONS_FILENAME == "predicciones_prueba.parquet"
    assert evaluate.TRIALS_FILENAME == "ensayos_prueba.parquet"
    assert Path(evaluate.DEFAULT_LOG_PATH) == ROOT / "reports" / "forecasting" / "bitacora_prueba.jsonl"


def test_without_recipe_fails_before_the_log(tmp_path):
    from src.forecasting.guard import FrozenRecipeError

    with pytest.raises(FrozenRecipeError):
        _run(tmp_path, _examples(), recipe=False)
    assert not (tmp_path / "bitacora.jsonl").exists() and not (tmp_path / "salida").exists()


@pytest.mark.parametrize("damage", ["f6_format", "missing_municipality", "same_set", "other_family", "unknown_set"])
def test_recipe_without_a_valid_ablation_fails_before_the_log(tmp_path, damage):
    config = _config()
    recipe = _recipe(config)
    if damage == "f6_format":
        changes = {"ablacion": None, "municipios": {code: {k: v for k, v in entry.items() if k != "conjunto_variables"}
                                                    for code, entry in recipe["municipios"].items()}}
    elif damage == "missing_municipality":
        changes = {"ablacion": {"00001": recipe["ablacion"]["00001"]}}
    elif damage == "same_set":
        changes = {"ablacion": {**recipe["ablacion"], "00001": {**recipe["ablacion"]["00001"],
                                                                "conjunto_variables": "sin_clima"}}}
    elif damage == "other_family":
        changes = {"ablacion": {**recipe["ablacion"], "00002": {**recipe["ablacion"]["00002"], "familia": "lightgbm"}}}
    else:
        changes = {"municipios": {**recipe["municipios"], "00002": {**recipe["municipios"]["00002"],
                                                                    "conjunto_variables": "parcial"}}}
    with pytest.raises(EpidemiologyValidationError):
        _run(tmp_path, _examples(), **changes)
    assert not (tmp_path / "bitacora.jsonl").exists() and not (tmp_path / "salida").exists()


def test_four_systems_on_exactly_the_trial_rows(first_run):
    from src.forecasting.splits import yearly_trials

    _, examples, run = first_run
    expected = set()
    for trial in yearly_trials(examples, range(2023, 2026), first_train_year=2012):
        expected |= set(zip(examples.loc[trial.evaluate, "municipality_code"], examples.loc[trial.evaluate, "t"]))
    predictions = run.predicciones
    assert list(predictions.columns) == PREDICTION_COLUMNS
    assert sorted(set(predictions["sistema"])) == sorted(SYSTEMS)
    for system in SYSTEMS:
        own = predictions[predictions["sistema"] == system]
        assert not own.duplicated(["municipality_code", "t"]).any()
        assert set(zip(own["municipality_code"], own["t"])) == expected
        assert all(2023 <= week.year <= 2025 for week in own["target_week"])
        assert (own["anio"] == own["target_week"].map(lambda week: week.year)).all()
    for code in CODES:
        own = predictions[predictions["municipality_code"] == code]
        model, ablation = own[own["sistema"] == "modelo"], own[own["sistema"] == "ablacion"]
        assert set(model["candidato"]) == {MODEL[code][0]} and set(model["conjunto_variables"]) == {MODEL[code][2]}
        assert set(ablation["candidato"]) == {ABLATION[code][0]}
        assert set(ablation["conjunto_variables"]) == {ABLATION[code][2]}
    assert predictions.loc[predictions["sistema"].isin(["B2", "B1"]), "conjunto_variables"].isna().all()


def test_annual_refit_with_d71_and_the_feature_sets_of_the_recipe(first_run):
    """Exact equality with an independent refit of each trial (D7.1 training rows, feature set of the recipe)."""
    from src.forecasting.baselines import b1_predict, b2_predict
    from src.forecasting.dataset import FEATURE_SETS
    from src.forecasting.guard import RecipeAccess
    from src.forecasting.models import fit_model, predict_positive
    from src.forecasting.splits import yearly_trials

    folder, examples, run = first_run
    access = RecipeAccess.from_config(_config(), recipe_path=folder / "receta.yaml")
    predictions = run.predicciones
    for trial in yearly_trials(examples, range(2023, 2026), first_train_year=2012):
        train, evaluate = examples.loc[trial.train], examples.loc[trial.evaluate]
        assert max(train["target_week"]) <= trial.t0
        code = trial.municipality_code
        expected = {
            "modelo": predict_positive(fit_model("logistica", MODEL[code][1], train,
                                                 features=FEATURE_SETS[MODEL[code][2]]), evaluate),
            "ablacion": predict_positive(fit_model("logistica", ABLATION[code][1], train,
                                                   features=FEATURE_SETS[ABLATION[code][2]]), evaluate),
            "B2": b2_predict(train, evaluate, access=access),
            "B1": b1_predict(train, evaluate, half_window=2, access=access)}
        for system, values in expected.items():
            own = predictions[(predictions["sistema"] == system) & (predictions["municipality_code"] == code)
                              & (predictions["anio"] == trial.year)].set_index("t")
            assert np.array_equal(own.loc[list(evaluate["t"]), "p"].to_numpy(), values), (system, code, trial.year)
        model = predictions[(predictions["sistema"] == "modelo") & (predictions["municipality_code"] == code)]
        ablation = predictions[(predictions["sistema"] == "ablacion") & (predictions["municipality_code"] == code)]
        assert not np.array_equal(model["p"].to_numpy(), ablation["p"].to_numpy())


def test_trials_table(first_run):
    from src.forecasting.splits import yearly_trials

    _, examples, run = first_run
    table = run.ensayos.set_index(["municipality_code", "anio"])
    assert {"t0", "filas_entrenamiento", "ultima_objetivo_entrenamiento", "filas_evaluadas"} <= set(table.columns)
    trials = yearly_trials(examples, range(2023, 2026), first_train_year=2012)
    assert len(table) == len(trials) == 6
    for trial in trials:
        row = table.loc[(trial.municipality_code, trial.year)]
        assert row["t0"] == trial.t0
        assert row["filas_entrenamiento"] == len(trial.train) and row["filas_evaluadas"] == len(trial.evaluate)
        assert row["ultima_objetivo_entrenamiento"] <= trial.t0


def test_outputs_are_written_to_the_folder(first_run):
    folder, _, run = first_run
    saved = pd.read_parquet(folder / "salida" / "predicciones_prueba.parquet")
    pd.testing.assert_frame_equal(saved, run.predicciones, check_exact=True)
    pd.testing.assert_frame_equal(pd.read_parquet(folder / "salida" / "ensayos_prueba.parquet"), run.ensayos,
                                  check_exact=True)


def test_log_registers_executions_and_requires_a_reason(tmp_path):
    from src.forecasting.guard import recipe_hash

    examples = _examples()
    run = _run(tmp_path, examples)
    entries = _log(tmp_path)
    assert len(entries) == 1 and set(entries[0]) == LOG_KEYS and entries[0] == run.entrada
    recipe = yaml.safe_load((tmp_path / "receta.yaml").read_text(encoding="utf-8"))
    assert entries[0]["ejecucion"] == 1 and entries[0]["razon"] is None
    assert entries[0]["receta_hash"] == recipe["hash_sha256"] == recipe_hash(recipe)
    assert entries[0]["version_dvc_canal"] == _config().datos.version_dvc_canal

    with pytest.raises(EpidemiologyValidationError):
        _run(tmp_path, examples, output="segunda")
    assert len(_log(tmp_path)) == 1 and not (tmp_path / "segunda").exists()

    _run(tmp_path, examples, output="segunda", reason="Repetición sintética de la prueba de aceptación")
    entries = _log(tmp_path)
    assert len(entries) == 2 and entries[1]["ejecucion"] == 2
    assert entries[1]["razon"] == "Repetición sintética de la prueba de aceptación"


def test_outputs_are_never_overwritten(tmp_path):
    examples = _examples()
    _run(tmp_path, examples)
    with pytest.raises(EpidemiologyValidationError):
        _run(tmp_path, examples, reason="Misma carpeta de salida")
    assert len(_log(tmp_path)) == 1
