"""F7 acceptance with synthetic inputs: the single test run (contrato §4.4, §6.1 with D7.1, §7.7; especificación
§4 «Barrera de prueba: detalle» and §6.F7). Approved by Jesús on 2026-11-12 (U7.1).

Rules:
- the test-period barrier is checked before anything else: without a valid frozen recipe, FrozenRecipeError and
  the log is not created;
- the log (JSON Lines, one entry per execution) is written when the run starts, after the barrier: a failed
  attempt also counts. A second execution needs an explicit reason, which is recorded; without it the run fails
  before writing anything. Outputs are never overwritten;
- one trial per (municipality, test year) with `splits.yearly_trials` (training rows with t + h <= t0, D7.1);
  in each trial the model is refitted with the recipe's family and hyperparameters of the municipality, the climate
  ablation with its own (separately selected) family and hyperparameters and the features without the 9 climate
  anomalies, and B2/B1 with the same training rows. All four systems predict exactly the same rows.

API frozen here:
- `src.forecasting.models.fit_model(family, params, train, *, features=FEATURES)`: optional keyword-only subset of
  the features (zona_t always included); without it the pipeline is the one of F5a;
- `src.forecasting.evaluate`: CLIMATE_FEATURES, SYSTEMS = ("modelo", "sin_clima", "B2", "B1"),
  PREDICTIONS_FILENAME = "predicciones_prueba.parquet", TRIALS_FILENAME = "ensayos_prueba.parquet" and
  `run_test(examples, *, config, access, ablation, log_path, output_dir, reason=None) -> TestRun` with the
  DataFrames predicciones (municipality_code, t, target_week, anio, y, zona_t, sistema, candidato, p) and ensayos,
  and the log entry `entrada`. ablation maps each municipality to {"familia", "candidato", "hiperparametros"}.
"""

from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.epidemiology.channel_epico import ZONES
from src.epidemiology.config import EpidemiologyValidationError

pytestmark = pytest.mark.acceptance

WEEK = timedelta(weeks=1)
CODES = ("00001", "00002")
CLIMATE = ("anom_temp_m1", "anom_temp_m2", "anom_temp_m3", "anom_prec_m1", "anom_prec_m2", "anom_prec_m3",
           "anom_hum_m1", "anom_hum_m2", "anom_hum_m3")
RECIPE_SPEC = {"familia": "logistica", "candidato": "logistica_03", "hiperparametros": {"C": 0.3, "max_iter": 5000}}
ABLATION_SPEC = {"familia": "logistica", "candidato": "logistica_02", "hiperparametros": {"C": 0.1, "max_iter": 5000}}
LOG_KEYS = {"ejecucion", "fecha_hora", "commit", "receta_hash", "version_dvc_canal", "ablacion_hash", "razon", "salida"}


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


def _write_recipe(path, config):
    from src.forecasting.guard import recipe_hash

    recipe = {"familia_comun": "logistica", "municipios": {code: RECIPE_SPEC for code in CODES},
              "decision": {"tau": 0.1, "k": 2, "W": 12},
              "procedencia": {"commit_codigo": "a" * 40, "commit_contrato": "b" * 40,
                              "version_dvc_canal": config.datos.version_dvc_canal},
              "aprobacion": {"nombre": "Receta sintética", "fecha": "2026-11-12"}}
    recipe["hash_sha256"] = recipe_hash(recipe)
    path.write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return recipe


def _access(config, recipe_path):
    from src.forecasting.guard import RecipeAccess

    return RecipeAccess.from_config(config, recipe_path=recipe_path)


def _run(tmp_path, examples, *, output="salida", reason=None, recipe=True):
    from src.forecasting.evaluate import run_test

    config = _config()
    recipe_path = tmp_path / "receta.yaml"
    if recipe and not recipe_path.exists():
        _write_recipe(recipe_path, config)
    return run_test(examples, config=config, access=_access(config, recipe_path),
                    ablation={code: ABLATION_SPEC for code in CODES}, log_path=tmp_path / "bitacora.jsonl",
                    output_dir=tmp_path / output, reason=reason)


def _log(tmp_path):
    return [json.loads(line) for line in (tmp_path / "bitacora.jsonl").read_text(encoding="utf-8").splitlines()]


_CACHE = {}


@pytest.fixture
def first_run(tmp_path_factory):
    if "run" not in _CACHE:
        folder = tmp_path_factory.mktemp("f7_evaluacion")
        examples = _examples()
        _CACHE["run"] = (folder, examples, _run(folder, examples))
    return _CACHE["run"]


def test_constants():
    from src.forecasting import evaluate

    assert evaluate.CLIMATE_FEATURES == CLIMATE
    assert evaluate.SYSTEMS == ("modelo", "sin_clima", "B2", "B1")
    assert evaluate.PREDICTIONS_FILENAME == "predicciones_prueba.parquet"
    assert evaluate.TRIALS_FILENAME == "ensayos_prueba.parquet"


def test_without_recipe_fails_before_the_log(tmp_path):
    from src.forecasting.guard import FrozenRecipeError

    with pytest.raises(FrozenRecipeError):
        _run(tmp_path, _examples(), recipe=False)
    assert not (tmp_path / "bitacora.jsonl").exists()
    assert not (tmp_path / "salida").exists()


def test_four_systems_on_exactly_the_trial_rows(first_run):
    from src.forecasting.splits import yearly_trials

    _, examples, run = first_run
    trials = yearly_trials(examples, range(2023, 2026), first_train_year=2012)
    expected = set()
    for trial in trials:
        expected |= set(zip(examples.loc[trial.evaluate, "municipality_code"], examples.loc[trial.evaluate, "t"]))
    predictions = run.predicciones
    assert list(predictions.columns) == ["municipality_code", "t", "target_week", "anio", "y", "zona_t", "sistema",
                                         "candidato", "p"]
    assert sorted(set(predictions["sistema"])) == sorted(("modelo", "sin_clima", "B2", "B1"))
    for system in ("modelo", "sin_clima", "B2", "B1"):
        own = predictions[predictions["sistema"] == system]
        assert not own.duplicated(["municipality_code", "t"]).any()
        assert set(zip(own["municipality_code"], own["t"])) == expected
        assert all(2023 <= week.year <= 2025 for week in own["target_week"])
        assert (own["anio"] == own["target_week"].map(lambda week: week.year)).all()


def test_annual_refit_with_d71_and_ablation(first_run):
    """Exact equality with an independent refit of each trial (D7.1 training rows)."""
    from src.forecasting.baselines import b1_predict, b2_predict
    from src.forecasting.dataset import FEATURES
    from src.forecasting.guard import RecipeAccess
    from src.forecasting.models import fit_model, predict_positive
    from src.forecasting.splits import yearly_trials

    folder, examples, run = first_run
    access = RecipeAccess.from_config(_config(), recipe_path=folder / "receta.yaml")
    no_climate = tuple(name for name in FEATURES if name not in CLIMATE)
    predictions = run.predicciones
    for trial in yearly_trials(examples, range(2023, 2026), first_train_year=2012):
        train, evaluate = examples.loc[trial.train], examples.loc[trial.evaluate]
        assert max(train["target_week"]) <= trial.t0
        expected = {
            "modelo": predict_positive(fit_model("logistica", RECIPE_SPEC["hiperparametros"], train), evaluate),
            "sin_clima": predict_positive(fit_model("logistica", ABLATION_SPEC["hiperparametros"], train,
                                                    features=no_climate), evaluate),
            "B2": b2_predict(train, evaluate, access=access),
            "B1": b1_predict(train, evaluate, half_window=2, access=access)}
        for system, values in expected.items():
            own = predictions[(predictions["sistema"] == system)
                              & (predictions["municipality_code"] == trial.municipality_code)
                              & (predictions["anio"] == trial.year)].set_index("t")
            assert np.array_equal(own.loc[list(evaluate["t"]), "p"].to_numpy(), values), (system, trial.year)
        model = predictions[(predictions["sistema"] == "modelo") & (predictions["anio"] == trial.year)]
        ablation = predictions[(predictions["sistema"] == "sin_clima") & (predictions["anio"] == trial.year)]
        assert set(model["candidato"]) == {"logistica_03"} and set(ablation["candidato"]) == {"logistica_02"}
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
