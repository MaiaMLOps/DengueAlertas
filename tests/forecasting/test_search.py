"""Unit tests of search.py (U5.3) with synthetic examples of two municipalities and a temporary MLflow file store:
structure of the outputs, interruption and resumption without duplicated rows or runs, reuse of a run whose
folder was not written, manifest identity, early checks, candidate expansion and the command line."""

from datetime import date, timedelta
import json

from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest

from src.epidemiology.channel_epico import ZONES
from src.epidemiology.config import EpidemiologyValidationError
from src.forecasting import search
from src.forecasting.config import SearchSection, load_config
from src.forecasting.dataset import FEATURES
from src.forecasting.models import HYPERPARAMETERS, make_model
from src.forecasting.search import Candidate, candidates_from_config, run_search, validate_score_table

CODES = ("11111", "22222")
YEARS = [2019, 2020, 2021, 2022]
LIGHTGBM = {"num_leaves": 4, "max_depth": 2, "learning_rate": 0.1, "n_estimators": 20, "min_child_samples": 10,
            "reg_lambda": 1.0, "subsample": 1.0, "subsample_freq": 0, "colsample_bytree": 1.0, "random_state": 3}
CANDIDATES = {
    "logistica": [Candidate("rl_a", {"C": 0.1, "max_iter": 2000}), Candidate("rl_b", {"C": 1.0, "max_iter": 2000})],
    "lightgbm": [Candidate("gb_a", LIGHTGBM)],
}
UNITS = len(CODES) * (2 + 3)   # B2, B1 and three candidates per municipality


@pytest.fixture(autouse=True)
def _restore_file_store_variable(monkeypatch):
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    monkeypatch.delenv("FORECAST_HORIZON", raising=False)


def _examples():
    rng = np.random.default_rng(5)
    rows = []
    for code in CODES:
        t = date(2011, 11, 7)
        while t + timedelta(weeks=8) <= date(2022, 12, 26):
            row = {"municipality_code": code, "t": t, "target_week": t + timedelta(weeks=8)}
            row.update({name: rng.normal() for name in FEATURES if name != "zona_t"})
            row["zona_t"] = str(rng.choice(ZONES))
            score = row["posicion_4s"] + {"seguridad": -1.0, "alerta": 0.0, "epidemia": 1.0}[row["zona_t"]]
            row["y"] = int(rng.uniform() < 1 / (1 + np.exp(-score)))
            rows.append(row)
            t += timedelta(weeks=1)
    frame = pd.DataFrame(rows)
    frame["y"] = frame["y"].astype("int8")
    return frame[["municipality_code", "t", "target_week", "y", *FEATURES]]


EXAMPLES = _examples()


def _run(tmp_path, folder="out", candidates=CANDIDATES, config=None):
    uri = (tmp_path / "mlruns").resolve().as_uri()
    table = run_search(EXAMPLES, candidates, config=config or load_config(), output_dir=tmp_path / folder,
                       tracking_uri=uri)
    return table, tmp_path / folder, uri


def _runs(uri):
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(load_config().seguimiento.experimentos.desarrollo)
    return [] if experiment is None else client.search_runs([experiment.experiment_id], max_results=1000)


def _finished_keys(uri):
    return sorted(run.data.tags["clave"] for run in _runs(uri) if run.info.status == "FINISHED")


def _fail_on_call(monkeypatch, target, number, error=RuntimeError("simulated failure")):
    original, calls = getattr(search, target), {"n": 0}

    def wrapper(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == number:
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(search, target, wrapper)


# --------------------------------------------------------------------------- outputs
def test_outputs_structure_and_runs(tmp_path):
    table, out, uri = _run(tmp_path)
    assert len(table) == len(CODES) * 3 * 4
    assert sorted(set(table["anio"])) == YEARS
    assert set(table["familia"]) == {"logistica", "lightgbm"}
    orders = table.drop_duplicates(["familia", "candidato"]).set_index("candidato")["orden"].to_dict()
    assert orders == {"rl_a": 0, "rl_b": 1, "gb_a": 0}
    assert json.loads(table.loc[table["candidato"] == "gb_a", "parametros"].iloc[0]) == LIGHTGBM
    assert (table.groupby(["municipality_code", "anio"])["logloss_b2"].nunique() == 1).all()
    assert table["mejora_b2"].tolist() == (1 - table["logloss"] / table["logloss_b2"]).tolist()

    predictions = pd.read_parquet(out / search.PREDICTIONS_FILENAME)
    sizes = predictions.groupby(["municipality_code", "familia", "candidato", "anio"]).size()
    assert sizes.to_dict() == table.set_index(["municipality_code", "familia", "candidato", "anio"])["n_filas"].to_dict()

    baselines = pd.read_parquet(out / search.BASELINES_FILENAME)
    assert len(baselines) == len(CODES) * 2 * 4 and set(baselines["familia"]) == {"B1", "B2"}
    b2 = baselines[baselines["familia"] == "B2"].set_index(["municipality_code", "anio"])["logloss"]
    assert table.drop_duplicates(["municipality_code", "anio"]).set_index(["municipality_code", "anio"])[
        "logloss_b2"].to_dict() == b2.to_dict()
    assert set(pd.read_parquet(out / search.BASELINE_PREDICTIONS_FILENAME)["familia"]) == {"B1", "B2"}

    runs = [run for run in _runs(uri) if run.info.status == "FINISHED"]
    assert len(runs) == UNITS
    model_run = next(run for run in runs if run.data.tags["candidato"] == "gb_a")
    assert {"municipio", "familia", "fase", "commit", "contrato_commit", "datos_dvc", "busqueda_id"} <= set(
        model_run.data.tags)
    assert model_run.data.tags["fase"] == "desarrollo"
    assert {f"{name}_{year}" for name in ("logloss", "brier", "mejora_b2") for year in YEARS} == set(
        model_run.data.metrics)
    assert model_run.data.params["n_estimators"] == "20"

    log = (out / search.LOG_FILENAME).read_text(encoding="utf-8")
    assert f"1/{UNITS} 11111 B2 B2" in log and "[00:" in log and "complete" in log
    assert not list((out / search.PARTS_DIRNAME).glob(".tmp-*"))


def test_validate_score_table_detects_missing_extra_and_duplicated_years(tmp_path):
    table, _, _ = _run(tmp_path)
    validate_score_table(table, municipalities=CODES, years=YEARS)
    with pytest.raises(ValueError, match="years"):
        validate_score_table(table[table["anio"] != 2021], municipalities=CODES, years=YEARS)
    with pytest.raises(ValueError, match="years"):
        validate_score_table(table, municipalities=CODES, years=YEARS + [2023])
    with pytest.raises(ValueError, match="duplicated"):
        validate_score_table(pd.concat([table, table.iloc[:1]]), municipalities=CODES, years=YEARS)


# --------------------------------------------------------------------------- interruption and resumption
def test_interrupted_search_resumes_to_the_same_table(tmp_path, monkeypatch):
    reference, _, _ = _run(tmp_path, folder="reference")

    store = tmp_path / "interrupted"
    store.mkdir()
    _fail_on_call(monkeypatch, "fit_model", 7)     # 11111: B2, B1, rl_a (fits 1-4), rl_b fails at its 3rd trial
    with pytest.raises(RuntimeError, match="simulated failure"):
        _run(store)
    out = store / "out"
    assert not (out / search.TABLE_FILENAME).exists()
    assert not list((out / search.PARTS_DIRNAME).glob(".tmp-*"))
    assert sorted(path.name for path in (out / search.PARTS_DIRNAME).iterdir()) == [
        "11111__B1__B1", "11111__B2__B2", "11111__logistica__rl_a"]
    uri = (store / "mlruns").resolve().as_uri()
    assert len(_finished_keys(uri)) == 3
    log = (out / search.LOG_FILENAME).read_text(encoding="utf-8")
    assert "11111__logistica__rl_b failed" in log and "year 2021" in log

    monkeypatch.undo()
    resumed, _, _ = _run(store)
    pd.testing.assert_frame_equal(resumed.drop(columns="segundos"), reference.drop(columns="segundos"))
    keys = _finished_keys(uri)
    assert len(keys) == len(set(keys)) == UNITS


def test_crash_between_run_and_folder_reuses_the_run(tmp_path, monkeypatch):
    _fail_on_call(monkeypatch, "_write_unit", 3)    # the run of 11111/rl_a is logged, its folder is not written
    with pytest.raises(RuntimeError):
        _run(tmp_path)
    uri = (tmp_path / "mlruns").resolve().as_uri()
    assert "11111__logistica__rl_a" in _finished_keys(uri)
    assert not (tmp_path / "out" / search.PARTS_DIRNAME / "11111__logistica__rl_a").exists()

    monkeypatch.undo()
    _run(tmp_path)
    keys = _finished_keys(uri)
    assert len(keys) == len(set(keys)) == UNITS
    log = (tmp_path / "out" / search.LOG_FILENAME).read_text(encoding="utf-8")
    assert "11111__logistica__rl_a reuses its FINISHED run" in log


def test_reused_run_with_different_metrics_fails(tmp_path, monkeypatch):
    _fail_on_call(monkeypatch, "_write_unit", 3)
    with pytest.raises(RuntimeError):
        _run(tmp_path)
    monkeypatch.undo()
    uri = (tmp_path / "mlruns").resolve().as_uri()
    run = next(r for r in _runs(uri) if r.data.tags["clave"] == "11111__logistica__rl_a")
    MlflowClient(tracking_uri=uri).log_metric(run.info.run_id, "logloss_2019", 9.0)
    with pytest.raises(EpidemiologyValidationError, match="differs from the recomputed unit"):
        _run(tmp_path)


def test_relaunching_a_complete_search_fits_nothing(tmp_path, monkeypatch):
    first, _, uri = _run(tmp_path)
    monkeypatch.setattr(search, "fit_model", lambda *a, **k: pytest.fail("a complete search must not refit"))
    again, _, _ = _run(tmp_path)
    pd.testing.assert_frame_equal(again, first)
    assert len(_finished_keys(uri)) == UNITS


def test_only_running_runs_of_the_same_search_are_marked_failed(tmp_path):
    _, out, uri = _run(tmp_path)
    search_id = json.loads((out / search.MANIFEST_FILENAME).read_text(encoding="utf-8"))["busqueda_id"]
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(load_config().seguimiento.experimentos.desarrollo).experiment_id
    same = client.create_run(experiment, tags={"busqueda_id": search_id, "clave": "11111__logistica__rl_a"})
    other = client.create_run(experiment, tags={"busqueda_id": "otra_busqueda", "clave": "x"})
    untagged = client.create_run(experiment, tags={"proyecto": "otro"})

    _run(tmp_path)
    assert client.get_run(same.info.run_id).info.status == "FAILED"
    assert client.get_run(other.info.run_id).info.status == "RUNNING"
    assert client.get_run(untagged.info.run_id).info.status == "RUNNING"


# --------------------------------------------------------------------------- identity and early checks
def test_a_different_search_in_the_same_folder_fails(tmp_path):
    _run(tmp_path)
    changed = {**CANDIDATES, "logistica": [Candidate("rl_a", {"C": 0.3, "max_iter": 2000})]}
    with pytest.raises(EpidemiologyValidationError, match="different search"):
        _run(tmp_path, candidates=changed)


@pytest.mark.parametrize(("candidates", "message"), [
    ({"logistica": [Candidate("a", {"C": 1.0, "max_iter": 100}), Candidate("a", {"C": 2.0, "max_iter": 100})]},
     "Repeated candidate names"),
    ({"gru": [Candidate("a", {})]}, "Unknown model family"),
    ({"logistica": [Candidate("a", {"C": 1.0, "max_iter": 100, "class_weight": "balanced"})]}, "class weights"),
    ({"logistica": [Candidate("a/b", {"C": 1.0, "max_iter": 100})]}, "names made of"),
])
def test_invalid_candidates_fail_before_anything(tmp_path, candidates, message):
    with pytest.raises(EpidemiologyValidationError, match=message):
        _run(tmp_path, candidates=candidates)
    assert not (tmp_path / "out").exists() and not (tmp_path / "mlruns").exists()


def test_development_years_touching_the_test_period_are_rejected(tmp_path):
    config = load_config()
    config = config.model_copy(update={"validacion": config.validacion.model_copy(update={"desarrollo": (2019, 2023)})})
    with pytest.raises(EpidemiologyValidationError, match="test years"):
        _run(tmp_path, config=config)
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------- candidates from the configuration
FULL_SPACE = {
    "logistica": {"C": [0.01, 0.1, 1.0], "max_iter": 3000},
    "lightgbm": {
        "n_candidatos": 25, "semilla_muestreo": 123, "semilla_modelo": 99, "fijos": {"subsample_freq": 1},
        "espacio": {
            "num_leaves": {"escala": "entera", "rango": [4, 31], "tope_por_profundidad": True},
            "max_depth": {"escala": "entera", "rango": [2, 5]},
            "learning_rate": {"escala": "log", "rango": [0.01, 0.1]},
            "n_estimators": {"escala": "entera", "rango": [100, 600]},
            "min_child_samples": {"escala": "entera", "rango": [10, 50]},
            "reg_lambda": {"escala": "real", "rango": [0, 10]},
            "subsample": {"escala": "real", "rango": [0.6, 1.0]},
            "colsample_bytree": {"escala": "real", "rango": [0.6, 1.0]},
        },
    },
}


def test_candidates_from_config_are_deterministic_and_within_the_space():
    section = SearchSection.model_validate(FULL_SPACE)
    first, second = candidates_from_config(section), candidates_from_config(section)
    assert first == second
    assert [c.name for c in first["logistica"]] == ["logistica_00", "logistica_01", "logistica_02"]
    assert [c.params["C"] for c in first["logistica"]] == [0.01, 0.1, 1.0]
    assert all(c.params["max_iter"] == 3000 for c in first["logistica"])
    drawn = first["lightgbm"]
    assert [c.name for c in drawn] == [f"lightgbm_{i:02d}" for i in range(25)]
    for candidate in drawn:
        p = candidate.params
        assert set(p) == set(HYPERPARAMETERS["lightgbm"])
        make_model("lightgbm", p)
        assert p["random_state"] == 99 and p["subsample_freq"] == 1
        assert 2 <= p["max_depth"] <= 5 and 4 <= p["num_leaves"] <= min(31, 2 ** p["max_depth"])
        assert 0.01 <= p["learning_rate"] <= 0.1 and 100 <= p["n_estimators"] <= 600
        assert isinstance(p["n_estimators"], int) and isinstance(p["reg_lambda"], float)
    assert len({json.dumps(c.params, sort_keys=True) for c in drawn}) == 25
    other = SearchSection.model_validate({**FULL_SPACE, "lightgbm": {**FULL_SPACE["lightgbm"], "semilla_muestreo": 124}})
    assert candidates_from_config(other)["lightgbm"] != drawn


def test_log_scale_is_log_uniform():
    space = {**FULL_SPACE["lightgbm"], "n_candidatos": 400}
    rates = [c.params["learning_rate"] for c in candidates_from_config(SearchSection.model_validate(
        {"lightgbm": space}))["lightgbm"]]
    below = np.mean(np.array(rates) < 10 ** -1.5)   # midpoint of [0.01, 0.1] on the log scale
    assert 0.4 < below < 0.6


# --------------------------------------------------------------------------- command line
def _config_file(tmp_path, with_search=True):
    import yaml

    from src.forecasting.config import DEFAULT_CONFIG_PATH

    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if with_search:
        data["busqueda"] = {"logistica": {"C": [0.1, 1.0], "max_iter": 2000}}
    path = tmp_path / "forecasting.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _fake_pipeline(monkeypatch, dirty=()):
    calls = {}
    monkeypatch.setattr(search, "_dirty_paths", lambda: list(dirty))
    monkeypatch.setattr(search, "load_channel_product", lambda path, *, expected_md5: "product")
    monkeypatch.setattr(search, "build_examples", lambda product, **kwargs: EXAMPLES)
    monkeypatch.setattr(search, "configure_tracking", lambda config: "file:///unused")

    def fake_run(examples, candidates, *, config, output_dir, tracking_uri):
        calls.update(candidates=candidates, output_dir=output_dir, uri=tracking_uri)
        return pd.DataFrame()

    monkeypatch.setattr(search, "run_search", fake_run)
    return calls


def test_cli_without_busqueda_fails_clearly(tmp_path, monkeypatch, capsys):
    _fake_pipeline(monkeypatch)
    assert search.main(["--config", str(_config_file(tmp_path, with_search=False))]) == 1
    assert "no busqueda section" in capsys.readouterr().err


def test_cli_passes_candidates_and_output(tmp_path, monkeypatch):
    calls = _fake_pipeline(monkeypatch)
    assert search.main(["--config", str(_config_file(tmp_path)), "--salida", str(tmp_path / "salida")]) == 0
    assert [c.name for c in calls["candidates"]["logistica"]] == ["logistica_00", "logistica_01"]
    assert calls["output_dir"] == tmp_path / "salida"


def test_cli_refuses_uncommitted_changes_unless_allowed(tmp_path, monkeypatch, capsys):
    _fake_pipeline(monkeypatch, dirty=[" M src/forecasting/search.py"])
    config = str(_config_file(tmp_path))
    assert search.main(["--config", config]) == 1
    assert "uncommitted changes" in capsys.readouterr().err
    assert search.main(["--config", config, "--permitir-cambios"]) == 0
