"""Unit tests of src/forecasting/evaluate.py beyond the F7b acceptance: scores of §7 and §8.4, outputs and MLflow
logging, on a synthetic world whose observed series, zones, labels and features are consistent (no real data)."""

from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.epidemiology.config import EpidemiologyValidationError
from src.forecasting import evaluate
from src.forecasting.comparison import combined_level, count_wins
from src.forecasting.config import load_config
from src.forecasting.dataset import FEATURES
from src.forecasting.events import channel_positive, evaluate_period, model_positive
from src.forecasting.guard import FrozenRecipeError, RecipeAccess, recipe_hash
from src.forecasting.metrics import brier

WEEK = timedelta(weeks=1)
CODES = ("00001", "00002")
TAU = 0.3


def world(seed=7):
    """Observed channel series 2011-11-07..2025-12-22 and the supervised rows of h = 8 for two municipalities."""
    rng = np.random.default_rng(seed)
    weeks = [date(2011, 11, 7) + i * WEEK for i in range(((date(2025, 12, 22) - date(2011, 11, 7)).days // 7) + 1)]
    observed, examples = [], []
    for c, code in enumerate(CODES):
        phase = [(i + 11 * c) % 40 for i in range(len(weeks))]
        exceed = {week: int(20 <= k < 26) for week, k in zip(weeks, phase)}
        zone = {week: "epidemia" if exceed[week] else ("alerta" if k in (17, 18, 19) else "seguridad")
                for week, k in zip(weeks, phase)}
        observed.append(pd.DataFrame({
            "municipality_code": code, "week_start": weeks,
            "target_u1e": np.array([exceed[w] for w in weeks], dtype="int8"), "channel_eligible": True,
            "channel_zone": [zone[w] for w in weeks], "confirmed_records_screening": [9 if exceed[w] else 2 for w in weeks],
            "epico_central": 3.0, "epico_upper": 8.0}))
        t = [w for w in weeks if w >= date(2012, 3, 19) and w + 8 * WEEK <= date(2025, 12, 22)]
        frame = pd.DataFrame({"municipality_code": code, "t": t, "target_week": [w + 8 * WEEK for w in t]})
        for name in FEATURES:
            if name != "zona_t":
                frame[name] = rng.normal(size=len(frame))
        frame["y"] = np.array([exceed[w] for w in frame["target_week"]], dtype="int8")
        frame["posicion_4s"] += 1.5 * frame["y"]
        frame["zona_t"] = [zone[w] for w in t]
        examples.append(frame)
    examples = pd.concat(examples, ignore_index=True)
    observed = pd.concat(observed, ignore_index=True)
    observed["channel_eligible"] = observed["channel_eligible"].astype(bool)
    return examples, observed


def write_recipe(path, config):
    recipe = {"familia_comun": "logistica",
              "municipios": {code: {"familia": "logistica", "candidato": "logistica_05",
                                    "hiperparametros": {"C": 3.0, "max_iter": 5000}, "conjunto_variables": "sin_clima"}
                             for code in CODES},
              "ablacion": {code: {"familia": "logistica", "candidato": "logistica_03",
                                  "hiperparametros": {"C": 0.3, "max_iter": 5000}, "conjunto_variables": "completo"}
                           for code in CODES},
              "decision": {"tau": TAU, "k": 2, "W": 12},
              "procedencia": {"commit_codigo": "a" * 40, "commit_contrato": "b" * 40,
                              "version_dvc_canal": config.datos.version_dvc_canal},
              "aprobacion": {"nombre": "Receta sintética", "fecha": "2026-11-12"}}
    recipe["hash_sha256"] = recipe_hash(recipe)
    path.write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return recipe


_CACHE = {}


@pytest.fixture
def scored(tmp_path_factory):
    if "scored" not in _CACHE:
        folder = tmp_path_factory.mktemp("evaluate")
        config = load_config()
        recipe = write_recipe(folder / "receta.yaml", config)
        access = RecipeAccess.from_config(config, recipe_path=folder / "receta.yaml")
        examples, observed = world()
        run = evaluate.run_test(examples, config=config, access=access, log_path=folder / "bitacora.jsonl",
                                output_dir=folder / "salida")
        rows = run.predicciones[run.predicciones["sistema"] == "modelo"][
            ["municipality_code", "t", "target_week", "zona_t", "y"]].reset_index(drop=True)
        cells, summary = evaluate.score_summary(rows, run.predicciones, observed, config=config, tau=TAU, access=access)
        _CACHE["scored"] = (folder, config, recipe, access, observed, run, rows, cells, summary)
    return _CACHE["scored"]


def test_cells_have_every_system_and_consistent_wins(scored):
    _, _, _, access, _, run, rows, cells, _ = scored
    assert len(cells) == len(CODES) * 3 and int(cells["filas"].sum()) == len(rows)
    predictions = run.predicciones
    for cell in cells.itertuples():
        own = predictions[(predictions["municipality_code"] == cell.municipality_code)
                          & (predictions["anio"] == cell.anio)]
        for system in evaluate.SYSTEMS:
            part = own[own["sistema"] == system]
            assert getattr(cell, f"brier_{system}") == pytest.approx(
                brier(part["y"].to_numpy(), part["p"].to_numpy(), list(part["target_week"]), access=access), rel=1e-12)
        for reference in ("B2", "B1", "ablacion"):
            assert getattr(cell, f"gana_brier_vs_{reference}") == (cell.brier_modelo < getattr(cell, f"brier_{reference}"))


def test_conclusions_follow_section_8_4(scored):
    _, _, _, _, _, _, _, cells, summary = scored
    for name, reference in (("vs_b2", "B2"), ("vs_b1", "B1"), ("ablacion_clima", "ablacion")):
        conclusion = summary["conclusiones"][name]
        assert conclusion["nivel"] == combined_level(conclusion["nivel_brier"], conclusion["nivel_logloss"])
        assert conclusion["casillas_ganadas_brier"] == count_wins(cells["brier_modelo"], cells[f"brier_{reference}"])
        assert conclusion["casillas"] == len(cells)
    estimation = summary["estimacion"]
    assert estimation["bss_vs_b2"] == pytest.approx(1 - estimation["brier"]["modelo"] / estimation["brier"]["B2"])


def test_events_are_those_of_evaluate_period(scored):
    _, config, _, access, observed, run, rows, _, summary = scored
    base = rows.sort_values(["municipality_code", "t"]).reset_index(drop=True)
    p = evaluate.aligned_systems(base, run.predicciones)
    model = evaluate_period(base, model_positive(p["modelo"], tau=TAU), observed, k=2, window=12, access=access)
    channel = evaluate_period(base, channel_positive(base["zona_t"]), observed, k=2, window=12, access=access)
    events = summary["eventos"]
    assert events["inicios_evaluables"] == model.summary["evaluable_onsets"] > 0
    assert (events["avisados_modelo"], events["avisados_canal"]) == (model.summary["warned_onsets"],
                                                                    channel.summary["warned_onsets"])
    assert events["falsas_alarmas"] == {"modelo": model.summary["false_alarms"], "canal": channel.summary["false_alarms"]}
    assert events["mcnemar"]["discrepantes"] >= events["mcnemar"]["a_favor_modelo"] >= 0


def test_scores_need_the_recipe(scored, tmp_path):
    _, config, _, _, observed, run, rows, _, _ = scored
    with pytest.raises(FrozenRecipeError):
        evaluate.score_summary(rows, run.predicciones, observed, config=config, tau=TAU,
                               access=RecipeAccess.from_config(config, recipe_path=tmp_path / "no.yaml"))


def test_outputs_and_mlflow_runs(scored, tmp_path, monkeypatch):
    from mlflow.tracking import MlflowClient

    folder, config, recipe, _, _, run, _, cells, summary = scored
    output = tmp_path / "salida"
    output.mkdir()
    run.predicciones.to_parquet(output / evaluate.PREDICTIONS_FILENAME, index=False)
    evaluate.write_outputs(output, cells, summary, run)
    with pytest.raises(EpidemiologyValidationError):
        evaluate.write_outputs(output, cells, summary, run)
    saved = json.loads((output / evaluate.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert saved["ejecucion"] == run.entrada and saved["tau"] == TAU

    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    local = config.model_copy(update={"seguimiento": config.seguimiento.model_copy(
        update={"modo": "local", "directorio_local": str(tmp_path / "mlruns")})})
    assert evaluate.log_test_runs(output, config=local, recipe=recipe, entry=run.entrada) == len(CODES) * 4 + 1
    client = MlflowClient(tracking_uri=(tmp_path / "mlruns").resolve().as_uri())
    experiment = client.get_experiment_by_name(config.seguimiento.experimentos.prueba)
    runs = client.search_runs([experiment.experiment_id], max_results=100)
    assert len(runs) == len(CODES) * 4 + 1 and {r.info.status for r in runs} == {"FINISHED"}
    model_run = next(r for r in runs if r.data.tags.get("sistema") == "modelo")
    assert model_run.data.tags["receta_hash"] == recipe["hash_sha256"]
    assert model_run.data.tags["conjunto_variables"] == "sin_clima" and model_run.data.params["C"] == "3.0"
    assert {f"{m}_{y}" for m in ("brier", "logloss") for y in (2023, 2024, 2025)} == set(model_run.data.metrics)


def test_recipe_systems_name_the_format_of_d72():
    with pytest.raises(EpidemiologyValidationError, match="DV-02"):
        evaluate.recipe_systems({"municipios": {"00001": {"familia": "logistica"}}}, ["00001"])


def test_cli_refuses_uncommitted_changes_before_reading_data(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("the data must not be read")

    monkeypatch.setattr(evaluate, "_dirty_paths", lambda: [" M src/forecasting/evaluate.py"])
    monkeypatch.setattr(evaluate, "load_channel_product", forbidden)
    monkeypatch.setattr(evaluate, "run_test", forbidden)
    code = evaluate.main(["--salida", str(tmp_path / "salida"), "--bitacora", str(tmp_path / "b.jsonl"),
                          "--sin-seguimiento"])
    assert code == 1 and "uncommitted changes" in capsys.readouterr().err
    assert not (tmp_path / "b.jsonl").exists() and not (tmp_path / "salida").exists()
