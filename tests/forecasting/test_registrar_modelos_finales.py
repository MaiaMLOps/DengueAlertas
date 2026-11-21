"""Dry run of scripts/registrar_modelos_finales.py with a simulated MLflow client and no network."""

import importlib.util
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("registrar_modelos_finales",
                                               ROOT / "scripts" / "registrar_modelos_finales.py")
reg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reg)

RECIPE = yaml.safe_load((ROOT / "configs" / "receta_congelada.yaml").read_text(encoding="utf-8"))
READ_ONLY_CALLS = {"search_registered_models", "search_model_versions", "get_experiment_by_name", "search_runs",
                   "search_logged_models"}
EXPECTED_NAMES = ["DengueAlertas_Barranquilla", "DengueAlertas_Soledad", "DengueAlertas_Neiva",
                  "DengueAlertas_Villavicencio", "DengueAlertas_Cucuta", "DengueAlertas_Cali"]


def _client(models=(), versions=None, runs=(), logged=(), experiment_id=None):
    """Simulated MlflowClient: registered `models`, `versions` per name, `runs` and `logged` models of the experiment."""
    versions = versions or {}
    client = mock.MagicMock(name="MlflowClient")

    def search_registered_models(filter_string=""):
        return [model for model in models if f"'{model.name}'" in filter_string]

    def search_model_versions(filter_string=""):
        return [v for name, items in versions.items() if f"'{name}'" in filter_string for v in items]

    def search_runs(experiment_ids, filter_string=""):
        return [run for run in runs if f"'{run.data.tags['md5']}'" in filter_string]

    client.search_registered_models.side_effect = search_registered_models
    client.search_model_versions.side_effect = search_model_versions
    client.search_runs.side_effect = search_runs
    client.search_logged_models.side_effect = lambda experiment_ids: list(logged)
    client.get_experiment_by_name.return_value = (
        None if experiment_id is None else SimpleNamespace(experiment_id=experiment_id, lifecycle_stage="active"))
    return client


def _run(run_id, md5, params=None, status="FINISHED"):
    return SimpleNamespace(info=SimpleNamespace(run_id=run_id, status=status),
                           data=SimpleNamespace(tags={"md5": md5}, params=params or {}))


def _logged(model_id, run_id):
    return SimpleNamespace(model_id=model_id, source_run_id=run_id, artifact_location=f"store/{model_id}")


def _verified():
    return reg.verify_files(reg.MODELS_DIR, reg.README_PATH, RECIPE)


def _written(client) -> list[str]:
    return sorted({call[0].split(".")[0] for call in client.mock_calls if call[0]} - READ_ONLY_CALLS)


# --------------------------------------------------------------------------- local files
def test_readme_lists_the_md5_of_the_seven_files():
    rows = reg.readme_md5(reg.README_PATH)
    assert sorted(rows) == sorted([f"{code}.joblib" for code in reg.MUNICIPALITIES] + ["manifiesto.json"])
    for name, md5 in rows.items():
        assert reg.file_md5(reg.MODELS_DIR / name) == md5


def test_verify_files_stops_on_a_changed_model(tmp_path):
    copy = tmp_path / "modelos"
    shutil.copytree(reg.MODELS_DIR, copy)
    target = copy / "41001.joblib"
    target.write_bytes(target.read_bytes() + b"\0")
    with pytest.raises(reg.RegistrationError, match="41001.joblib"):
        reg.verify_files(copy, copy / "README.md", RECIPE)


def test_verify_files_stops_on_a_manifest_of_another_recipe():
    other = dict(RECIPE, hash_sha256="0" * 64)
    with pytest.raises(reg.RegistrationError, match="receta_hash"):
        reg.verify_files(reg.MODELS_DIR, reg.README_PATH, other)


# --------------------------------------------------------------------------- plan
def test_plan_for_an_empty_registry_registers_six_ascii_names_without_alias():
    manifest, md5s = _verified()
    client = _client()
    plan = reg.build_plan(client, manifest, md5s, RECIPE, commit="a" * 40, alias=None, enabled=False)
    assert [item.name for item in plan] == EXPECTED_NAMES
    assert all(item.name.isascii() for item in plan)
    assert {item.action for item in plan} == {"registrar"}
    assert {item.alias_action for item in plan} == {"ninguno"}
    cucuta = plan[4]
    assert cucuta.model_tags == {"enabled": "false", "municipality_code": "54001", "department": "Norte de Santander",
                                 "depto_id": "CONSA", "receta_hash": RECIPE["hash_sha256"], "familia": "logistica"}
    assert cucuta.version_tags["md5"] == md5s["54001"]
    assert cucuta.version_tags["commit"] == "a" * 40
    assert cucuta.run_tags["fase"] == "F8" and cucuta.run_tags["municipio"] == "54001"
    assert cucuta.params["C"] == "3.0" and cucuta.params["candidato"] == "logistica_05"
    assert _written(client) == []


def test_habilitar_sets_enabled_true():
    manifest, md5s = _verified()
    plan = reg.build_plan(_client(), manifest, md5s, RECIPE, commit="a" * 40, alias="staging", enabled=True)
    assert {item.model_tags["enabled"] for item in plan} == {"true"}
    assert {item.alias_action for item in plan} == {"asignar"}


def test_a_version_with_the_same_md5_is_skipped_and_existing_tags_are_not_changed():
    manifest, md5s = _verified()
    neiva = SimpleNamespace(name="DengueAlertas_Neiva", tags={"enabled": "true"}, aliases={"staging": "2"})
    versions = {"DengueAlertas_Neiva": [SimpleNamespace(version="1", tags={"md5": "f" * 32}),
                                        SimpleNamespace(version="2", tags={"md5": md5s["41001"]})]}
    client = _client([neiva], versions)
    plan = {item.code: item for item in reg.build_plan(client, manifest, md5s, RECIPE, commit="a" * 40,
                                                        alias="staging", enabled=False)}
    item = plan["41001"]
    assert item.exists and item.action == "omitir" and item.duplicate_version == "2"
    assert item.alias_action == "ya_asignado"
    assert [v["version"] for v in item.versions] == ["1", "2"]
    assert any("no se modifican" in note for note in item.notes)
    assert plan["08001"].action == "registrar"
    assert _written(client) == []


def test_an_alias_held_by_another_version_stops_the_plan():
    manifest, md5s = _verified()
    cali = SimpleNamespace(name="DengueAlertas_Cali", tags={}, aliases={"staging": "1"})
    client = _client([cali], {"DengueAlertas_Cali": [SimpleNamespace(version="1", tags={})]})
    with pytest.raises(reg.RegistrationError, match="never moved"):
        reg.build_plan(client, manifest, md5s, RECIPE, commit="a" * 40, alias="staging", enabled=False)


def test_experiment_state_and_versions():
    client = _client()
    assert reg.experiment_state(client, "dengue_h8_final").startswith("no existe")
    client.get_experiment_by_name.return_value = SimpleNamespace(lifecycle_stage="deleted", experiment_id="9")
    with pytest.raises(reg.RegistrationError, match="deleted"):
        reg.experiment_state(client, "dengue_h8_final")
    reg.check_versions("3.17.0", "3.17.0")
    with pytest.raises(reg.RegistrationError, match="differs"):
        reg.check_versions("3.17.0", "3.18.0")


# --------------------------------------------------------------------------- command line
def test_dry_run_writes_nothing_and_never_registers(capsys):
    client = _client()
    rows = pd.DataFrame()
    with mock.patch("mlflow.MlflowClient", return_value=client), \
            mock.patch("src.forecasting.tracking.configure_tracking", return_value="https://servidor.invalid"), \
            mock.patch.object(reg, "server_version", return_value="3.17.0"), \
            mock.patch("mlflow.__version__", "3.17.0"), \
            mock.patch.object(reg, "dirty_paths", return_value=[]), \
            mock.patch.object(reg, "emission_rows", return_value=(rows, None)), \
            mock.patch.object(reg, "signature_and_example", return_value=(None, None)), \
            mock.patch.object(reg, "register") as register, \
            mock.patch.object(reg, "verify_registered") as verify:
        assert reg.main([]) == 0
    register.assert_not_called()
    verify.assert_not_called()
    assert _written(client) == []
    out = capsys.readouterr().out
    assert "no se escribió nada" in out
    assert all(name in out for name in EXPECTED_NAMES)


def test_ejecutar_refuses_uncommitted_changes():
    with mock.patch.object(reg, "dirty_paths", return_value=[" M scripts/registrar_modelos_finales.py"]), \
            mock.patch.object(reg, "register") as register:
        with pytest.raises(reg.RegistrationError, match="Uncommitted"):
            reg.main(["--ejecutar"])
    register.assert_not_called()


# --------------------------------------------------------------------------- runs (--solo-runs)
def test_find_run_reuses_one_finished_run_and_refuses_ambiguity():
    client = _client(runs=[_run("r1", "a" * 32)], experiment_id="13")
    assert reg.find_run(client, "13", "a" * 32).info.run_id == "r1"
    assert reg.find_run(client, "13", "b" * 32) is None
    assert reg.find_run(client, None, "a" * 32) is None
    with pytest.raises(reg.RegistrationError, match="at most one"):
        reg.find_run(_client(runs=[_run("r1", "a" * 32), _run("r2", "a" * 32)]), "13", "a" * 32)
    with pytest.raises(reg.RegistrationError, match="FAILED"):
        reg.find_run(_client(runs=[_run("r1", "a" * 32, status="FAILED")]), "13", "a" * 32)


def test_solo_runs_reuses_the_run_with_the_same_md5_and_never_calls_the_registry():
    manifest, md5s = _verified()
    plan = reg.build_plan(_client(), manifest, md5s, RECIPE, commit="a" * 40, alias=None, enabled=False)
    first = plan[0]
    existing = _run("run08001", first.md5, params=first.params)
    logged = [_logged("m-08001", "run08001")] + [_logged(f"m-{item.code}", f"run{item.code}") for item in plan[1:]]
    client = _client(runs=[existing], logged=logged, experiment_id="13")
    with mock.patch("mlflow.set_experiment"), \
            mock.patch.object(reg, "log_run", side_effect=lambda item, **_: f"run{item.code}") as log_run:
        out = reg.log_runs_only(plan, client=client, experiment_name="dengue_h8_final", rows=None, last=None,
                                features=())
    assert [call.args[0].code for call in log_run.call_args_list] == [item.code for item in plan[1:]]
    assert out["08001"]["reutilizado"] == "sí" and out["08001"]["run_id"] == "run08001"
    assert out["54001"] == {"nombre": "DengueAlertas_Cucuta", "run_id": "run54001", "modelo": "modelo",
                            "model_id": "m-54001", "model_uri": "models:/m-54001",
                            "artifact_location": "store/m-54001", "reutilizado": "no"}
    assert _written(client) == []


def test_a_run_with_the_same_md5_and_other_parameters_is_not_reused():
    manifest, md5s = _verified()
    plan = reg.build_plan(_client(), manifest, md5s, RECIPE, commit="a" * 40, alias=None, enabled=False)
    client = _client(runs=[_run("r1", plan[0].md5, params={"C": "1.0"})], experiment_id="13")
    with mock.patch("mlflow.set_experiment"), mock.patch.object(reg, "log_run") as log_run:
        with pytest.raises(reg.RegistrationError, match="other parameters"):
            reg.log_runs_only(plan[:1], client=client, experiment_name="dengue_h8_final", rows=None, last=None,
                              features=())
    log_run.assert_not_called()


# --------------------------------------------------------------------------- registration from the interface (--verificar)
def _interface_registry(md5s, *, wrong_run_for=None, missing_version_for=None):
    runs, logged, models, versions = [], [], [], {}
    for code in reg.MUNICIPALITIES:
        name = reg.registered_name(code)
        runs.append(_run(f"run{code}", md5s[code]))
        logged.append(_logged(f"m-{code}", f"run{code}"))
        models.append(SimpleNamespace(name=name, tags={}, aliases={}))
        run_id = "otro" if code == wrong_run_for else f"run{code}"
        number = "2" if code == missing_version_for else "1"
        versions[name] = [SimpleNamespace(version=number, run_id=run_id, model_id=f"m-{code}",
                                          source=f"models:/m-{code}", tags={})]
    return _client(models, versions, runs=runs, logged=logged, experiment_id="13")


def test_verificar_accepts_the_six_versions_from_the_right_runs_without_writing():
    _, md5s = _verified()
    client = _interface_registry(md5s)
    registered = reg.check_interface_registration(client, experiment_name="dengue_h8_final", md5s=md5s)
    assert registered == {name: "1" for name in EXPECTED_NAMES}
    assert _written(client) == []


def test_verificar_refuses_a_version_from_another_run_or_a_missing_version_1():
    _, md5s = _verified()
    with pytest.raises(reg.RegistrationError, match="comes from run otro"):
        reg.check_interface_registration(_interface_registry(md5s, wrong_run_for="41001"),
                                         experiment_name="dengue_h8_final", md5s=md5s)
    with pytest.raises(reg.RegistrationError, match="no version 1"):
        reg.check_interface_registration(_interface_registry(md5s, missing_version_for="76001"),
                                         experiment_name="dengue_h8_final", md5s=md5s)


def test_modes_are_exclusive_and_alias_only_goes_with_ejecutar():
    with pytest.raises(SystemExit):
        reg.parse_args(["--solo-runs", "--verificar"])
    with pytest.raises(SystemExit):
        reg.parse_args(["--solo-runs", "--alias", "staging"])
    with pytest.raises(SystemExit):
        reg.parse_args(["--verificar", "--habilitar"])
    assert reg.parse_args(["--solo-runs"]).solo_runs


def test_solo_runs_refuses_uncommitted_changes():
    with mock.patch.object(reg, "dirty_paths", return_value=[" M scripts/registrar_modelos_finales.py"]), \
            mock.patch.object(reg, "log_runs_only") as log_runs_only:
        with pytest.raises(reg.RegistrationError, match="Uncommitted"):
            reg.main(["--solo-runs"])
    log_runs_only.assert_not_called()
