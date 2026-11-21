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
READ_ONLY_CALLS = {"search_registered_models", "search_model_versions", "get_experiment_by_name"}
EXPECTED_NAMES = ["DengueAlertas_Barranquilla", "DengueAlertas_Soledad", "DengueAlertas_Neiva",
                  "DengueAlertas_Villavicencio", "DengueAlertas_Cucuta", "DengueAlertas_Cali"]


def _client(models=(), versions=None):
    """Simulated MlflowClient: `models` are registered models, `versions` maps a name to its versions."""
    versions = versions or {}
    client = mock.MagicMock(name="MlflowClient")

    def search_registered_models(filter_string=""):
        return [model for model in models if f"'{model.name}'" in filter_string]

    def search_model_versions(filter_string=""):
        return [v for name, items in versions.items() if f"'{name}'" in filter_string for v in items]

    client.search_registered_models.side_effect = search_registered_models
    client.search_model_versions.side_effect = search_model_versions
    client.get_experiment_by_name.return_value = None
    return client


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
