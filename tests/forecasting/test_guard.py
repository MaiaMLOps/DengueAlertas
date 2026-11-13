"""Unit tests of the test-period barrier and the frozen-recipe format (U3.3), with synthetic recipes only."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.epidemiology.config import EpidemiologyValidationError
from src.forecasting.guard import (FROZEN_RECIPE_PATH, FrozenRecipeError, RecipeAccess, check_test_access,
                                   load_frozen_recipe, recipe_hash)

CHANNEL = "0123456789abcdef0123456789abcdef"


def _recipe(**approval):
    recipe = {
        "familia": "lightgbm",
        "decision": {"tau": 0.25, "k": 2, "W": 12},
        "procedencia": {"commit_codigo": "c" * 40, "commit_contrato": "d" * 40, "version_dvc_canal": CHANNEL},
        "aprobacion": {"nombre": "Receta sintética", "fecha": date(2026, 11, 13), **approval},
    }
    recipe["hash_sha256"] = recipe_hash(recipe)
    return recipe


def _write(path: Path, recipe) -> Path:
    path.write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _access(path):
    return RecipeAccess(test_years=(2023, 2025), channel_version=CHANNEL, recipe_path=path)


# --------------------------------------------------------------------------- period edges and inputs
def test_period_edges(tmp_path):
    access = _access(tmp_path / "missing.yaml")
    check_test_access([date(2022, 12, 26)], access)                 # last development target week
    check_test_access([], access)
    for week in (date(2023, 1, 2), date(2025, 12, 22)):
        with pytest.raises(FrozenRecipeError, match="No frozen recipe"):
            check_test_access([week], access)


def test_date_containers_are_equivalent(tmp_path):
    access = _access(_write(tmp_path / "r.yaml", _recipe()))
    weeks = [date(2023, 1, 2), date(2024, 6, 3)]
    for container in (weeks, pd.Series(weeks), pd.to_datetime(pd.Series(weeks)), tuple(pd.Timestamp(w) for w in weeks)):
        check_test_access(container, access)
    with pytest.raises(FrozenRecipeError):
        check_test_access(pd.Series(weeks), _access(tmp_path / "missing.yaml"))


@pytest.mark.parametrize("bad", ["2023-01-02", 2023, None, pd.NaT])
def test_non_dates_are_input_errors(tmp_path, bad):
    with pytest.raises(EpidemiologyValidationError):
        check_test_access([date(2019, 1, 7), bad], _access(tmp_path / "missing.yaml"))


def test_recipe_is_read_on_every_call(tmp_path):
    path = _write(tmp_path / "r.yaml", _recipe())
    access = _access(path)
    check_test_access([date(2023, 1, 2)], access)
    path.write_text(path.read_text(encoding="utf-8").replace("tau: 0.25", "tau: 0.3"), encoding="utf-8")
    with pytest.raises(FrozenRecipeError, match="stored hash"):
        check_test_access([date(2023, 1, 2)], access)


# --------------------------------------------------------------------------- recipe format
def test_valid_recipe_is_returned(tmp_path):
    loaded = load_frozen_recipe(_write(tmp_path / "r.yaml", _recipe()), channel_version=CHANNEL)
    assert loaded["familia"] == "lightgbm" and loaded["hash_sha256"] == _recipe()["hash_sha256"]


def test_quoted_date_gives_the_same_hash(tmp_path):
    quoted = _recipe(fecha="2026-11-13")
    assert quoted["hash_sha256"] == _recipe()["hash_sha256"]
    load_frozen_recipe(_write(tmp_path / "r.yaml", quoted), channel_version=CHANNEL)


def _mutated(change):
    recipe = _recipe()
    change(recipe)
    recipe["hash_sha256"] = recipe_hash(recipe)
    return recipe


@pytest.mark.parametrize("label, recipe, message", [
    ("blank name", _mutated(lambda r: r["aprobacion"].update(nombre="   ")), "aprobacion.nombre"),
    ("name not text", _mutated(lambda r: r["aprobacion"].update(nombre=7)), "aprobacion.nombre"),
    ("invalid date", _mutated(lambda r: r["aprobacion"].update(fecha="13/11/2026")), "aprobacion.fecha"),
    ("short commit", _mutated(lambda r: r["procedencia"].update(commit_codigo="abc123")), "commit_codigo"),
    ("uppercase commit", _mutated(lambda r: r["procedencia"].update(commit_contrato="D" * 40)), "commit_contrato"),
    ("bad channel format", _mutated(lambda r: r["procedencia"].update(version_dvc_canal="xyz")), "version_dvc_canal"),
    ("section not a mapping", _mutated(lambda r: r.update(aprobacion="Jesús")), "aprobacion"),
    ("hash not sha256", {**_recipe(), "hash_sha256": "abc"}, "hash_sha256"),
])
def test_invalid_formats_are_rejected(tmp_path, label, recipe, message):
    with pytest.raises(FrozenRecipeError, match=message):
        load_frozen_recipe(_write(tmp_path / "r.yaml", recipe), channel_version=CHANNEL)


def test_not_a_mapping_and_invalid_yaml(tmp_path):
    (tmp_path / "list.yaml").write_text("- a\n- b\n", encoding="utf-8")
    (tmp_path / "broken.yaml").write_text("a: [1, 2\n", encoding="utf-8")
    with pytest.raises(FrozenRecipeError, match="mapping"):
        load_frozen_recipe(tmp_path / "list.yaml", channel_version=CHANNEL)
    with pytest.raises(FrozenRecipeError, match="not valid YAML"):
        load_frozen_recipe(tmp_path / "broken.yaml", channel_version=CHANNEL)


def test_duplicate_keys_are_rejected(tmp_path):
    path = _write(tmp_path / "r.yaml", _recipe())
    text = path.read_text(encoding="utf-8").replace("familia: lightgbm\n", "familia: lightgbm\nfamilia: regresion_logistica\n")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(FrozenRecipeError, match="duplicate key"):
        load_frozen_recipe(path, channel_version=CHANNEL)


def test_unserializable_content_is_rejected():
    with pytest.raises(FrozenRecipeError, match="serialized"):
        recipe_hash({"tau": float("nan")})
    with pytest.raises(FrozenRecipeError, match="serialized"):
        recipe_hash({"valores": {1, 2}})
    with pytest.raises(FrozenRecipeError, match="mapping"):
        recipe_hash(["no", "mapping"])


def test_errors_name_both_hashes_and_both_channels(tmp_path):
    altered = {**_recipe(), "hash_sha256": "0" * 64}
    with pytest.raises(FrozenRecipeError, match="0{64}.*differs"):
        load_frozen_recipe(_write(tmp_path / "a.yaml", altered), channel_version=CHANNEL)
    other = "f" * 32
    with pytest.raises(FrozenRecipeError, match=f"{CHANNEL}.*{other}"):
        load_frozen_recipe(_write(tmp_path / "b.yaml", _recipe()), channel_version=other)


# --------------------------------------------------------------------------- access
@pytest.mark.parametrize("years, channel", [((2025, 2023), CHANNEL), ((2023,), CHANNEL), ((2023.0, 2025), CHANNEL),
                                             ((2023, 2025), "ABC"), ((2023, 2025), None)])
def test_access_is_validated(years, channel):
    with pytest.raises(EpidemiologyValidationError):
        RecipeAccess(test_years=years, channel_version=channel, recipe_path="r.yaml")


def test_access_from_config_uses_the_default_path():
    from src.forecasting.config import load_config

    config = load_config()
    access = RecipeAccess.from_config(config)
    assert access.recipe_path == FROZEN_RECIPE_PATH == Path("configs/receta_congelada.yaml")
    assert access.test_years == tuple(config.validacion.prueba)
    assert access.channel_version == config.datos.version_dvc_canal
    if FROZEN_RECIPE_PATH.exists():                             # recipe in force (D7.2, 6fc68f6), approved by the person responsible for the implementation: must be valid
        load_frozen_recipe(FROZEN_RECIPE_PATH, channel_version=config.datos.version_dvc_canal)
