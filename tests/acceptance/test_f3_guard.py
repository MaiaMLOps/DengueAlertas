"""F3 acceptance: test-period barrier and minimal frozen-recipe format (CLAUDE.md rule 1; especificación §4
«Barrera de prueba: detalle», §6.F3 and §8; contrato §6.1 and §7.1).

API frozen here:
- `guard.RecipeAccess(test_years, channel_version, recipe_path)`: what the barrier needs;
- `guard.check_test_access(target_weeks, access)`: the single verification function. If any target week
  falls in the test years it requires a valid frozen recipe at `recipe_path`, otherwise it raises
  `guard.FrozenRecipeError`; rows outside the test years need no recipe;
- `guard.recipe_hash(recipe)`: SHA-256 of the recipe without its `hash_sha256` field;
- the scoring functions call the barrier with the target weeks of the rows they score:
  `metrics.log_loss(y, p, target_weeks, *, access, clip)`, `metrics.brier(y, p, target_weeks, *, access)`,
  `metrics.average_precision(y, p, target_weeks, *, access)`, and `baselines.b1_predict` /
  `baselines.b2_predict` with the target weeks of `evaluate` (training rows are not scored).

Minimal recipe (especificación §8): `procedencia` {commit_codigo, commit_contrato, version_dvc_canal},
`aprobacion` {nombre, fecha} and `hash_sha256`. It is valid if those fields exist and are not empty, `version_dvc_canal`
equals the configured channel version and the hash matches the content. Serialization of the hash (approved
by Jesús in U3.1): json.dumps(recipe without hash_sha256, sort_keys=True, separators=(",", ":"),
ensure_ascii=False), dates in ISO 8601 (YYYY-MM-DD), encoded in UTF-8.

Only synthetic recipes inside tmp_path are used; configs/receta_congelada.yaml is never created.
"""

from datetime import date, timedelta
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import yaml

pytestmark = pytest.mark.acceptance

H = 8
TEST_YEARS = (2023, 2024, 2025)
SYSTEMS = ("check_test_access", "log_loss", "brier", "average_precision", "b1_predict", "b2_predict")


def _config():
    from src.forecasting.config import load_config

    return load_config()


def _iso(value):
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Not serializable: {type(value).__name__}")


def _independent_hash(recipe: dict) -> str:
    content = {key: value for key, value in recipe.items() if key != "hash_sha256"}
    text = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_iso)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _recipe() -> dict:
    recipe = {
        "familia": "regresion_logistica",
        "hiperparametros": {"08001": {"C": 0.1}, "50001": {"C": 0.3}},
        "decision": {"tau": 0.3, "k": 2, "W": 12},
        "procedencia": {"commit_codigo": "a" * 40, "commit_contrato": "b" * 40,
                        "version_dvc_canal": _config().datos.version_dvc_canal},
        "aprobacion": {"nombre": "Receta sintética de prueba (no es una aprobación real)", "fecha": date(2026, 11, 13)},
    }
    recipe["hash_sha256"] = _independent_hash(recipe)
    return recipe


def _write(path, recipe):
    path.write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _access(recipe_path):
    from src.forecasting.guard import RecipeAccess

    config = _config()
    return RecipeAccess(test_years=tuple(config.validacion.prueba), channel_version=config.datos.version_dvc_canal,
                        recipe_path=recipe_path)


def _rows(years):
    """Four scored rows per target year (two positive), with zona_t; t = target_week - h."""
    rows = []
    for year in years:
        for k in range(4):
            week = date(year, 3, 6) - timedelta(days=date(year, 3, 6).weekday()) + timedelta(weeks=k)
            rows.append(("08001", week - timedelta(weeks=H), week, int(k % 2 == 0), "alerta"))
    return pd.DataFrame(rows, columns=["municipality_code", "t", "target_week", "y", "zona_t"]).astype({"y": "int8"})


def _train():
    rows = [("08001", date(2015, 3, 2) + timedelta(weeks=k), date(2015, 4, 27) + timedelta(weeks=k), k % 2, "alerta")
            for k in range(6)]
    return pd.DataFrame(rows, columns=["municipality_code", "t", "target_week", "y", "zona_t"]).astype({"y": "int8"})


def _score(system, rows, access):
    from src.forecasting import baselines, guard, metrics

    y, p, weeks = rows["y"].to_numpy(), np.linspace(0.2, 0.8, len(rows)), list(rows["target_week"])
    if system == "check_test_access":
        return guard.check_test_access(weeks, access)
    if system == "log_loss":
        return metrics.log_loss(y, p, weeks, access=access, clip=_config().metricas.recorte_log_loss)
    if system == "brier":
        return metrics.brier(y, p, weeks, access=access)
    if system == "average_precision":
        return metrics.average_precision(y, p, weeks, access=access)
    if system == "b1_predict":
        return baselines.b1_predict(_train(), rows, half_window=_config().lineas_base.b1_semiventana, access=access)
    if system == "b2_predict":
        return baselines.b2_predict(_train(), rows, access=access)
    raise AssertionError(system)


def _error():
    from src.forecasting.guard import FrozenRecipeError

    return FrozenRecipeError


def test_configured_test_years_match_contract():
    assert tuple(_config().validacion.prueba) == (2023, 2025)


# --------------------------------------------------------------------------- without a recipe
@pytest.mark.parametrize("year", TEST_YEARS)
@pytest.mark.parametrize("system", SYSTEMS)
def test_test_rows_without_recipe_raise(tmp_path, system, year):
    with pytest.raises(_error()):
        _score(system, _rows([year]), _access(tmp_path / "receta_inexistente.yaml"))


@pytest.mark.parametrize("system", SYSTEMS)
def test_mixed_development_and_test_rows_raise(tmp_path, system):
    with pytest.raises(_error()):
        _score(system, _rows([2022, 2023]), _access(tmp_path / "receta_inexistente.yaml"))


@pytest.mark.parametrize("system", SYSTEMS)
def test_development_rows_need_no_recipe(tmp_path, system):
    _score(system, _rows([2019, 2020, 2021, 2022]), _access(tmp_path / "receta_inexistente.yaml"))


# --------------------------------------------------------------------------- with a recipe
@pytest.mark.parametrize("system", SYSTEMS)
def test_valid_synthetic_recipe_allows_test_rows(tmp_path, system):
    path = _write(tmp_path / "receta.yaml", _recipe())
    _score(system, _rows(list(TEST_YEARS)), _access(path))


def _altered_hash(recipe):
    recipe["hash_sha256"] = "0" * 64


def _value_changed(recipe):
    recipe["decision"]["tau"] = 0.35


def _without(*keys):
    def mutate(recipe):
        target = recipe
        for key in keys[:-1]:
            target = target[key]
        del target[keys[-1]]
        if keys != ("hash_sha256",):
            recipe["hash_sha256"] = _independent_hash(recipe)
    return mutate


def _empty_approval_name(recipe):
    recipe["aprobacion"]["nombre"] = ""
    recipe["hash_sha256"] = _independent_hash(recipe)


def _other_channel(recipe):
    recipe["procedencia"]["version_dvc_canal"] = "f" * 32
    recipe["hash_sha256"] = _independent_hash(recipe)


INVALID = {
    "hash alterado": _altered_hash,
    "valor cambiado sin rehacer el hash": _value_changed,
    "sin hash": _without("hash_sha256"),
    "sin aprobacion": _without("aprobacion"),
    "sin nombre de aprobacion": _without("aprobacion", "nombre"),
    "sin fecha de aprobacion": _without("aprobacion", "fecha"),
    "sin commit del codigo": _without("procedencia", "commit_codigo"),
    "sin commit del contrato": _without("procedencia", "commit_contrato"),
    "sin version del canal": _without("procedencia", "version_dvc_canal"),
    "otra version del canal": _other_channel,
    "nombre de aprobacion vacio": _empty_approval_name,
}


@pytest.mark.parametrize("system", ["check_test_access", "log_loss"])
@pytest.mark.parametrize("case", list(INVALID))
def test_invalid_recipe_raises(tmp_path, case, system):
    recipe = _recipe()
    INVALID[case](recipe)
    path = _write(tmp_path / "receta.yaml", recipe)
    with pytest.raises(_error()):
        _score(system, _rows([2024]), _access(path))


# --------------------------------------------------------------------------- hash serialization
def test_recipe_hash_matches_frozen_serialization():
    from src.forecasting.guard import recipe_hash

    recipe = _recipe()
    without_field = {key: value for key, value in recipe.items() if key != "hash_sha256"}
    assert recipe_hash(recipe) == recipe["hash_sha256"]
    assert recipe_hash(without_field) == recipe["hash_sha256"]


def _reversed(value):
    if isinstance(value, dict):
        return {key: _reversed(value[key]) for key in reversed(list(value))}
    return value


def test_reordering_the_yaml_does_not_change_the_hash(tmp_path):
    from src.forecasting.guard import recipe_hash

    recipe = _recipe()
    original = _write(tmp_path / "receta.yaml", recipe)
    reordered = _write(tmp_path / "receta_reordenada.yaml", _reversed(recipe))
    assert original.read_text(encoding="utf-8") != reordered.read_text(encoding="utf-8")

    loaded = yaml.safe_load(reordered.read_text(encoding="utf-8"))
    assert recipe_hash(loaded) == recipe["hash_sha256"]
    _score("check_test_access", _rows([2023]), _access(reordered))


def test_changing_a_value_changes_the_hash():
    from src.forecasting.guard import recipe_hash

    recipe = _recipe()
    changed = copy.deepcopy(recipe)
    changed["aprobacion"]["fecha"] = date(2026, 11, 14)
    assert recipe_hash(changed) != recipe_hash(recipe)
