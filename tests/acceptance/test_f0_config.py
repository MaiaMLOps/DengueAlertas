"""F0 acceptance: forecasting configuration (especificación §6.F0; contrato §10.5 and §10.14).

Literal values come from the contract: h = 8 (D4, §3.5), sensitivity h in {2, 4, 12} (§3.6),
k = 2 and W = 12 (D8, §7.4), tau grid 0.05..0.95 step 0.05 (§7.5), development 2019-2022 and
test 2023-2025 (D7). The YAML structure mirrors the attribute paths checked here.
"""

import shutil
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.acceptance

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "forecasting.yaml"
EXPECTED_TAU_GRID = [round(0.05 * i, 2) for i in range(1, 20)]


@pytest.fixture(autouse=True)
def _no_horizon_override(monkeypatch):
    monkeypatch.delenv("FORECAST_HORIZON", raising=False)


def _config_with(tmp_path, dotted_key, value):
    """Copy the real YAML and replace a single nested key."""
    target = tmp_path / "forecasting.yaml"
    shutil.copyfile(CONFIG_PATH, target)
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    node = data
    *parents, leaf = dotted_key.split(".")
    for part in parents:
        node = node[part]
    assert leaf in node, f"{dotted_key} must exist in configs/forecasting.yaml"
    node[leaf] = value
    target.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return target


def test_contract_values_load_and_validate():
    from src.forecasting.config import load_config

    config = load_config()

    assert config.horizonte.h == 8
    assert config.decision.k == 2
    assert config.decision.W == 12
    assert len(config.decision.tau_grid) == 19
    assert [round(value, 2) for value in config.decision.tau_grid] == EXPECTED_TAU_GRID
    assert tuple(config.validacion.desarrollo) == (2019, 2022)
    assert tuple(config.validacion.prueba) == (2023, 2025)


@pytest.mark.parametrize(
    ("dotted_key", "value", "named_field"),
    [
        ("horizonte.h", 0, "horizonte.h"),
        ("horizonte.h", 5, "horizonte.h"),
        ("decision.k", 0, "decision.k"),
        ("decision.tau_grid", [0.05, 1.2], "decision.tau_grid"),
        ("validacion.prueba", [2022, 2025], "validacion"),
    ],
    ids=["h_zero", "h_five_not_allowed", "k_zero", "tau_outside_unit_interval", "test_overlaps_development"],
)
def test_invalid_value_raises_clear_error(tmp_path, dotted_key, value, named_field):
    from src.forecasting.config import load_config

    path = _config_with(tmp_path, dotted_key, value)

    with pytest.raises(ValueError) as error:
        load_config(path)

    assert named_field in str(error.value)


def test_forecast_horizon_env_overrides_h(monkeypatch):
    from src.forecasting.config import load_config

    monkeypatch.setenv("FORECAST_HORIZON", "4")

    config = load_config()

    assert config.horizonte.h == 4
    assert config.decision.k == 2
    assert config.decision.W == 12


@pytest.mark.parametrize("raw", ["abc", "4.5", "4.0", "5"], ids=["text", "decimal", "float_literal", "not_allowed"])
def test_forecast_horizon_env_invalid_fails(monkeypatch, raw):
    from src.forecasting.config import load_config

    monkeypatch.setenv("FORECAST_HORIZON", raw)

    with pytest.raises(ValueError) as error:
        load_config()

    assert "FORECAST_HORIZON" in str(error.value)
