"""Unit checks of the configuration schema beyond the frozen F0 acceptance tests."""

import copy

import pytest
import yaml
from pydantic import ValidationError

from src.forecasting.config import DEFAULT_CONFIG_PATH, load_config


@pytest.fixture(autouse=True)
def _no_horizon_override(monkeypatch):
    monkeypatch.delenv("FORECAST_HORIZON", raising=False)


@pytest.fixture
def base_data():
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def _write(tmp_path, data):
    path = tmp_path / "forecasting.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_unknown_top_level_key_is_rejected(tmp_path, base_data):
    data = copy.deepcopy(base_data)
    data["busqueda_typo"] = {}
    with pytest.raises(ValidationError, match="busqueda_typo"):
        load_config(_write(tmp_path, data))


def test_unknown_nested_key_is_rejected(tmp_path, base_data):
    data = copy.deepcopy(base_data)
    data["decision"]["tau"] = 0.5
    with pytest.raises(ValidationError, match=r"decision\.tau"):
        load_config(_write(tmp_path, data))


@pytest.mark.parametrize("grid", [[0.10, 0.05], [0.05, 0.05, 0.10]], ids=["unordered", "repeated"])
def test_tau_grid_must_be_strictly_increasing(tmp_path, base_data, grid):
    data = copy.deepcopy(base_data)
    data["decision"]["tau_grid"] = grid
    with pytest.raises(ValidationError, match=r"decision\.tau_grid"):
        load_config(_write(tmp_path, data))


def test_first_training_year_after_development_start_is_rejected(tmp_path, base_data):
    data = copy.deepcopy(base_data)
    data["validacion"]["primer_anio_entrenamiento"] = 2020
    with pytest.raises(ValidationError, match="primer_anio_entrenamiento"):
        load_config(_write(tmp_path, data))


def test_feature_block_after_t_is_rejected(tmp_path, base_data):
    data = copy.deepcopy(base_data)
    data["features"]["bloques"]["m1"] = [-3, 1]
    with pytest.raises(ValidationError, match=r"features\.bloques"):
        load_config(_write(tmp_path, data))


@pytest.mark.parametrize("value", [0, 26, "2", True, 2.0])
def test_b1_half_window_must_be_an_integer_from_1_to_25(tmp_path, base_data, value):
    data = copy.deepcopy(base_data)
    data["lineas_base"]["b1_semiventana"] = value
    with pytest.raises(ValidationError, match=r"lineas_base\.b1_semiventana"):
        load_config(_write(tmp_path, data))


def test_baselines_section_is_required(tmp_path, base_data):
    data = copy.deepcopy(base_data)
    del data["lineas_base"]
    with pytest.raises(ValidationError, match="lineas_base"):
        load_config(_write(tmp_path, data))


@pytest.mark.parametrize("value", [0, 0.0, 0.5, "1e-15", True])
def test_log_loss_clip_must_be_a_float_in_open_interval(tmp_path, base_data, value):
    data = copy.deepcopy(base_data)
    data["metricas"]["recorte_log_loss"] = value
    with pytest.raises(ValidationError, match=r"metricas\.recorte_log_loss"):
        load_config(_write(tmp_path, data))


def test_metrics_section_is_required(tmp_path, base_data):
    data = copy.deepcopy(base_data)
    del data["metricas"]
    with pytest.raises(ValidationError, match="metricas"):
        load_config(_write(tmp_path, data))


def test_unknown_tracking_mode_is_rejected(tmp_path, base_data):
    data = copy.deepcopy(base_data)
    data["seguimiento"]["modo"] = "automatico"
    with pytest.raises(ValidationError, match=r"seguimiento\.modo"):
        load_config(_write(tmp_path, data))


def test_horizon_env_with_whitespace_is_rejected(monkeypatch):
    monkeypatch.setenv("FORECAST_HORIZON", " 4")
    with pytest.raises(ValueError, match="FORECAST_HORIZON"):
        load_config()


def test_loaded_config_is_immutable():
    config = load_config()
    with pytest.raises(ValidationError):
        config.horizonte.h = 4


def test_feature_and_channel_values_match_yaml():
    config = load_config()
    assert config.features.w == 4
    assert config.features.bloques == {"m1": (-3, 0), "m2": (-7, -4), "m3": (-11, -8)}
    assert config.features.ventana_semanas == 12
    assert (config.canal.n, config.canal.desplazamiento, config.canal.cuantil_t) == (5, 1, 0.025)
    assert config.validacion.primer_anio_entrenamiento == 2012
    assert config.seguimiento.modo == "servidor"   # since the pre-registration (F5b)


# --------------------------------------------------------------------------- busqueda (U5.3; values from the pre-registration)
SEARCH = {
    "logistica": {"C": [0.1, 1.0], "max_iter": 2000},
    "lightgbm": {
        "n_candidatos": 3, "semilla_muestreo": 11, "semilla_modelo": 7, "fijos": {"subsample_freq": 1},
        "espacio": {"max_depth": {"escala": "entera", "rango": [2, 5]},
                    "num_leaves": {"escala": "entera", "rango": [4, 31], "tope_por_profundidad": True},
                    "learning_rate": {"escala": "log", "rango": [0.01, 0.1]}},
    },
}


def test_busqueda_holds_the_pre_registration():
    search = load_config().busqueda
    assert search.logistica.C == (0.01, 0.03, 0.1, 0.3, 1.0, 3.0) and search.logistica.max_iter == 5000
    lgb = search.lightgbm
    assert (lgb.n_candidatos, lgb.semilla_muestreo, lgb.semilla_modelo) == (25, 20261112, 20261112)
    assert lgb.fijos == {"subsample_freq": 1}
    assert {name: (space.escala, space.rango, space.tope_por_profundidad) for name, space in lgb.espacio.items()} == {
        "max_depth": ("entera", (2, 5), False), "num_leaves": ("entera", (4, 31), True),
        "learning_rate": ("log", (0.01, 0.1), False), "n_estimators": ("entera", (100, 600), False),
        "min_child_samples": ("entera", (10, 50), False), "reg_lambda": ("real", (0.0, 10.0), False),
        "subsample": ("real", (0.6, 1.0), False), "colsample_bytree": ("real", (0.6, 1.0), False)}


def test_valid_busqueda_loads(tmp_path, base_data):
    data = copy.deepcopy(base_data)
    data["busqueda"] = copy.deepcopy(SEARCH)
    config = load_config(_write(tmp_path, data))
    assert config.busqueda.logistica.C == (0.1, 1.0)
    assert config.busqueda.lightgbm.espacio["num_leaves"].tope_por_profundidad


@pytest.mark.parametrize(("change", "message"), [
    (lambda s: s["logistica"].update(C=[1.0, 0.1]), "strictly increasing"),
    (lambda s: s["logistica"].update(C=[0.0, 1.0]), "positive"),
    (lambda s: s["lightgbm"]["espacio"]["learning_rate"].update(rango=[0.0, 0.1]), "positive range"),
    (lambda s: s["lightgbm"]["espacio"]["max_depth"].update(rango=[2.5, 5]), "integer bounds"),
    (lambda s: s["lightgbm"]["espacio"]["max_depth"].update(rango=[5, 2]), "not ordered"),
    (lambda s: s["lightgbm"]["fijos"].update(max_depth=3), "both fixed and sampled"),
    (lambda s: s["lightgbm"]["fijos"].update(random_state=1), "semilla_modelo"),
    (lambda s: s["lightgbm"]["espacio"]["learning_rate"].update(tope_por_profundidad=True), "only applies"),
    (lambda s: s.update(logistica=None, lightgbm=None), "at least one family"),
])
def test_invalid_busqueda_is_rejected(tmp_path, base_data, change, message):
    data = copy.deepcopy(base_data)
    search = copy.deepcopy(SEARCH)
    change(search)
    data["busqueda"] = search
    with pytest.raises(ValidationError, match=message):
        load_config(_write(tmp_path, data))


@pytest.mark.parametrize("alpha", [0, 1, 1.5, "0.05"])
def test_comparison_alpha_must_be_a_float_in_the_unit_interval(tmp_path, base_data, alpha):
    data = copy.deepcopy(base_data)
    data["comparacion"]["alfa"] = alpha
    with pytest.raises(ValidationError, match="comparacion"):
        load_config(_write(tmp_path, data))
