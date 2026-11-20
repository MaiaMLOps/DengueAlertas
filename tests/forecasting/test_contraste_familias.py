"""Family contrast (análisis complementario): configuration, candidates and pipelines, without fitting any model."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.pipeline import Pipeline

from src.forecasting.config import (
    CONTRAST_CONFIG_PATH,
    CONTRAST_FAMILIES,
    ROOT,
    ContrastSearchSection,
    RandomSearch,
    SearchSection,
    contrast_base_config,
    contrast_search_config,
    load_config,
    load_contrast_config,
)
from src.forecasting.dataset import FEATURE_SETS
from src.forecasting.models import FAMILIES, MODEL_FAMILIES, make_model
from src.forecasting.search import candidates_from_config

RANKING = ROOT / "reports" / "tesis_contraste_modelos" / "tabla_01_ranking_global_sin_clima.csv"
SCRIPTS = ("run_contrast_search", "run_seq_dl", "run_horizon_sensitivity", "run_horizon_climate_mlflow",
           "log_seq_dl_mlflow", "build_thesis_report")
OPTIONAL = {"catboost": "catboost", "xgboost": "xgboost"}


@pytest.fixture(scope="module")
def contrast():
    return load_contrast_config(CONTRAST_CONFIG_PATH)


@pytest.fixture(scope="module")
def base(contrast):
    return contrast_base_config(contrast)


def _space(**extra):
    return {"n_candidatos": 2, "semilla_muestreo": 1, "semilla_modelo": 1,
            "espacio": {"C": {"escala": "log", "rango": [0.1, 1.0]}}, **extra}


# --------------------------------------------------------------------------- configuration
def test_contract_families_unchanged():
    assert FAMILIES == ("logistica", "lightgbm")
    assert MODEL_FAMILIES == FAMILIES + CONTRAST_FAMILIES


def test_contrast_config_validates(contrast):
    assert contrast.experimento == "dengue_h8_contraste_familias"
    assert contrast.variante == "sin_clima"
    assert set(contrast.corridas) == {"sin_clima", "logistica_ampliada", "cb_xgb", "svm_ampliado", "mlp_sin_clima",
                                      "dl_secuencial"}
    assert all(Path(run.carpeta).name == name for name, run in contrast.corridas.items())
    grid = contrast.corridas["logistica_ampliada"].busqueda.logistica.C
    assert len(grid) == 19 and grid[0] == 0.001 and grid[-1] == 100.0


def test_main_configuration_has_only_the_contract_families():
    config = load_config()
    assert type(config.busqueda) is SearchSection
    assert set(candidates_from_config(config.busqueda)) == {"logistica", "lightgbm"}


def test_contrast_search_config_changes_only_search_and_experiment(contrast, base):
    merged = contrast_search_config(base, contrast, "svm_ampliado")
    assert merged.busqueda == contrast.corridas["svm_ampliado"].busqueda
    assert merged.seguimiento.experimentos.desarrollo == contrast.experimento
    assert merged.seguimiento.experimentos.prueba == base.seguimiento.experimentos.prueba
    for section in ("datos", "canal", "horizonte", "features", "validacion", "lineas_base", "metricas", "decision"):
        assert getattr(merged, section) == getattr(base, section)
    with pytest.raises(ValueError, match="no tabular search"):
        contrast_search_config(base, contrast, "dl_secuencial")
    with pytest.raises(ValueError, match="Unknown contrast run"):
        contrast_search_config(base, contrast, "otra")


def test_random_search_validation():
    RandomSearch.model_validate(_space())
    with pytest.raises(ValidationError, match="both fixed and sampled"):
        RandomSearch.model_validate(_space(fijos={"C": 1.0}))
    with pytest.raises(ValidationError, match="random_state"):
        RandomSearch.model_validate(_space(fijos={"random_state": 3}))
    capped = _space()
    capped["espacio"] = {"num_leaves": {"escala": "entera", "rango": [4, 8], "tope_por_profundidad": True}}
    with pytest.raises(ValidationError, match="tope_por_profundidad"):
        RandomSearch.model_validate(capped)


def test_contrast_search_section_accepts_a_contrast_family_alone():
    section = ContrastSearchSection.model_validate({"svm": _space()})
    assert section.logistica is None and section.svm is not None
    with pytest.raises(ValidationError, match="at least one family"):
        ContrastSearchSection.model_validate({})
    with pytest.raises(ValidationError):
        SearchSection.model_validate({"svm": _space()})   # the pre-registered section has no contrast families


# --------------------------------------------------------------------------- candidates and pipelines
def test_winning_candidates_regenerate_from_the_configuration(contrast, base):
    ranking = pd.read_csv(RANKING)
    tabular = ranking[ranking["corrida"].isin([name for name, run in contrast.corridas.items()
                                               if run.busqueda is not None])]
    assert len(tabular) == 5
    for row in tabular.itertuples(index=False):
        candidates = candidates_from_config(contrast_search_config(base, contrast, row.corrida).busqueda)
        family = row.candidato.rsplit("_", 1)[0]
        params = dict(next(c.params for c in candidates[family] if c.name == row.candidato))
        stored = json.loads(row.parametros)
        added = {name: params.pop(name) for name in set(params) - set(stored)}
        assert params == stored, row.candidato
        assert added in ({}, {"gamma": 0.0, "reg_alpha": 0.0}), row.candidato   # XGBoost defaults (cb_xgb)


def _first_candidates(contrast, base):
    out = {}
    for name, run in contrast.corridas.items():
        section = run.busqueda if run.busqueda is not None else base.busqueda
        if section is None:
            continue
        for family, candidates in candidates_from_config(section).items():
            out.setdefault(family, candidates[0])
    return out


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_pipeline_of_each_tabular_family_builds_without_fitting(contrast, base, family):
    if family in OPTIONAL:
        pytest.importorskip(OPTIONAL[family])
    candidate = _first_candidates(contrast, base)[family]
    for features in (FEATURE_SETS["sin_clima"], FEATURE_SETS["completo"]):
        model = make_model(family, candidate.params, features=features)
        assert isinstance(model, Pipeline)
        estimator = model.named_steps["modelo"]
        if family == "catboost":   # CatBoost defines classes_ before fitting
            assert not estimator.is_fitted()
        else:
            assert not hasattr(estimator, "classes_")   # not fitted


def test_mlp_layers():
    params = {"hidden_units": 21, "n_layers": 2, "alpha": 0.1, "learning_rate_init": 0.001, "max_iter": 200,
              "batch_size": 32, "random_state": 1}
    assert make_model("mlp", params).named_steps["modelo"].hidden_layer_sizes == (21, 10)
    params["n_layers"] = 1
    assert make_model("mlp", params).named_steps["modelo"].hidden_layer_sizes == (21,)


def test_sequential_candidates_reproduce_the_reported_winners(contrast):
    sys.path.insert(0, str(ROOT))
    from scripts.run_seq_dl import sample_candidates

    _, search = contrast.secuencial
    gru = dict(sample_candidates("gru", search.espacio, search.n_candidatos, search.semilla))
    cnn = dict(sample_candidates("cnn1d", search.espacio, search.n_candidatos, search.semilla))
    assert gru["gru_00"] == {"hidden_size": 8, "num_layers": 1, "dropout": 0.1547, "lr": 0.009185,
                             "weight_decay": 0.000234, "epochs": 115, "batch_size": 256}
    assert cnn["cnn1d_03"] == {"channels": 16, "kernel_size": 3, "num_layers": 2, "dropout": 0.0398, "lr": 0.00059,
                               "weight_decay": 0.000309, "epochs": 153, "batch_size": 128}


@pytest.mark.parametrize("script", SCRIPTS)
def test_contrast_scripts_answer_help(script):
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / f"{script}.py"), "--help"],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
