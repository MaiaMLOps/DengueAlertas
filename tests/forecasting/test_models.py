"""Unit tests of models.py (U5.2) with synthetic rows: preprocessing fitted on train only, two zone indicators,
categorical zone in LightGBM, probabilistic output without class weights, determinism and clear errors."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.epidemiology.channel_epico import ZONES
from src.epidemiology.config import EpidemiologyValidationError
from src.forecasting.dataset import FEATURES
from src.forecasting.models import FAMILIES, NUMERIC_FEATURES, fit_model, make_model, predict_positive

LOGISTIC = {"C": 0.3, "max_iter": 2000}
LIGHTGBM = {"num_leaves": 7, "max_depth": 3, "learning_rate": 0.05, "n_estimators": 80, "min_child_samples": 10,
            "reg_lambda": 1.0, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "random_state": 7}
PARAMS = {"logistica": LOGISTIC, "lightgbm": LIGHTGBM}
REL = 1e-9   # criterion approved for two implementations of the same quantity


def _rows(n, seed, zones=ZONES):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({name: rng.normal(loc=i, scale=1 + i, size=n) for i, name in enumerate(NUMERIC_FEATURES)})
    frame.insert(0, "zona_t", rng.choice(list(zones), size=n))
    score = 0.8 * frame["posicion_4s"] - 0.3 + frame["zona_t"].map({"seguridad": -1.0, "alerta": 0.0, "epidemia": 1.5})
    frame["y"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-score))).astype("int8")
    return frame


@pytest.fixture
def train():
    return _rows(300, 1)


@pytest.fixture
def evaluate():
    return _rows(60, 2)


# --------------------------------------------------------------------------- logistic regression
def test_standardization_uses_train_only(train, evaluate):
    model = fit_model("logistica", LOGISTIC, train)
    scaler = model.named_steps["preparar"].named_transformers_["estandarizar"]
    assert np.allclose(scaler.mean_, train[list(NUMERIC_FEATURES)].mean().to_numpy(), rtol=REL, atol=0)
    assert np.allclose(scaler.scale_, train[list(NUMERIC_FEATURES)].std(ddof=0).to_numpy(), rtol=REL, atol=0)

    base = predict_positive(model, evaluate)
    shifted = evaluate.copy()
    shifted.loc[shifted.index[0], "posicion_4s"] += 100.0                  # change one evaluated row
    extended = pd.concat([evaluate, _rows(40, 3)], ignore_index=True)       # add evaluated rows
    assert np.array_equal(predict_positive(model, shifted)[1:], base[1:])
    assert np.array_equal(predict_positive(model, extended)[: len(evaluate)], base)


def test_zone_has_two_indicators_with_seguridad_as_reference(train):
    model = fit_model("logistica", LOGISTIC, train)
    prepare = model.named_steps["preparar"]
    probe = _rows(3, 4).assign(zona_t=list(ZONES))
    matrix = prepare.transform(probe[list(FEATURES)])
    assert matrix.shape == (3, 16)
    assert matrix[:, -2:].tolist() == [[0, 0], [1, 0], [0, 1]]
    assert model.named_steps["modelo"].coef_.shape == (1, 16)


def test_missing_zone_in_train_keeps_both_indicators():
    train = _rows(200, 5, zones=("seguridad", "alerta"))
    model = fit_model("logistica", LOGISTIC, train)
    probe = _rows(2, 6).assign(zona_t=["epidemia", "seguridad"])
    assert model.named_steps["preparar"].transform(probe[list(FEATURES)]).shape == (2, 16)
    assert predict_positive(model, probe).shape == (2,)


def test_logistic_matches_manual_construction(train, evaluate):
    numeric = list(NUMERIC_FEATURES)
    mean, std = train[numeric].mean(), train[numeric].std(ddof=0)

    def design(frame):
        indicators = np.column_stack([(frame["zona_t"] == "alerta"), (frame["zona_t"] == "epidemia")]).astype(float)
        return np.column_stack([((frame[numeric] - mean) / std).to_numpy(), indicators])

    manual = LogisticRegression(C=LOGISTIC["C"], max_iter=LOGISTIC["max_iter"]).fit(design(train), train["y"])
    expected = manual.predict_proba(design(evaluate))[:, 1]
    observed = predict_positive(fit_model("logistica", LOGISTIC, train), evaluate)
    assert observed == pytest.approx(expected, rel=REL, abs=0)


# --------------------------------------------------------------------------- LightGBM
def test_lightgbm_zone_is_categorical_with_fixed_categories():
    train = _rows(300, 8, zones=("seguridad", "alerta"))
    model = fit_model("lightgbm", LIGHTGBM, train)
    booster = model.named_steps["modelo"].booster_
    assert booster.pandas_categorical == [list(ZONES)]
    info = booster.dump_model()
    assert info["feature_names"][0] == "zona_t"
    assert predict_positive(model, _rows(3, 9).assign(zona_t=list(ZONES))).shape == (3,)


def test_lightgbm_separates_a_zone_with_fewer_than_100_rows():
    rng = np.random.default_rng(10)
    train = _rows(330, 11, zones=("seguridad", "alerta"))
    train.loc[train.index[:30], "zona_t"] = "epidemia"
    train["y"] = (train["zona_t"] == "epidemia").astype("int8")
    for name in NUMERIC_FEATURES:                                   # numeric features carry no signal
        train[name] = rng.normal(size=len(train))
    model = fit_model("lightgbm", LIGHTGBM, train)
    probe = _rows(3, 12).assign(zona_t=list(ZONES))
    for name in NUMERIC_FEATURES:
        probe[name] = 0.0
    p = predict_positive(model, probe)
    assert p[2] > 0.5 > max(p[0], p[1])


# --------------------------------------------------------------------------- both families
@pytest.mark.parametrize("family", FAMILIES)
def test_output_is_probability_in_row_order(family, train, evaluate):
    model = fit_model(family, PARAMS[family], train)
    p = predict_positive(model, evaluate)
    assert p.shape == (len(evaluate),) and p.dtype == float
    assert ((p > 0) & (p < 1)).all()
    assert np.array_equal(p, model.predict_proba(evaluate[list(FEATURES)])[:, 1])
    reversed_rows = evaluate.iloc[::-1]
    assert np.array_equal(predict_positive(model, reversed_rows), p[::-1])


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("weight", [{"class_weight": "balanced"}, {"is_unbalance": True}, {"scale_pos_weight": 3.0}])
def test_class_weights_are_rejected(family, weight):
    with pytest.raises(EpidemiologyValidationError, match="class weights"):
        make_model(family, {**PARAMS[family], **weight})


@pytest.mark.parametrize("family", FAMILIES)
def test_estimators_have_no_class_weight(family):
    assert make_model(family, PARAMS[family]).named_steps["modelo"].get_params()["class_weight"] is None


@pytest.mark.parametrize("family", FAMILIES)
def test_two_fits_are_identical(family, train, evaluate):
    first = predict_positive(fit_model(family, PARAMS[family], train), evaluate)
    second = predict_positive(fit_model(family, PARAMS[family], train), evaluate)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("family", FAMILIES)
def test_column_order_does_not_matter(family, train, evaluate):
    p = predict_positive(fit_model(family, PARAMS[family], train), evaluate)
    shuffled_train = train[list(train.columns[::-1])]
    shuffled_rows = evaluate[list(evaluate.columns[::-1])]
    assert np.array_equal(predict_positive(fit_model(family, PARAMS[family], shuffled_train), shuffled_rows), p)


# --------------------------------------------------------------------------- errors
def test_unknown_family():
    with pytest.raises(EpidemiologyValidationError, match="Unknown model family"):
        make_model("gru", {})


@pytest.mark.parametrize(("family", "params", "message"), [
    ("logistica", {**LOGISTIC, "tol": 1e-6}, "not allowed"),
    ("logistica", {"C": 1.0}, "lacks hyperparameters"),
    ("lightgbm", {**LIGHTGBM, "early_stopping_rounds": 10}, "early stopping"),
    ("lightgbm", {k: v for k, v in LIGHTGBM.items() if k != "random_state"}, "lacks hyperparameters"),
])
def test_hyperparameter_names_are_checked(family, params, message):
    with pytest.raises(EpidemiologyValidationError, match=message):
        make_model(family, params)


def _without(frame, column):
    return frame.drop(columns=[column])


@pytest.mark.parametrize(("alter", "message"), [
    (lambda f: _without(f, "anom_hum_m3"), "lacks feature columns"),
    (lambda f: f.assign(posicion_4s=np.r_[np.nan, f["posicion_4s"].to_numpy()[1:]]), "missing or infinite"),
    (lambda f: f.assign(zona_t=["brote"] + list(f["zona_t"].iloc[1:])), "outside"),
    (lambda f: f.assign(y=[2] + list(f["y"].iloc[1:])), "0 or 1"),
    (lambda f: f.assign(y=0), "single class"),
])
@pytest.mark.parametrize("family", FAMILIES)
def test_invalid_training_rows_fail_clearly(family, alter, message, train):
    with pytest.raises(EpidemiologyValidationError, match=message):
        fit_model(family, PARAMS[family], alter(train))


def test_non_convergence_is_an_error(train):
    with pytest.raises(EpidemiologyValidationError, match="did not converge"):
        fit_model("logistica", {"C": 1.0, "max_iter": 1}, train)


# --------------------------------------------------------------------------- feature subsets (F7, climate ablation)
NO_CLIMATE = tuple(name for name in FEATURES if not name.startswith("anom_"))


@pytest.mark.parametrize("family", FAMILIES)
def test_default_features_give_the_same_pipeline_bit_for_bit(family, train, evaluate):
    default = predict_positive(fit_model(family, PARAMS[family], train), evaluate)
    explicit = predict_positive(fit_model(family, PARAMS[family], train, features=FEATURES), evaluate)
    assert np.array_equal(default, explicit)
    assert [name for name, _ in make_model(family, PARAMS[family]).steps] == \
        [name for name, _ in make_model(family, PARAMS[family], features=FEATURES).steps]


@pytest.mark.parametrize("family", FAMILIES)
def test_subset_ignores_the_removed_columns(family, train, evaluate):
    model = fit_model(family, PARAMS[family], train, features=NO_CLIMATE)
    base = predict_positive(model, evaluate)
    noisy_train = train.copy()
    noisy_evaluate = evaluate.copy()
    rng = np.random.default_rng(11)
    for name in FEATURES:
        if name.startswith("anom_"):
            noisy_train[name] = rng.normal(size=len(train))
            noisy_evaluate[name] = rng.normal(size=len(evaluate))
    refit = fit_model(family, PARAMS[family], noisy_train, features=NO_CLIMATE)
    assert np.array_equal(predict_positive(refit, noisy_evaluate), base)
    assert not np.array_equal(predict_positive(fit_model(family, PARAMS[family], noisy_train), noisy_evaluate),
                              predict_positive(fit_model(family, PARAMS[family], train), evaluate))


def test_logistic_subset_has_one_coefficient_per_kept_feature(train):
    model = fit_model("logistica", LOGISTIC, train, features=NO_CLIMATE)
    assert model.named_steps["modelo"].coef_.shape == (1, len(NO_CLIMATE) - 1 + 2)   # numeric + two zone indicators


def test_feature_order_does_not_matter(train, evaluate):
    reversed_order = tuple(reversed(NO_CLIMATE))
    assert np.array_equal(predict_positive(fit_model("logistica", LOGISTIC, train, features=reversed_order), evaluate),
                          predict_positive(fit_model("logistica", LOGISTIC, train, features=NO_CLIMATE), evaluate))


@pytest.mark.parametrize("features, message", [
    (tuple(name for name in FEATURES if name != "zona_t"), "zona_t"),
    (("zona_t", "posicion_4s", "posicion_4s"), "repeated"),
    (("zona_t", "otra"), "Unknown features"),
    ("zona_t", "not a string")])
def test_invalid_feature_subsets_fail_clearly(features, message, train):
    with pytest.raises(EpidemiologyValidationError, match=message):
        fit_model("logistica", LOGISTIC, train, features=features)
