"""Unit tests of B2 and B1 (U3.4) with synthetic rows; t = target_week - 8 weeks."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.epidemiology.config import EpidemiologyValidationError
from src.epidemiology.ewt import week_ordinal
from src.forecasting.baselines import b1_ordinal, b1_predict, b1_window, b2_predict, laplace
from src.forecasting.guard import FrozenRecipeError, RecipeAccess

H = 8
CHANNEL = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def access(tmp_path):
    return RecipeAccess(test_years=(2023, 2025), channel_version=CHANNEL, recipe_path=tmp_path / "missing.yaml")


def _monday(year, ordinal):
    first = date(year, 1, 1)
    return first + timedelta(days=(-first.weekday()) % 7) + timedelta(weeks=ordinal - 1)


def _frame(rows, index=None):
    return pd.DataFrame([(code, week - timedelta(weeks=H), week, y, zone) for code, week, y, zone in rows],
                        columns=["municipality_code", "t", "target_week", "y", "zona_t"], index=index)


def test_laplace():
    assert laplace(0, 0) == 0.5 and laplace(3, 7) == 4 / 9 and laplace(7, 7) == 8 / 9


# --------------------------------------------------------------------------- B2
def test_b2_three_municipalities_and_zones(access):
    train = _frame([("08001", _monday(2015, 10 + k), int(k < 2), zone)
                    for zone in ("seguridad", "alerta", "epidemia") for k in range(4)]
                   + [("41001", _monday(2016, 5), 1, "epidemia"), ("76001", _monday(2016, 5), 0, "seguridad")])
    week = _monday(2019, 30)
    evaluate = _frame([("76001", week, 0, "seguridad"), ("08001", week, 0, "epidemia"),
                       ("41001", week, 0, "epidemia"), ("41001", week, 0, "alerta")])
    assert b2_predict(train, evaluate, access=access).tolist() == [1 / 3, 3 / 6, 2 / 3, 1 / 2]


def test_output_follows_evaluate_order_with_any_index(access):
    train = _frame([("08001", _monday(2015, 10), 1, "alerta"), ("08001", _monday(2015, 11), 0, "seguridad")])
    week = _monday(2019, 30)
    evaluate = _frame([("08001", week, 0, "seguridad"), ("08001", week, 0, "alerta")], index=["z", "a"])
    assert b2_predict(train, evaluate, access=access).tolist() == [1 / 3, 2 / 3]


# --------------------------------------------------------------------------- B1
def test_b1_ordinal_and_window():
    assert b1_ordinal(date(2018, 12, 31)) == 52 and week_ordinal(date(2018, 12, 31)) == 53
    assert b1_ordinal(date(2019, 1, 7)) == 1
    assert b1_window(4, 2) == (2, 3, 4, 5, 6)
    assert b1_window(1, 2) == (51, 52, 1, 2, 3)
    assert b1_window(52, 3) == (49, 50, 51, 52, 1, 2, 3)
    assert b1_window(26, 25) == tuple(range(1, 52))


def test_b1_half_window_one_and_three(access):
    rows = [("08001", _monday(2013, ordinal), int(ordinal % 2 == 0), "alerta") for ordinal in range(45, 53)]
    rows += [("08001", date(2018, 12, 31), 1, "alerta")]                      # ordinal 53 counts as 52
    rows += [("08001", _monday(2014, ordinal), 1, "alerta") for ordinal in (1, 2, 3)]
    train = _frame(rows)
    evaluate = _frame([("08001", _monday(2020, 52), 0, "alerta")])
    # k = 1: ordinals 51, 52, 1 -> 51 (y 0), 52 (y 1) + 53 (y 1), 1 (y 1): 3 of 4
    assert b1_predict(train, evaluate, half_window=1, access=access)[0] == laplace(3, 4)
    # k = 3: 49..52 (y 0, 1, 0, 1), 53 (1), 1..3 (1, 1, 1): 6 of 8
    assert b1_predict(train, evaluate, half_window=3, access=access)[0] == laplace(6, 8)


@pytest.mark.parametrize("half_window", [0, 26, True, 2.0])
def test_b1_invalid_half_window(access, half_window):
    train = _frame([("08001", _monday(2015, 10), 1, "alerta")])
    with pytest.raises(EpidemiologyValidationError, match="half_window"):
        b1_predict(train, _frame([("08001", _monday(2019, 10), 0, "alerta")]), half_window=half_window, access=access)


# --------------------------------------------------------------------------- validation (both predictors)
def _call(predictor, train, evaluate, access):
    if predictor == "b1":
        return b1_predict(train, evaluate, half_window=2, access=access)
    return b2_predict(train, evaluate, access=access)


TRAIN = [("08001", _monday(2015, 10), 1, "alerta"), ("08001", _monday(2015, 11), 0, "alerta")]
EVAL = [("08001", _monday(2019, 10), 0, "alerta")]


@pytest.mark.parametrize("predictor", ["b1", "b2"])
@pytest.mark.parametrize("label, train, evaluate, message", [
    ("y not binary", _frame(TRAIN).assign(y=[1, 2]), _frame(EVAL), "train.y"),
    ("y missing", _frame(TRAIN).assign(y=[1.0, np.nan]), _frame(EVAL), "train.y"),
    ("unknown zone", _frame(TRAIN).assign(zona_t=["alerta", "brote"]), _frame(EVAL), "zona_t"),
    ("missing column", _frame(TRAIN).drop(columns="y"), _frame(EVAL), "train lacks"),
    ("evaluate without t", _frame(TRAIN), _frame(EVAL).drop(columns="t"), "evaluate lacks"),
    ("municipality without training", _frame(TRAIN), _frame(EVAL + [("50001", _monday(2019, 10), 0, "alerta")]),
     "without training rows"),
    ("training label after t0", _frame(TRAIN + [("08001", _monday(2019, 3), 1, "alerta")]), _frame(EVAL), "D7.1"),
])
def test_invalid_inputs_fail_clearly(access, predictor, label, train, evaluate, message):
    with pytest.raises(EpidemiologyValidationError, match=message):
        _call(predictor, train, evaluate, access)


@pytest.mark.parametrize("predictor", ["b1", "b2"])
def test_training_label_equal_to_t0_is_allowed(access, predictor):
    t0 = _monday(2019, 10) - timedelta(weeks=H)
    train = _frame(TRAIN + [("08001", t0, 1, "alerta")])
    assert _call(predictor, train, _frame(EVAL), access).shape == (1,)


@pytest.mark.parametrize("predictor", ["b1", "b2"])
def test_barrier_runs_before_validation(access, predictor):
    malformed_test_rows = _frame([("08001", _monday(2023, 10), 0, "brote")])
    with pytest.raises(FrozenRecipeError):
        _call(predictor, _frame(TRAIN).drop(columns="y"), malformed_test_rows, access)
