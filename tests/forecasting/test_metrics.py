"""Unit tests of the estimation metrics (U3.5), with synthetic rows in development years."""

from datetime import date, timedelta
import math

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss as sklearn_log_loss

from src.epidemiology.config import EpidemiologyValidationError
from src.forecasting.config import load_config
from src.forecasting.guard import FrozenRecipeError, RecipeAccess
from src.forecasting.metrics import average_precision, brier, improvement, log_loss

CHANNEL = "0123456789abcdef0123456789abcdef"
CLIP = 1e-15
TOLERANCE = {"rel": 1e-9, "abs": 1e-12}


@pytest.fixture
def access(tmp_path):
    return RecipeAccess(test_years=(2023, 2025), channel_version=CHANNEL, recipe_path=tmp_path / "missing.yaml")


def _weeks(n, year=2015):                                       # 200 weeks from 2015 stay before 2023
    return [date(year, 1, 6) + timedelta(weeks=k) for k in range(n)]


def test_configured_clip_is_the_approved_value():
    assert load_config().metricas.recorte_log_loss == 1e-15


# --------------------------------------------------------------------------- values by hand
def test_log_loss_and_brier_by_hand(access):
    y, p = [1, 0], [0.8, 0.3]
    assert log_loss(y, p, _weeks(2), access=access, clip=CLIP) == pytest.approx(
        (-math.log(0.8) - math.log(0.7)) / 2, **TOLERANCE)                       # 0.289909...
    assert brier(y, p, _weeks(2), access=access) == pytest.approx(0.065, **TOLERANCE)


def test_clip_bounds_confident_errors(access):
    assert log_loss([1], [0.0], _weeks(1), access=access, clip=CLIP) == pytest.approx(-math.log(1e-15), **TOLERANCE)
    # 1 − ε is rounded in float64, so 1 − (1 − ε) = 9.992e-16 and the cap is 34.5396..., not 34.5388...
    assert log_loss([0], [1.0], _weeks(1), access=access, clip=CLIP) == pytest.approx(-math.log(1 - (1 - 1e-15)), **TOLERANCE)
    assert log_loss([0], [1.0], _weeks(1), access=access, clip=CLIP) == pytest.approx(34.53957599234088, **TOLERANCE)
    perfect = log_loss([1], [1.0], _weeks(1), access=access, clip=CLIP)
    assert math.isfinite(perfect) and 0 < perfect < 1e-14


def test_same_clip_for_every_system(access):
    y = [1, 0, 0]
    confident, cautious = [0.0, 0.0, 0.0], [0.2, 0.2, 0.2]
    worst = log_loss(y, confident, _weeks(3), access=access, clip=CLIP)
    assert worst == pytest.approx(-math.log(1e-15) / 3 + 2 * -math.log(1 - 1e-15) / 3, **TOLERANCE)
    assert worst > log_loss(y, cautious, _weeks(3), access=access, clip=CLIP)


def test_matches_sklearn_on_interior_probabilities(access):
    rng = np.random.default_rng(20261111)
    p = rng.uniform(0.01, 0.99, 200)
    y = (rng.uniform(size=200) < p).astype(int)
    assert log_loss(y, p, _weeks(200), access=access, clip=CLIP) == pytest.approx(sklearn_log_loss(y, p), **TOLERANCE)
    assert brier(y, p, _weeks(200), access=access) == pytest.approx(brier_score_loss(y, p), **TOLERANCE)


def test_average_precision_by_hand_and_without_positives(access):
    assert average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], _weeks(4), access=access) == \
        pytest.approx((1 + 2 / 3) / 2, **TOLERANCE)
    assert math.isnan(average_precision([0, 0, 0], [0.1, 0.2, 0.3], _weeks(3), access=access))


def test_improvement():
    assert improvement(0.3, 0.5) == pytest.approx(0.4, **TOLERANCE)
    assert improvement(0.6, 0.5) == pytest.approx(-0.2, **TOLERANCE)
    assert improvement(np.float64(0.5), 0.5) == 0.0


@pytest.mark.parametrize("score, reference", [(0.3, 0.0), (0.3, -1.0), (math.nan, 0.5), (0.3, math.inf), (True, 0.5)])
def test_improvement_rejects_invalid_values(score, reference):
    with pytest.raises(EpidemiologyValidationError):
        improvement(score, reference)


# --------------------------------------------------------------------------- validation and barrier
def _call(metric, y, p, weeks, access):
    if metric == "log_loss":
        return log_loss(y, p, weeks, access=access, clip=CLIP)
    return {"brier": brier, "average_precision": average_precision}[metric](y, p, weeks, access=access)


METRICS = ("log_loss", "brier", "average_precision")


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("label, y, p, n_weeks", [
    ("y not binary", [1, 2], [0.5, 0.5], 2),
    ("y missing", [1.0, np.nan], [0.5, 0.5], 2),
    ("y boolean", [True, False], [0.5, 0.5], 2),
    ("p above one", [1, 0], [1.2, 0.5], 2),
    ("p negative", [1, 0], [-0.1, 0.5], 2),
    ("p missing", [1, 0], [np.nan, 0.5], 2),
    ("lengths differ", [1, 0], [0.5, 0.5], 3),
    ("empty", [], [], 0),
])
def test_invalid_rows_fail_clearly(access, metric, label, y, p, n_weeks):
    with pytest.raises(EpidemiologyValidationError):
        _call(metric, y, p, _weeks(n_weeks), access)


@pytest.mark.parametrize("clip", [0.0, 0.5, 1, "1e-15", None])
def test_log_loss_rejects_invalid_clip(access, clip):
    with pytest.raises(EpidemiologyValidationError, match="clip"):
        log_loss([1, 0], [0.5, 0.5], _weeks(2), access=access, clip=clip)


@pytest.mark.parametrize("metric", METRICS)
def test_barrier_runs_before_validation(access, metric):
    with pytest.raises(FrozenRecipeError):
        _call(metric, [7], [3.0], _weeks(2, year=2024), access)


@pytest.mark.parametrize("metric", METRICS)
def test_target_weeks_may_be_a_generator(access, metric):
    weeks = (week for week in _weeks(2))
    assert math.isfinite(_call(metric, [1, 0], [0.7, 0.2], weeks, access))
