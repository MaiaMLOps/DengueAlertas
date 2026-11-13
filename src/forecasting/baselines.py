"""Baselines B2 (channel state) and B1 (seasonal climatology) (contrato §4.1, §6.1 amended by D7.1, §10.8).

Both are empirical frequencies per municipality with Laplace smoothing, (positives + 1)/(cases + 2),
estimated only with the training rows of a trial (`splits.yearly_trials`):
    B2: P(y = 1 | municipality, zona_t), the zone observed at the emission t
    B1: P(y = 1 | municipality, ordinal of t + h in a circular window of ±half_window weeks)

Flow of each predictor:
    check_test_access(target weeks of `evaluate`)   (test-period barrier, before anything else)
      -> validate columns, y in {0, 1}, zones, municipalities with training, no training label after t0
      -> integer counts (cases, positives) per group in `train`
      -> laplace(...) for each evaluated row, in the order of `evaluate`

B1 window edges (approved by the person responsible for the implementation in U3.1): ordinal 53 counts as 52, both for training rows and for the
evaluated row, and the window is circular over 52 weeks, {((o - 1 + d) mod 52) + 1 : d = -k..k}.
The descriptive channel alert (§4.1, third row) belongs to the events module (F4).
"""

import numpy as np
import pandas as pd

from src.epidemiology.channel_epico import ZONES
from src.epidemiology.ewt import week_ordinal
from src.epidemiology.validation import require

from .guard import RecipeAccess, check_test_access

# Length of the B1 cycle: part of the approved edge rule (53 counts as 52), not a tunable parameter.
B1_CYCLE_WEEKS = 52
_MAX_HALF_WINDOW = (B1_CYCLE_WEEKS - 1) // 2   # 25: a wider window would count weeks twice
_TRAIN_COLUMNS = ("municipality_code", "target_week", "zona_t", "y")
_EVALUATE_COLUMNS = ("municipality_code", "t", "target_week", "zona_t")


def laplace(positives: int, cases: int) -> float:
    """(positives + 1)/(cases + 2) (contrato §4.1)."""
    return (positives + 1) / (cases + 2)


def _check_frame(frame, columns, name: str) -> None:
    require(isinstance(frame, pd.DataFrame), f"{name} must be a pandas DataFrame.")
    missing = [column for column in columns if column not in frame.columns]
    require(not missing, f"{name} lacks columns {missing}.")


def _check_inputs(train: pd.DataFrame, evaluate: pd.DataFrame) -> None:
    _check_frame(train, _TRAIN_COLUMNS, "train")
    _check_frame(evaluate, _EVALUATE_COLUMNS, "evaluate")
    y = train["y"]
    require(not y.isna().any() and y.isin([0, 1]).all(), "train.y must be 0 or 1 without missing values.")
    for name, frame in (("train", train), ("evaluate", evaluate)):
        unknown = sorted(set(frame["zona_t"]) - set(ZONES))
        require(not unknown, f"{name}.zona_t has values outside {ZONES}: {unknown}.")

    trained = set(train["municipality_code"])
    untrained = sorted(set(evaluate["municipality_code"]) - trained)
    require(not untrained, f"Evaluated municipalities without training rows: {untrained}.")

    first_emission = evaluate.groupby("municipality_code")["t"].min()
    last_label = train.groupby("municipality_code")["target_week"].max()
    for code, t0 in first_emission.items():
        require(last_label[code] <= t0,
                f"Municipality {code}: a training label (target week {last_label[code]}) is later than the first "
                f"evaluated emission t0 = {t0}; training must only use t + h <= t0 (contrato §6.1, D7.1).")


def b2_predict(train: pd.DataFrame, evaluate: pd.DataFrame, *, access: RecipeAccess) -> np.ndarray:
    """B2: Laplace frequency of y per (municipality, zona_t) over the trial's training rows."""
    check_test_access(evaluate["target_week"], access)
    _check_inputs(train, evaluate)
    counts = train.groupby(["municipality_code", "zona_t"])["y"].agg(["size", "sum"])
    cases, positives = counts["size"].to_dict(), counts["sum"].to_dict()
    keys = zip(evaluate["municipality_code"], evaluate["zona_t"])
    return np.array([laplace(int(positives.get(key, 0)), int(cases.get(key, 0))) for key in keys], dtype=float)


def b1_ordinal(week) -> int:
    """Ordinal of a target week for B1: Mondays of its year from one, with 53 counted as 52."""
    return min(week_ordinal(week), B1_CYCLE_WEEKS)


def b1_window(ordinal: int, half_window: int) -> tuple[int, ...]:
    """Circular window of ordinals {((o - 1 + d) mod 52) + 1 : d = -k..k}."""
    return tuple(((ordinal - 1 + d) % B1_CYCLE_WEEKS) + 1 for d in range(-half_window, half_window + 1))


def _check_half_window(half_window) -> None:
    require(isinstance(half_window, int) and not isinstance(half_window, bool)
            and 1 <= half_window <= _MAX_HALF_WINDOW,
            f"half_window must be an integer from 1 to {_MAX_HALF_WINDOW}; got {half_window!r}.")


def b1_predict(train: pd.DataFrame, evaluate: pd.DataFrame, *, half_window: int, access: RecipeAccess) -> np.ndarray:
    """B1: Laplace frequency of y per municipality over training rows whose target ordinal is in the window."""
    check_test_access(evaluate["target_week"], access)
    _check_half_window(half_window)
    _check_inputs(train, evaluate)

    cases, positives = {}, {}
    for code, week, y in zip(train["municipality_code"], train["target_week"], train["y"]):
        key = (code, b1_ordinal(week))
        cases[key] = cases.get(key, 0) + 1
        positives[key] = positives.get(key, 0) + int(y)

    predictions = []
    for code, week in zip(evaluate["municipality_code"], evaluate["target_week"]):
        window = b1_window(b1_ordinal(week), half_window)
        predictions.append(laplace(sum(positives.get((code, o), 0) for o in window),
                                   sum(cases.get((code, o), 0) for o in window)))
    return np.array(predictions, dtype=float)
