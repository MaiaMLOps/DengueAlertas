"""Unit tests of yearly_trials with synthetic rows (U3.2; contrato §6.1 amended by D7.1).

Synthetic rows have t weekly and target_week = t + 8 weeks. Dates worked by hand:
- first Monday of 2013 is 2013-01-07, so t0 of trial 2013 is 2012-11-12 (target 2013-01-07);
- first Monday of 2014 is 2014-01-06, so t0 of trial 2014 is 2013-11-11.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from src.epidemiology.config import EpidemiologyValidationError
from src.forecasting.splits import Trial, yearly_trials

H = 8


def _rows(code, first_t, last_t, skip=()):
    rows, t = [], first_t
    while t <= last_t:
        if t not in skip:
            rows.append({"municipality_code": code, "t": t, "target_week": t + timedelta(weeks=H), "y": 0})
        t += timedelta(weeks=1)
    return rows


def _examples(*groups):
    return pd.DataFrame([row for group in groups for row in group])


A = _rows("08001", date(2011, 9, 5), date(2014, 10, 27))       # targets from 2011-10-31 to 2014-12-22
B_SKIP = (date(2013, 11, 11), date(2013, 11, 18), date(2013, 11, 25))
B = _rows("50001", date(2011, 9, 5), date(2014, 10, 27), skip=B_SKIP)


def _labels(examples, code, condition):
    own = examples[examples["municipality_code"] == code]
    return set(own.index[condition(own)])


def _one(trials, code, year):
    found = [trial for trial in trials if trial.municipality_code == code and trial.year == year]
    assert len(found) == 1
    return found[0]


def test_training_closes_at_first_emission():
    examples = _examples(A)
    trial = _one(yearly_trials(examples, [2013], first_train_year=2012), "08001", 2013)

    assert isinstance(trial, Trial)
    assert trial.t0 == date(2012, 11, 12)
    assert set(trial.evaluate) == _labels(examples, "08001", lambda g: g["target_week"].map(lambda w: w.year) == 2013)
    train = examples.loc[trial.train]
    assert min(train["target_week"]) == date(2012, 1, 2)       # targets of 2011 are left out
    assert max(train["target_week"]) == date(2012, 11, 12)
    assert max(train["t"]) == date(2012, 9, 17)
    assert len(train) == 46                                    # targets 2012-01-02 .. 2012-11-12


def test_gap_rows_enter_the_next_trial_only():
    examples = _examples(A)
    trials = yearly_trials(examples, [2013, 2014], first_train_year=2012)
    first, second = _one(trials, "08001", 2013), _one(trials, "08001", 2014)
    gap = _labels(examples, "08001", lambda g: (g["target_week"] > date(2012, 11, 12)) & (g["target_week"] <= date(2012, 12, 31)))

    assert len(gap) == 7                                       # targets 2012-11-19 .. 2012-12-31
    assert not gap & set(first.train) and not gap & set(first.evaluate)
    assert gap <= set(second.train)
    assert second.t0 == date(2013, 11, 11)


def test_t0_is_per_municipality():
    examples = _examples(A, B)
    trials = yearly_trials(examples, [2014], first_train_year=2012)
    a, b = _one(trials, "08001", 2014), _one(trials, "50001", 2014)

    assert a.t0 == date(2013, 11, 11)
    assert b.t0 == date(2013, 12, 2)                           # B lacks its first three emissions of 2014
    assert max(examples.loc[a.train, "target_week"]) == date(2013, 11, 11)
    assert max(examples.loc[b.train, "target_week"]) == date(2013, 12, 2)
    assert set(examples.loc[a.train, "municipality_code"]) == {"08001"}
    assert set(examples.loc[b.evaluate, "municipality_code"]) == {"50001"}
    assert len(b.evaluate) == len(a.evaluate) - 3


def test_sets_are_disjoint_and_order_is_deterministic():
    examples = _examples(B, A)                                 # input order does not set trial order
    trials = yearly_trials(examples, [2014, 2013], first_train_year=2012)

    assert [(t.municipality_code, t.year) for t in trials] == [("08001", 2014), ("08001", 2013),
                                                               ("50001", 2014), ("50001", 2013)]
    for trial in trials:
        assert not set(trial.train) & set(trial.evaluate)
        assert list(trial.train) == sorted(trial.train)       # labels follow the order of examples


def test_index_labels_are_preserved():
    examples = _examples(A)
    examples.index = [f"fila{i:04d}" for i in range(len(examples))]
    trial = yearly_trials(examples, [2013], first_train_year=2012)[0]
    assert all(label.startswith("fila") for label in trial.train)
    assert examples.loc[trial.evaluate, "target_week"].map(lambda w: w.year).eq(2013).all()


@pytest.mark.parametrize("label, make, years, first", [
    ("year without rows", lambda: _examples(A), [2016], 2012),
    ("no training rows", lambda: _examples(_rows("08001", date(2012, 11, 12), date(2013, 6, 3))), [2013], 2012),
    ("empty years", lambda: _examples(A), [], 2012),
    ("year not after first training year", lambda: _examples(A), [2012], 2012),
    ("repeated years", lambda: _examples(A), [2013, 2013], 2012),
    ("non-integer year", lambda: _examples(A), [2013.0], 2012),
    ("duplicate index", lambda: _examples(A).set_axis([0] * len(A)), [2013], 2012),
    ("duplicate origin", lambda: _examples(A, A[:1]), [2013], 2012),
    ("missing column", lambda: _examples(A).drop(columns="target_week"), [2013], 2012),
    ("timestamp instead of date", lambda: _examples(A).assign(t=lambda f: pd.to_datetime(f["t"])), [2013], 2012),
])
def test_invalid_inputs_fail_clearly(label, make, years, first):
    with pytest.raises(EpidemiologyValidationError):
        yearly_trials(make(), years, first_train_year=first)
