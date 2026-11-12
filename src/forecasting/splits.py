"""Yearly trials by the year of the target week (contrato §6.1 amended by D7.1, §10.11; especificación §4).

Flow, per municipality and evaluated year Y (rows come from `build_examples`; nothing is rebuilt):
    year of a row = calendar year of the Monday of its target week t + h
      -> evaluate: the municipality's rows whose target week falls in Y
      -> t0: the earliest emission t among them (the trial's first emission)
      -> train: the municipality's rows with target year >= first_train_year and t + h <= t0,
         the only labels that already existed at t0 (D7.1)

Rows emitted in t0 < t + h < 1 January of Y are in neither set of trial Y; they enter the training of
Y + 1. Trials only return index labels, in the order of `examples`, so every candidate, baseline and
ablation uses exactly the same rows (contrato §6.1, «Mismas condiciones para todos»). Building trials is
not scoring, so the test-period barrier (guard.py) is applied by whoever scores the evaluated rows.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from src.epidemiology.validation import require

_REQUIRED_COLUMNS = ("municipality_code", "t", "target_week")


@dataclass(frozen=True)
class Trial:
    """One municipality-year trial: index labels of `examples` for training and evaluation."""

    year: int
    municipality_code: str
    t0: date
    train: pd.Index
    evaluate: pd.Index


def _is_plain_date(value) -> bool:
    return isinstance(value, date) and not isinstance(value, datetime)


def _check_examples(examples: pd.DataFrame) -> None:
    require(isinstance(examples, pd.DataFrame), "examples must be a pandas DataFrame (rows of build_examples).")
    missing = [column for column in _REQUIRED_COLUMNS if column not in examples.columns]
    require(not missing, f"examples lacks columns {missing}.")
    require(examples.index.is_unique, "examples must have a unique index; trials return its labels.")
    require(not examples.duplicated(["municipality_code", "t"]).any(), "Duplicate (municipality_code, t) rows.")
    for column in ("t", "target_week"):
        bad = examples.loc[~examples[column].map(_is_plain_date), column]
        if not bad.empty:
            require(False, f"Column {column} must hold datetime.date values; found {bad.iloc[0]!r}.")


def _check_years(years: Iterable[int], first_train_year: int) -> list[int]:
    require(isinstance(first_train_year, int) and not isinstance(first_train_year, bool),
            f"first_train_year must be an integer; got {first_train_year!r}.")
    checked = list(years)
    require(checked, "years must not be empty.")
    for year in checked:
        require(isinstance(year, int) and not isinstance(year, bool), f"Evaluated years must be integers; got {year!r}.")
        require(year > first_train_year,
                f"Evaluated year {year} must be after the first training year {first_train_year}.")
    require(len(set(checked)) == len(checked), f"Repeated evaluated years: {checked}.")
    return checked


def yearly_trials(examples: pd.DataFrame, years: Iterable[int], *, first_train_year: int) -> list[Trial]:
    """One trial per (municipality, year), ordered by municipality and then by year (contrato §6.1, D7.1).

    Fails if a municipality has no evaluated rows in a year or no training rows for it: either would
    make the comparisons across municipalities and systems silently unpaired.
    """
    _check_examples(examples)
    checked_years = _check_years(years, first_train_year)
    target_year = examples["target_week"].map(lambda week: week.year)

    trials = []
    for code in sorted(examples["municipality_code"].unique()):
        own = examples["municipality_code"] == code
        for year in checked_years:
            evaluate = examples.index[own & (target_year == year)]
            require(len(evaluate) > 0, f"Municipality {code} has no rows with target week in {year}.")
            t0 = min(examples.loc[evaluate, "t"])
            train = examples.index[own & (target_year >= first_train_year) & (examples["target_week"] <= t0)]
            require(len(train) > 0,
                    f"Municipality {code}, trial {year}: no training rows with target year >= "
                    f"{first_train_year} and target week <= t0 = {t0}.")
            trials.append(Trial(year=year, municipality_code=code, t0=t0, train=train, evaluate=evaluate))
    return trials
