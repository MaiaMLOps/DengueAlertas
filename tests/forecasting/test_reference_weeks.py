"""Unit tests of reference_weeks, the single D1 alignment for the channel and clim_s (U2.4)."""

from datetime import date, datetime, timedelta

import pytest

from src.epidemiology.channel_epico import channel_for_week, epico_channel, reference_weeks
from src.epidemiology.config import EpidemiologyValidationError

PARAMS = {"n_years": 5, "shift": 1, "alpha": 0.025}


def mondays(first: date, last: date) -> set[date]:
    weeks, current = set(), first
    while current <= last:
        weeks.add(current)
        current += timedelta(weeks=1)
    return weeks


@pytest.fixture
def weeks():
    return mondays(date(2007, 1, 1), date(2013, 12, 30))


def test_same_ordinal_in_the_five_prior_years_in_year_order(weeks):
    # 2013-03-04 is ordinal 9 of 2013.
    assert reference_weeks(weeks, date(2013, 3, 4), n_years=5) == (
        date(2008, 3, 3), date(2009, 3, 2), date(2010, 3, 1), date(2011, 2, 28), date(2012, 2, 27))


def test_week_53_falls_back_to_52_only_in_years_without_it(weeks):
    assert reference_weeks(weeks, date(2012, 12, 31), n_years=5) == (
        date(2007, 12, 31), date(2008, 12, 29), date(2009, 12, 28), date(2010, 12, 27), date(2011, 12, 26))


def test_existing_week_53_missing_from_the_set_gives_none(weeks):
    weeks.discard(date(2007, 12, 31))

    assert reference_weeks(weeks, date(2012, 12, 31), n_years=5) is None


def test_excluded_partial_week_gives_none():
    complete = mondays(date(2019, 1, 7), date(2025, 12, 22))  # 2025-12-29 (partial) excluded

    assert reference_weeks(complete, date(2026, 12, 28), n_years=5) is None
    assert reference_weeks(complete, date(2026, 2, 16), n_years=5) == (
        date(2021, 2, 15), date(2022, 2, 14), date(2023, 2, 13), date(2024, 2, 12), date(2025, 2, 17))


def test_insufficient_history_gives_none(weeks):
    assert reference_weeks(weeks, date(2011, 12, 26), n_years=5) is None


@pytest.mark.parametrize("bad", [datetime(2013, 1, 7), date(2013, 1, 8)], ids=["datetime", "not_monday"])
def test_invalid_weeks_fail_clearly(weeks, bad):
    with pytest.raises(EpidemiologyValidationError):
        reference_weeks(weeks | {bad}, date(2013, 3, 4), n_years=5)


def test_invalid_week_start_and_years_fail_clearly(weeks):
    with pytest.raises(EpidemiologyValidationError):
        reference_weeks(weeks, date(2013, 3, 5), n_years=5)
    with pytest.raises(EpidemiologyValidationError):
        reference_weeks(weeks, date(2013, 3, 4), n_years=4)
    with pytest.raises(TypeError):
        reference_weeks(weeks, date(2013, 3, 4), 5)


def test_channel_for_week_uses_these_references(weeks):
    counts = {week: (week.toordinal() * 7) % 37 for week in weeks}
    references = reference_weeks(counts, date(2012, 12, 31), n_years=5)
    central, _, upper = epico_channel([counts[week] for week in references], **PARAMS)

    assert channel_for_week(counts, date(2012, 12, 31), **PARAMS) == (central, upper)
