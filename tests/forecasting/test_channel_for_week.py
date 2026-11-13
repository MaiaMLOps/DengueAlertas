"""Unit tests of channel_for_week with synthetic counts (especificación §4; U2.2).

Reference weeks are written as literal dates (ordinal of the Monday within its year, D1).
"""

from datetime import date, datetime, timedelta

import pandas as pd
import pyarrow as pa
import pytest

from src.epidemiology.channel_epico import build_channel_table, channel_for_week, epico_channel
from src.epidemiology.config import FIRST_LABEL_YEAR, EpidemiologyValidationError
from src.epidemiology.validation import SOURCE_SCHEMA

PARAMS = {"n_years": 5, "shift": 1, "alpha": 0.025}


def synthetic_count(start: date) -> int:
    return (start.toordinal() * 7 + start.year) % 41


def weekly_counts(first: date, last: date) -> dict[date, int]:
    counts, current = {}, first
    while current <= last:
        counts[current] = synthetic_count(current)
        current += timedelta(weeks=1)
    return counts


def expected(reference_weeks, counts):
    central, _, upper = epico_channel([counts[week] for week in reference_weeks], **PARAMS)
    return central, upper


@pytest.fixture
def counts_2007_2013():
    return weekly_counts(date(2007, 1, 1), date(2013, 12, 30))


def test_same_ordinal_in_the_five_prior_years(counts_2007_2013):
    # 2013-03-04 is ordinal 9 of 2013.
    references = [date(2008, 3, 3), date(2009, 3, 2), date(2010, 3, 1), date(2011, 2, 28), date(2012, 2, 27)]

    assert channel_for_week(counts_2007_2013, date(2013, 3, 4), **PARAMS) == expected(references, counts_2007_2013)


def test_missing_reference_gives_none(counts_2007_2013):
    del counts_2007_2013[date(2010, 3, 1)]

    assert channel_for_week(counts_2007_2013, date(2013, 3, 4), **PARAMS) is None


def test_initialization_years_have_no_channel(counts_2007_2013):
    assert channel_for_week(counts_2007_2013, date(2011, 12, 26), **PARAMS) is None
    assert channel_for_week(counts_2007_2013, date(2012, 1, 2), **PARAMS) is not None


def test_week_53_falls_back_to_52_year_by_year(counts_2007_2013):
    # 2012-12-31 is ordinal 53; 2007 has its own ordinal 53 (2007-12-31), 2008-2011 do not.
    counts_2007_2013[date(2007, 12, 31)] = 97  # distinct from 2007-12-24 (ordinal 52)
    references = [date(2007, 12, 31), date(2008, 12, 29), date(2009, 12, 28), date(2010, 12, 27), date(2011, 12, 26)]

    assert channel_for_week(counts_2007_2013, date(2012, 12, 31), **PARAMS) == expected(references, counts_2007_2013)


def test_missing_week_53_is_not_replaced_by_52(counts_2007_2013):
    del counts_2007_2013[date(2007, 12, 31)]

    assert channel_for_week(counts_2007_2013, date(2012, 12, 31), **PARAMS) is None


def test_week_after_the_panel_needs_no_row():
    counts = weekly_counts(date(2019, 1, 7), date(2025, 12, 22))  # complete weeks only
    # 2026-02-16 is ordinal 7 of 2026.
    references = [date(2021, 2, 15), date(2022, 2, 14), date(2023, 2, 13), date(2024, 2, 12), date(2025, 2, 17)]

    assert date(2026, 2, 16) not in counts
    assert channel_for_week(counts, date(2026, 2, 16), **PARAMS) == expected(references, counts)


def test_excluded_partial_week_leaves_its_ordinal_without_history():
    counts = weekly_counts(date(2019, 1, 7), date(2025, 12, 22))  # 2025-12-29 (partial) excluded

    # 2026-12-28 is ordinal 52 of 2026; its 2025 reference would be the partial week.
    assert channel_for_week(counts, date(2026, 12, 28), **PARAMS) is None


def test_parameters_are_keyword_only_without_defaults(counts_2007_2013):
    with pytest.raises(TypeError):
        channel_for_week(counts_2007_2013, date(2013, 3, 4), 5, 1, 0.025)
    with pytest.raises(TypeError):
        channel_for_week(counts_2007_2013, date(2013, 3, 4), n_years=5, shift=1)


@pytest.mark.parametrize("change", [
    lambda c: {datetime(2013, 1, 7): 1, **c},
    lambda c: {pd.Timestamp(2013, 1, 7): 1, **c},
    lambda c: {date(2013, 1, 8): 1, **c},
    lambda c: list(c.items()),
], ids=["datetime_key", "timestamp_key", "not_monday_key", "not_mapping"])
def test_invalid_counts_fail_clearly(counts_2007_2013, change):
    with pytest.raises(EpidemiologyValidationError):
        channel_for_week(change(counts_2007_2013), date(2013, 3, 4), **PARAMS)


@pytest.mark.parametrize("week", [date(2013, 3, 5), datetime(2013, 3, 4)], ids=["not_monday", "datetime"])
def test_invalid_week_fails_clearly(counts_2007_2013, week):
    with pytest.raises(EpidemiologyValidationError):
        channel_for_week(counts_2007_2013, week, **PARAMS)


def test_invalid_parameters_and_counts_fail_clearly(counts_2007_2013):
    with pytest.raises(EpidemiologyValidationError):
        channel_for_week(counts_2007_2013, date(2013, 3, 4), n_years=4, shift=1, alpha=0.025)
    with pytest.raises(EpidemiologyValidationError):
        channel_for_week(counts_2007_2013, date(2013, 3, 4), n_years=5, shift=0, alpha=0.025)
    counts_2007_2013[date(2012, 2, 27)] = -1
    with pytest.raises(EpidemiologyValidationError):
        channel_for_week(counts_2007_2013, date(2013, 3, 4), **PARAMS)


def _panel(last: date) -> pa.Table:
    rows, current = [], date(2007, 1, 1)
    while current <= last:
        value = synthetic_count(current)
        nominal = current + timedelta(days=6)
        rows.append({"municipality_code": "08001", "municipality": "Synthetic", "week_start": current,
                     "population": 1000, "area_km2": 10.0, "population_density_km2": 100.0,
                     "temperature_c": 25.0, "precipitation_mm": 0.0, "relative_humidity_pct": 70.0,
                     "confirmed_records_screening": value, "no_eligible_records_observed": value == 0,
                     "week_end_nominal": nominal, "week_end_effective": nominal, "n_days": 7,
                     "partial_week": False})
        current += timedelta(weeks=1)
    return pa.Table.from_pylist(rows, schema=SOURCE_SCHEMA)


def test_reproduces_build_channel_table_exactly():
    panel = _panel(date(2013, 12, 30))
    product = build_channel_table(panel, first_year=FIRST_LABEL_YEAR, **PARAMS).to_pylist()
    counts = {row["week_start"]: row["confirmed_records_screening"] for row in product}

    eligible = 0
    for row in product:
        result = channel_for_week(counts, row["week_start"], **PARAMS)
        if row["channel_eligible"]:
            eligible += 1
            assert result == (row["epico_central"], row["epico_upper"]), row["week_start"]
        else:
            assert result is None, row["week_start"]
    assert eligible == 105  # 2012 (53 Mondays) and 2013 (52 Mondays)
