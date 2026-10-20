import pandas as pd
import pytest

from src.data.normalization import municipality_codes, normalize_component, parse_source_dates, week_start


@pytest.mark.parametrize("department,municipality,expected", [
    ("5", "1", "05001"), (" 05 ", " 001 ", "05001"), ("08", "078", "08078"), ("76", "001", "76001"),
])
def test_leading_zeros(department, municipality, expected):
    assert municipality_codes(pd.Series([department]), pd.Series([municipality])).iloc[0] == expected


@pytest.mark.parametrize("value", [None, "", "5.0", "005", "AB"])
def test_ambiguous_or_invalid_components_rejected(value):
    with pytest.raises(ValueError, match="Invalid territorial"):
        normalize_component(pd.Series([value]), 2)


def test_explicit_day_first_meridiem_and_missing_dates():
    parsed, qc = parse_source_dates(pd.Series([
        "02/03/2007 12:00:00 a.\u00a0m.", "02/03/2007 12:00:00 p. m.",
        "02/03/2007 01:30:00 p.\u202fm.", "", "not a date", "31/02/2007 12:00:00 a. m.",
    ]))
    assert parsed.iloc[0] == pd.Timestamp("2007-03-02 00:00:00")
    assert parsed.iloc[1] == pd.Timestamp("2007-03-02 12:00:00")
    assert parsed.iloc[2] == pd.Timestamp("2007-03-02 13:30:00")
    assert qc == {"blank": 1, "parse_fail_nonblank": 2}


def test_sunday_monday_and_cross_year_are_symptom_based():
    dates = pd.Series(pd.to_datetime(["2007-12-30 23:59:59", "2007-12-31", "2008-01-01", "2025-12-31"], format="ISO8601"))
    assert week_start(dates).tolist() == list(pd.to_datetime(["2007-12-24", "2007-12-31", "2007-12-31", "2025-12-29"]))
