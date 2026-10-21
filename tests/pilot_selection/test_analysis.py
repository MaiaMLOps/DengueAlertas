import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.pilot_selection.analysis import (
    annual_profiles, comparison_15, complete_weeks, distribution_summary,
    longest_zero_run, municipal_indicators, peak_year_distribution, period_frame, prepare_panel,
)
from src.pilot_selection.config import COUNT, INDICATORS


def test_cross_year_run_and_period_clipping(short_panel):
    frame = complete_weeks(prepare_panel(short_panel))
    metric = municipal_indicators(frame).iloc[0]
    assert metric.max_zero_run == 3
    assert metric.weekly_median == 0
    assert metric.active_weeks == 2
    assert metric.active_pct == pytest.approx(100 / 3)
    hist = municipal_indicators(period_frame(frame, (2007, 2022))).iloc[0]
    assert hist.max_zero_run == 2
    assert hist.total_records == 2
    assert hist.n_weeks == 3


@pytest.mark.parametrize("values,expected", [([], 0), ([1, 2], 0), ([0, 0], 2),
                                            ([0, 0, 2, 0, 0, 0], 3), ([1, 0, 2], 1)])
def test_zero_run_edges(values, expected):
    assert longest_zero_run(values) == expected


def test_partial_exclusion_and_annual_denominators(synthetic_master):
    source = prepare_panel(synthetic_master)
    complete = complete_weeks(source)
    assert source.partial_week.sum() == 18
    assert complete.groupby("municipality_code").size().eq(991).all()
    annual = annual_profiles(complete)
    assert annual.loc[annual.year.eq(2025), "n_weeks"].eq(51).all()
    assert annual.loc[annual.year.eq(2007), "n_weeks"].eq(53).all()
    np.testing.assert_allclose(annual.records_per_week * annual.n_weeks, annual.total_records)
    np.testing.assert_allclose(annual.active_proportion * annual.n_weeks, annual.active_weeks)


def test_historical_isolation(synthetic_master):
    baseline = complete_weeks(prepare_panel(synthetic_master))
    changed = baseline.copy()
    mask = changed.week_start.dt.year.ge(2023)
    changed.loc[mask, COUNT] = 999999
    before = municipal_indicators(period_frame(baseline, (2007, 2022)))
    after = municipal_indicators(period_frame(changed, (2007, 2022)))
    assert_frame_equal(before, after)


def test_annual_ties_zeros_and_quantiles(short_panel):
    frame = prepare_panel(short_panel)
    frame[COUNT] = [2, 0, 0, 0, 2, 0]
    frame["no_eligible_records_observed"] = frame[COUNT].eq(0)
    metrics = municipal_indicators(complete_weeks(frame))
    row = metrics.iloc[0]
    assert row.peak_year == 2022
    assert row.peak_years == "2022|2023"
    assert row.min_years == "2022|2023"
    assert row.peak_concentration_pct == 50
    frame[COUNT] = 0
    frame["no_eligible_records_observed"] = True
    zero = municipal_indicators(complete_weeks(frame)).iloc[0]
    assert pd.isna(zero.peak_year)
    assert pd.isna(zero.peak_concentration_pct)
    assert zero.max_zero_run == 6
    distribution = peak_year_distribution(pd.DataFrame([zero]), (2022, 2023))
    assert distribution.municipalities.sum() == 0
    artificial = pd.DataFrame({c: [0, 10, 20, 30] for c in INDICATORS})
    summary = distribution_summary(artificial)
    assert summary.p25.eq(7.5).all()
    assert summary.p75.eq(22.5).all()


def test_comparison_distance_ties_use_code():
    metrics = pd.DataFrame({"municipality_code": [f"01{i:03d}" for i in range(1, 21)],
                            "active_weeks": list(range(1, 21)), "n_weeks": 20})
    ordered = comparison_15(metrics)
    assert ordered.loc[ordered.comparison_group.eq("median"), "municipality_code"].tolist() == [
        "01010", "01011", "01009", "01012", "01008"]
    assert_frame_equal(ordered, comparison_15(metrics.sample(frac=1, random_state=19)))
    metrics["active_weeks"] = 1
    with pytest.raises(ValueError, match="overlap"):
        comparison_15(metrics)


@pytest.mark.parametrize("defect", ["duplicate", "missing_week", "numeric_code", "null", "negative",
                                    "float_count", "wrong_flag", "wrong_duration", "wrong_date", "non_monday"])
def test_invalid_panel_is_rejected(short_panel, defect):
    frame = short_panel.copy()
    if defect == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif defect == "missing_week":
        frame = frame.drop(index=2)
    elif defect == "numeric_code":
        frame["municipality_code"] = 8001
    elif defect == "null":
        frame.loc[0, COUNT] = np.nan
    elif defect == "negative":
        frame.loc[0, COUNT] = -1
    elif defect == "float_count":
        frame[COUNT] = frame[COUNT].astype(float)
    elif defect == "wrong_flag":
        frame.loc[0, "no_eligible_records_observed"] = True
    elif defect == "wrong_duration":
        frame.loc[0, "n_days"] = 3
    elif defect == "wrong_date":
        frame["week_start"] = frame.week_start.astype(object)
        frame.loc[0, "week_start"] = "not a date"
    else:
        frame.loc[0, "week_start"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        prepare_panel(frame)


def test_sorting_and_text_codes_are_preserved(short_panel):
    frame = prepare_panel(short_panel.sample(frac=1, random_state=11))
    assert frame.municipality_code.eq("08001").all()
    assert frame.week_start.is_monotonic_increasing
