"""Observed references for the current dataset; not production calculation constants."""

from itertools import groupby

import pandas as pd
import pytest

from src.pilot_selection.analysis import build_tables, complete_weeks, period_frame, prepare_panel
from src.pilot_selection.config import COUNT, PERIODS, PILOTS
from src.pilot_selection.reporting import sha256


@pytest.fixture(scope="module")
def observed(master_path):
    before = sha256(master_path)
    source = prepare_panel(pd.read_parquet(master_path))
    complete = complete_weeks(source)
    historical = period_frame(complete, PERIODS["historical"])
    tables = build_tables(complete)
    yield source, complete, historical, tables
    assert sha256(master_path) == before


def test_known_panel_totals(observed):
    source, complete, historical, tables = observed
    assert source.shape == (77376, 15)
    assert source.municipality_code.nunique() == 78
    assert complete.groupby("municipality_code").size().eq(991).all()
    assert len(complete) == 77298
    assert complete[COUNT].sum() == 636865
    assert historical.groupby("municipality_code").size().eq(835).all()
    assert len(historical) == 65130
    assert historical[COUNT].sum() == 384008
    assert len(tables["annual_profiles"]) == 1482
    peaks = tables["peak_year_distribution"]
    assert peaks.loc[peaks.period.eq("full") & peaks.year.eq(2024), "municipalities"].item() == 34
    assert source.partial_week.sum() == 78
    assert source.loc[source.partial_week, COUNT].sum() == 406


def test_comparison_and_top_ten_codes(observed):
    tables = observed[3]
    assert tables["comparison_15"].municipality_code.tolist() == [
        "54001", "08001", "41001", "50001", "76001", "25875", "76111", "15572",
        "44430", "05088", "05579", "08296", "05360", "41396", "19698"]
    assert tables["sensitivity_top10"].municipality_code.tolist() == [
        "54001", "08001", "41001", "50001", "76001", "68001", "08758", "05001", "20001", "73001"]


def test_known_between_municipality_quantiles(observed):
    summary = observed[3]["distribution_summary"].query("period == 'full'").set_index("indicator")
    expected = {
        "total_records": [1035, 2443, 3768.5, 7269.5, 93894],
        "active_weeks": [447, 544.25, 639, 799.25, 975],
        "annual_active_median": [18, 28, 35.5, 47, 52],
        "annual_active_min": [1, 3, 6, 12.75, 44],
        "max_zero_run": [2, 13.5, 25, 37, 52],
        "peak_concentration_pct": [12.479910787497, 17.464698201756, 22.291054166102,
                                   34.382967581711, 59.12925170068],
    }
    for indicator, quantiles in expected.items():
        assert summary.loc[indicator, ["min", "p25", "median", "p75", "max"]].tolist() == pytest.approx(quantiles)


def test_selected_pilot_regressions(observed):
    _, complete, _, tables = observed
    selected = complete.loc[complete.municipality_code.isin([pilot.code for pilot in PILOTS])]
    assert len(selected) == 5946
    assert selected[COUNT].sum() == 231281
    full = tables["selected_pilots"].query("period == 'full'")
    hist = tables["selected_pilots"].query("period == 'historical'")
    assert full.municipality_code.tolist() == [pilot.code for pilot in PILOTS]
    assert full.total_records.tolist() == [30489, 39758, 27717, 25684, 93894, 13739]
    assert hist.total_records.tolist() == [21928, 28059, 17699, 16294, 55635, 9309]
    assert hist.active_weeks.tolist() == [819, 818, 816, 808, 807, 780]
    assert full.max_zero_run.tolist() == [2, 3, 2, 3, 9, 5]


def test_all_zero_runs_against_independent_block_reference(observed):
    _, complete, historical, tables = observed
    metrics = tables["municipal_indicators"].set_index(["period", "municipality_code"])
    for label, frame in (("full", complete), ("historical", historical)):
        for code, group in frame.groupby("municipality_code"):
            longest = max((sum(1 for _ in block) for zero, block in groupby(group[COUNT].eq(0)) if zero), default=0)
            assert metrics.loc[(label, code), "max_zero_run"] == longest
