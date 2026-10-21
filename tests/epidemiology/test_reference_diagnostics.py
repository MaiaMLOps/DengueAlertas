from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.epidemiology.characterization import positive_runs
from src.epidemiology.config import COUNT
from src.epidemiology.ewt import week_ordinal
from src.epidemiology.labels import label_weeks
from src.epidemiology.reference_diagnostic_plots import zero_figure
from src.epidemiology.reference_diagnostic_report import export_diagnostics
from src.epidemiology.reference_diagnostics import (
    build_diagnostics, combination_summary, long_run_details, municipal_quartiles,
    quartile_grid, reference_frame, zero_summary,
)


def test_zero_counts_follow_exact_five_references_including_53_fallback(make_panel):
    def counts(code, start):
        if start.year >= 2012:
            return 2
        n = (week_ordinal(start) - 1) % 6
        return 0 if start.year - 2007 < n else 16
    source = make_panel(end=date(2012, 12, 31), counts=counts)
    original = source.to_pylist()
    product, _ = label_weeks(source)
    frame, audit = reference_frame(source, product)
    assert len(frame) == 53 and frame.label_eligible.all()
    assert frame.historical_zero_count.iloc[:6].tolist() == [0, 1, 2, 3, 4, 5]
    assert frame.historical_zero_count.iloc[-1] == 3
    assert audit['historical_references_checked'] == 265
    assert audit['week_53_fallback_references'] == 4
    assert source.to_pylist() == original
    assert product.select(source.column_names).equals(source, check_metadata=False)
    tables, _, controls = build_diagnostics(source, product)
    assert controls['long_runs'] == 0
    assert tables['zeros_global'].eligible_weeks.sum() == 53
    assert tables['zeros_global'].historical_zero_count.tolist() == list(range(6))
    five = tables['zeros_global'].iloc[5]
    assert five.ewt_median == 0 and five.positive_pct_in_group == 100


def simple_frame():
    return pd.DataFrame({'municipality_code': ['08001'] * 8, 'municipality': ['Synthetic'] * 8,
                         'week_start': pd.date_range('2012-01-02', periods=8, freq='W-MON'),
                         'label_eligible': True, 'excess_label': pd.array([0, 1, 0, 1, 1, 1, 1, 1], dtype='Int8'),
                         'ewt': [0., 1., 1., 1., 2., 4., 9., 100.], COUNT: [0, 1, 1, 1, 2, 4, 9, 100],
                         'historical_zero_count': [0, 0, 1, 1, 2, 3, 4, 5]})


def test_empty_zero_groups_are_explicit_and_not_reported_as_zero_thresholds():
    frame = simple_frame()
    frame['historical_zero_count'] = 0
    table = zero_summary(frame)
    assert len(table) == 6 and table.iloc[0].eligible_weeks == 8
    assert table.iloc[1].eligible_weeks == 0 and table.iloc[1].positive_weeks == 0
    assert pd.isna(table.iloc[1].ewt_median) and pd.isna(table.iloc[1].positive_pct_in_group)


def test_quartiles_keep_ties_together_and_use_municipal_scales():
    first = simple_frame()
    second = first.copy()
    second['municipality_code'], second['municipality'] = '08758', 'Other'
    second[['ewt', COUNT]] *= 100
    frame = pd.concat([first, second], ignore_index=True)
    enriched, cutoffs = municipal_quartiles(frame)
    assert enriched.loc[:7, 'ewt_quartile'].tolist() == enriched.loc[8:, 'ewt_quartile'].tolist()
    assert enriched.loc[enriched.ewt.eq(1), 'ewt_quartile'].tolist() == [1, 1, 1]
    assert cutoffs.iloc[1].observed_p75 == 100 * cutoffs.iloc[0].observed_p75
    assert quartile_grid(enriched).eligible_weeks.sum() == 16
    assert quartile_grid(enriched, True).positive_weeks.sum() == frame.excess_label.sum()
    combos = combination_summary(enriched)
    total = combos.loc[combos.municipality_code.eq('ALL')]
    assert total.eligible_weeks.sum() == 16 and total.positive_weeks.sum() == frame.excess_label.sum()


def test_long_run_statistics_are_exact_and_noneligible_row_cuts_sequences():
    frame = pd.concat([simple_frame().iloc[[0]]] * 60, ignore_index=True)
    frame['week_start'] = pd.date_range('2012-01-02', periods=60, freq='W-MON')
    frame['excess_label'] = pd.array([1] * 60, dtype='Int8')
    frame['ewt'], frame[COUNT] = 5., 50
    frame['historical_zero_count'] = [1] * 10 + [0] * 50
    enriched, cutoffs = municipal_quartiles(frame)
    details, annual = long_run_details(enriched, positive_runs(frame), cutoffs)
    row = details.iloc[0]
    assert row.duration_weeks == 60 and row.ewt_median == 5 and row.observed_median == 50
    assert row.weeks_with_historical_zero == row.historical_zero_references == 10
    assert row.pct_weeks_with_historical_zero == pytest.approx(100 / 6)
    assert row.pct_ewt_le_municipal_p25 == 100 and row.pct_observed_gt_municipal_p75 == 0
    assert annual.weeks.sum() == 60
    frame.loc[30, 'label_eligible'] = False
    frame.loc[30, 'excess_label'] = pd.NA
    assert positive_runs(frame).duration_weeks.tolist() == [30, 29]


def test_zero_plot_shows_observed_groups_only_and_handles_exact_zero():
    fig = zero_figure(simple_frame())
    try:
        assert fig.axes[0].get_yscale() == 'linear'
        assert len(fig.axes[0].get_xticklabels()) == 6
    finally:
        plt.close(fig)


def test_export_refuses_existing_directory_before_reading_inputs(tmp_path):
    destination = tmp_path / 'reports/epidemiology/runs/existing'
    destination.mkdir(parents=True)
    marker = destination / 'keep.txt'
    marker.write_text('keep', encoding='utf-8')
    with pytest.raises(FileExistsError):
        export_diagnostics(Path('reports/epidemiology/runs/existing'), root=tmp_path)
    assert marker.read_text(encoding='utf-8') == 'keep'
