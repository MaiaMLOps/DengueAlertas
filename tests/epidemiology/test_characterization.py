from datetime import date
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.sources import sha256
from src.epidemiology.characterization import (
    build_tables, descriptive_statistics, label_counts, positive_runs, prepare_labels,
    run_summary, validate_label_frame, validate_run_coverage,
)
from src.epidemiology.characterization_plots import weekly_figure
from src.epidemiology.characterization_report import audit_numerics, export_characterization
from src.epidemiology.config import COUNT, EpidemiologyValidationError
from src.epidemiology.labels import label_weeks
from src.epidemiology.pipeline import export_labels
from src.pilot_selection.config import PILOTS


def short_frame(labels, dates=None, code='08001'):
    dates = pd.to_datetime(dates) if dates is not None else pd.date_range('2012-12-17', periods=len(labels), freq='W-MON')
    eligible = [value is not None for value in labels]
    return pd.DataFrame({'municipality_code': code, 'municipality': 'Synthetic ' + code,
                         'week_start': dates, 'excess_label': pd.array(labels, dtype='Int8'),
                         'label_eligible': eligible, COUNT: [2 if value == 1 else 0 for value in labels],
                         'ewt': [1. if flag else np.nan for flag in eligible]})


def test_nulls_never_enter_negative_counts_or_percentage_denominators():
    frame = short_frame([1, None, 0, 1, 0])
    result = label_counts(frame, []).iloc[0]
    assert result.total_weeks == 5 and result.eligible_weeks == 4 and result.ineligible_weeks == 1
    assert result.positive_weeks == 2 and result.negative_weeks == 2
    assert result.positive_pct == result.negative_pct == 50
    empty = label_counts(short_frame([None, None]), []).iloc[0]
    assert empty.negative_weeks == 0 and pd.isna(empty.negative_pct)


def test_runs_break_at_ineligible_and_negative_weeks():
    frame = short_frame([1, 1, None, 1, 0, 1, 1])
    result = positive_runs(frame)
    assert result.duration_weeks.tolist() == [2, 1, 2]
    assert result.right_boundary.tolist() == ['ineligible_week', 'negative_week', 'series_end']
    assert result.left_boundary.tolist() == ['series_start', 'ineligible_week', 'negative_week']
    validate_run_coverage(frame, result)


def test_calendar_gap_separates_positive_sequences_without_fabricating_weeks():
    frame = short_frame([1, 1, 1], ['2012-12-17', '2012-12-31', '2013-01-07'])
    result = positive_runs(frame)
    assert result.duration_weeks.tolist() == [1, 2]
    assert result.iloc[0].right_boundary == 'calendar_gap'
    assert result.iloc[1].left_boundary == 'calendar_gap'
    validate_run_coverage(frame, result)


def test_cross_year_positive_weeks_are_one_run_and_input_order_does_not_matter():
    frame = short_frame([0, 1, 1, 1, 0], ['2012-12-17', '2012-12-24', '2012-12-31', '2013-01-07', '2013-01-14'])
    result = positive_runs(frame.sample(frac=1, random_state=9))
    assert result.duration_weeks.tolist() == [3]
    assert result.iloc[0].start_week == pd.Timestamp('2012-12-24')
    assert result.iloc[0].end_week == pd.Timestamp('2013-01-07')


def test_independent_municipalities_and_zero_positive_run_summary():
    frame = pd.concat([short_frame([1, 1], code='08001'), short_frame([0, 0], code='08758')], ignore_index=True)
    runs = positive_runs(frame)
    summary = run_summary(frame, runs, ['municipality_code', 'municipality']).set_index('municipality_code')
    assert len(runs) == 1 and summary.loc['08001', 'positive_weeks'] == 2
    assert summary.loc['08758', 'n_runs'] == 0 and summary.loc['08758', 'positive_weeks'] == 0
    assert pd.isna(summary.loc['08758', 'duration_median'])
    validate_run_coverage(frame, runs)


def test_run_coverage_rejects_a_bridge_over_null():
    frame = short_frame([1, None, 1])
    invalid = positive_runs(frame).iloc[[0]].copy()
    invalid['end_week'] = frame.week_start.iloc[-1]
    invalid['duration_weeks'] = 3
    with pytest.raises(EpidemiologyValidationError):
        validate_run_coverage(frame, invalid)


@pytest.mark.parametrize('defect', ['negative_for_null', 'missing_label', 'duplicate', 'wrong_day'])
def test_invalid_descriptive_labels_are_not_repaired(defect):
    frame = short_frame([1, None, 0])
    if defect == 'negative_for_null':
        frame.loc[1, 'excess_label'] = 0
    elif defect == 'missing_label':
        frame.loc[0, 'excess_label'] = pd.NA
    elif defect == 'duplicate':
        frame = pd.concat([frame, frame.iloc[[0]]])
    else:
        frame.loc[0, 'week_start'] += pd.Timedelta(days=1)
    with pytest.raises(EpidemiologyValidationError):
        validate_label_frame(frame)


def test_statistics_and_annual_aggregates_use_eligible_weeks_only():
    frame = short_frame([1, None, 0, 1, 0])
    frame.loc[1, COUNT] = 999999
    stats = descriptive_statistics(frame, []).set_index('variable')
    assert stats.loc[COUNT, 'count'] == 4 and stats.loc[COUNT, 'mean'] == 1
    assert stats.loc[COUNT, 'std'] == pytest.approx(np.std([2, 0, 2, 0], ddof=1))
    tables, controls = build_tables(frame)
    assert controls['eligible_weeks'] == 4 and controls['positive_weeks'] == 2
    assert tables['labels_by_year'].eligible_weeks.sum() == 4
    assert tables['positive_runs'].duration_weeks.sum() == 2


def test_numeric_audit_rejects_changed_saved_labels_and_thresholds(make_panel):
    source = make_panel(counts=lambda code, start: start.year - 2000)
    product, _ = label_weeks(source)
    assert audit_numerics(source, product)['exact_integer_label_comparison'] == 'passed'
    rows = product.to_pylist()
    i = next(i for i, row in enumerate(rows) if row['label_eligible'])
    rows[i]['excess_label'] = 1 - rows[i]['excess_label']
    with pytest.raises(EpidemiologyValidationError, match='exact integer'):
        audit_numerics(source, pa.Table.from_pylist(rows, schema=product.schema))
    rows = product.to_pylist()
    rows[i]['ewt'] += .5
    with pytest.raises(EpidemiologyValidationError, match='60-digit'):
        audit_numerics(source, pa.Table.from_pylist(rows, schema=product.schema))


def test_weekly_plots_keep_real_dates_values_null_gaps_and_independent_axes(make_panel):
    source = make_panel(codes=tuple(p.code for p in PILOTS), end=date(2025, 12, 29))
    product, _ = label_weeks(source)
    frame = prepare_labels(product)
    for zoom in [False, True]:
        fig, checks = weekly_figure(frame, zoom)
        try:
            assert len(fig.axes) == 6 and all(row['plotted_ewt_positions'] == 730 for row in checks)
            assert all(row['plotted_count_positions'] == (730 if zoom else 991) for row in checks)
            for i, (ax, pilot) in enumerate(zip(fig.axes, PILOTS)):
                expected = frame.loc[frame.municipality_code.eq(pilot.code)]
                if zoom:
                    expected = expected.loc[expected.week_start.dt.year.ge(2012)]
                assert np.array_equal(ax.lines[0].get_xdata(), expected.week_start.to_numpy())
                assert np.array_equal(ax.lines[1].get_ydata(), expected.ewt.to_numpy(), equal_nan=True)
                assert ax.get_yscale() == 'linear' and np.isnan(ax.lines[0].get_ydata()[-1])
                assert all(not ax.get_shared_y_axes().joined(ax, prior) for prior in fig.axes[:i])
        finally:
            plt.close(fig)


def test_characterization_export_preserves_preliminary_and_refuses_overwrites(tmp_path, make_panel):
    source = tmp_path / 'data/processed/pilots.parquet'
    source.parent.mkdir(parents=True)
    pq.write_table(make_panel(codes=tuple(p.code for p in PILOTS), end=date(2025, 12, 29)), source)
    preliminary = export_labels(Path('data/processed/epidemiology/preliminary'), source, root=tmp_path)
    product = tmp_path / preliminary['output']['path']
    before = {p: sha256(p) for p in [source, product, product.with_name('run_manifest.json')]}
    output_dir = Path('reports/epidemiology/runs/synthetic')
    manifest = export_characterization(output_dir, product, root=tmp_path)
    directory = tmp_path / output_dir
    assert manifest['controls']['eligible_weeks'] == 4380
    assert manifest['controls']['positive_weeks'] == 0 and manifest['controls']['negative_weeks'] == 4380
    assert json.loads((directory / 'run_manifest.json').read_text(encoding='utf-8')) == manifest
    assert all(sha256(directory / name) == record['sha256'] for name, record in manifest['artifacts'].items())
    reread = pd.read_csv(directory / 'labels_by_municipality.csv', dtype={'municipality_code': str})
    assert '08001' in reread.municipality_code.to_list()
    with pytest.raises(FileExistsError):
        export_characterization(output_dir, product, root=tmp_path)
    assert all(sha256(p) == value for p, value in before.items())
