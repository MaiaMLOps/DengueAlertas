from datetime import date

import pyarrow as pa
import pytest

from src.epidemiology.config import COUNT, EpidemiologyValidationError
from src.epidemiology.labels import label_weeks


@pytest.mark.parametrize("observed,expected", [(0, 0), (3, 0), (4, 0), (5, 1)])
def test_strict_comparison_including_exact_equality(make_panel, observed, expected):
    table = make_panel(counts=lambda code, start: 4 if start.year < 2012 else observed)
    result, _ = label_weeks(table)
    row = next(r for r in result.to_pylist() if r['week_start'] == date(2012, 1, 2))
    assert row['ewt'] == 4.
    assert row['excess_label'] == expected
    assert row['label_eligible']


@pytest.mark.parametrize("observed,expected", [(0, 0), (1, 1)])
def test_zero_threshold_does_not_create_false_positive(make_panel, observed, expected):
    source = make_panel(counts=lambda code, start: 0 if start.year < 2012 else observed)
    result, _ = label_weeks(source)
    row = next(r for r in result.to_pylist() if r['week_start'] == date(2012, 1, 2))
    assert row['ewt'] == 0.
    assert row['excess_label'] == expected


def test_no_threshold_rounding_or_minimum_floor(make_panel):
    source = make_panel(counts=lambda code, start: 0 if start.year < 2011 else 1)
    result, _ = label_weeks(source)
    row = next(r for r in result.to_pylist() if r['week_start'] == date(2012, 1, 2))
    assert 0 < row['ewt'] < 0.01
    assert row['excess_label'] == 1


def test_nonconstant_exact_equality_is_not_a_false_excess(make_panel):
    history = dict(zip(range(2007, 2012), [0, 0, 0, 1, 1004006004]))
    source = make_panel(counts=lambda code, start: history.get(start.year, 1))
    result, _ = label_weeks(source)
    row = next(r for r in result.to_pylist() if r['week_start'] == date(2012, 1, 2))
    assert row['ewt'] == 1.
    assert row['excess_label'] == 0


@pytest.mark.parametrize('difference,expected', [(-1, 1), (1, 0)])
def test_legitimate_comparisons_on_both_sides_of_equality(make_panel, difference, expected):
    history = dict(zip(range(2007, 2012), [0, 0, 0, 1, 1004006004 + difference]))
    source = make_panel(counts=lambda code, start: history.get(start.year, 1))
    result, _ = label_weeks(source)
    row = next(r for r in result.to_pylist() if r['week_start'] == date(2012, 1, 2))
    assert row['ewt'] != 1.
    assert row['excess_label'] == expected


def test_initialization_and_last_partial_are_retained_but_null(make_panel):
    source = make_panel(end=date(2025, 12, 29))
    result, diagnostics = label_weeks(source)
    for row in result.to_pylist():
        if row['week_start'].year < 2012 or row['partial_week']:
            assert not row['label_eligible']
            assert row['ewt'] is None and row['excess_label'] is None
        else:
            assert row['label_eligible'] and row['excess_label'] == 0
    assert diagnostics['validation']['eligible'] == 730
    assert diagnostics['validation']['ineligible'] == 262
    assert diagnostics['references']['historical_references'] == 3650
    assert diagnostics['references']['week_53_fallback_references'] == 14
    last = result.to_pylist()[-1]
    assert last['week_start'] == date(2025, 12, 29) and last['n_days'] == 3


def test_calculations_are_independent_between_municipalities(make_panel):
    source = make_panel(codes=('08001', '08758'), counts=lambda code, start: 2 if code == '08001' else 10000)
    before, _ = label_weeks(source)
    rows = source.to_pylist()
    for row in rows:
        if row['municipality_code'] == '08758':
            row[COUNT] = 0
            row['no_eligible_records_observed'] = True
    after, _ = label_weeks(pa.Table.from_pylist(rows, schema=source.schema))
    mask = pa.array([code == '08001' for code in source['municipality_code'].to_pylist()])
    assert before.filter(mask).equals(after.filter(mask), check_metadata=True)


def test_original_columns_values_types_and_unsorted_row_order_are_preserved(make_panel):
    source = make_panel(codes=('08758', '08001'))
    source = source.take(pa.array(list(reversed(range(source.num_rows)))))
    before = source.to_pylist()
    result, _ = label_weeks(source)
    assert result.column_names == source.column_names + ['ewt', 'excess_label', 'label_eligible']
    assert result.select(source.column_names).equals(source, check_metadata=False)
    assert all(a.equals(b, check_metadata=True) for a, b in zip(source.schema, result.schema))
    assert source.to_pylist() == before


@pytest.mark.parametrize('defect', ['gap', 'duplicate', 'null', 'negative', 'numeric_code', 'wrong_calendar', 'wrong_zero_flag'])
def test_source_incidents_are_rejected_without_repair(make_panel, defect):
    source = make_panel()
    rows = source.to_pylist()
    if defect == 'gap':
        rows.pop(200)
    elif defect == 'duplicate':
        rows.append(rows[0].copy())
    elif defect == 'null':
        rows[0][COUNT] = None
    elif defect == 'negative':
        rows[0][COUNT] = -1
    elif defect == 'wrong_calendar':
        rows[0]['n_days'] = 3
    elif defect == 'wrong_zero_flag':
        rows[0]['no_eligible_records_observed'] = True
    elif defect == 'numeric_code':
        source = source.set_column(0, 'municipality_code', source['municipality_code'].cast(pa.int64()))
    if defect != 'numeric_code':
        source = pa.Table.from_pylist(rows, schema=source.schema)
    with pytest.raises(EpidemiologyValidationError):
        label_weeks(source)
