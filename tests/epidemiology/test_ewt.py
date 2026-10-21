from datetime import date
from decimal import Decimal, localcontext

import pyarrow as pa
import pytest

from src.epidemiology.config import COUNT, EpidemiologyValidationError
from src.epidemiology.ewt import (
    calculate_ewt, historical_references, history_index, shifted_geometric_mean, week_ordinal,
)


def decimal_reference(values):
    with localcontext() as context:
        context.prec = 60
        epsilon = Decimal("0.001")
        return float((sum((Decimal(value) + epsilon).ln() for value in values) / 5).exp() - epsilon)


@pytest.mark.parametrize("values", [[1, 2, 4, 8, 16], [0, 0, 1, 20, 400], [0, 0, 0, 0, 1],
                                   [100, 200, 300, 400, 1000000]])
def test_shifted_geometric_mean_against_high_precision_reference(values):
    assert shifted_geometric_mean(values) == pytest.approx(decimal_reference(values), rel=3e-14, abs=1e-16)


@pytest.mark.parametrize("value", [0, 1, 2, 13, 1000])
def test_constant_histories_are_exact_without_rounding(value):
    assert shifted_geometric_mean([value] * 5) == value


def test_mixed_zero_history_keeps_small_positive_threshold():
    assert 0 < shifted_geometric_mean([0, 0, 0, 0, 1]) < 0.01


@pytest.mark.parametrize('values', [[0, 0, 0, 1, 1004006004], [0, 0, 0, 0, 1005010010005]])
def test_nonconstant_exact_integer_mean_has_no_float_equality_residue(values):
    # product(1000*C + 1) == 1001**5, so the exact shifted mean is one.
    assert shifted_geometric_mean(values) == 1.


@pytest.mark.parametrize('difference', [-1, 1])
def test_exact_identity_does_not_round_legitimate_neighboring_thresholds(difference):
    values = [0, 0, 0, 1, 1004006004 + difference]
    threshold = shifted_geometric_mean(values)
    assert threshold != 1.
    assert (threshold < 1.) == (difference < 0)
    assert threshold == pytest.approx(decimal_reference(values), rel=3e-14)


@pytest.mark.parametrize("values", [[], [1] * 4, [1] * 6, [-1, 0, 0, 0, 0], [True] * 5, [1.] * 5])
def test_invalid_histories_stop(values):
    with pytest.raises(EpidemiologyValidationError):
        shifted_geometric_mean(values)


@pytest.mark.parametrize("start,ordinal", [(date(2007, 1, 1), 1), (date(2007, 12, 31), 53),
                                           (date(2008, 1, 7), 1), (date(2012, 12, 31), 53),
                                           (date(2025, 12, 29), 52)])
def test_monday_ordinals_follow_monday_year_not_iso(start, ordinal):
    assert week_ordinal(start) == ordinal


def test_non_monday_cannot_receive_ordinal():
    with pytest.raises(EpidemiologyValidationError):
        week_ordinal(date(2008, 1, 1))


def test_exact_five_years_epidemic_counts_and_53_fallback(make_panel):
    table = make_panel(counts=lambda code, start: (start.year - 2000) * 100 + week_ordinal(start))
    index = history_index(table)
    refs = historical_references(index, "08001", 2012, 53)
    assert [r.week_start.year for r in refs] == [2007, 2008, 2009, 2010, 2011]
    assert [r.ordinal for r in refs] == [53, 52, 52, 52, 52]
    assert [r.week_start for r in refs] == [date(2007, 12, 31), date(2008, 12, 29),
                                           date(2009, 12, 28), date(2010, 12, 27), date(2011, 12, 26)]
    thresholds, audit = calculate_ewt(table)
    i = table['week_start'].to_pylist().index(date(2012, 12, 31))
    assert thresholds[i] == pytest.approx(decimal_reference([753, 852, 952, 1052, 1152]))
    assert audit['week_53_fallback_references'] == 4
    later = historical_references(index, "08001", 2013, 1)
    assert [r.week_start.year for r in later] == [2008, 2009, 2010, 2011, 2012]


def test_missing_regular_week_never_substitutes_a_neighbor(make_panel):
    index = history_index(make_panel())
    del index[("08001", 2009, 10)]
    assert historical_references(index, "08001", 2012, 10) == ()
    assert historical_references(index, "08758", 2012, 10) == ()


def test_temporal_causality_changing_target_and_future_years(make_panel):
    source = make_panel(counts=lambda code, start: start.year - 2000)
    before, _ = calculate_ewt(source)
    rows = source.to_pylist()
    for row in rows:
        if row['week_start'].year >= 2012:
            row[COUNT] = 999999
            row['no_eligible_records_observed'] = False
    after, _ = calculate_ewt(pa.Table.from_pylist(rows, schema=source.schema))
    for i, start in enumerate(source['week_start'].to_pylist()):
        if start.year == 2012:
            assert before[i] == after[i]
    first_2013 = source['week_start'].to_pylist().index(date(2013, 1, 7))
    assert before[first_2013] != after[first_2013]  # A changed prior year must matter.
