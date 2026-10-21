"""Opt-in real-panel audit using an independent calendar and Decimal formula."""

from collections import defaultdict
from decimal import Decimal, localcontext

import pyarrow.parquet as pq
import pytest

from src.data.sources import sha256
from src.epidemiology.config import COUNT
from src.epidemiology.labels import label_weeks


def test_local_pilots_against_independent_reference(pilot_path):
    before = sha256(pilot_path)
    source = pq.read_table(pilot_path)
    result, diagnostics = label_weeks(source)
    groups = defaultdict(list)
    for row in source.to_pylist():
        groups[(row['municipality_code'], row['week_start'].year)].append(row)
    # Enumerate sorted dates, rather than calling the production ordinal helper.
    lookup = {key: sorted(rows, key=lambda row: row['week_start']) for key, rows in groups.items()}
    eligible = fallbacks = 0
    with localcontext() as context:
        context.prec = 60
        epsilon = Decimal('0.001')
        for row in result.to_pylist():
            year = row['week_start'].year
            if year < 2012 or row['partial_week']:
                assert row['ewt'] is None and row['excess_label'] is None and not row['label_eligible']
                continue
            current_year = lookup[(row['municipality_code'], year)]
            ordinal = [r['week_start'] for r in current_year].index(row['week_start']) + 1
            references = []
            for historical_year in range(year - 5, year):
                weeks = lookup[(row['municipality_code'], historical_year)]
                position = ordinal
                if ordinal == 53 and len(weeks) == 52:
                    position = 52
                    fallbacks += 1
                reference = weeks[position - 1]
                assert reference['week_start'].year < year and not reference['partial_week']
                references.append(reference[COUNT])
            exact = (sum((Decimal(c) + epsilon).ln() for c in references) / 5).exp() - epsilon
            expected = float(exact)
            assert row['ewt'] == pytest.approx(expected, rel=3e-14, abs=1e-16)
            # Exact constant histories avoid Decimal's own exponent/log residue.
            expected_label = int(row[COUNT] > references[0]) if len(set(references)) == 1 else int(Decimal(row[COUNT]) > exact)
            assert row['excess_label'] == expected_label
            assert row['label_eligible']
            eligible += 1
    assert source.shape == (5952, 15) and result.shape == (5952, 18)
    assert eligible == 4380 and fallbacks == 84
    assert diagnostics['references']['historical_references'] == eligible * 5
    assert diagnostics['references']['missing_history'] == 0
    assert result.select(source.column_names).equals(source, check_metadata=False)
    assert sha256(pilot_path) == before
