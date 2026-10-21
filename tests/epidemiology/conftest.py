"""Synthetic full-calendar panels; local data require an explicit pytest option."""

from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

from src.epidemiology.validation import SOURCE_SCHEMA


def pytest_addoption(parser):
    parser.addoption("--pilot-parquet", default=None, help="Opt in to read-only local six-pilot regression.")
    parser.addoption("--labels-parquet", default=None, help="Opt in to the consolidated epidemiological product regression.")


@pytest.fixture
def pilot_path(request):
    value = request.config.getoption("--pilot-parquet")
    if value is None:
        pytest.skip("Local six-pilot regression not requested.")
    path = Path(value).resolve()
    if not path.is_file():
        pytest.fail("Explicit pilot Parquet is unavailable.")
    return path


@pytest.fixture
def labels_path(request):
    value = request.config.getoption('--labels-parquet')
    if value is None:
        pytest.skip('Consolidated product regression not requested.')
    path = Path(value).resolve()
    if not path.is_file():
        pytest.fail('Explicit consolidated labels Parquet is unavailable.')
    return path


@pytest.fixture
def make_panel():
    def make(codes=("08001",), end=date(2014, 12, 29), counts=None):
        rows = []
        for code in codes:
            current = date(2007, 1, 1)
            while current <= end:
                value = counts(code, current) if counts else 4
                nominal = current + timedelta(days=6)
                effective = min(nominal, date(2025, 12, 31))
                days = (effective - current).days + 1
                rows.append({"municipality_code": code, "municipality": "Synthetic " + code,
                             "week_start": current, "population": 1000, "area_km2": 10.,
                             "population_density_km2": 100., "temperature_c": 25.,
                             "precipitation_mm": 0., "relative_humidity_pct": 70.,
                             "confirmed_records_screening": value, "no_eligible_records_observed": value == 0,
                             "week_end_nominal": nominal, "week_end_effective": effective,
                             "n_days": days, "partial_week": days < 7})
                current += timedelta(days=7)
        return pa.Table.from_pylist(rows, schema=SOURCE_SCHEMA)
    return make
