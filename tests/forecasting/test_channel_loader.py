"""Unit tests of the channel loader with a synthetic product (especificación §4; U2.2)."""

from datetime import date, timedelta
import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.epidemiology.channel_epico import build_channel_table
from src.epidemiology.config import FIRST_LABEL_YEAR
from src.epidemiology.validation import SOURCE_SCHEMA
from src.forecasting.dataset import CHANNEL_PRODUCT_COLUMNS, ChannelVersionError, load_channel_product


def _product() -> pa.Table:
    rows, current = [], date(2007, 1, 1)
    while current <= date(2012, 12, 31):
        value = current.toordinal() % 13
        nominal = current + timedelta(days=6)
        rows.append({"municipality_code": "08001", "municipality": "Synthetic", "week_start": current,
                     "population": 1000, "area_km2": 10.0, "population_density_km2": 100.0,
                     "temperature_c": 25.0, "precipitation_mm": 0.0, "relative_humidity_pct": 70.0,
                     "confirmed_records_screening": value, "no_eligible_records_observed": value == 0,
                     "week_end_nominal": nominal, "week_end_effective": nominal, "n_days": 7,
                     "partial_week": False})
        current += timedelta(weeks=1)
    panel = pa.Table.from_pylist(rows, schema=SOURCE_SCHEMA)
    return build_channel_table(panel, n_years=5, shift=1, alpha=0.025, first_year=FIRST_LABEL_YEAR)


def _write(table: pa.Table, path) -> str:
    pq.write_table(table, path)
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture
def product_file(tmp_path):
    table = _product()
    path = tmp_path / "channel.parquet"
    return table, path, _write(table, path)


def test_accepts_the_product_with_its_md5(product_file):
    table, path, md5 = product_file

    loaded = load_channel_product(path, expected_md5=md5)

    assert loaded.column_names == CHANNEL_PRODUCT_COLUMNS
    assert loaded.equals(table)


def test_rejects_a_different_md5_naming_both(product_file):
    _, path, md5 = product_file
    other = "0" * 32

    with pytest.raises(ChannelVersionError) as error:
        load_channel_product(path, expected_md5=other)
    assert isinstance(error.value, ValueError)
    assert md5 in str(error.value) and other in str(error.value)


def test_missing_file_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_channel_product(tmp_path / "absent.parquet", expected_md5="0" * 32)


@pytest.mark.parametrize("value", [None, "", "ABCDEF" * 5 + "AB", "0" * 31, 12345], ids=str)
def test_malformed_expected_md5_fails(product_file, value):
    _, path, _ = product_file

    with pytest.raises(ValueError, match="version_dvc_canal"):
        load_channel_product(path, expected_md5=value)


def test_unexpected_columns_fail(tmp_path):
    table = _product().drop_columns(["channel_zone"])
    path = tmp_path / "channel.parquet"

    with pytest.raises(ValueError, match="columns"):
        load_channel_product(path, expected_md5=_write(table, path))
