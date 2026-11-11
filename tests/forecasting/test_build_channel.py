"""Unit tests of the channel product builder with a synthetic panel (contrato §10.2; U1.3)."""

from datetime import date, timedelta
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.epidemiology.config import FIRST_LABEL_YEAR
from src.epidemiology.validation import SOURCE_SCHEMA
from src.forecasting.build_channel import export_channel, main

PARAMS = {"n_years": 5, "shift": 1, "alpha": 0.025, "first_year": FIRST_LABEL_YEAR}


def make_panel(end=date(2013, 12, 30), codes=("08001", "41001")):
    rows = []
    for code in codes:
        current = date(2007, 1, 1)
        while current <= end:
            value = 1 + (current.toordinal() // 7 + int(code)) % 9
            nominal = current + timedelta(days=6)
            effective = min(nominal, date(2025, 12, 31))
            days = (effective - current).days + 1
            rows.append({"municipality_code": code, "municipality": "Synthetic " + code,
                         "week_start": current, "population": 1000, "area_km2": 10.0,
                         "population_density_km2": 100.0, "temperature_c": 25.0,
                         "precipitation_mm": 0.0, "relative_humidity_pct": 70.0,
                         "confirmed_records_screening": value, "no_eligible_records_observed": value == 0,
                         "week_end_nominal": nominal, "week_end_effective": effective,
                         "n_days": days, "partial_week": days < 7})
            current += timedelta(days=7)
    return pa.Table.from_pylist(rows, schema=SOURCE_SCHEMA)


@pytest.fixture
def panel_file(tmp_path):
    path = tmp_path / "panel.parquet"
    pq.write_table(make_panel(), path)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_writes_product_and_consistent_manifest(tmp_path, panel_file):
    output, manifest_path = tmp_path / "out" / "channel.parquet", tmp_path / "out" / "channel_manifest.json"

    manifest = export_channel(panel_file, output, manifest_path, **PARAMS)

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert manifest["output"]["sha256"] == sha(output)
    assert manifest["source"]["sha256"] == sha(panel_file)
    assert {key: manifest["parameters"][key] for key in PARAMS} == PARAMS
    assert set(manifest["git"]) >= {"commit", "src_dirty"}
    product = pq.read_table(output)
    counts = manifest["counts"]
    assert (counts["rows"], counts["columns"]) == (product.num_rows, 20)
    eligible = sum(product.column("channel_eligible").to_pylist())
    assert counts["eligible"] == eligible == 2 * (53 + 52)  # 2012 (53 Mondays) and 2013, two municipalities
    assert counts["positives"] == sum(v for v in product.column("target_u1e").to_pylist() if v is not None)
    assert sum(counts["zones"].values()) == eligible


def test_product_metadata_documents_channel(tmp_path, panel_file):
    output = tmp_path / "channel.parquet"
    export_channel(panel_file, output, tmp_path / "manifest.json", **PARAMS)

    metadata = pq.read_schema(output).metadata
    dictionary = json.loads(metadata[b"denguealertas.data_dictionary"])
    assert {"epico_central", "epico_upper", "target_u1e", "channel_zone", "channel_eligible"} <= set(dictionary)
    assert json.loads(metadata[b"denguealertas.derivation"])["source_sha256"] == sha(panel_file)
    assert json.loads(metadata[b"denguealertas.channel"])["alpha"] == 0.025


def test_existing_output_is_never_overwritten(tmp_path, panel_file):
    output, manifest_path = tmp_path / "channel.parquet", tmp_path / "manifest.json"
    output.write_bytes(b"previous")

    with pytest.raises(FileExistsError):
        export_channel(panel_file, output, manifest_path, **PARAMS)

    assert output.read_bytes() == b"previous"
    assert not manifest_path.exists()


def test_export_is_deterministic(tmp_path, panel_file):
    first = export_channel(panel_file, tmp_path / "a.parquet", tmp_path / "a.json", **PARAMS)
    second = export_channel(panel_file, tmp_path / "b.parquet", tmp_path / "b.json", **PARAMS)

    assert first["output"]["sha256"] == second["output"]["sha256"]


def test_cli_rejects_a_panel_that_is_not_the_six_pilots(tmp_path, panel_file, capsys):
    output = tmp_path / "channel.parquet"

    assert main(["--input", str(panel_file), "--output", str(output)]) == 1

    assert "six-pilot panel" in capsys.readouterr().err
    assert not output.exists()
