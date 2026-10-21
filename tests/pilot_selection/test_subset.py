"""Territorial-only export, including partial weeks, without touching integration."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from src.pilot_selection import config, subset
from src.pilot_selection.reporting import sha256


@pytest.fixture
def arrow_master(synthetic_master):
    frame = synthetic_master.copy()
    frame["population"] = 1000
    for name, value in (("area_km2", 10.), ("population_density_km2", 100.), ("temperature_c", 25.),
                        ("precipitation_mm", 0.), ("relative_humidity_pct", 70.)):
        frame[name] = value
    columns = ["municipality_code", "municipality", "week_start", "population", "area_km2",
               "population_density_km2", "temperature_c", "precipitation_mm", "relative_humidity_pct",
               config.COUNT, "no_eligible_records_observed", "week_end_nominal", "week_end_effective",
               "n_days", "partial_week"]
    table = pa.Table.from_pandas(frame[columns], preserve_index=False)
    for name in ("week_start", "week_end_nominal", "week_end_effective"):
        index = table.schema.get_field_index(name)
        table = table.set_column(index, pa.field(name, pa.date32()), table[name].cast(pa.date32()))
    metadata = {
        b"pandas": b"obsolete master index metadata must not survive",
        b"denguealertas.master_scope": b"18 municipalities; source-only aggregate metadata",
        b"denguealertas.warnings": json.dumps(["Final municipalities remain undecided."]).encode(),
        b"denguealertas.rule": json.dumps({"event": "210", "confirmed": "1", "calendar": "Monday-Sunday W-SUN",
                                           "municipality_count": 18}).encode(),
        b"denguealertas.data_dictionary": json.dumps({name: "Synthetic definition of " + name
                                                      for name in columns}).encode(),
    }
    return table.replace_schema_metadata(metadata)


def write_source(tmp_path, table):
    source = tmp_path / "data/processed/master.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, source)
    return source


def independent_subset(master):
    codes = {pilot.code for pilot in config.PILOTS}
    source_codes = master["municipality_code"].to_pylist()
    dates = master["week_start"].to_pylist()
    indices = sorted((i for i, code in enumerate(source_codes) if code in codes),
                     key=lambda i: (source_codes[i], dates[i]))
    return master.take(pa.array(indices, type=pa.int64())).replace_schema_metadata(None)


def test_filter_keeps_every_original_field_value_and_partial_week(arrow_master):
    # Deliberately reverse the master's storage order.
    master = arrow_master.take(pa.array(list(range(arrow_master.num_rows - 1, -1, -1))))
    result = subset.filter_pilots(master)
    assert result.equals(independent_subset(master), check_metadata=True)
    assert result.num_rows == 5952 and result.num_columns == 15
    profile = subset.panel_profile(result)
    assert profile["partial_positions"] == 6
    assert set(profile["weeks_per_municipality"].values()) == {992}
    assert profile["week_start_max"] == "2025-12-29"
    assert pc.sum(result["n_days"]).as_py() == 6 * (991 * 7 + 3)
    assert all(a.equals(b, check_metadata=True) for a, b in zip(master.schema, result.schema))


def test_export_provenance_exact_readback_and_source_immutability(tmp_path, arrow_master):
    source = write_source(tmp_path, arrow_master)
    before = sha256(source)
    output = tmp_path / "data/processed/pilots.parquet"
    result = subset.export_pilot_subset(source, output, root=tmp_path)
    restored = pq.read_table(output)
    assert restored.equals(independent_subset(arrow_master), check_metadata=False)
    assert result["source_sha256"] == before == sha256(source)
    assert result["output_sha256"] == sha256(output)
    assert result["readback_equal"] and result["exact_subset_equal"]
    metadata = restored.schema.metadata
    assert b"pandas" not in metadata and b"denguealertas.master_scope" not in metadata
    assert b"undecided" not in metadata[b"denguealertas.warnings"]
    assert metadata[b"denguealertas.data_dictionary"] == arrow_master.schema.metadata[b"denguealertas.data_dictionary"]
    assert "municipality_count" not in json.loads(metadata[b"denguealertas.rule"])
    provenance = json.loads(metadata[b"denguealertas.derivation"])
    assert provenance["source"]["sha256"] == before
    assert provenance["source"]["profile"]["municipalities"] == 18
    assert provenance["subset"]["municipalities"] == 6
    assert provenance["selection"]["codes"] == list(subset.pilot_codes())
    assert provenance["subset"]["partial_positions"] == 6
    serialized = json.dumps(provenance)
    assert str(tmp_path) not in serialized and tmp_path.as_posix() not in serialized


def test_configuration_is_read_from_pilots_without_a_second_list(monkeypatch, arrow_master):
    new_pilot = config.Municipality("01001", "Synthetic replacement", "#000000")
    previous = config.PILOTS[0].code
    monkeypatch.setattr(config, "PILOTS", (new_pilot, *config.PILOTS[1:]))
    result = subset.filter_pilots(arrow_master)
    codes = set(result["municipality_code"].to_pylist())
    assert new_pilot.code in codes and previous not in codes
    assert codes == {pilot.code for pilot in config.PILOTS}


def test_existing_output_and_source_are_never_overwritten(tmp_path, arrow_master):
    source = write_source(tmp_path, arrow_master)
    before = sha256(source)
    output = source.with_name("existing.parquet")
    output.write_bytes(b"PREEXISTING DATA")
    with pytest.raises(FileExistsError):
        subset.export_pilot_subset(source, output, root=tmp_path)
    assert output.read_bytes() == b"PREEXISTING DATA"
    with pytest.raises(ValueError, match="cannot replace"):
        subset.export_pilot_subset(source, source, root=tmp_path)
    assert sha256(source) == before


@pytest.mark.parametrize("defect", ["null", "duplicate", "missing_week", "numeric_code", "wrong_partial",
                                    "wrong_date_type", "wrong_column_count"])
def test_reject_invalid_source_without_repair(arrow_master, defect):
    table = arrow_master
    if defect == "null":
        values = table["population"].to_pylist()
        values[0] = None
        table = table.set_column(table.schema.get_field_index("population"), "population", pa.array(values, type=pa.int64()))
    elif defect == "duplicate":
        table = pa.concat_tables([table, table.slice(0, 1)])
    elif defect == "missing_week":
        table = table.slice(1)
    elif defect == "numeric_code":
        table = table.set_column(0, "municipality_code", table["municipality_code"].cast(pa.int64()))
    elif defect == "wrong_partial":
        table = table.set_column(table.schema.get_field_index("partial_week"), "partial_week",
                                 pa.array([False] * table.num_rows))
    elif defect == "wrong_date_type":
        index = table.schema.get_field_index("week_start")
        table = table.set_column(index, "week_start", table["week_start"].cast(pa.timestamp("ns")))
    else:
        table = table.drop(["population"])
    with pytest.raises(ValueError):
        subset.panel_profile(table)


def test_missing_pilot_is_not_silently_omitted(arrow_master):
    missing = subset.pilot_codes()[0]
    table = arrow_master.filter(pc.not_equal(arrow_master["municipality_code"], missing))
    with pytest.raises(ValueError, match="missing"):
        subset.filter_pilots(table)


def test_failed_encode_reconciliation_creates_no_output(tmp_path, arrow_master, monkeypatch):
    source = write_source(tmp_path, arrow_master)
    output = source.with_name("pilots.parquet")
    original = pq.write_table
    def corrupt(table, destination, **kwargs):
        index = table.schema.get_field_index("population")
        table = table.set_column(index, "population", pa.array([0] * table.num_rows, type=pa.int64()))
        return original(table, destination, **kwargs)
    monkeypatch.setattr(subset.pq, "write_table", corrupt)
    with pytest.raises(ValueError, match="Encoded"):
        subset.export_pilot_subset(source, output, root=tmp_path)
    assert not output.exists()


def test_concurrent_destination_creation_is_protected(tmp_path, arrow_master, monkeypatch):
    source = write_source(tmp_path, arrow_master)
    output = source.with_name("pilots.parquet")
    original = pq.write_table
    def competing_creation(table, destination, **kwargs):
        output.write_bytes(b"CONCURRENT PREEXISTING DATA")
        return original(table, destination, **kwargs)
    monkeypatch.setattr(subset.pq, "write_table", competing_creation)
    with pytest.raises(FileExistsError):
        subset.export_pilot_subset(source, output, root=tmp_path)
    assert output.read_bytes() == b"CONCURRENT PREEXISTING DATA"


def test_changed_source_hash_cancels_publication(tmp_path, arrow_master, monkeypatch):
    source = write_source(tmp_path, arrow_master)
    output = source.with_name("pilots.parquet")
    original = subset.sha256
    calls = []
    def changed(path):
        calls.append(path)
        return original(path) if len(calls) == 1 else "changed"
    monkeypatch.setattr(subset, "sha256", changed)
    with pytest.raises(ValueError, match="Source changed"):
        subset.export_pilot_subset(source, output, root=tmp_path)
    assert not output.exists()


def test_destination_cannot_escape_processed(tmp_path, arrow_master):
    source = write_source(tmp_path, arrow_master)
    with pytest.raises(ValueError, match="inside data/processed"):
        subset.export_pilot_subset(source, tmp_path / "outside.parquet", root=tmp_path)


def test_cli_reports_external_errors_without_private_details(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise OSError("SYNTHETIC PRIVATE DETAILS")
    monkeypatch.setattr(subset, "export_pilot_subset", fail)
    assert subset.main([]) == 1
    error = capsys.readouterr().err
    assert "OSError" in error and "PRIVATE" not in error


def test_optional_real_master_exact_subset_and_partial_positions(master_path):
    before = sha256(master_path)
    master = pq.read_table(master_path)
    table = subset.filter_pilots(master)
    profile = subset.panel_profile(table)
    assert table.equals(independent_subset(master), check_metadata=True)
    assert table.num_rows == 5952 and table.num_columns == 15
    assert profile["municipalities"] == 6
    assert set(profile["weeks_per_municipality"].values()) == {992}
    assert profile["partial_positions"] == 6
    assert profile["duplicate_keys"] == 0
    assert not any(profile["nulls"].values())
    assert sha256(master_path) == before
