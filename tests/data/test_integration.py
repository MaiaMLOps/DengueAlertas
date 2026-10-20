import json

import pandas as pd
import pytest

from src.data.config import COUNT, KEYS, PipelineConfig, YEARS
from src.data.pipeline import run_pipeline
from src.data.sources import aggregate_epidemiology, inventory_sources, sha256
from src.data.validation import acceptance_checks, integrate, validate_covariates


def test_aggregation_across_files_exclusions_and_2013_retained(tmp_path, write_source, covariates):
    first = write_source(2007, [
        {"CONSECUTIVE": "SYNTHETIC-REUSED-ID"},
        {"INI_SIN": "07/01/2007 11:59:59 p. m."},
        {"confirmados": "0", "Estado_final_de_caso": "2"},
        {"INI_SIN": "01/01/2006 12:00:00 a. m."},
        {"INI_SIN": ""}, {"INI_SIN": "not a date"},
        {"COD_MUN_O": "999"},
        {"INI_SIN": "31/12/2025 11:59:59 p. m."},
        {"INI_SIN": "01/01/2026 12:00:00 a. m."},
    ])
    second = write_source(2008, [{"CONSECUTIVE": "SYNTHETIC-REUSED-ID"}, {"INI_SIN": "01/01/2008 12:00:00 a. m."}])
    third = write_source(2013, [{"INI_SIN": "01/06/2013 12:00:00 a. m.", "FEC_NOT": "01/05/2013 12:00:00 a. m."}])
    hashes = {e["file"]: sha256(tmp_path / e["file"]) for e in [first, second, third]}
    report = {}
    counts = aggregate_epidemiology(tmp_path, [first, second, third], {"05001"}, report)
    assert int(counts[COUNT].sum()) == 6  # includes repeated cross-file ID: no deduplication
    assert counts.set_index("week_start").loc[pd.Timestamp("2007-01-01"), COUNT] == 3
    assert counts.set_index("week_start").loc[pd.Timestamp("2007-12-31"), COUNT] == 1
    assert report["keys_with_multiple_nominal_files"] == 1
    assert report["files"][2]["notification_before_symptoms"] == 1
    assert report["files"][2]["exclusions_sequential"]["included"] == 1
    exclusions = report["exclusions_sequential"]
    assert exclusions == {"not_confirmed": 1, "confirmed_missing_symptom_date": 1,
                          "confirmed_unparseable_symptom_date": 1, "confirmed_outside_date_window": 2,
                          "confirmed_valid_date_outside_candidates": 1, "included": 6}
    assert sum(exclusions.values()) == 12
    panel, merge = integrate(validate_covariates(covariates), counts)
    assert len(panel) == 992 and panel[COUNT].sum() == 6
    assert panel["no_eligible_records_observed"].eq(panel[COUNT].eq(0)).all()
    assert merge["rows_multiplied"] == 0 and merge["record_total_before"] == merge["record_total_after"]
    assert not panel.duplicated(KEYS).any()
    assert all(sha256(tmp_path / e["file"]) == hashes[e["file"]] for e in [first, second, third])
    serialized = json.dumps(report)
    assert "SYNTHETIC-REUSED-ID" not in serialized and "Synthetic final classification" not in serialized


@pytest.mark.parametrize("defect", ["duplicate", "missing_week", "null", "numeric_code", "sunday", "density", "humidity", "precipitation"])
def test_covariate_defects_are_rejected(covariates, defect):
    frame = covariates.copy()
    if defect == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif defect == "missing_week":
        frame = frame.iloc[1:]
    elif defect == "null":
        frame.loc[0, "temperature_c"] = None
    elif defect == "numeric_code":
        frame["municipality_code"] = 5001
    elif defect == "sunday":
        frame.loc[0, "week_start"] += pd.Timedelta(days=6)
    elif defect == "density":
        frame.loc[0, "population_density_km2"] = 200.0
    elif defect == "humidity":
        frame.loc[0, "relative_humidity_pct"] = 101.0
    else:
        frame.loc[0, "precipitation_mm"] = -1.0
    with pytest.raises(ValueError):
        validate_covariates(frame)


def test_merge_rejects_duplicate_or_unmatched_count_keys(covariates):
    counts = pd.DataFrame({"municipality_code": ["05001", "05001"], "week_start": pd.to_datetime(["2007-01-01"] * 2), COUNT: [1, 2]})
    with pytest.raises(ValueError, match="duplicated"):
        integrate(covariates, counts)
    counts = counts.iloc[[0]].assign(municipality_code="99999")
    with pytest.raises(ValueError, match="outside covariates"):
        integrate(covariates, counts)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "object"])
def test_missing_count_is_rejected_instead_of_becoming_zero(covariates, dtype):
    counts = pd.DataFrame({
        "municipality_code": ["05001"], "week_start": pd.to_datetime(["2007-01-01"]),
        COUNT: pd.Series([None], dtype=dtype),
    })
    with pytest.raises(ValueError, match="Weekly counts contain missing values"):
        integrate(covariates, counts)
    assert counts[COUNT].isna().all()


def test_zero_fill_is_limited_to_keys_without_eligible_records(covariates):
    counts = pd.DataFrame({
        "municipality_code": ["05001"], "week_start": pd.to_datetime(["2007-01-01"]),
        COUNT: pd.Series([7], dtype="Int64"),
    })
    panel, merge = integrate(covariates, counts)
    assert panel.iloc[0][COUNT] == 7
    assert panel.iloc[1:][COUNT].eq(0).all()
    assert panel["no_eligible_records_observed"].sum() == 991
    assert merge["positions_filled_with_zero"] == 991
    assert merge["record_total_before"] == merge["record_total_after"] == 7


def test_acceptance_rejects_small_panel_without_forcing_values(covariates):
    counts = pd.DataFrame({"municipality_code": ["05001"], "week_start": pd.to_datetime(["2007-01-01"]), COUNT: [1]})
    panel, _ = integrate(covariates, counts)
    checks = acceptance_checks(panel)
    assert checks["municipalities"] == {"observed": 1, "expected": 78, "passed": False}
    assert checks["partial_week_coverage"]["passed"]
    assert panel[COUNT].sum() == 1


def test_inventory_and_failed_run_write_only_aggregate_report(tmp_path, write_source, covariates):
    sources = tmp_path / "inputs"
    entries = [write_source(y, [{}], sources) for y in YEARS]
    manifest = pd.DataFrame([{
        "year_nominal": e["year"], "source_file": e["source_file"], "source_rows": e["rows"],
        "cache_file": e["file"], "cache_rows": e["rows"], "cache_size_mb": 0.001, "elapsed_seconds": 0,
    } for e in entries])
    manifest.to_csv(sources / "cache_manifest.csv", index=False)
    inventory, _ = inventory_sources(sources)
    assert len(inventory) == 19 and sum(e["rows"] for e in inventory) == 19
    workbook = tmp_path / "covariates.xlsx"
    covariates.to_excel(workbook, index=False, engine="openpyxl")
    cfg = PipelineConfig(sources, workbook, tmp_path / "data/processed/test.parquet",
                         tmp_path / "reports/data_quality/test.json", tmp_path)
    with pytest.raises(RuntimeError, match="failed at integration"):
        run_pipeline(cfg)
    assert not cfg.output.exists()
    report = json.loads(cfg.report.read_text(encoding="utf-8"))
    assert report["status"] == "failed" and "municipalities" in report["discrepancies"]
    assert report["acceptance"]["eligible_records"]["observed"] == 19
    assert "SYNTHETIC-" not in cfg.report.read_text(encoding="utf-8")
    manifest.loc[0, "cache_rows"] = 2
    manifest.to_csv(sources / "cache_manifest.csv", index=False)
    with pytest.raises(ValueError, match="Manifest row mismatch"):
        inventory_sources(sources)


def test_outputs_cannot_escape_repo_or_overwrite(tmp_path):
    output = tmp_path / "data/processed/existing.parquet"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"keep this artifact")
    cfg = PipelineConfig(tmp_path / "inputs", tmp_path / "input.xlsx", output,
                         tmp_path / "reports/data_quality/new.json", tmp_path)
    with pytest.raises(FileExistsError):
        cfg.validate_destinations()
    assert output.read_bytes() == b"keep this artifact"
    escaped = PipelineConfig(output=tmp_path.parent / "outside.parquet", repo_root=tmp_path).resolved()
    with pytest.raises(ValueError, match="Output must be inside"):
        escaped.validate_destinations()
