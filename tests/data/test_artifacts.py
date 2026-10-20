"""Fault injection with synthetic aggregates only; no health microdata."""

import json
from pathlib import PureWindowsPath

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data import artifacts, pipeline
from src.data.config import COUNT, PipelineConfig
from src.data.sources import sha256


@pytest.fixture
def publication(tmp_path):
    table = pa.table({"municipality_code": ["05001"], COUNT: [7]})
    output = tmp_path / "data/processed/synthetic.parquet"
    report_path = tmp_path / "reports/data_quality/synthetic.json"
    report = {"configuration": {"output": "data/processed/synthetic.parquet"}}
    return table, output, report_path, report


def assert_no_temporaries(tmp_path):
    assert not list(tmp_path.rglob("*.tmp"))


def test_publish_complete_pair_after_verification(publication, tmp_path):
    table, output, report_path, report = publication
    artifacts.publish_panel(table, output, report_path, report)
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["status"] == "passed"
    assert saved["output"]["sha256"] == sha256(output)
    assert saved["output"]["readback_equal"]
    assert pq.read_table(output).equals(table)
    assert_no_temporaries(tmp_path)


def test_partial_parquet_write_does_not_publish(publication, tmp_path, monkeypatch):
    table, output, report_path, report = publication

    def interrupt(table, stream, **kwargs):
        stream.write(b"SYNTHETIC PARTIAL PARQUET")
        raise OSError("synthetic interrupted write")

    monkeypatch.setattr(artifacts.pq, "write_table", interrupt)
    with pytest.raises(OSError):
        artifacts.publish_panel(table, output, report_path, report)
    assert not output.exists() and not report_path.exists()
    assert_no_temporaries(tmp_path)


def test_failed_parquet_readback_does_not_publish(publication, tmp_path, monkeypatch):
    table, output, report_path, report = publication
    monkeypatch.setattr(artifacts.pq, "read_table", lambda path: pa.table({COUNT: [99]}))
    with pytest.raises(ValueError, match="Parquet readback differs"):
        artifacts.publish_panel(table, output, report_path, report)
    assert not output.exists() and not report_path.exists()
    assert_no_temporaries(tmp_path)


def test_partial_json_write_does_not_publish_either_file(publication, tmp_path, monkeypatch):
    table, output, report_path, report = publication

    def interrupt(path, value):
        path.write_text('{"synthetic":', encoding="utf-8")
        raise OSError("synthetic interrupted JSON write")

    monkeypatch.setattr(artifacts, "_serialize_report", interrupt)
    with pytest.raises(OSError):
        artifacts.publish_panel(table, output, report_path, report)
    assert not output.exists() and not report_path.exists()
    assert_no_temporaries(tmp_path)


def test_report_publication_error_rolls_back_owned_dataset(publication, tmp_path, monkeypatch):
    table, output, report_path, report = publication
    original_publish = artifacts.publish_exclusive
    verified_before_report = []

    def interrupt(temporary, destination):
        if destination == report_path:
            verified_before_report.append(pq.read_table(output).equals(table))
            raise OSError("synthetic publication failure")
        original_publish(temporary, destination)

    monkeypatch.setattr(artifacts, "publish_exclusive", interrupt)
    with pytest.raises(OSError):
        artifacts.publish_panel(table, output, report_path, report)
    assert verified_before_report == [True]
    assert not output.exists() and not report_path.exists()
    assert_no_temporaries(tmp_path)


@pytest.mark.parametrize("existing", ["dataset", "report"])
def test_existing_artifacts_are_never_replaced(publication, tmp_path, existing):
    table, output, report_path, report = publication
    destination = output if existing == "dataset" else report_path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"SYNTHETIC PREEXISTING ARTIFACT")
    with pytest.raises(FileExistsError):
        artifacts.publish_panel(table, output, report_path, report)
    assert destination.read_bytes() == b"SYNTHETIC PREEXISTING ARTIFACT"
    assert not (report_path if existing == "dataset" else output).exists()
    assert_no_temporaries(tmp_path)


def test_failure_report_is_also_atomic(tmp_path, monkeypatch):
    destination = tmp_path / "quality.json"

    def interrupt(path, value):
        path.write_text('{"synthetic":', encoding="utf-8")
        raise OSError("synthetic JSON failure")

    monkeypatch.setattr(artifacts, "_serialize_report", interrupt)
    with pytest.raises(OSError):
        artifacts.write_report(destination, {"status": "failed"})
    assert not destination.exists()
    assert_no_temporaries(tmp_path)


def test_existing_failure_report_is_preserved(tmp_path):
    destination = tmp_path / "quality.json"
    destination.write_bytes(b"SYNTHETIC PREEXISTING REPORT")
    with pytest.raises(FileExistsError):
        artifacts.write_report(destination, {"status": "failed"})
    assert destination.read_bytes() == b"SYNTHETIC PREEXISTING REPORT"
    assert_no_temporaries(tmp_path)


def test_same_filesystem_reference_is_relative(tmp_path):
    assert pipeline.relative(tmp_path / "input/book.xlsx", tmp_path) == "input/book.xlsx"


@pytest.mark.parametrize("path,root", [
    (PureWindowsPath("C:/synthetic-private-input/book.xlsx"), PureWindowsPath("D:/repo")),
    (PureWindowsPath("//synthetic-server/share/book.xlsx"), PureWindowsPath("D:/repo")),
])
def test_different_drive_or_unc_uses_private_path_free_alias(path, root):
    assert pipeline.relative(path, root, label="covariates") == "external:covariates"


def test_unrepresentable_relative_path_uses_alias(tmp_path, monkeypatch):
    def cannot_represent(*args):
        raise ValueError("synthetic different volumes")

    monkeypatch.setattr(pipeline.os.path, "relpath", cannot_represent)
    assert pipeline.relative(tmp_path / "input", tmp_path, label="sivigila_dir") == "external:sivigila_dir"


def config_for_test(tmp_path):
    return PipelineConfig(
        tmp_path / "inputs", tmp_path / "covariates.xlsx",
        tmp_path / "data/processed/test.parquet", tmp_path / "reports/data_quality/test.json", tmp_path,
    )


@pytest.mark.parametrize("failing_function", ["relative", "git_state"])
def test_initialization_errors_generate_safe_failure_report(tmp_path, monkeypatch, failing_function):
    cfg = config_for_test(tmp_path)

    def interrupt(*args, **kwargs):
        raise ValueError("SYNTHETIC PRIVATE ERROR TEXT MUST NOT APPEAR")

    monkeypatch.setattr(pipeline, failing_function, interrupt)
    with pytest.raises(RuntimeError, match="failed at initialization"):
        pipeline.run_pipeline(cfg)
    saved = json.loads(cfg.report.read_text(encoding="utf-8"))
    assert saved["status"] == "failed" and saved["failed_stage"] == "initialization"
    assert "PRIVATE ERROR TEXT" not in cfg.report.read_text(encoding="utf-8")
    assert not cfg.output.exists()


def test_failure_to_persist_diagnostics_is_reported_safely(tmp_path, monkeypatch):
    cfg = config_for_test(tmp_path)

    def interrupt(*args):
        raise OSError("SYNTHETIC PRIVATE ERROR TEXT MUST NOT APPEAR")

    monkeypatch.setattr(pipeline, "write_report", interrupt)
    with pytest.raises(RuntimeError, match="failed at inventory.*quality report could not be saved") as error:
        pipeline.run_pipeline(cfg)
    assert "PRIVATE ERROR TEXT" not in str(error.value)
    assert not cfg.output.exists() and not cfg.report.exists()
