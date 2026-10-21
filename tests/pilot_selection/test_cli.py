import hashlib
import io
import json

import pandas as pd
import pytest

from src.pilot_selection.__main__ import main
from src.pilot_selection.config import RunConfig
from src.pilot_selection.reporting import csv_bytes, run_analysis, sha256


def test_csv_preserves_leading_zero_and_unrounded_values():
    source = pd.DataFrame({"municipality_code": ["08001"], "active_pct": [100 * 818 / 835]})
    data = csv_bytes(source)
    assert b'"08001"' in data
    reread = pd.read_csv(io.BytesIO(data), dtype={"municipality_code": str})
    assert reread.municipality_code.iloc[0] == "08001"
    assert reread.active_pct.iloc[0] == pytest.approx(source.active_pct.iloc[0], abs=1e-12)


def test_export_manifest_and_repeatability(tmp_path, synthetic_master):
    source = tmp_path / "master.parquet"
    synthetic_master.to_parquet(source, index=False)
    source_digest = sha256(source)
    output = tmp_path / "reports/pilot_selection/runs/one"
    first = run_analysis(RunConfig(source, output, False, tmp_path))
    assert first["status"] == "passed"
    assert len(first["artifacts"]) == 8
    assert first["controls"]["input_sha256_unchanged"]
    assert sha256(source) == source_digest
    for name, details in first["artifacts"].items():
        assert sha256(output / name) == details["sha256"]
    serialized = (output / "run_manifest.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert tmp_path.as_posix() not in serialized
    assert json.loads(serialized)["input"]["path"] == "master.parquet"
    old_manifest = hashlib.sha256(serialized.encode()).hexdigest()
    with pytest.raises(FileExistsError):
        run_analysis(RunConfig(source, output, False, tmp_path))
    assert hashlib.sha256((output / "run_manifest.json").read_bytes()).hexdigest() == old_manifest
    second_dir = output.with_name("two")
    second = run_analysis(RunConfig(source, second_dir, False, tmp_path))
    assert second["artifacts"] == first["artifacts"]
    assert set(p.name for p in output.iterdir()) == set(first["artifacts"]) | {"run_manifest.json"}


def test_outputs_cannot_escape_or_replace_parent(tmp_path):
    for destination in (tmp_path / "outside", tmp_path / "reports/pilot_selection/runs",
                        tmp_path / "reports/pilot_selection/evidence"):
        with pytest.raises(ValueError):
            RunConfig(tmp_path / "master.parquet", destination, root=tmp_path).resolved()


def test_cli_errors_are_safe(capsys):
    assert main(["--input", "missing.parquet", "--output-dir", "reports/pilot_selection/runs/missing_input_test"]) == 1
    output = capsys.readouterr()
    assert "Analysis stopped" in output.err
    assert "Traceback" not in output.err


def test_external_error_details_are_not_exposed(monkeypatch, capsys):
    from src.pilot_selection import __main__ as cli
    def fail(*args, **kwargs):
        raise ValueError("SYNTHETIC PRIVATE FILESYSTEM DETAILS")
    monkeypatch.setattr(cli, "run_analysis", fail)
    assert cli.main([]) == 1
    error = capsys.readouterr().err
    assert "ValueError" in error
    assert "PRIVATE" not in error


def test_failure_does_not_certify_completion(tmp_path, synthetic_master, monkeypatch):
    source = tmp_path / "master.parquet"
    synthetic_master.to_parquet(source, index=False)
    output = tmp_path / "reports/pilot_selection/runs/failure"
    from src.pilot_selection import reporting
    original = reporting.sha256
    def corrupt_artifact_checksum(path):
        return "bad" if path.suffix == ".csv" else original(path)
    monkeypatch.setattr(reporting, "sha256", corrupt_artifact_checksum)
    with pytest.raises(ValueError, match="checksum"):
        run_analysis(RunConfig(source, output, False, tmp_path))
    assert not (output / "run_manifest.json").exists()
