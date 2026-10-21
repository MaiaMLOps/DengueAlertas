from datetime import date
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.data.sources import sha256
from src.epidemiology import pipeline
from src.epidemiology.__main__ import main
from src.epidemiology.config import EpidemiologyValidationError
from src.pilot_selection.config import PILOTS


@pytest.fixture
def source_path(tmp_path, make_panel):
    source = tmp_path / 'data/processed/pilots.parquet'
    source.parent.mkdir(parents=True)
    pq.write_table(make_panel(codes=tuple(p.code for p in PILOTS), end=date(2025, 12, 29)), source)
    return source


def test_export_readback_provenance_source_preservation_and_no_overwrite(tmp_path, source_path):
    output_dir = Path('data/processed/epidemiology/synthetic_run')
    original_hash = sha256(source_path)
    manifest = pipeline.export_labels(output_dir, source_path, root=tmp_path)
    output = tmp_path / manifest['output']['path']
    restored = pq.read_table(output)
    source = pq.read_table(source_path)
    assert restored.select(source.column_names).equals(source, check_metadata=False)
    assert manifest['diagnostics']['validation']['eligible'] == 4380
    assert manifest['diagnostics']['validation']['ineligible'] == 1572
    assert manifest['source']['sha256'] == original_hash == sha256(source_path)
    assert manifest['output']['sha256'] == sha256(output)
    assert json.loads((output.parent / 'run_manifest.json').read_text(encoding='utf-8')) == manifest
    metadata = restored.schema.metadata
    assert json.loads(metadata[b'denguealertas.epidemiology'])['epsilon'] == 0.001
    assert json.loads(metadata[b'denguealertas.derivation'])['source_sha256'] == original_hash
    assert str(tmp_path) not in json.dumps(manifest)
    saved_hash = sha256(output)
    with pytest.raises(FileExistsError):
        pipeline.export_labels(output_dir, source_path, root=tmp_path)
    assert saved_hash == sha256(output)


def test_destinations_cannot_escape_or_replace_source(tmp_path, source_path):
    for output_dir in [Path('data/processed/epidemiology'), source_path, tmp_path / 'outside']:
        with pytest.raises(EpidemiologyValidationError):
            pipeline.export_labels(output_dir, source_path, root=tmp_path)


def test_changed_source_hash_cancels_before_publication(tmp_path, source_path, monkeypatch):
    original = pipeline.sha256
    calls = []
    def changed(path):
        if path == source_path:
            calls.append(path)
            return original(path) if len(calls) == 1 else 'changed'
        return original(path)
    monkeypatch.setattr(pipeline, 'sha256', changed)
    output_dir = tmp_path / 'data/processed/epidemiology/changed'
    with pytest.raises(EpidemiologyValidationError, match='Source changed'):
        pipeline.export_labels(output_dir, source_path, root=tmp_path)
    assert not output_dir.exists()


def test_failed_saved_verification_has_no_completion_manifest(tmp_path, source_path, monkeypatch):
    original_read = pipeline.pq.read_table
    def wrong_read(path, *args, **kwargs):
        table = original_read(path, *args, **kwargs)
        if isinstance(path, Path) and path.name == 'pilot_weekly_labels.parquet':
            return table.slice(0, 1)
        return table
    monkeypatch.setattr(pipeline.pq, 'read_table', wrong_read)
    output_dir = tmp_path / 'data/processed/epidemiology/incomplete'
    with pytest.raises(EpidemiologyValidationError, match='Saved product failed'):
        pipeline.export_labels(output_dir, source_path, root=tmp_path)
    assert not (output_dir / 'run_manifest.json').exists()


def test_cli_error_does_not_expose_external_details(monkeypatch, capsys):
    import src.epidemiology.__main__ as cli
    def fail(*args):
        raise OSError('PRIVATE SYNTHETIC PATH')
    monkeypatch.setattr(cli, 'export_labels', fail)
    assert main(['--output-dir', 'data/processed/epidemiology/cli']) == 1
    assert 'PRIVATE' not in capsys.readouterr().err
