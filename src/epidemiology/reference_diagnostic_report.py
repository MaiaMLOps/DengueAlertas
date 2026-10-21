"""Export only new diagnostic tables, figures and provenance from immutable inputs."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import pandas as pd
import pyarrow.parquet as pq

from src.data.sources import sha256
from src.pilot_selection.reporting import csv_bytes, git_state
from .characterization import positive_runs
from .characterization_report import DEFAULT_PRODUCT, audit_numerics
from .config import ROOT, EpidemiologyValidationError, method_contract
from .reference_diagnostics import build_diagnostics
from .validation import require

DEFAULT_CHARACTERIZATION = Path('reports/epidemiology/evidence/labels_20261021')


def export_diagnostics(output_dir: Path, product_path: Path = DEFAULT_PRODUCT,
                       characterization_dir: Path = DEFAULT_CHARACTERIZATION, *, root: Path = ROOT) -> dict:
    root = root.resolve()
    output_dir, product_path, characterization_dir = [(root / path).resolve() for path in [output_dir, product_path, characterization_dir]]
    allowed = root / 'reports/epidemiology/runs'
    require(output_dir != allowed and output_dir.is_relative_to(allowed), 'Choose a new child results directory.')
    require(product_path.is_relative_to(root) and characterization_dir.is_relative_to(root), 'Inputs must remain inside the repository.')
    if output_dir.exists():
        raise FileExistsError('Results directory exists; choose a new directory.')
    preliminary_manifest = product_path.with_name('run_manifest.json')
    characterization_manifest = characterization_dir / 'run_manifest.json'
    old = json.loads(preliminary_manifest.read_text(encoding='utf-8'))
    earlier = json.loads(characterization_manifest.read_text(encoding='utf-8'))
    require(old['status'] == earlier['status'] == 'passed', 'An input manifest is not passed.')
    source_path = (root / old['source']['path']).resolve()
    require(source_path.is_relative_to(root), 'Original source must remain inside the repository.')
    require(sha256(product_path) == old['output']['sha256'] and sha256(source_path) == old['source']['sha256'],
            'Preliminary product/source checksum mismatch.')
    for name, expected in earlier['inputs'].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and sha256(path) == expected, 'Earlier characterization input changed.')
    for name, record in earlier['artifacts'].items():
        path = (characterization_dir / name).resolve()
        require(path.is_relative_to(characterization_dir) and sha256(path) == record['sha256'], 'Earlier evidence changed.')
    watched = [product_path, source_path, preliminary_manifest, characterization_manifest]
    watched += [(characterization_dir / name).resolve() for name in earlier['artifacts']]
    inputs = {path: sha256(path) for path in watched}
    product, source = pq.read_table(product_path), pq.read_table(source_path)
    require(json.loads(product.schema.metadata[b'denguealertas.epidemiology']) == method_contract(), 'Saved EWT method changed.')
    numerical_audit = audit_numerics(source, product)
    tables, frame, controls = build_diagnostics(source, product)
    for key, expected in [('eligible_weeks', 4380), ('positive_weeks', 2870), ('negative_weeks', 1510),
                          ('positive_runs', 271), ('long_runs', 20), ('weeks_in_long_runs', 1881)]:
        require(controls[key] == expected, 'Observed results differ from the established characterization; stop.')
    from .characterization import prepare_labels
    actual_runs = positive_runs(prepare_labels(product))
    saved_runs = pd.read_csv(characterization_dir / 'positive_runs.csv', dtype={'municipality_code': str},
                             parse_dates=['start_week', 'end_week'])
    keys = ['municipality_code', 'start_week', 'end_week', 'duration_weeks']
    require(actual_runs[keys].equals(saved_runs[keys]), 'Run boundaries disagree with the prior evidence.')
    from .reference_diagnostic_plots import figures
    files = {name + '.csv': csv_bytes(table) for name, table in tables.items()}
    files.update(figures(frame, tables))
    code = sorted((root / 'src/epidemiology').glob('*.py')) + sorted((root / 'tests/epidemiology').glob('*.py'))
    code += [root / 'src/pilot_selection' / name for name in ['config.py', 'reporting.py']]
    code += [root / 'requirements' / name for name in ['data.txt', 'data-test.txt', 'pilot-selection.txt']]
    manifest = {
        'schema_version': 1, 'status': 'passed', 'purpose': 'descriptive historical-zero and long-run review; saved labels unchanged',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'git': git_state(root),
        'output_dir': output_dir.relative_to(root).as_posix(),
        'inputs_sha256': {path.relative_to(root).as_posix(): value for path, value in inputs.items()},
        'method': method_contract(), 'numerical_audit': numerical_audit, 'controls': controls,
        'description_rules': {
            'statistics_population': 'eligible target weeks only; reference counts may belong to initialization years',
            'historical_zeros': 'number of zero counts among the exact five approved aligned references, including 53->52 fallback',
            'empty_zero_groups': 'show zero weeks; undefined percentages and EWT summaries are empty, not zero',
            'quartiles': 'within each municipality using all eligible 2012-2025 weeks; linear interpolation',
            'quartile_intervals': 'Q1<=P25; P25<Q2<=P50; P50<Q3<=P75; Q4>P75; ties retained together',
            'low_ewt': '<= municipal EWT P25; descriptive only',
            'high_observed_count': '> municipal observed-count P75; descriptive only',
            'pooled_grid': 'aggregate municipal-relative quartiles, not pooled absolute cutoffs',
            'long_runs': 'existing maximal positive runs with duration >=52 weeks; annual profiles use year of week_start',
            'statistics': 'sample std ddof=1; linear quantiles; original values unmodified',
            'interpretation': 'association only; no counterfactual EWT, causal attribution, outbreak declaration or new target',
            'modeling_warning': 'full-period descriptive quantiles are not training features or approved model thresholds',
        },
        'environment': {'python': platform.python_version(), **{name: importlib.metadata.version(name)
                        for name in ['pandas', 'numpy', 'pyarrow', 'matplotlib']}},
        'code_and_dependencies_sha256': {path.relative_to(root).as_posix(): sha256(path) for path in code if path.is_file()},
        'tables': {name + '.csv': {'rows': len(table), 'columns': list(table.columns)} for name, table in tables.items()},
        'artifacts': {name: {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)} for name, content in files.items()},
        'inputs_unchanged': True,
    }
    content = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    require(all(sha256(path) == value for path, value in inputs.items()), 'An input changed before publication; stop.')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    for name, data in files.items():
        with (output_dir / name).open('xb') as stream:
            stream.write(data)
        require(sha256(output_dir / name) == manifest['artifacts'][name]['sha256'], 'Saved diagnostic checksum mismatch.')
    require(all(sha256(path) == value for path, value in inputs.items()), 'Input changed; no completion marker published.')
    with (output_dir / 'run_manifest.json').open('xb') as stream:
        stream.write(content)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description='Audit historical zeros and existing long positive runs.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--input', type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument('--characterization-dir', type=Path, default=DEFAULT_CHARACTERIZATION)
    args = parser.parse_args(argv)
    try:
        result = export_diagnostics(args.output_dir, args.input, args.characterization_dir)
    except (EpidemiologyValidationError, FileExistsError) as exc:
        print(str(exc) if isinstance(exc, EpidemiologyValidationError) else 'Output exists; choose a new directory.', file=sys.stderr)
        return 1
    except Exception as exc:
        print(f'Diagnostic failed ({type(exc).__name__}); completion not certified; inspect locally.', file=sys.stderr)
        return 1
    print(json.dumps(result['controls']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
