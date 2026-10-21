"""Read-only audit and reproducible local evidence from the saved preliminary labels."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

import pyarrow.parquet as pq

from src.data.sources import sha256
from src.pilot_selection.config import PILOTS
from src.pilot_selection.reporting import csv_bytes, git_state

from .characterization import build_tables, prepare_labels
from .config import COUNT, ROOT, EpidemiologyValidationError, method_contract
from .validation import require

DEFAULT_PRODUCT = Path('data/processed/epidemiology/pilot_weekly_labels.parquet')


def audit_numerics(source, product) -> dict:
    """Audit existing values; do not call the production EWT or labelling functions."""
    grouped = defaultdict(list)
    for row in source.to_pylist():
        grouped[(row['municipality_code'], row['week_start'].year)].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row['week_start'])
    positions = {(code, row['week_start']): ordinal
                 for (code, year), rows in grouped.items() for ordinal, row in enumerate(rows, 1)}
    checked = equalities = references_checked = fallbacks = 0
    with localcontext() as context:
        context.prec = 60
        epsilon = Decimal('0.001')
        for row in product.to_pylist():
            if not row['label_eligible']:
                continue
            year = row['week_start'].year
            ordinal = positions[(row['municipality_code'], row['week_start'])]
            counts = []
            for historical_year in range(year - 5, year):
                historical = grouped[(row['municipality_code'], historical_year)]
                position = 52 if ordinal == 53 and len(historical) == 52 else ordinal
                require(len(historical) >= position, 'Incomplete historical alignment in numerical audit.')
                reference = historical[position - 1]
                require(reference['week_start'].year == historical_year < year and not reference['partial_week'],
                        'Audit reference is not a complete prior-year week.')
                counts.append(reference[COUNT])
                fallbacks += ordinal == 53 and position == 52
            # This strict comparison is exact for integer counts and eps=1/1000.
            left = (1000 * row[COUNT] + 1) ** 5
            right = math.prod(1000 * count + 1 for count in counts)
            require(row['excess_label'] == int(left > right), 'Saved label disagrees with exact integer comparison; stop.')
            if left == right:
                equalities += 1
                require(row['ewt'] == row[COUNT], 'Saved exact equality has a floating-point residue; stop.')
            expected = (float(counts[0]) if len(set(counts)) == 1 else
                        float((sum((Decimal(count) + epsilon).ln() for count in counts) / 5).exp() - epsilon))
            require(math.isclose(row['ewt'], expected, rel_tol=3e-14, abs_tol=1e-16),
                    'Saved EWT disagrees with 60-digit reference; stop.')
            checked += 1
            references_checked += len(counts)
    return {'eligible_labels_checked': checked, 'historical_references_checked': references_checked,
            'week_53_fallback_references': fallbacks, 'exact_observed_equalities': equalities,
            'exact_integer_label_comparison': 'passed', 'decimal_60_digit_ewt_comparison': 'passed',
            'ewt_verification_tolerance_only': {'relative': 3e-14, 'absolute': 1e-16},
            'label_comparison_tolerance': 0}


def export_characterization(output_dir: Path, product_path: Path = DEFAULT_PRODUCT, *, root: Path = ROOT) -> dict:
    root = root.resolve()
    product_path, output_dir = (root / product_path).resolve(), (root / output_dir).resolve()
    allowed = root / 'reports/epidemiology/runs'
    require(product_path.is_relative_to(root) and product_path.suffix == '.parquet', 'Expected a repository-local labelled Parquet.')
    require(output_dir != allowed and output_dir.is_relative_to(allowed), 'Choose a new child directory inside reports/epidemiology/runs.')
    if output_dir.exists():
        raise FileExistsError('Results directory exists; choose a new directory.')
    preliminary_manifest = product_path.with_name('run_manifest.json')
    require(product_path.is_file() and preliminary_manifest.is_file(), 'Preliminary product or paired manifest missing.')
    product_hash, manifest_hash = sha256(product_path), sha256(preliminary_manifest)
    earlier = json.loads(preliminary_manifest.read_text(encoding='utf-8'))
    require(earlier['status'] == 'passed' and earlier['output']['sha256'] == product_hash,
            'Preliminary manifest does not certify this product.')
    source_path = (root / earlier['source']['path']).resolve()
    require(source_path.is_relative_to(root) and source_path.is_file(), 'Original pilot input unavailable.')
    source_hash = sha256(source_path)
    require(source_hash == earlier['source']['sha256'], 'Original pilot hash disagrees with preliminary provenance.')
    product, source = pq.read_table(product_path), pq.read_table(source_path)
    frame = prepare_labels(product)
    require(product.select(source.column_names).equals(source, check_metadata=False), 'Original pilot fields differ.')
    require(json.loads(product.schema.metadata[b'denguealertas.epidemiology']) == method_contract(),
            'Saved method metadata differs from the approved contract.')
    require(set(frame.municipality_code) == {p.code for p in PILOTS} and len(frame) == 5952
            and int(frame.label_eligible.sum()) == 4380, 'Expected the six-pilot preliminary product and 4380 eligible weeks.')
    # All precision checks must pass before any characterization is computed or saved.
    numerical_audit = audit_numerics(source, product)
    tables, controls = build_tables(frame)
    from .characterization_plots import figure_bytes
    figures, plot_checks = figure_bytes(frame, tables)
    files = {name + '.csv': csv_bytes(table) for name, table in tables.items()}
    files.update(figures)
    notes = (
        'Caracterización descriptiva de etiquetas semanales existentes, 2012–2025.\n'
        'Positivos y negativos usan únicamente semanas elegibles; los nulos no son negativos.\n'
        'Años: año del lunes inicial. Porcentajes anuales: denominador municipal/anual elegible.\n'
        'Estadísticas: semanas elegibles; std muestral (ddof=1), cuantiles con interpolación lineal.\n'
        'Rachas: secuencias máximas de etiquetas 1, con separación exacta de siete días.\n'
        'Una negativa, semana no elegible o salto de calendario corta la racha; un cambio de año no.\n'
        'Los límites iniciales/finales y no elegibles limitan la duración observable. No son episodios validados.\n'
        'Las rachas no redefinen la variable objetivo ni constituyen un clasificador de episodios.\n'
        'Figuras: conteos sin suavizado, fechas reales, escalas Y independientes; semana parcial no graficada.\n'
        'Los porcentajes anotados en la figura anual se redondean solo para presentación; los CSV conservan precisión.\n'
        'EWT central retrospectivo; no equivale a brote oficial. Se mantienen todas las limitaciones del screening.\n'
    )
    files['README.txt'] = notes.encode('utf-8')
    inputs = {product_path: product_hash, preliminary_manifest: manifest_hash, source_path: source_hash}
    require(all(sha256(path) == value for path, value in inputs.items()), 'An input changed during preparation; stop.')
    paths = sorted((root / 'src/epidemiology').glob('*.py')) + sorted((root / 'tests/epidemiology').glob('*.py'))
    paths += [root / 'requirements' / name for name in ['data.txt', 'data-test.txt', 'pilot-selection.txt']]
    paths += [root / 'src/pilot_selection' / name for name in ['config.py', 'reporting.py']]
    manifest = {
        'schema_version': 1, 'status': 'passed', 'purpose': 'descriptive characterization for team review; unchanged saved labels',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': {path.relative_to(root).as_posix(): value for path, value in inputs.items()},
        'output_dir': output_dir.relative_to(root).as_posix(), 'git': git_state(root),
        'method': method_contract(), 'numerical_audit': numerical_audit, 'controls': controls,
        'description_rules': {'denominator': 'eligible weeks only', 'year': 'year of week_start',
                              'statistics': 'eligible weeks; sample std ddof=1; linear quantiles',
                              'runs': 'maximal positive weekly sequences; negative/ineligible/gap breaks; years do not break',
                              'run_boundary': 'boundaries describe observation, not confirmed episode onset/end'},
        'plot_checks': plot_checks, 'inputs_unchanged': True,
        'code_and_dependencies_sha256': {path.relative_to(root).as_posix(): sha256(path) for path in paths if path.is_file()},
        'environment': {'python': platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ['pandas', 'numpy', 'pyarrow', 'matplotlib', 'Pillow']}},
        'tables': {name + '.csv': {'rows': len(table), 'columns': list(table.columns)} for name, table in tables.items()},
        'artifacts': {name: {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)} for name, content in files.items()},
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    for name, content in files.items():
        with (output_dir / name).open('xb') as stream:
            stream.write(content)
        require(sha256(output_dir / name) == manifest['artifacts'][name]['sha256'], 'Saved evidence checksum mismatch.')
    require(all(sha256(path) == value for path, value in inputs.items()), 'Input changed; no completion manifest published.')
    # As with the preliminary product, an interrupted directory without this marker is incomplete.
    with (output_dir / 'run_manifest.json').open('xb') as stream:
        stream.write(manifest_bytes)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Characterize existing weekly epidemiological labels without modifying them.')
    parser.add_argument('--input', type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = export_characterization(args.output_dir, args.input)
    except (EpidemiologyValidationError, FileExistsError) as exc:
        message = str(exc) if isinstance(exc, EpidemiologyValidationError) else 'Output directory exists; choose a new directory.'
        print(f'Characterization stopped: {message}', file=sys.stderr)
        return 1
    except Exception as exc:
        print(f'Characterization failed ({type(exc).__name__}); inspect locally; completion not certified.', file=sys.stderr)
        return 1
    print(json.dumps({'output_dir': manifest['output_dir'], 'controls': manifest['controls']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
