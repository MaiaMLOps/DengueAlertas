"""Opt-in release contract; no dependency on ignored exploratory directories."""

import json

import pyarrow.parquet as pq

from src.data.sources import sha256
from src.epidemiology.characterization_report import DEFAULT_PRODUCT, audit_numerics
from src.epidemiology.config import ROOT, method_contract
from src.epidemiology.validation import validate_product, validate_source


def test_consolidated_product_matches_approved_contract(labels_path, pilot_path):
    before = {path: sha256(path) for path in [labels_path, pilot_path]}
    manifest_path = labels_path.with_name('run_manifest.json')
    manifest_before = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    source, product = pq.read_table(pilot_path), pq.read_table(labels_path)
    validate_source(source)
    profile = validate_product(source, product)
    assert (ROOT / DEFAULT_PRODUCT).resolve() == labels_path
    assert product.shape == (5952, 18)
    assert profile['eligible'] == 4380 and profile['ineligible'] == 1572
    assert all(item['rows'] == 992 and item['eligible'] == 730 for item in profile['by_municipality'].values())
    labels = product['excess_label'].to_pylist()
    assert labels.count(1) == 2870 and labels.count(0) == 1510 and labels.count(None) == 1572
    assert manifest['status'] == 'passed' and manifest['method'] == method_contract()
    assert manifest['output']['sha256'] == before[labels_path] == manifest['promotion']['source_sha256']
    assert manifest['source']['sha256'] == before[pilot_path]
    assert manifest['promotion']['operation'] == 'exact_binary_copy'
    assert audit_numerics(source, product)['historical_references_checked'] == 21900
    assert product.select(source.column_names).equals(source, check_metadata=False)
    assert all(sha256(path) == value for path, value in before.items())
    assert sha256(manifest_path) == manifest_before
