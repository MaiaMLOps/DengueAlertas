"""Publish a new local preliminary product, preserving the source and provenance."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.sources import sha256
from src.pilot_selection.config import PILOTS
from src.pilot_selection.reporting import git_state
from src.pilot_selection.subset import panel_profile

from .config import DEFAULT_INPUT, ROOT, method_contract
from .labels import label_weeks
from .validation import require, validate_product


def export_labels(output_dir: Path, source: Path = DEFAULT_INPUT, *, root: Path = ROOT) -> dict:
    root = root.resolve()
    source, output_dir = (root / source).resolve(), (root / output_dir).resolve()
    allowed = root / "data/processed/epidemiology"
    require(source.is_relative_to(root) and source.suffix == ".parquet", "Input must be a local repository Parquet.")
    require(output_dir != allowed and output_dir.is_relative_to(allowed),
            "Choose a new child directory inside data/processed/epidemiology.")
    if output_dir.exists():
        raise FileExistsError("Output directory exists; choose a new preliminary run directory.")
    require(source.is_file(), "Pilot source Parquet is unavailable.")
    source_hash = sha256(source)
    table = pq.read_table(source)
    source_profile = panel_profile(table)
    require(source_profile["municipality_codes"] == sorted(pilot.code for pilot in PILOTS)
            and source_profile["rows"] == 5952, "Expected the existing six-pilot panel with 5952 rows.")
    product, diagnostics = label_weeks(table)
    provenance = {"source": source.relative_to(root).as_posix(), "source_sha256": source_hash,
                  "method": method_contract(), "status": "preliminary_for_team_review"}
    metadata = dict(product.schema.metadata or {})
    metadata[b"denguealertas.derivation"] = json.dumps(provenance, sort_keys=True).encode("utf-8")
    product = product.replace_schema_metadata(metadata)
    sink = pa.BufferOutputStream()
    pq.write_table(product, sink, compression="zstd")
    content = sink.getvalue().to_pybytes()
    restored = pq.read_table(pa.BufferReader(content))
    require(restored.equals(product, check_metadata=True), "Encoded product failed exact readback.")
    require(sha256(source) == source_hash, "Source changed; publication cancelled.")
    code_paths = sorted((root / "src/epidemiology").glob("*.py"))
    manifest = {
        "schema_version": 1, "status": "passed", "product_status": "preliminary_for_team_review",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "method": method_contract(),
        "source": {"path": source.relative_to(root).as_posix(), "sha256": source_hash,
                   "profile": source_profile, "metadata_sha256": {
                       key.decode("utf-8"): hashlib.sha256(value).hexdigest()
                       for key, value in (table.schema.metadata or {}).items()}},
        "git": git_state(root),
        "code_sha256": {path.relative_to(root).as_posix(): sha256(path) for path in code_paths},
        "environment": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("pandas", "numpy", "pyarrow")}},
        "diagnostics": diagnostics,
        "output": {"path": (output_dir / "pilot_weekly_labels.parquet").relative_to(root).as_posix(),
                   "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                   "schema": {field.name: str(field.type) for field in product.schema}},
        "controls": {"source_sha256_unchanged": True, "encoded_readback_equal": True,
                     "saved_readback_equal": True, "original_columns_preserved": True},
    }
    # Prepare everything before creating a new directory. The manifest is the final
    # completion marker; an interrupted directory without it is never reused.
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    output = output_dir / "pilot_weekly_labels.parquet"
    with output.open("xb") as stream:
        stream.write(content)
    saved = pq.read_table(output)
    require(saved.equals(product, check_metadata=True) and sha256(output) == manifest["output"]["sha256"],
            "Saved product failed verification; no completion manifest published.")
    validate_product(table, saved)
    require(sha256(source) == source_hash, "Source changed; no completion manifest published.")
    with (output_dir / "run_manifest.json").open("xb") as stream:
        stream.write(manifest_bytes)
    return manifest
