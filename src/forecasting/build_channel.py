"""Build the channel product data/processed/epidemiology/pilot_weekly_channel.parquet (contrato §10.2).

Flow: load configs/forecasting.yaml (section `canal` and paths `datos`) -> read the six-pilot
panel and hash it -> `channel_epico.build_channel_table` (pure, no I/O) -> encode and verify
the Parquet -> write it without overwriting -> write the manifest last as completion marker.

The dependency runs from forecasting to epidemiology only; `channel_epico` never reads the
configuration. `first_year` is the D1 constant `src.epidemiology.config.FIRST_LABEL_YEAR`.

Usage:
    python -m src.forecasting.build_channel [--input P] [--output P] [--manifest P]
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.sources import sha256
from src.epidemiology.channel_epico import CHANNEL_SCHEMA, ROUND_DECIMALS, ZONES, build_channel_table
from src.epidemiology.config import FIRST_LABEL_YEAR, EpidemiologyValidationError
from src.epidemiology.validation import require
from src.pilot_selection.config import PILOTS

from .config import DEFAULT_CONFIG_PATH, ROOT, load_config

MANIFEST_NAME = "pilot_weekly_channel_manifest.json"
EPICO_REFERENCE = "epiverse-trace/epiCo commit 39a45ef, endemic_channel, method = geometric"
CODE_FILES = [
    "src/epidemiology/channel_epico.py",
    "src/epidemiology/ewt.py",
    "src/epidemiology/validation.py",
    "src/epidemiology/config.py",
    "src/forecasting/build_channel.py",
    "src/forecasting/config.py",
]
DICTIONARY = {
    "epico_central": "epiCo central line (geometric, shift): round(exp(mean(log(x + shift))) - shift, 5) over the "
                     "same ordinal in the five prior years; null when ineligible.",
    "epico_upper": "epiCo upper limit U1e = central * exp(|qt(alpha; n - 1)| * gsd / sqrt(n)), 0 if central = 0; "
                   "not rounded; null when ineligible.",
    "target_u1e": "1 iff confirmed_records_screening > epico_upper; 0 otherwise if eligible; null when ineligible.",
    "channel_zone": "seguridad if C <= central; alerta if central < C <= U1e; epidemia if C > U1e; null when ineligible.",
    "channel_eligible": "True only for complete weeks from 2012 with five valid historical references (D1).",
}
WARNINGS = [
    "Retrospective weekly state against the epiCo endemic channel, not an officially confirmed outbreak.",
    "Final event-210 confirmed records; historical availability is not established; zeros are not verified "
    "negative notifications.",
    "Transcription of the epiCo formula verified in Python; epiCo was not executed in R.",
    "No forecast, horizon or supervised features are produced; the legacy ewt and excess_label are not included.",
]


def _git(*args: str) -> str:
    result = subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def code_state() -> dict:
    """Commit, branch and whether src/ had uncommitted or untracked changes at build time."""
    try:
        return {"commit": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"),
                "src_dirty": bool(_git("status", "--porcelain=v1", "--untracked-files=all", "--", "src"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "src_dirty": None, "available": False}


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def _counts(product: pa.Table) -> dict:
    rows = product.select(["municipality_code", "channel_eligible", "target_u1e", "channel_zone"]).to_pylist()
    by_municipality = {}
    for row in rows:
        summary = by_municipality.setdefault(row["municipality_code"], {
            "rows": 0, "eligible": 0, "positives": 0, "zones": dict.fromkeys(ZONES, 0)})
        summary["rows"] += 1
        if row["channel_eligible"]:
            summary["eligible"] += 1
            summary["positives"] += row["target_u1e"]
            summary["zones"][row["channel_zone"]] += 1
    eligible = sum(row["channel_eligible"] for row in rows)
    zones = Counter(row["channel_zone"] for row in rows if row["channel_eligible"])
    return {"rows": product.num_rows, "columns": product.num_columns, "eligible": eligible,
            "ineligible": product.num_rows - eligible,
            "positives": sum(row["target_u1e"] for row in rows if row["channel_eligible"]),
            "zones": {zone: zones.get(zone, 0) for zone in ZONES},
            "by_municipality": dict(sorted(by_municipality.items()))}


def export_channel(source: Path, output: Path, manifest_path: Path, *, n_years: int, shift: float,
                   alpha: float, first_year: int) -> dict:
    """Write the channel product and its manifest; never overwrite. Returns the manifest."""
    source, output, manifest_path = Path(source).resolve(), Path(output).resolve(), Path(manifest_path).resolve()
    require(source.is_file() and source.suffix == ".parquet", "Input panel Parquet is unavailable.")
    require(output.suffix == ".parquet" and manifest_path.suffix == ".json", "Unexpected output suffixes.")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Channel product or manifest already exists; nothing was overwritten.")
    source_hash = sha256(source)
    table = pq.read_table(source)
    parameters = {"n_years": n_years, "shift": shift, "alpha": alpha, "first_year": first_year,
                  "round_decimals": ROUND_DECIMALS, "reference": EPICO_REFERENCE,
                  "formula": "central = round(exp(mean(log(x + shift))) - shift, 5); "
                             "gsd = round(sd(log(x + shift)), 5); "
                             "U1e = central * exp(|qt(alpha; n - 1)| * gsd / sqrt(n)), 0 if central = 0"}
    product = build_channel_table(table, n_years=n_years, shift=shift, alpha=alpha, first_year=first_year)

    original = table.schema.metadata or {}
    dictionary = json.loads(original.get(b"denguealertas.data_dictionary", b"{}"))
    dictionary.update(DICTIONARY)
    metadata = {
        b"denguealertas.data_dictionary": json.dumps(dictionary, ensure_ascii=False).encode("utf-8"),
        b"denguealertas.channel": json.dumps(parameters, sort_keys=True).encode("utf-8"),
        b"denguealertas.derivation": json.dumps({"source": _relative(source), "source_sha256": source_hash},
                                                sort_keys=True).encode("utf-8"),
        b"denguealertas.warnings": json.dumps(WARNINGS).encode("utf-8"),
    }
    if b"denguealertas.rule" in original:
        metadata[b"denguealertas.rule"] = original[b"denguealertas.rule"]
    product = product.replace_schema_metadata(metadata)

    sink = pa.BufferOutputStream()
    pq.write_table(product, sink, compression="zstd")
    content = sink.getvalue().to_pybytes()
    require(pq.read_table(pa.BufferReader(content)).equals(product, check_metadata=True),
            "Encoded product failed exact readback.")
    product_hash = hashlib.sha256(content).hexdigest()
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters,
        "source": {"path": _relative(source), "sha256": source_hash, "rows": table.num_rows},
        "config": {"path": _relative(DEFAULT_CONFIG_PATH), "sha256": sha256(DEFAULT_CONFIG_PATH)},
        "git": code_state(),
        "code_sha256": {name: sha256(ROOT / name) for name in CODE_FILES},
        "environment": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow", "scipy")}},
        "counts": _counts(product),
        "output": {"path": _relative(output), "bytes": len(content), "sha256": product_hash,
                   "schema": {field.name: str(field.type) for field in product.schema}},
        "controls": {"source_sha256_unchanged": True, "encoded_readback_equal": True,
                     "saved_readback_equal": True, "channel_columns": CHANNEL_SCHEMA.names},
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(content)
    require(sha256(output) == product_hash and pq.read_table(output).equals(product, check_metadata=True),
            "Saved product failed verification; no manifest written.")
    require(sha256(source) == source_hash, "Input panel changed during the build; no manifest written.")
    with manifest_path.open("xb") as stream:
        stream.write(manifest_bytes)
    return manifest


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    default_output = ROOT / config.datos.canal
    parser = argparse.ArgumentParser(description="Build the epiCo channel product for the six pilots.")
    parser.add_argument("--input", type=Path, default=ROOT / config.datos.panel)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--manifest", type=Path, default=None,
                        help=f"Defaults to {MANIFEST_NAME} next to the output.")
    args = parser.parse_args(argv)
    manifest_path = args.manifest if args.manifest is not None else args.output.parent / MANIFEST_NAME
    try:
        require(args.input.is_file(), "Input panel Parquet is unavailable.")
        codes = sorted(set(pq.read_table(args.input, columns=["municipality_code"])
                           .column("municipality_code").to_pylist()))
        require(codes == sorted(pilot.code for pilot in PILOTS),
                "Input is not the six-pilot panel (pilot_municipalities_weekly.parquet).")
        manifest = export_channel(args.input, args.output, manifest_path,
                                  n_years=config.canal.n, shift=config.canal.desplazamiento,
                                  alpha=config.canal.cuantil_t, first_year=FIRST_LABEL_YEAR)
    except (EpidemiologyValidationError, FileExistsError) as exc:
        print(f"Channel build stopped: {exc}", file=sys.stderr)
        return 1
    counts = manifest["counts"]
    print(f"Channel product: {counts['rows']} rows, {counts['columns']} columns; {counts['eligible']} eligible, "
          f"{counts['positives']} with target_u1e = 1.")
    print(f"Output: {manifest['output']['path']} (sha256 {manifest['output']['sha256']})")
    print(f"Manifest: {_relative(Path(manifest_path).resolve())}; commit {manifest['git']['commit']}, "
          f"src_dirty={manifest['git']['src_dirty']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
