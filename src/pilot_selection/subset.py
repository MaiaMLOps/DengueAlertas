"""Export the exact territorial subset of the master, including partial weeks.

Run from the repository root: python -m src.pilot_selection.subset
Arrow arrays are filtered and sorted directly; no pandas conversion or recasting.
"""

import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
import json
import os
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import config
from .analysis import require
from .reporting import sha256


def pilot_codes() -> tuple[str, ...]:
    """PILOTS is the sole source of the team's territorial configuration."""
    codes = tuple(pilot.code for pilot in config.PILOTS)
    require(len(codes) == 6 and len(set(codes)) == 6, "PILOTS must identify six distinct municipalities.")
    require(all(isinstance(code, str) and len(code) == 5 and code.isascii() and code.isdigit()
                for code in codes), "PILOTS must contain five-digit text codes.")
    return codes


def panel_profile(table: pa.Table) -> dict:
    """Validate the existing calendar and schema; report only aggregate controls."""
    required = set(config.KEYS + [config.COUNT, "partial_week", "n_days", "week_end_nominal",
                                 "week_end_effective", "no_eligible_records_observed"])
    require(table.num_columns == 15 and len(set(table.column_names)) == 15,
            "Expected fifteen distinct original columns.")
    require(required.issubset(table.column_names), "Missing required panel columns.")
    require(table.num_rows > 0, "Panel is empty.")
    nulls = {name: table[name].null_count for name in table.column_names}
    require(not any(nulls.values()), "Panel contains nulls; no values are imputed.")
    require(pa.types.is_string(table.schema.field("municipality_code").type),
            "DIVIPOLA must retain its original Arrow string type.")
    for name in ("week_start", "week_end_nominal", "week_end_effective"):
        require(pa.types.is_date32(table.schema.field(name).type), "Expected original Arrow date32 calendar.")
    for name in ("partial_week", "no_eligible_records_observed"):
        require(pa.types.is_boolean(table.schema.field(name).type), "Expected original boolean indicators.")
    require(pa.types.is_integer(table.schema.field(config.COUNT).type), "Counts must be integers.")
    require(pa.types.is_integer(table.schema.field("n_days").type), "n_days must be an integer.")
    codes = table["municipality_code"].to_pylist()
    require(all(len(code) == 5 and code.isascii() and code.isdigit() for code in codes),
            "Invalid five-digit DIVIPOLA codes.")
    starts = table["week_start"].to_pylist()
    keys = list(zip(codes, starts))
    require(len(set(keys)) == len(keys), "Duplicate municipality-week keys.")
    expected = []
    current = date(2007, 1, 1)
    end = date(2025, 12, 31)
    while current <= end:
        expected.append(current)
        current += timedelta(days=7)
    calendars = defaultdict(list)
    for code, start in keys:
        calendars[code].append(start)
    require(all(sorted(values) == expected for values in calendars.values()),
            "Municipalities must retain the full 2007-2025 weekly calendar, including its partial interval.")
    counts = table[config.COUNT].to_pylist()
    partial = table["partial_week"].to_pylist()
    days = table["n_days"].to_pylist()
    nominal = table["week_end_nominal"].to_pylist()
    effective = table["week_end_effective"].to_pylist()
    zero_flags = table["no_eligible_records_observed"].to_pylist()
    require(all(value >= 0 for value in counts), "Negative screening counts.")
    require(all(flag == (value == 0) for flag, value in zip(zero_flags, counts)), "Inconsistent zero flags.")
    require(all(nom == start + timedelta(days=6) and eff == min(nom, end)
                and duration == (eff - start).days + 1 and flag == (duration < 7)
                for start, nom, eff, duration, flag in zip(starts, nominal, effective, days, partial)),
            "Calendar durations or partial-week flags disagree with the master window.")
    numeric_ranges = {}
    for field in table.schema:
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            require(pc.all(pc.is_finite(table[field.name])).as_py(), "Nonfinite numeric values.")
            numeric_ranges[field.name] = pc.min_max(table[field.name]).as_py()
    return {
        "rows": table.num_rows, "columns": table.num_columns,
        "municipalities": len(calendars), "municipality_codes": sorted(calendars),
        "weeks_per_municipality": dict(sorted(Counter(codes).items())),
        "partial_positions": sum(partial), "duplicate_keys": 0, "nulls": nulls,
        "types": {field.name: str(field.type) for field in table.schema},
        "numeric_ranges": numeric_ranges, "week_start_min": str(min(starts)),
        "week_start_max": str(max(starts)), "week_end_effective_max": str(max(effective)),
        "total_records": sum(counts),
        "partial_records": sum(count for count, flag in zip(counts, partial) if flag),
    }


def filter_pilots(master: pa.Table) -> pa.Table:
    """Only an is-in territorial filter and a deterministic sort; all columns retained."""
    codes = pilot_codes()
    require("municipality_code" in master.column_names, "Missing territorial key.")
    require(pa.types.is_string(master.schema.field("municipality_code").type), "DIVIPOLA must be text.")
    subset = master.filter(pc.is_in(master["municipality_code"], value_set=pa.array(codes, type=pa.string())))
    require(set(subset["municipality_code"].to_pylist()) == set(codes), "Some configured pilots are missing.")
    subset = subset.sort_by([(key, "ascending") for key in config.KEYS])
    # Master-wide schema metadata must not silently describe this six-municipality subset.
    return subset.replace_schema_metadata(None)


def derived_metadata(master: pa.Table, source_path: str, source_hash: str,
                     source_profile: dict, subset_profile: dict) -> dict[bytes, bytes]:
    source_metadata = master.schema.metadata or {}
    metadata = {}
    # These two inspected definitions are independent of the number of municipalities.
    # Do not inherit pandas index metadata, old territorial warnings, or arbitrary master metadata.
    name = b"denguealertas.data_dictionary"
    if name in source_metadata:
        json.loads(source_metadata[name])
        metadata[name] = source_metadata[name]
    name = b"denguealertas.rule"
    if name in source_metadata:
        rule = json.loads(source_metadata[name])
        require(rule.get("event") == "210" and rule.get("confirmed") == "1",
                "Source metadata does not describe the authorized event-210 screening.")
        safe_keys = {"calendar", "confirmed", "date", "deduplication", "end_inclusive", "event",
                     "partial_week", "start_inclusive", "status", "territory"}
        metadata[name] = json.dumps({key: value for key, value in rule.items() if key in safe_keys},
                                    sort_keys=True).encode("utf-8")
    if b"denguealertas.data_dictionary" in metadata:
        require(set(json.loads(metadata[b"denguealertas.data_dictionary"])) == set(master.column_names),
                "Source dictionary disagrees with original columns.")
    warnings = [
        "Six team-selected pilots; the territorial filter is explicit, not an automatic scientific selection.",
        "Retrospective provisional event-210 screening; not a supervised dataset or epidemiological target.",
        "All weeks are retained, including the final three-day partial interval, without extrapolation.",
        "Counts are eligible records, not certified unique people or episodes; no deduplication is added.",
        "Zeros do not establish absence of infection or verified negative notification.",
        "Historical prospective availability and equivalence to the official INS calendar are not established.",
        "The documented 2013 notification/symptom anomaly is retained without correction.",
        "Pilot selection and the original candidate screening used retrospective information.",
    ]
    provenance = {
        "schema_version": 1, "operation": "territorial_filter_and_key_sort_only",
        "source": {"path": source_path, "sha256": source_hash, "profile": source_profile},
        "selection": {"configuration": "src/pilot_selection/config.py:PILOTS",
                      "codes": list(pilot_codes()), "selection_date": config.SELECTION_DATE,
                      "authority": "explicit team decision"},
        "subset": subset_profile,
        "preserved": {"columns_types_and_values": True, "partial_weeks": True,
                      "order": config.KEYS, "definitions": "unchanged from the source dictionary and screening rule"},
        "metadata_policy": "keep rule and dictionary; replace master-wide warnings; discard other master schema metadata",
        "software": {"pyarrow": pa.__version__},
    }
    for name, value in ((b"denguealertas.warnings", warnings), (b"denguealertas.derivation", provenance)):
        metadata[name] = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return metadata


def export_pilot_subset(source: Path = Path("data/processed/master_weekly_screening.parquet"),
                        output: Path = Path("data/processed/pilot_municipalities_weekly.parquet"),
                        *, root: Path = config.ROOT) -> dict:
    root = root.resolve()
    source, output = (root / source).resolve(), (root / output).resolve()
    require(source.is_relative_to(root) and source.suffix == ".parquet", "Source must be a local repository Parquet.")
    require(output.is_relative_to(root / "data/processed") and output.suffix == ".parquet",
            "Output must be a Parquet inside data/processed.")
    require(source != output, "Output cannot replace the source.")
    if output.exists():
        raise FileExistsError("Output exists; choose a new output filename. No overwrites are permitted.")
    require(source.is_file(), "Source Parquet is unavailable.")
    source_hash = sha256(source)
    master = pq.read_table(source)
    source_profile = panel_profile(master)
    subset = filter_pilots(master)
    profile = panel_profile(subset)
    require(profile["municipalities"] == len(pilot_codes()), "Unexpected derived territorial scope.")
    require(profile["partial_positions"] == len(pilot_codes()), "One partial interval per pilot is required.")
    require(subset.column_names == master.column_names, "Column order changed.")
    require(all(a.equals(b, check_metadata=True) for a, b in zip(subset.schema, master.schema)),
            "Original Arrow fields or their types changed.")
    subset = subset.replace_schema_metadata(derived_metadata(
        master, source.relative_to(root).as_posix(), source_hash, source_profile, profile))
    sink = pa.BufferOutputStream()
    pq.write_table(subset, sink, compression="zstd")
    content = sink.getvalue()
    restored = pq.read_table(pa.BufferReader(content))
    require(restored.equals(subset, check_metadata=True), "Encoded Parquet does not match the derived table.")
    require(restored.equals(filter_pilots(master), check_metadata=False), "Encoded subset differs from the master's rows.")
    require(sha256(source) == source_hash, "Source changed during preparation; export cancelled.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacing even a file created concurrently after the preflight check.
    # A failure may leave an incomplete file; no success is certified and it is never overwritten automatically.
    with output.open("xb") as stream:
        stream.write(content.to_pybytes())
        stream.flush()
        os.fsync(stream.fileno())
    reread = pq.read_table(output)
    require(reread.equals(subset, check_metadata=True), "Saved Parquet differs from the verified subset.")
    require(sha256(source) == source_hash, "Source changed before final verification.")
    return {"source": source.relative_to(root).as_posix(), "source_sha256": source_hash,
            "source_sha256_unchanged": True, "output": output.relative_to(root).as_posix(),
            "output_sha256": sha256(output), "bytes": output.stat().st_size,
            "profile": profile, "exact_subset_equal": True, "readback_equal": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Territorial-only pilot Parquet; retains partial weeks and all original columns.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/master_weekly_screening.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/pilot_municipalities_weekly.parquet"))
    args = parser.parse_args(argv)
    try:
        result = export_pilot_subset(args.input, args.output)
    except config.AnalysisValidationError as exc:
        print(f"Export stopped: {exc}", file=sys.stderr)
        return 1
    except FileExistsError:
        print("Export stopped: output exists; choose a new filename. Nothing was overwritten.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Export failed ({type(exc).__name__}); inspect locally. No completion is certified.", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
