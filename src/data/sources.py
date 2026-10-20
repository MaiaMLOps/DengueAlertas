"""Immutable source inventory and per-file aggregation; never export microdata."""

import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import COUNT, END, KEYS, SOURCE_FIELDS, START, YEARS
from .normalization import municipality_codes, parse_source_dates, week_start
from .validation import frame_profile, require


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_sources(directory: Path) -> tuple[list[dict], dict]:
    require(directory.is_dir(), "SIVIGILA input directory is not accessible.")
    manifest_path = directory / "cache_manifest.csv"
    require(manifest_path.is_file(), "Missing cache_manifest.csv.")
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    fields = ["year_nominal", "source_file", "source_rows", "cache_file", "cache_rows", "cache_size_mb", "elapsed_seconds"]
    require(list(manifest.columns) == fields, "Unexpected manifest schema.")
    require(len(manifest) == len(YEARS) and manifest["year_nominal"].is_unique, "Manifest must contain 19 distinct years.")
    require(set(manifest["year_nominal"]) == {str(y) for y in YEARS}, "Manifest years differ from 2007-2025.")
    expected_names = {f"sivigila_210_{y}.parquet" for y in YEARS}
    require({p.name for p in directory.glob("*.parquet")} == expected_names, "Unexpected/missing annual Parquet files.")
    inventory = []
    for year in YEARS:
        entry = manifest.loc[manifest["year_nominal"].eq(str(year))].iloc[0]
        filename = f"sivigila_210_{year}.parquet"
        require(entry["cache_file"] == filename and entry["source_file"] == f"Datos_{year}_210.xlsx", "Manifest filename mismatch.")
        path = directory / filename
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        expected = ["_year_nominal", "_source_file", "_excel_row"] + SOURCE_FIELDS
        require(schema.names == expected, f"Unexpected source columns in nominal year {year}.")
        for name in expected:
            target = pa.int64() if name in ["_year_nominal", "_excel_row"] else pa.string()
            require(schema.field(name).type == target, f"Unexpected source type in nominal year {year} ({name}).")
        rows = parquet.metadata.num_rows
        require(str(rows) == entry["cache_rows"] == entry["source_rows"], f"Manifest row mismatch in nominal year {year}.")
        inventory.append({
            "year": year, "file": filename, "source_file": entry["source_file"],
            "rows": rows, "bytes": path.stat().st_size, "sha256": sha256(path),
            "schema": {f.name: str(f.type) for f in schema},
            "row_groups": parquet.metadata.num_row_groups,
            "compression": sorted({parquet.metadata.row_group(i).column(j).compression
                                   for i in range(parquet.metadata.num_row_groups)
                                   for j in range(parquet.metadata.num_columns)}),
        })
        parquet.close()
    return inventory, {"file": manifest_path.name, "rows": len(manifest), "sha256": sha256(manifest_path)}


def aggregate_epidemiology(directory: Path, inventory: list[dict], candidate_codes: set[str], report: dict) -> pd.DataFrame:
    """Read one annual file at a time; retain only weekly aggregates afterwards."""
    report.update({"files": [], "exclusions_sequential": {}, "input_rows": 0})
    parts = []
    totals: dict[str, int] = {}
    all_codes: set[str] = set()
    for item in inventory:
        year = item["year"]
        frame = pq.read_table(directory / item["file"]).to_pandas()
        item_report = {"year": year, "input_profile": frame_profile(frame)}
        report["files"].append(item_report)
        require(len(frame) == item["rows"], f"Read row count changed in year {year}.")
        require(frame["_year_nominal"].eq(year).all(), f"Year trace mismatch in {year}.")
        require(frame["_source_file"].eq(item["source_file"]).all(), f"Source trace mismatch in {year}.")
        require(frame["_excel_row"].notna().all() and frame["_excel_row"].is_unique, f"Invalid row trace in {year}.")
        codes = municipality_codes(frame["COD_DPTO_O"], frame["COD_MUN_O"])
        all_codes.update(codes.unique())
        require(frame["COD_EVE"].astype("string").fillna("").str.strip().eq("210").all(), f"Non-210 or missing event in {year}.")
        confirm = frame["confirmados"].astype("string").str.strip()
        state = frame["Estado_final_de_caso"].astype("string").str.strip()
        require(confirm.isin(["0", "1"]).all() and state.isin(["2", "3", "5"]).all(), f"Unknown case classification in {year}.")
        require(confirm.eq("1").eq(state.isin(["3", "5"])).all(), f"Confirmation/state mismatch in {year}.")
        symptoms, symptom_qc = parse_source_dates(frame["INI_SIN"])
        notification, notification_qc = parse_source_dates(frame["FEC_NOT"])
        for column, dates, qc in [("INI_SIN", symptoms, symptom_qc), ("FEC_NOT", notification, notification_qc)]:
            qc.update({"outside_nominal_year": int((dates.notna() & dates.dt.year.ne(year)).sum()),
                       "min": dates.min().isoformat() if dates.notna().any() else None,
                       "max": dates.max().isoformat() if dates.notna().any() else None})
            item_report[column] = qc
        item_report["notification_before_symptoms"] = int((notification < symptoms).sum())
        ids = frame["CONSECUTIVE"].astype("string").fillna("").str.strip()
        item_report["identifier_blank"] = int(ids.eq("").sum())
        item_report["identifier_duplicates_within_file"] = int(ids.duplicated().sum())
        # Observe duplicates only: no deletion or unapproved deduplication rule.
        item_report["deduplication_applied"] = False
        item_report["territorial_codes"] = int(codes.nunique())
        confirmed = confirm.eq("1")
        blank = frame["INI_SIN"].astype("string").fillna("").str.strip().eq("")
        valid = symptoms.ge(pd.Timestamp(START)) & symptoms.lt(pd.Timestamp(END) + pd.Timedelta(days=1))
        selected = codes.isin(candidate_codes)
        stages = {
            "not_confirmed": int((~confirmed).sum()),
            "confirmed_missing_symptom_date": int((confirmed & blank).sum()),
            "confirmed_unparseable_symptom_date": int((confirmed & ~blank & symptoms.isna()).sum()),
            "confirmed_outside_date_window": int((confirmed & symptoms.notna() & ~valid).sum()),
            "confirmed_valid_date_outside_candidates": int((confirmed & valid & ~selected).sum()),
            "included": int((confirmed & valid & selected).sum()),
        }
        require(sum(stages.values()) == len(frame), f"Exclusions do not reconcile in {year}.")
        item_report["exclusions_sequential"] = stages
        for name, n in stages.items():
            totals[name] = totals.get(name, 0) + n
        keep = confirmed & valid & selected
        eligible = pd.DataFrame({"municipality_code": codes.loc[keep], "week_start": week_start(symptoms.loc[keep])})
        counts = eligible.groupby(KEYS, sort=True).size().rename(COUNT).reset_index()
        require(int(counts[COUNT].sum()) == stages["included"], f"Aggregation lost records in {year}.")
        item_report["weekly_profile"] = frame_profile(counts, KEYS)
        parts.append(counts)
        report["input_rows"] += len(frame)
        del frame, eligible
    combined = pd.concat(parts, ignore_index=True)
    contributions = combined.groupby(KEYS).size()
    weekly = combined.groupby(KEYS, as_index=False, sort=True)[COUNT].sum()
    require(int(weekly[COUNT].sum()) == totals["included"], "Cross-file aggregation changed the record total.")
    report.update({
        "exclusions_sequential": totals, "observed_territorial_codes": len(all_codes),
        "candidate_codes_without_any_source_record": len(candidate_codes - all_codes),
        "weekly_keys_before_cross_file_aggregation": len(combined),
        "weekly_keys_after_cross_file_aggregation": len(weekly),
        "keys_with_multiple_nominal_files": int(contributions.gt(1).sum()),
        "weekly_profile": frame_profile(weekly, KEYS),
        "deduplication_applied": False, "cross_file_person_episode_uniqueness_verified": False,
    })
    return weekly
