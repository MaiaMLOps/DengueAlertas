"""Local publication and provenance, with exclusive output directories."""

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .analysis import build_tables, complete_weeks, period_frame, prepare_panel, require
from .config import COUNT, KEYS, PERIODS, PILOTS, RunConfig, SELECTION_DATE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_profile(frame: pd.DataFrame) -> dict:
    """Aggregate controls only; never serialize observed weekly rows."""
    profile = {"rows": len(frame), "columns": len(frame.columns),
               "types": {c: str(t) for c, t in frame.dtypes.items()},
               "nulls": {c: int(n) for c, n in frame.isna().sum().items()}}
    if "municipality_code" in frame:
        profile["municipalities"] = int(frame.municipality_code.nunique())
    if set(KEYS).issubset(frame):
        profile["duplicate_keys"] = int(frame.duplicated(KEYS).sum())
        profile["week_start_min"] = str(pd.Timestamp(frame.week_start.min()).date())
        profile["week_start_max"] = str(pd.Timestamp(frame.week_start.max()).date())
    if COUNT in frame:
        profile["total_records"] = int(frame[COUNT].sum())
        profile["active_positions"] = int(frame[COUNT].gt(0).sum())
    profile["numeric_ranges"] = {}
    for column in frame.select_dtypes(include="number"):
        values = frame[column].dropna()
        if len(values):
            require(np.isfinite(values.astype(float)).all(), "Nonfinite numeric values detected.")
            profile["numeric_ranges"][column] = [float(values.min()), float(values.max())]
    return profile


def git_state(root: Path) -> dict:
    def git(*args):
        result = subprocess.run(["git", "--no-optional-locks", *args], cwd=root,
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                "dirty": bool(git("status", "--porcelain=v1", "--untracked-files=all"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None, "available": False}


def csv_bytes(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g", quoting=csv.QUOTE_NONNUMERIC)
    # Preserve codes on a typed read-back. Generic CSV readers may still infer numbers.
    if "municipality_code" in frame:
        reread = pd.read_csv(io.StringIO(text), dtype={"municipality_code": str})
        require(reread.municipality_code.tolist() == frame.municipality_code.tolist(),
                "CSV round-trip changes DIVIPOLA codes.")
    return text.encode("utf-8")


def run_analysis(configuration: RunConfig) -> dict:
    cfg = configuration.resolved()
    require(cfg.input.is_file(), "Input Parquet is unavailable.")
    input_hash = sha256(cfg.input)
    table = pq.read_table(cfg.input)
    source = prepare_panel(table.to_pandas())
    # Contract of this retrospective product, derived from the authorized calendar.
    expected = pd.date_range("2007-01-01", "2025-12-31", freq="W-MON")
    require(source.week_start.drop_duplicates().tolist() == expected.tolist(),
            "Input does not cover the authorized 2007-2025 weekly calendar.")
    expected_days = ((source.week_end_nominal.clip(upper=pd.Timestamp("2025-12-31"))
                      - source.week_start).dt.days + 1)
    require(source.n_days.eq(expected_days).all(), "Input duration differs from the authorized window.")
    complete = complete_weeks(source)
    historical = period_frame(complete, PERIODS["historical"])
    tables = build_tables(complete)
    files = {name + ".csv": csv_bytes(frame) for name, frame in tables.items()}
    plot_checks = []
    if cfg.figures:
        from .plots import figure_bytes
        figures, plot_checks = figure_bytes(complete)
        files.update(figures)
    require(sha256(cfg.input) == input_hash, "Input changed during analysis; nothing published.")
    code_paths = sorted((cfg.root / "src/pilot_selection").glob("*.py"))
    dependency_paths = [cfg.root / "requirements" / name
                        for name in ("data.txt", "data-test.txt", "pilot-selection.txt")]
    supporting_paths = sorted((cfg.root / "tests/pilot_selection").glob("*.py")) + [
        cfg.root / "docs/pilot_selection.md", cfg.root / ".gitignore"]
    manifest = {
        "schema_version": 1, "status": "passed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "descriptive evidence; team-adopted selection, no automatic selection",
        "input": {"path": cfg.input.relative_to(cfg.root).as_posix(), "sha256": input_hash,
                  "arrow_schema": {field.name: str(field.type) for field in table.schema}},
        "output_dir": cfg.output_dir.relative_to(cfg.root).as_posix(), "git": git_state(cfg.root),
        "code_sha256": {p.relative_to(cfg.root).as_posix(): sha256(p) for p in code_paths},
        "dependency_sha256": {p.relative_to(cfg.root).as_posix(): sha256(p) for p in dependency_paths if p.is_file()},
        "supporting_sha256": {p.relative_to(cfg.root).as_posix(): sha256(p) for p in supporting_paths if p.is_file()},
        "environment": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("pandas", "numpy", "pyarrow")}},
        "configuration": {"periods_by_week_start_year": PERIODS, "figures": cfg.figures,
                          "pilots": [vars(pilot) for pilot in PILOTS],
                          "selection_authority": "team decision communicated by user",
                          "selection_date": SELECTION_DATE},
        "rules": {"partial_weeks": "excluded", "annual_group": "year of week_start",
                  "active_week": "confirmed_records_screening > 0", "weekly_median": "includes zeros",
                  "zero_run": "consecutive observed complete weeks, crosses years, clipped to period",
                  "quantiles": "linear interpolation", "annual_ties": "earliest; all tied years retained",
                  "zero_total_concentration": "undefined (empty CSV field)",
                  "comparison_15": "5 highest; 5 nearest median; 5 lowest; DIVIPOLA ascending breaks ties"},
        "warnings": [
            "Retrospective eligible records, not certified unique people or episodes; event 210 only.",
            "Zero records do not establish absence of infection or verified negative notification.",
            "Persistence does not certify surveillance quality or prospective information availability.",
            "Historical subset uses week_start: 2022-12-26 includes the calendar day 2023-01-01.",
            "Both the initial candidates and the team selection use retrospective information.",
            "No endemic channel, epidemiological label, supervised data or model is generated.",
            "The documented 2013 notification/symptom anomaly is not corrected by this analysis.",
        ],
        "controls": {"source": frame_profile(source), "complete": frame_profile(complete),
                     "historical": frame_profile(historical),
                     "excluded_partial": {"rows": int(source.partial_week.sum()),
                                          "records": int(source.loc[source.partial_week, COUNT].sum())},
                     "weeks_per_municipality": {
                         label: sorted(int(v) for v in frame.groupby("municipality_code").size().unique())
                         for label, frame in (("full", complete), ("historical", historical))},
                     "plots": plot_checks, "input_sha256_unchanged": True},
        "artifacts": {name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
                      for name, content in files.items()},
        "table_profiles": {name + ".csv": frame_profile(frame) for name, frame in tables.items()},
    }
    if cfg.figures:
        manifest["environment"]["matplotlib"] = importlib.metadata.version("matplotlib")
        manifest["environment"]["Pillow"] = importlib.metadata.version("Pillow")
    # All calculations and serialization finish before creating a new destination.
    json.dumps(manifest, allow_nan=False)
    cfg.output_dir.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(exist_ok=False)
    for name, content in files.items():
        with (cfg.output_dir / name).open("xb") as stream:
            stream.write(content)
        require(sha256(cfg.output_dir / name) == manifest["artifacts"][name]["sha256"],
                "An exported artifact failed checksum verification.")
    require(sha256(cfg.input) == input_hash, "Input changed before completion.")
    # Completion marker is last. A directory without it is incomplete and never overwritten.
    with (cfg.output_dir / "run_manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest
