"""Orchestrate local retrospective integration, traceability and safe outputs."""

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path, PurePath
import platform
import subprocess

import pandas as pd
import pyarrow as pa

from .artifacts import publish_panel, write_report
from .config import COUNT, END, KEYS, START, DataValidationError, PipelineConfig
from .sources import aggregate_epidemiology, inventory_sources, sha256
from .validation import acceptance_checks, frame_profile, integrate, require, validate_covariates

WARNINGS = [
    "Historical screening only: confirmed=1 is provisional and is not the final scientific outcome definition.",
    "Counts represent final retrospectively confirmed records, not information proven available at a historical prediction time.",
    "A zero means no eligible records observed in this extraction; negative notification and absence of infections are not verified.",
    "The 2013 notification/symptom anomaly is retained without correction or exclusion based on notification dates.",
    "No cross-file, person or episode deduplication is applied or certified.",
    "W-SUN Monday-Sunday has not been validated as equivalent to the official INS epidemiological calendar.",
    "n_days measures the authorized calendar-window intersection, not verified daily surveillance or raster coverage.",
    "The 78 candidate municipalities were selected retrospectively using the full 2007-2025 period; final municipalities remain undecided.",
    "Covariate completeness does not certify meteorological accuracy or historical operational availability.",
    "Manifest source row counts are checked against caches; original source Excel files are not re-audited.",
    "Population discontinuities, including Montelibano 2008, and fixed MGN-2025 areas are preserved.",
    "No labels, endemic channels, lags, predictive horizons, train/test splits or models are constructed.",
]
DICTIONARY = {
    "municipality_code": "Five-digit occurrence DIVIPOLA code; integration key, including leading zeros.",
    "municipality": "Descriptive name supplied by integrated covariates; never a join key.",
    "week_start": "Monday start of W-SUN interval; date, integration key.",
    "population": "Annual DANE population assigned by year of week_start; not certified INS at-risk population.",
    "area_km2": "Fixed municipal MGN-2025 area in square kilometres.",
    "population_density_km2": "Annual population / fixed area, inhabitants per square kilometre.",
    "temperature_c": "Municipal weekly ERA5-Land mean air temperature at 2 m, degrees Celsius.",
    "precipitation_mm": "Municipal mean CHIRPS accumulated precipitation over available interval days, millimetres.",
    "relative_humidity_pct": "Estimated relative humidity from ERA5-Land daily mean temperature/dew point, percent.",
    COUNT: "Count of retrospectively confirmed eligible source rows under the provisional historical screening rule; not unique people/episodes.",
    "no_eligible_records_observed": "True exactly when screening count is zero; does not assert verified negative notification.",
    "week_end_nominal": "Sunday nominal end, inclusive; can lie beyond the authorized window.",
    "week_end_effective": "Inclusive nominal end clipped to 2025-12-31.",
    "n_days": "Number of calendar days intersecting authorized window, inclusive; does not measure source completeness.",
    "partial_week": "True when n_days < 7; the last interval has three days and is retained without extrapolation.",
}


def relative(path: PurePath, root: PurePath, label: str = "input") -> str:
    """Describe inputs without private absolute paths, including other drives."""
    if path.drive.casefold() != root.drive.casefold():
        return f"external:{label}"
    try:
        return Path(os.path.relpath(path, root)).as_posix()
    except ValueError:
        return f"external:{label}"


def git_state(root: Path) -> dict:
    def read(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", "--no-optional-locks", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = read("status", "--porcelain", "--untracked-files=all")
    return {"commit": read("rev-parse", "HEAD"), "branch": read("branch", "--show-current"),
            "working_tree_dirty": bool(status) if status is not None else None}


def run_pipeline(config: PipelineConfig) -> dict:
    cfg = config.resolved()
    cfg.validate_destinations()
    report = {
        "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": {"status": "authorized_provisional_historical_screening", "event": "210",
                 "confirmed": "1", "territory": "COD_DPTO_O + COD_MUN_O (occurrence)",
                 "date": "INI_SIN", "start_inclusive": START, "end_inclusive": END,
                 "calendar": "Monday-Sunday W-SUN", "deduplication": "none", "partial_week": "retained, not extrapolated"},
        "warnings": WARNINGS, "data_dictionary": DICTIONARY,
    }
    stage = "initialization"
    try:
        report.update({
            "configuration": {name: relative(getattr(cfg, name), cfg.repo_root, label=name)
                              for name in ["sivigila_dir", "covariates", "output", "report"]},
            "configuration_path_convention": "Repository-relative paths; external:<input_name> aliases require explicit local CLI paths and are identified by source fingerprints.",
            "runtime": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ["pandas", "numpy", "pyarrow", "openpyxl"]}},
            "git": git_state(cfg.repo_root),
            "pipeline_source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        })
        stage = "inventory"
        inventory, manifest = inventory_sources(cfg.sivigila_dir)
        report["source_inventory"] = inventory
        report["manifest"] = manifest
        require(cfg.covariates.is_file(), "Covariate workbook is not accessible.")
        cov_hash = sha256(cfg.covariates)
        report["covariate_source"] = {"file": cfg.covariates.name, "bytes": cfg.covariates.stat().st_size, "sha256": cov_hash}
        stage = "covariates"
        workbook = pd.ExcelFile(cfg.covariates, engine="openpyxl")
        try:
            require(len(workbook.sheet_names) == 1, "Expected exactly one covariate worksheet.")
            frame = pd.read_excel(workbook, dtype={"municipality_code": "string"})
            report["covariates_before_validation"] = frame_profile(frame, KEYS) if all(c in frame for c in KEYS) else frame_profile(frame)
            covariates = validate_covariates(frame)
        finally:
            workbook.close()
        report["covariates_validated"] = frame_profile(covariates, KEYS)
        stage = "epidemiology"
        report["epidemiology"] = {}
        counts = aggregate_epidemiology(cfg.sivigila_dir, inventory, set(covariates["municipality_code"]), report["epidemiology"])
        stage = "integration"
        panel, report["merge"] = integrate(covariates, counts)
        report["panel"] = frame_profile(panel, KEYS)
        report["acceptance"] = acceptance_checks(panel)
        municipal = panel.groupby("municipality_code").agg(
            positions=(COUNT, "size"), eligible_records=(COUNT, "sum"),
            positions_without_records=("no_eligible_records_observed", "sum"), partial_positions=("partial_week", "sum"),
        )
        report["by_municipality"] = municipal.reset_index().to_dict(orient="records")
        report["discrepancies"] = [k for k, check in report["acceptance"].items() if not check["passed"]]
        if report["discrepancies"]:
            report["possible_origins_to_investigate"] = ["Changed sources or candidate keys", "Changed date representations or missing dates", "Calendar/key coverage differences"]
        require(not report["discrepancies"], "Acceptance values differ; no panel will be saved.")
        stage = "source_immutability"
        unchanged = all(sha256(cfg.sivigila_dir / item["file"]) == item["sha256"] for item in inventory)
        unchanged = unchanged and sha256(cfg.sivigila_dir / manifest["file"]) == manifest["sha256"] and sha256(cfg.covariates) == cov_hash
        report["source_hashes_unchanged"] = unchanged
        require(unchanged, "Input content changed during execution.")
        stage = "output"
        table = pa.Table.from_pandas(panel, preserve_index=False)
        for c in ["week_start", "week_end_nominal", "week_end_effective"]:
            index = table.schema.get_field_index(c)
            table = table.set_column(index, c, pa.array(panel[c].dt.date, type=pa.date32()))
        metadata = dict(table.schema.metadata or {})
        metadata[b"denguealertas.rule"] = json.dumps(report["rule"], sort_keys=True).encode()
        metadata[b"denguealertas.warnings"] = json.dumps(WARNINGS).encode()
        metadata[b"denguealertas.data_dictionary"] = json.dumps(DICTIONARY, ensure_ascii=False).encode("utf-8")
        table = table.replace_schema_metadata(metadata)
        publish_panel(table, cfg.output, cfg.report, report)
    except Exception as exc:
        # No raw exception text/traceback: third-party readers can expose source values.
        report.update({"status": "failed", "failed_stage": stage, "error_type": type(exc).__name__})
        if isinstance(exc, DataValidationError):
            report["validation_failure"] = str(exc)
        report["failure_guidance"] = "Inspect aggregate stage diagnostics and input schema/coverage; do not change scientific filters to force expected totals."
        # A failed publication is never represented as a completed output pair.
        report.pop("output", None)
        try:
            write_report(cfg.report, report)
        except Exception as report_exc:
            raise RuntimeError(
                f"Integration failed at {stage} ({type(exc).__name__}); "
                f"quality report could not be saved ({type(report_exc).__name__})."
            ) from None
        raise RuntimeError(f"Integration failed at {stage} ({type(exc).__name__}); inspect the aggregate quality report.") from None
    return report
