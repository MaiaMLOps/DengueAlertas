"""Explicit team decisions and descriptive periods, separate from calculations."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COUNT = "confirmed_records_screening"
KEYS = ["municipality_code", "week_start"]
PERIODS = {"full": (2007, 2025), "historical": (2007, 2022)}
SELECTION_DATE = "2026-10-20"


class AnalysisValidationError(ValueError):
    """A validation message containing only safe aggregate or contract details."""


@dataclass(frozen=True)
class Municipality:
    code: str
    name: str
    color: str


# Adopted by the team; never derived from an indicator or ranking.
PILOTS = (
    Municipality("54001", "Cúcuta", "#18657A"),
    Municipality("08001", "Barranquilla", "#9A512F"),
    Municipality("41001", "Neiva", "#3E5E9B"),
    Municipality("50001", "Villavicencio", "#657943"),
    Municipality("76001", "Cali", "#77558D"),
    Municipality("08758", "Soledad", "#9C4F64"),
)
INDICATORS = (
    "total_records", "active_weeks", "active_pct", "weekly_median",
    "annual_active_median", "annual_active_min", "max_zero_run",
    "peak_concentration_pct",
)


@dataclass(frozen=True)
class RunConfig:
    input: Path = Path("data/processed/master_weekly_screening.parquet")
    output_dir: Path = Path("reports/pilot_selection/runs/exploration_v1")
    figures: bool = False
    root: Path = ROOT

    def resolved(self) -> "RunConfig":
        root = self.root.resolve()
        source = (root / self.input).resolve()
        output = (root / self.output_dir).resolve()
        if not source.is_relative_to(root) or source.suffix != ".parquet":
            raise AnalysisValidationError("Input must be a local repository Parquet.")
        allowed = [root / "reports/pilot_selection" / folder for folder in ("runs", "evidence")]
        if not any(output != folder and output.is_relative_to(folder) for folder in allowed):
            raise AnalysisValidationError("Choose a new child directory inside reports/pilot_selection/runs or evidence.")
        if output.exists():
            raise FileExistsError("Output directory exists; choose a new run directory.")
        return RunConfig(source, output, self.figures, root)
