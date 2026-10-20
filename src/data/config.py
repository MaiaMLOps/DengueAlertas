"""Paths are configurable; the authorized historical screening rule is fixed."""

from dataclasses import dataclass, field
from pathlib import Path


class DataValidationError(ValueError):
    """A validation failure whose message contains only safe aggregate details."""


START = "2007-01-01"
END = "2025-12-31"
YEARS = tuple(range(2007, 2026))
KEYS = ["municipality_code", "week_start"]
COUNT = "confirmed_records_screening"
COVARIATE_COLUMNS = [
    "municipality_code", "municipality", "week_start", "population", "area_km2",
    "population_density_km2", "temperature_c", "precipitation_mm", "relative_humidity_pct",
]
SOURCE_FIELDS = [
    "CONSECUTIVE", "COD_EVE", "FEC_NOT", "SEMANA", "ANO", "COD_DPTO_O", "COD_MUN_O",
    "INI_SIN", "TIP_CAS", "confirmados", "Estado_final_de_caso", "nom_est_f_caso",
]
EXPECTED = {
    "municipalities": 78,
    "weeks_per_municipality_min": 992,
    "weeks_per_municipality_max": 992,
    "rows": 77376,
    "eligible_records": 637271,
    "positions_with_records": 52831,
    "positions_without_records": 24545,
    "partial_positions": 78,
}


@dataclass(frozen=True)
class PipelineConfig:
    sivigila_dir: Path = Path("../Datos SIVIGILA")
    covariates: Path = Path("../Covariables/covariates_weekly_78_2007_2025.xlsx")
    output: Path = Path("data/processed/master_weekly_screening.parquet")
    report: Path = Path("reports/data_quality/integration_quality.json")
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    def resolved(self) -> "PipelineConfig":
        root = self.repo_root.resolve()

        def absolute(path: Path) -> Path:
            return (root / path).resolve()

        return PipelineConfig(*(absolute(p) for p in (
            self.sivigila_dir, self.covariates, self.output, self.report
        )), repo_root=root)

    def validate_destinations(self) -> None:
        for path, allowed, suffix in (
            (self.output, self.repo_root / "data/processed", ".parquet"),
            (self.report, self.repo_root / "reports/data_quality", ".json"),
        ):
            if not path.is_relative_to(allowed.resolve()) or path.suffix != suffix:
                raise ValueError(f"Output must be inside {allowed.relative_to(self.repo_root)} ({suffix}).")
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path.name}; choose a new output/report path.")
            if path == self.covariates or path.is_relative_to(self.sivigila_dir):
                raise ValueError("An output overlaps an input location.")
