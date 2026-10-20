"""Synthetic fixtures only: no local health sources are read by unit tests."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.calendar import build_calendar
from src.data.config import COVARIATE_COLUMNS, SOURCE_FIELDS


@pytest.fixture
def covariates():
    calendar = build_calendar(["05001"])
    return pd.DataFrame({
        "municipality_code": calendar["municipality_code"],
        "municipality": "Synthetic municipality", "week_start": calendar["week_start"],
        "population": 1000, "area_km2": 10.0, "population_density_km2": 100.0,
        "temperature_c": 22.0, "precipitation_mm": 0.0, "relative_humidity_pct": 80.0,
    })[COVARIATE_COLUMNS]


def source_frame(year: int, specifications: list[dict]) -> pd.DataFrame:
    rows = []
    for index, spec in enumerate(specifications):
        row = {name: "" for name in SOURCE_FIELDS}
        row.update({
            "CONSECUTIVE": f"SYNTHETIC-{year}-{index}", "COD_EVE": "210",
            "FEC_NOT": "10/01/2013 12:00:00 a. m.", "SEMANA": "1", "ANO": str(year),
            "COD_DPTO_O": "05", "COD_MUN_O": "001", "INI_SIN": "01/01/2007 12:00:00 a. m.",
            "TIP_CAS": "3", "confirmados": "1", "Estado_final_de_caso": "3",
            "nom_est_f_caso": "Synthetic final classification",
        })
        row.update(spec)
        rows.append({"_year_nominal": year, "_source_file": f"Datos_{year}_210.xlsx", "_excel_row": index + 2, **row})
    return pd.DataFrame(rows)


@pytest.fixture
def write_source(tmp_path):
    def write(year: int, specifications: list[dict], directory: Path | None = None):
        folder = directory or tmp_path
        folder.mkdir(parents=True, exist_ok=True)
        frame = source_frame(year, specifications)
        filename = f"sivigila_210_{year}.parquet"
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), folder / filename, compression="zstd")
        return {"year": year, "file": filename, "source_file": f"Datos_{year}_210.xlsx", "rows": len(frame)}
    return write
