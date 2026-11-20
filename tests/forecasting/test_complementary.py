"""Summaries of the complementary logistic analyses on a synthetic single table."""

import json
from datetime import date

import pandas as pd
import pytest

from src.epidemiology.config import EpidemiologyValidationError
from src.forecasting.complementary import (
    candidate_c,
    common_by_year,
    common_choice,
    fixed_cells,
    grid_candidates,
    out_of_year_cells,
    paired_wins,
    summarize,
)


def _table() -> pd.DataFrame:
    # Two municipalities, three years, two values of C; logistica_01 is better in every cell except (A, 2021).
    rows = []
    for code in ("A", "B"):
        for year in (2019, 2020, 2021):
            for position, c in enumerate((1.0, 3.0)):
                loss = 0.40 if position == 0 else 0.30
                if code == "A" and year == 2021:
                    loss = 0.20 if position == 0 else 0.50
                rows.append({"municipality_code": code, "familia": "logistica", "candidato": f"logistica_{position:02d}",
                             "orden": position, "parametros": json.dumps({"C": c, "max_iter": 100}), "anio": year,
                             "logloss": loss, "logloss_b2": 0.40, "mejora_b2": 1 - loss / 0.40,
                             "brier": loss / 4, "brier_b2": 0.10, "n_filas": 52,
                             "primera_semana_objetivo": date(year, 1, 7), "ultima_semana_objetivo": date(year, 12, 30)})
    return pd.DataFrame(rows)


def test_grid_candidates_keep_the_order():
    names = [(c.name, c.params["C"]) for c in grid_candidates([0.1, 1.0, 3.0], max_iter=50)]
    assert names == [("logistica_00", 0.1), ("logistica_01", 1.0), ("logistica_02", 3.0)]
    with pytest.raises(EpidemiologyValidationError):
        grid_candidates([1.0, 0.5], max_iter=50)


def test_out_of_year_cells_and_summary():
    table = _table()
    assert candidate_c(table) == {"logistica_00": 1.0, "logistica_01": 3.0}
    assert common_choice(table) == "logistica_01"
    assert common_by_year(table) == {2019: "logistica_01", 2020: "logistica_01", 2021: "logistica_01"}
    cells = out_of_year_cells(table)
    assert len(cells) == 6 and set(cells["C"]) == {3.0}
    summary = summarize(cells)
    assert summary["victorias_b2"] == 5          # (A, 2021) loses to B2
    assert summary["logloss_medio"] == pytest.approx((5 * 0.30 + 0.50) / 6)


def test_fixed_cells_and_paired_wins():
    table = _table()
    low, high = fixed_cells(table, "logistica_00"), fixed_cells(table, "logistica_01")
    assert list(low["C"].unique()) == [1.0]
    assert paired_wins(high, low) == {"ganadas": 5, "perdidas": 1, "empates": 0}
    with pytest.raises(EpidemiologyValidationError):
        fixed_cells(table, "logistica_09")
