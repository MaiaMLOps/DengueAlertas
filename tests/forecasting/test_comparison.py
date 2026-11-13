"""Unit tests of comparison.py (U5.4) beyond the frozen F5a acceptance: tie-break by pre-registered position,
choices against an independent computation, levels with 24 cells and clear errors."""

import json

import numpy as np
import pandas as pd
import pytest

from src.epidemiology.config import EpidemiologyValidationError
from src.forecasting.comparison import (CONSISTENT, NONE, TENDENCY, chance_probability, combined_level, count_wins,
                                        discordant_wins, event_level, out_of_year_scores, weekly_level,
                                        win_threshold)

ALPHA = 0.05
YEARS = (2019, 2020, 2021, 2022)


def _table(values, codes, names, orders=None):
    """values[(code, name, year)] = mejora_b2; logloss_b2 = 0.5; rows in the given name order."""
    orders = orders or {name: i for i, name in enumerate(names)}
    rows = [{"municipality_code": code, "familia": "logistica", "candidato": name, "orden": orders[name],
             "parametros": json.dumps({"C": orders[name]}), "anio": year, "logloss": 0.5 * (1 - values[(code, name, year)]),
             "logloss_b2": 0.5, "mejora_b2": values[(code, name, year)], "n_filas": 52}
            for code in codes for name in names for year in YEARS]
    return pd.DataFrame(rows)


def test_exact_tie_goes_to_the_smallest_pre_registered_position():
    codes, names = ("A1", "A2"), ("late", "early")
    values = {(c, n, y): 0.1 for c in codes for n in names for y in YEARS}
    scores = out_of_year_scores(_table(values, codes, names, orders={"late": 1, "early": 0}), familia="logistica")
    assert set(scores["candidato_comun"]) == {"early"} and set(scores["candidato_propio"]) == {"early"}


def test_choices_match_an_independent_computation():
    rng = np.random.default_rng(3)
    codes, names = ("A1", "A2", "A3"), ("c0", "c1", "c2", "c3")
    values = {(c, n, y): float(rng.uniform(-0.3, 0.3)) for c in codes for n in names for y in YEARS}
    scores = out_of_year_scores(_table(values, codes, names), familia="logistica").set_index(
        ["municipality_code", "anio"])
    for year in YEARS:
        others = [y for y in YEARS if y != year]
        common = max(names, key=lambda n: np.mean([np.mean([values[(c, n, y)] for y in others]) for c in codes]))
        for code in codes:
            own = max(names, key=lambda n: np.mean([values[(code, n, y)] for y in others]))
            cell = scores.loc[(code, year)]
            assert cell["candidato_comun"] == common and cell["candidato_propio"] == own
            assert cell["mejora_comun"] == values[(code, common, year)]
            assert cell["logloss_propio"] == 0.5 * (1 - values[(code, own, year)])


@pytest.mark.parametrize(("wins", "expected"), [(17, CONSISTENT), (16, TENDENCY), (13, TENDENCY), (12, NONE)])
def test_weekly_levels_with_24_cells(wins, expected):
    assert win_threshold(24, alpha=ALPHA) == 17
    assert weekly_level(True, wins, 24, alpha=ALPHA) == expected


def test_probability_edges():
    assert chance_probability(0, 7) == 1.0
    assert win_threshold(4, alpha=0.01) is None
    assert event_level(0, 0, alpha=ALPHA) == NONE


@pytest.mark.parametrize(("call", "message"), [
    (lambda: count_wins([0.1, 0.2], [0.1]), "same length"),
    (lambda: count_wins([0.1, np.nan], [0.1, 0.2]), "finite"),
    (lambda: discordant_wins([1, 0], [True, False]), "booleans"),
    (lambda: discordant_wins([True], [True, False]), "same onsets"),
    (lambda: chance_probability(5, 4), "from 0 to n"),
    (lambda: win_threshold(0, alpha=ALPHA), "positive integer"),
    (lambda: win_threshold(18, alpha=1.0), "alpha"),
    (lambda: weekly_level(True, 19, 18, alpha=ALPHA), "from 0 to n"),
    (lambda: weekly_level(1, 10, 18, alpha=ALPHA), "boolean"),
    (lambda: event_level(4, 3, alpha=ALPHA), "from 0 to d"),
    (lambda: combined_level(CONSISTENT, "mejora"), "Unknown level"),
])
def test_invalid_inputs_fail_clearly(call, message):
    with pytest.raises(EpidemiologyValidationError, match=message):
        call()


def test_invalid_tables_fail_clearly():
    codes, names = ("A1", "A2"), ("c0", "c1")
    table = _table({(c, n, y): 0.1 for c in codes for n in names for y in YEARS}, codes, names)
    with pytest.raises(EpidemiologyValidationError, match="not in the score table"):
        out_of_year_scores(table, familia="lightgbm")
    missing_year = table[~((table["municipality_code"] == "A2") & (table["anio"] == 2020))]
    with pytest.raises(EpidemiologyValidationError, match="years"):
        out_of_year_scores(missing_year, familia="logistica")
    with pytest.raises(EpidemiologyValidationError, match="lacks columns"):
        out_of_year_scores(table.drop(columns="orden"), familia="logistica")
