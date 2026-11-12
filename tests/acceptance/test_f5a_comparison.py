"""F5a acceptance with synthetic inputs: D9 counting rules, out-of-year scores and the candidate list
(contrato §6.2, §8.2–§8.4 and §10.12, §10.17; especificación §6.F5a). Approved by Jesús on 2026-11-11
(groups A and B of U5.1).

Literal figures (contrato §8.2, docs/forecasting/verificacion/umbrales_d9.py), with α = comparacion.alfa:
- 13 of 18 (chance 0.048), 17 of 24 (chance 0.032), 4 of 4 = 0.0625;
- one-sided exact McNemar threshold by number of discordant onsets d: d < 5 impossible (None);
  5→5, 6→6, 7→7, 8→7, 9→8, 10→9.

Rules:
- a win is a strictly smaller loss on the same unit; a tie is not a win (§8.2);
- out-of-year scores (§8.3): for each municipality m and development year y, the common configuration is
  chosen with the other three years and the six municipalities (mean improvement over B2, first over
  years, then over municipalities), the own configuration with the other three years of m; both are
  scored in y. The year y never enters the choice made for y;
- levels (§8.4): weekly, "mejora consistente" if the global metric is favourable and wins reach the
  threshold, "tendencia favorable" if favourable and wins are a majority below the threshold, "sin mejora"
  otherwise; the conclusion with Brier and log-loss adopts the lower level; events, the same with the
  McNemar threshold over the discordant onsets.

API frozen here (`src.forecasting.comparison`):
- `win_threshold(n, *, alpha) -> int | None`, `chance_probability(wins, n) -> float`;
- `count_wins(losses, reference) -> int`, `discordant_wins(option_warned, reference_warned) -> (wins, d)`;
- `out_of_year_scores(table, *, familia) -> DataFrame`, one row per (municipality_code, anio), with columns
  candidato_comun, candidato_propio, logloss_comun, logloss_propio, mejora_comun, mejora_propia;
- `weekly_level(favorable, wins, n, *, alpha)`, `event_level(wins, d, *, alpha)`, `combined_level(a, b)`,
  returning "mejora consistente", "tendencia favorable" or "sin mejora".
And `src.forecasting.search.validate_score_table(table, *, municipalities, years)`, which fails when the
candidate list (name, position and parameters) differs between municipalities.

Single table minimum columns: municipality_code, familia, candidato, orden (position in the pre-registered
list, the tie-break of a choice), parametros (canonical JSON), anio, logloss, logloss_b2, mejora_b2 =
1 − logloss/logloss_b2, n_filas, primera_semana_objetivo, ultima_semana_objetivo.
"""

from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.acceptance

CODES = ("08001", "08758", "41001", "50001", "54001", "76001")
YEARS = (2019, 2020, 2021, 2022)
FAMILY = "logistica"
CANDIDATES = (("A", {"C": 0.1}), ("B", {"C": 1.0}), ("C", {"C": 3.0}))
CONSISTENT, TENDENCY, NONE = "mejora consistente", "tendencia favorable", "sin mejora"


def _alpha():
    from src.forecasting.config import load_config

    return load_config().comparacion.alfa


# --------------------------------------------------------------------------- synthetic single table
def _table(improvement, reference=0.5):
    """improvement(code, name, year) -> mejora_b2; logloss is derived from it with logloss_b2 = reference."""
    rows = []
    for code in CODES:
        for position, (name, params) in enumerate(CANDIDATES):
            for year in YEARS:
                gain = improvement(code, name, year)
                rows.append({
                    "municipality_code": code, "familia": FAMILY, "candidato": name, "orden": position,
                    "parametros": json.dumps(params, sort_keys=True), "anio": year,
                    "logloss": reference * (1 - gain), "logloss_b2": reference, "mejora_b2": gain,
                    "n_filas": 52, "primera_semana_objetivo": date(year, 1, 7),
                    "ultima_semana_objetivo": date(year, 12, 30),
                })
    return pd.DataFrame(rows)


def _designed(code, name, year):
    """A is steady; B is excellent only in 2020; C is the own best of 08001 outside 2020's B choice."""
    if name == "A":
        return 0.10
    if name == "B":
        return 0.99 if year == 2020 else -0.20
    return 0.15 if code == "08001" else 0.05


# --------------------------------------------------------------------------- thresholds (§8.2)
def test_alpha_is_contract_value():
    assert _alpha() == 0.05


@pytest.mark.parametrize(("n", "threshold", "chance"), [(18, 13, 0.048), (24, 17, 0.032)])
def test_cell_thresholds(n, threshold, chance):
    from src.forecasting.comparison import chance_probability, win_threshold

    alpha = _alpha()
    assert win_threshold(n, alpha=alpha) == threshold
    assert round(chance_probability(threshold, n), 3) == chance
    assert chance_probability(threshold, n) < alpha <= chance_probability(threshold - 1, n)


def test_mcnemar_thresholds():
    from src.forecasting.comparison import win_threshold

    alpha = _alpha()
    observed = {d: win_threshold(d, alpha=alpha) for d in range(1, 11)}
    assert observed == {1: None, 2: None, 3: None, 4: None, 5: 5, 6: 6, 7: 7, 8: 7, 9: 8, 10: 9}


def test_four_of_four_years():
    from src.forecasting.comparison import chance_probability

    assert chance_probability(4, 4) == 0.0625


# --------------------------------------------------------------------------- ties (§8.2)
def test_tie_is_not_a_win():
    from src.forecasting.comparison import count_wins

    assert count_wins([0.5, 0.4, 0.3], [0.5, 0.5, 0.2]) == 1
    assert count_wins(np.array([0.3, 0.3]), np.array([0.3, 0.3])) == 0


def test_mcnemar_counts_only_discordant_onsets():
    from src.forecasting.comparison import discordant_wins

    option = [True, True, False, False, True]
    reference = [True, False, False, True, False]
    assert discordant_wins(option, reference) == (2, 3)
    assert discordant_wins([True, False], [True, False]) == (0, 0)


# --------------------------------------------------------------------------- out-of-year scores (§8.3)
def test_out_of_year_choice_ignores_the_scored_year():
    from src.forecasting.comparison import out_of_year_scores

    table = _table(_designed)
    scores = out_of_year_scores(table, familia=FAMILY).set_index(["municipality_code", "anio"])
    assert len(scores) == len(CODES) * len(YEARS)

    for code in CODES:
        cell = scores.loc[(code, 2020)]
        assert cell["candidato_comun"] == "A"               # B's 2020 advantage is not used for 2020
        assert cell["candidato_propio"] == ("C" if code == "08001" else "A")
        assert cell["logloss_comun"] == 0.5 * (1 - 0.10)    # scored in 2020 with A's 2020 loss
        assert cell["mejora_comun"] == 0.10
        for year in (2019, 2021, 2022):                     # positive control: 2020 is used for other years
            assert scores.loc[(code, year), "candidato_comun"] == "B"
            assert scores.loc[(code, year), "candidato_propio"] == "B"
            assert scores.loc[(code, year), "logloss_comun"] == 0.5 * (1 - (-0.20))
    assert scores.loc[("08001", 2020), "logloss_propio"] == 0.5 * (1 - 0.15)


def test_perturbing_the_scored_year_does_not_change_its_choice():
    from src.forecasting.comparison import out_of_year_scores

    rng = np.random.default_rng(20261112)
    base_values = {(c, n, y): rng.uniform(-0.3, 0.4) for c in CODES for n, _ in CANDIDATES for y in YEARS}
    base = out_of_year_scores(_table(lambda c, n, y: base_values[(c, n, y)]), familia=FAMILY)
    base = base.set_index(["municipality_code", "anio"])

    for year in YEARS:
        for _ in range(5):
            noise = {key: rng.uniform(-0.9, 0.9) for key in base_values if key[2] == year}
            values = {**base_values, **noise}
            table = _table(lambda c, n, y: values[(c, n, y)])
            perturbed = out_of_year_scores(table, familia=FAMILY).set_index(["municipality_code", "anio"])
            for code in CODES:
                cell, before = perturbed.loc[(code, year)], base.loc[(code, year)]
                assert cell["candidato_comun"] == before["candidato_comun"]
                assert cell["candidato_propio"] == before["candidato_propio"]
                assert cell["mejora_comun"] == values[(code, cell["candidato_comun"], year)]
                assert cell["mejora_propia"] == values[(code, cell["candidato_propio"], year)]


# --------------------------------------------------------------------------- candidate list (§6.2, §10.12)
def test_identical_candidate_lists_are_accepted():
    from src.forecasting.search import validate_score_table

    validate_score_table(_table(_designed), municipalities=CODES, years=YEARS)


def _without_candidate(table):
    return table[~((table["municipality_code"] == "54001") & (table["candidato"] == "C"))]


def _with_extra_candidate(table):
    extra = table[(table["municipality_code"] == "54001") & (table["candidato"] == "C")].copy()
    extra["candidato"], extra["orden"], extra["parametros"] = "D", 3, json.dumps({"C": 10.0})
    return pd.concat([table, extra], ignore_index=True)


def _with_other_parameters(table):
    changed = table.copy()
    selected = (changed["municipality_code"] == "76001") & (changed["candidato"] == "B")
    changed.loc[selected, "parametros"] = json.dumps({"C": 0.3}, sort_keys=True)
    return changed


@pytest.mark.parametrize("alter", [_without_candidate, _with_extra_candidate, _with_other_parameters])
def test_different_candidate_lists_fail(alter):
    from src.forecasting.search import validate_score_table

    with pytest.raises(ValueError, match="candidat"):
        validate_score_table(alter(_table(_designed)), municipalities=CODES, years=YEARS)


# --------------------------------------------------------------------------- levels (§8.4), group B
@pytest.mark.parametrize(("favorable", "wins", "expected"), [
    (True, 13, CONSISTENT), (True, 12, TENDENCY), (True, 11, TENDENCY), (True, 10, TENDENCY),
    (True, 9, NONE), (False, 18, NONE),
])
def test_weekly_levels(favorable, wins, expected):
    from src.forecasting.comparison import weekly_level

    assert weekly_level(favorable, wins, 18, alpha=_alpha()) == expected


@pytest.mark.parametrize(("first", "second", "expected"), [
    (CONSISTENT, TENDENCY, TENDENCY), (TENDENCY, CONSISTENT, TENDENCY), (TENDENCY, NONE, NONE),
    (CONSISTENT, CONSISTENT, CONSISTENT),
])
def test_combined_level_adopts_the_lower(first, second, expected):
    from src.forecasting.comparison import combined_level

    assert combined_level(first, second) == expected


@pytest.mark.parametrize(("wins", "d", "expected"), [
    (7, 8, CONSISTENT), (5, 5, CONSISTENT), (6, 8, TENDENCY), (3, 4, TENDENCY), (4, 8, NONE), (0, 0, NONE),
])
def test_event_levels(wins, d, expected):
    from src.forecasting.comparison import event_level

    assert event_level(wins, d, alpha=_alpha()) == expected
