"""Counting rules of D9: out-of-year scores, paired wins, one-sided exact binomial, McNemar and conclusion levels
(contrato §8.2–§8.4 and §10.17; especificación §6.F5a).

Flow:
    single table (search.run_search)
      -> out_of_year_scores(table, familia): for each municipality m and development year y,
           common choice = argmax of the mean mejora_b2 over the other years of each municipality, then over
                           the municipalities; own choice = argmax over the other years of m;
           an exact tie goes to the smallest `orden` (pre-registered position; especificación §6.F5a);
           both are scored with their logloss and mejora_b2 in (m, y). The year y never enters its own choice.
      -> losses per unit -> count_wins (strictly smaller loss; a tie is not a win)
      -> win_threshold / chance_probability (P(X >= k | n, 1/2), one tail) -> weekly_level
    warned onsets of two systems -> discordant_wins -> event_level (McNemar exact, one tail)
    Brier and log-loss levels -> combined_level (the lower one)

Levels (contrato §8.4): "mejora consistente" when the global metric is favourable and the wins reach the
threshold; "tendencia favorable" when favourable and the wins are a strict majority below it (10 to 12 of 18);
"sin mejora" otherwise. For events, the same with the McNemar threshold of the d discordant onsets and no global
metric. Whether the global metric is favourable (BSS > 0, lower Brier in the ablation) is decided by the caller.
Hyperparameter and family decisions built on these functions belong to selection.py (F6).
"""

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.epidemiology.validation import require

from .search import validate_score_table

CONSISTENT = "mejora consistente"
TENDENCY = "tendencia favorable"
NONE = "sin mejora"
_RANK = {NONE: 0, TENDENCY: 1, CONSISTENT: 2}
_SCORE_COLUMNS = ("municipality_code", "familia", "candidato", "orden", "parametros", "anio", "logloss", "mejora_b2")


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_alpha(alpha) -> None:
    require(isinstance(alpha, (float, int)) and not isinstance(alpha, bool) and 0 < alpha < 1,
            f"alpha must be a number in (0, 1); got {alpha!r}.")


def chance_probability(wins: int, n: int) -> float:
    """P(X >= wins) for X ~ Binomial(n, 1/2): the chance of at least `wins` wins between equivalent options."""
    require(_is_int(n) and n >= 1, f"n must be a positive integer; got {n!r}.")
    require(_is_int(wins) and 0 <= wins <= n, f"wins must be an integer from 0 to n = {n}; got {wins!r}.")
    return float(binom.sf(int(wins) - 1, int(n), 0.5))


def win_threshold(n: int, *, alpha: float) -> int | None:
    """Smallest k with P(X >= k | n, 1/2) < alpha (one tail), or None when no k reaches it."""
    require(_is_int(n) and n >= 1, f"n must be a positive integer; got {n!r}.")
    _check_alpha(alpha)
    return next((k for k in range(int(n) + 1) if chance_probability(k, int(n)) < alpha), None)


def _vector(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    require(array.ndim == 1 and len(array) > 0, f"{name} must be a non-empty one-dimensional sequence.")
    return array


def count_wins(losses, reference) -> int:
    """Units where the option's loss is strictly smaller than the reference's (a tie is not a win)."""
    option, other = _vector(losses, "losses"), _vector(reference, "reference")
    require(len(option) == len(other), f"losses and reference must have the same length; got {len(option)} and "
                                       f"{len(other)}.")
    option, other = option.astype(float), other.astype(float)
    require(np.isfinite(option).all() and np.isfinite(other).all(), "losses must be finite numbers.")
    return int((option < other).sum())


def discordant_wins(option_warned, reference_warned) -> tuple[int, int]:
    """(onsets warned by the option and not by the reference, number of discordant onsets d)."""
    option, other = _vector(option_warned, "option_warned"), _vector(reference_warned, "reference_warned")
    require(option.dtype == bool and other.dtype == bool, "Warned onsets must be booleans.")
    require(len(option) == len(other), f"Both systems must cover the same onsets; got {len(option)} and "
                                       f"{len(other)}.")
    return int((option & ~other).sum()), int((option != other).sum())


def _choose(rows: pd.DataFrame) -> str:
    """Candidate with the largest mean mejora_b2 (over years within each municipality, then over municipalities);
    an exact tie goes to the smallest pre-registered position."""
    per_municipality = rows.groupby(["municipality_code", "candidato", "orden"])["mejora_b2"].mean()
    means = per_municipality.groupby(["candidato", "orden"]).mean().reset_index()
    best = means["mejora_b2"].max()
    return str(means.loc[means["mejora_b2"] == best].sort_values("orden")["candidato"].iloc[0])


def out_of_year_scores(table: pd.DataFrame, *, familia: str) -> pd.DataFrame:
    """Common and own configuration of each (municipality, year), chosen without that year and scored in it."""
    require(isinstance(table, pd.DataFrame), "table must be the single score table (pandas DataFrame).")
    missing = [column for column in _SCORE_COLUMNS if column not in table.columns]
    require(not missing, f"Score table lacks columns {missing}.")
    own = table[table["familia"] == familia]
    require(len(own) > 0, f"Family {familia!r} is not in the score table.")
    codes, years = sorted(set(own["municipality_code"])), sorted(set(own["anio"]))
    validate_score_table(own, municipalities=codes, years=years)
    require(len(years) >= 2, f"Out-of-year scores need at least two years; got {years}.")
    for column in ("logloss", "mejora_b2"):
        require(np.isfinite(own[column].to_numpy(dtype=float)).all(), f"Column {column} must be finite.")

    cells = own.set_index(["municipality_code", "anio", "candidato"])
    rows = []
    for year in years:
        others = own[own["anio"] != year]
        common = _choose(others)
        for code in codes:
            mine = _choose(others[others["municipality_code"] == code])
            rows.append({
                "municipality_code": code, "anio": year,
                "candidato_comun": common, "candidato_propio": mine,
                "logloss_comun": float(cells.loc[(code, year, common), "logloss"]),
                "logloss_propio": float(cells.loc[(code, year, mine), "logloss"]),
                "mejora_comun": float(cells.loc[(code, year, common), "mejora_b2"]),
                "mejora_propia": float(cells.loc[(code, year, mine), "mejora_b2"]),
            })
    return pd.DataFrame(rows).sort_values(["municipality_code", "anio"]).reset_index(drop=True)


def weekly_level(favorable: bool, wins: int, n: int, *, alpha: float) -> str:
    """Conclusion of a weekly comparison over n cells (contrato §8.4)."""
    require(isinstance(favorable, (bool, np.bool_)), f"favorable must be a boolean; got {favorable!r}.")
    threshold = win_threshold(n, alpha=alpha)
    chance_probability(wins, n)   # validates wins against n
    if favorable and threshold is not None and wins >= threshold:
        return CONSISTENT
    if favorable and wins > n / 2:
        return TENDENCY
    return NONE


def event_level(wins: int, d: int, *, alpha: float) -> str:
    """Conclusion of the McNemar comparison over d discordant onsets (contrato §8.4)."""
    _check_alpha(alpha)
    require(_is_int(d) and d >= 0, f"d must be a non-negative integer; got {d!r}.")
    require(_is_int(wins) and 0 <= wins <= d, f"wins must be an integer from 0 to d = {d}; got {wins!r}.")
    if d == 0:
        return NONE
    threshold = win_threshold(d, alpha=alpha)
    if threshold is not None and wins >= threshold:
        return CONSISTENT
    if wins > d / 2:
        return TENDENCY
    return NONE


def combined_level(first: str, second: str) -> str:
    """The lower of two levels (Brier and log-loss, contrato §8.4)."""
    for level in (first, second):
        require(level in _RANK, f"Unknown level {level!r}; expected one of {list(_RANK)}.")
    return first if _RANK[first] <= _RANK[second] else second

