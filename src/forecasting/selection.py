"""Selection of F6: hyperparameters -> family -> exceptions -> tau, from the saved development search
(contrato §4.3, §6.2, §7.5 and §8.3; especificación §6.F6 and §7, «Orden de selección para F6»).

Nothing is retrained: the inputs are the single table and the per-row predictions of F5b.

Flow:
    single table (tabla_unica_desarrollo)
      1. common_configuration(table, familia): with the 4 years, the candidate with the largest mean mejora_b2,
         first over the years of each municipality, then over the municipalities; an exact tie goes to the
         smallest `orden` (comparison._choose, the same rule as the out-of-year choices)
      2. compare_families(table, alpha): out_of_year_scores of each family (common configuration chosen
         without y, scored in y); LightGBM replaces logistic regression (parsimony order of models.FAMILIES)
         only if its mean out-of-year improvement is strictly greater AND it wins at least
         win_threshold(cells, alpha) cells (17 of 24). A win is a strictly smaller log-loss (`_loss`)
      3-4. municipality_choices(table, familia_comun), per municipality:
         3. family exception: the alternative family's out-of-year common configuration beats the common
            family's in every development year (4 of 4)
         4. hyperparameter exception within the final family: the own candidate chosen without y beats the
            common one chosen without y in every year; then the own candidate chosen with the 4 years is used
            (nominal exception when it equals the common one)
         every comparison is reported with its annual differences (logloss option − logloss reference)
    per-row predictions -> chosen_predictions (the configuration of each municipality)
      -> tau_curve: the development rows of the six municipalities together, through events.evaluate_period
         for the channel (channel_positive) and for every tau of the grid (model_positive), same rows
      -> choose_tau: maximum warned onsets with false alarms <= the channel's; tie -> larger tau; none
         feasible -> fewest false alarms (tie -> larger tau) and restriccion_cumplida = False
    select(...) chains everything and checks that no input has a test-period target week.

Wins are counted on log-loss, the deciding metric in development (contrato §8.2), never on mejora_b2:
1 − a/c can tie in floating point while a < b. The table is put in a canonical order first, so the result
does not depend on the order of its rows.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.epidemiology.validation import require

from .comparison import _choose, count_wins, out_of_year_scores, win_threshold
from .events import channel_positive, evaluate_period, model_positive
from .guard import RecipeAccess
from .models import FAMILIES
from .search import validate_score_table

FAMILY_RULE = "familia"
HYPERPARAMETER_RULE = "hiperparametros"
_TABLE_COLUMNS = ("municipality_code", "familia", "candidato", "orden", "parametros", "anio", "logloss",
                  "mejora_b2")
_PREDICTION_COLUMNS = ("municipality_code", "t", "target_week", "familia", "candidato", "anio", "y", "p")
_ROW_COLUMNS = ("municipality_code", "t", "target_week", "zona_t", "y")
_KEYS = ["municipality_code", "t"]
_CANONICAL_ORDER = ["municipality_code", "familia", "orden", "anio"]
CURVE_COLUMNS = ("evaluable_onsets", "warned_onsets", "detection", "false_alarms", "municipality_years",
                 "false_alarms_per_municipality_year", "judgeable_alerts", "continuity_alerts")


@dataclass(frozen=True)
class FamilyComparison:
    """Step 2: the common family and the evidence that decided it (out-of-year, contrato §8.3)."""

    familia_comun: str
    mejora: dict          # family -> mean out-of-year improvement over B2 (years, then municipalities)
    victorias: int        # cells where the more complex family has a strictly smaller log-loss
    casillas: int
    umbral: int | None    # win_threshold(casillas, alpha)
    reemplaza: bool


@dataclass(frozen=True)
class TauChoice:
    tau: float
    restriccion_cumplida: bool


@dataclass(frozen=True)
class Selection:
    """Everything F6 decides from the development search, plus the evidence to report it."""

    familias: FamilyComparison
    elecciones: pd.DataFrame      # one row per municipality: family, candidate and exceptions
    comparaciones: pd.DataFrame   # (municipality, rule, year): option vs reference log-loss
    predicciones: pd.DataFrame    # development predictions of the chosen configurations
    curva: pd.DataFrame           # one row per tau of the grid
    canal: dict                   # channel summary on the same rows (events.evaluate_period)
    tau: TauChoice


# --------------------------------------------------------------------------- the single table
def _canonical(table: pd.DataFrame) -> pd.DataFrame:
    """Validate the single table and return it in a canonical row order."""
    require(isinstance(table, pd.DataFrame), "table must be the single score table (pandas DataFrame).")
    missing = [column for column in _TABLE_COLUMNS if column not in table.columns]
    require(not missing, f"Score table lacks columns {missing}.")
    absent = [family for family in FAMILIES if family not in set(table["familia"])]
    require(not absent, f"Score table lacks the families {absent}; selection compares {list(FAMILIES)}.")
    unknown = sorted(set(table["familia"]) - set(FAMILIES))
    require(not unknown, f"Score table has families outside the selection {unknown}.")
    validate_score_table(table, municipalities=sorted(set(table["municipality_code"])),
                         years=sorted(set(table["anio"])))
    return table.sort_values(_CANONICAL_ORDER, kind="mergesort").reset_index(drop=True)


def _loss(scores: pd.DataFrame, which: str) -> np.ndarray:
    """Loss of every win comparison: the log-loss of the common ("comun") or own ("propio") configuration."""
    return scores[f"logloss_{which}"].to_numpy(dtype=float)


def _mean_improvement(scores: pd.DataFrame) -> float:
    return float(scores.groupby("municipality_code")["mejora_comun"].mean().mean())


def common_configuration(table: pd.DataFrame, *, familia: str) -> str:
    """Step 1: the common configuration of a family, chosen with all the development years."""
    table = _canonical(table)
    require(familia in FAMILIES, f"Unknown family {familia!r}; expected one of {list(FAMILIES)}.")
    return _choose(table[table["familia"] == familia])


def compare_families(table: pd.DataFrame, *, alpha: float) -> FamilyComparison:
    """Step 2: the more complex family replaces the simpler one only with a strictly greater mean out-of-year
    improvement and at least win_threshold wins over the cells (contrato §4.3 and §8.3)."""
    table = _canonical(table)
    simple, complex_ = FAMILIES
    simple_scores = out_of_year_scores(table, familia=simple)
    complex_scores = out_of_year_scores(table, familia=complex_)
    keys = ["municipality_code", "anio"]
    require(simple_scores[keys].equals(complex_scores[keys]),
            "Both families must have the same (municipality, year) cells.")
    wins = count_wins(_loss(complex_scores, "comun"), _loss(simple_scores, "comun"))
    cells = len(simple_scores)
    threshold = win_threshold(cells, alpha=alpha)
    means = {simple: _mean_improvement(simple_scores), complex_: _mean_improvement(complex_scores)}
    replaces = bool(means[complex_] > means[simple] and threshold is not None and wins >= threshold)
    return FamilyComparison(familia_comun=complex_ if replaces else simple, mejora=means, victorias=wins,
                            casillas=cells, umbral=threshold, reemplaza=replaces)


def _comparison_rows(code, rule, option_family, option, option_names, reference_family, reference,
                     reference_names) -> list[dict]:
    rows = []
    for year, a, b, name_a, name_b in zip(option.index, option, reference, option_names, reference_names):
        rows.append({"municipality_code": code, "regla": rule, "anio": int(year),
                     "familia_opcion": option_family, "candidato_opcion": str(name_a),
                     "familia_referencia": reference_family, "candidato_referencia": str(name_b),
                     "logloss_opcion": float(a), "logloss_referencia": float(b), "diferencia": float(a - b),
                     "gana": bool(a < b)})
    return rows


def municipality_choices(table: pd.DataFrame, *, familia_comun: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Steps 3 and 4: family exception per municipality, then hyperparameter exception within the final family.
    Returns (choices, comparisons); every municipality gets both comparisons with their annual differences."""
    table = _canonical(table)
    require(familia_comun in FAMILIES, f"Unknown family {familia_comun!r}; expected one of {list(FAMILIES)}.")
    alternative = next(family for family in FAMILIES if family != familia_comun)
    scores = {family: out_of_year_scores(table, familia=family).set_index(["municipality_code", "anio"])
              for family in FAMILIES}
    common = {family: _choose(table[table["familia"] == family]) for family in FAMILIES}
    n_years = len(set(table["anio"]))

    choices, comparisons = [], []
    for code in sorted(set(table["municipality_code"])):
        reference, option = scores[familia_comun].loc[code], scores[alternative].loc[code]
        reference_loss = pd.Series(_loss(reference, "comun"), index=reference.index)
        option_loss = pd.Series(_loss(option, "comun"), index=option.index)
        family_exception = count_wins(option_loss, reference_loss) == n_years
        comparisons += _comparison_rows(code, FAMILY_RULE, alternative, option_loss, option["candidato_comun"],
                                        familia_comun, reference_loss, reference["candidato_comun"])

        final = alternative if family_exception else familia_comun
        own = scores[final].loc[code]
        own_loss = pd.Series(_loss(own, "propio"), index=own.index)
        common_loss = pd.Series(_loss(own, "comun"), index=own.index)
        hyperparameter_exception = count_wins(own_loss, common_loss) == n_years
        comparisons += _comparison_rows(code, HYPERPARAMETER_RULE, final, own_loss, own["candidato_propio"],
                                        final, common_loss, own["candidato_comun"])

        rows = table[(table["familia"] == final) & (table["municipality_code"] == code)]
        name = _choose(rows) if hyperparameter_exception else common[final]
        described = rows[rows["candidato"] == name].iloc[0]
        choices.append({"municipality_code": code, "familia": final, "candidato": name,
                        "orden": int(described["orden"]), "parametros": str(described["parametros"]),
                        "excepcion_familia": bool(family_exception),
                        "excepcion_hiperparametros": bool(hyperparameter_exception),
                        "excepcion_hiperparametros_nominal": bool(hyperparameter_exception and name == common[final])})
    return pd.DataFrame(choices), pd.DataFrame(comparisons)


# --------------------------------------------------------------------------- predictions and tau
def _check_columns(frame, columns, name: str) -> None:
    require(isinstance(frame, pd.DataFrame), f"{name} must be a pandas DataFrame.")
    missing = [column for column in columns if column not in frame.columns]
    require(not missing, f"{name} lacks columns {missing}.")


def chosen_predictions(predictions: pd.DataFrame, choices: pd.DataFrame) -> pd.DataFrame:
    """Development predictions of the configuration chosen in each municipality, sorted by (municipality, t)."""
    _check_columns(predictions, _PREDICTION_COLUMNS, "predictions")
    _check_columns(choices, ("municipality_code", "familia", "candidato"), "choices")
    require(not choices["municipality_code"].duplicated().any(), "choices must have one row per municipality.")
    keys = choices[["municipality_code", "familia", "candidato"]]
    chosen = predictions.merge(keys, on=["municipality_code", "familia", "candidato"], how="inner")
    found = set(chosen["municipality_code"])
    require(found == set(keys["municipality_code"]),
            f"No predictions for the chosen configuration of {sorted(set(keys['municipality_code']) - found)}.")
    return chosen.sort_values(_KEYS, kind="mergesort").reset_index(drop=True)


def _aligned_probabilities(rows: pd.DataFrame, predictions: pd.DataFrame) -> np.ndarray:
    """p in the order of rows; predictions must hold exactly the rows' (municipality, t), target week and y."""
    _check_columns(rows, _ROW_COLUMNS, "rows")
    _check_columns(predictions, ("municipality_code", "t", "target_week", "y", "p"), "predictions")
    require(not rows.duplicated(_KEYS).any(), "rows have duplicated (municipality_code, t).")
    require(not predictions.duplicated(_KEYS).any(), "predictions have duplicated (municipality_code, t).")
    require(len(predictions) == len(rows),
            f"predictions have {len(predictions)} rows and the development rows {len(rows)}; they must match.")
    merged = rows[_KEYS + ["target_week", "y"]].merge(
        predictions[_KEYS + ["target_week", "y", "p"]], on=_KEYS, how="left", suffixes=("", "_prediccion"))
    require(bool(merged["p"].notna().all()), "Some development rows have no prediction.")
    require(bool((merged["target_week"] == merged["target_week_prediccion"]).all()),
            "A prediction has a target week different from its row.")
    require(bool((merged["y"].astype(int) == merged["y_prediccion"].astype(int)).all()),
            "A prediction has a label y different from its row.")
    return merged["p"].to_numpy(dtype=float)


def tau_curve(rows: pd.DataFrame, predictions: pd.DataFrame, observed: pd.DataFrame, *, tau_grid: Sequence[float],
              k: int, window: int, access: RecipeAccess) -> tuple[pd.DataFrame, dict]:
    """Event metrics of the model for every tau and of the channel, on the same rows (contrato §7.4–§7.5)."""
    probabilities = _aligned_probabilities(rows, predictions)
    grid = list(tau_grid)
    require(len(grid) > 0, "tau_grid must not be empty.")
    channel = evaluate_period(rows, channel_positive(rows["zona_t"]), observed, k=k, window=window,
                              access=access).summary
    records = []
    for tau in grid:
        summary = evaluate_period(rows, model_positive(probabilities, tau=tau), observed, k=k, window=window,
                                  access=access).summary
        require(summary["municipality_years"] == channel["municipality_years"]
                and summary["evaluable_onsets"] == channel["evaluable_onsets"],
                "Model and channel were not evaluated on the same rows.")
        records.append({"tau": float(tau), **{column: summary[column] for column in CURVE_COLUMNS}})
    return pd.DataFrame(records), dict(channel)


def choose_tau(curve: pd.DataFrame, *, channel_false_alarms: int) -> TauChoice:
    """Rule of contrato §7.5 over the curve (counts share the channel's municipality-years)."""
    _check_columns(curve, ("tau", "warned_onsets", "false_alarms"), "curve")
    require(len(curve) > 0, "curve must not be empty.")
    require(not curve["tau"].duplicated().any(), "curve has repeated tau values.")
    require(isinstance(channel_false_alarms, (int, np.integer)) and not isinstance(channel_false_alarms, bool)
            and channel_false_alarms >= 0,
            f"channel_false_alarms must be a non-negative integer; got {channel_false_alarms!r}.")
    feasible = curve[curve["false_alarms"] <= channel_false_alarms]
    if len(feasible):
        best = feasible["warned_onsets"].max()
        return TauChoice(tau=float(feasible.loc[feasible["warned_onsets"] == best, "tau"].max()),
                         restriccion_cumplida=True)
    fewest = curve["false_alarms"].min()
    return TauChoice(tau=float(curve.loc[curve["false_alarms"] == fewest, "tau"].max()), restriccion_cumplida=False)


# --------------------------------------------------------------------------- the whole selection
def _check_development_only(table, predictions, rows, config) -> None:
    first, last = config.validacion.desarrollo
    years = list(range(first, last + 1))
    test_day = date(config.validacion.prueba[0], 1, 1)
    require(sorted(set(table["anio"])) == years,
            f"Score table years {sorted(set(table['anio']))} must be the development years {years}.")
    require(all(week < test_day for week in predictions["target_week"]),
            f"Predictions have target weeks on or after {test_day}; selection never reads the test period.")
    require(set(predictions["anio"]) <= set(years), "Predictions have years outside development.")
    require(all(week < test_day for week in rows["target_week"]),
            f"Rows have target weeks on or after {test_day}; selection uses the development rows only.")


def select(table: pd.DataFrame, predictions: pd.DataFrame, rows: pd.DataFrame, observed: pd.DataFrame, *,
           config, access: RecipeAccess) -> Selection:
    """Run the whole F6 selection from the saved development search (especificación §7)."""
    _check_columns(table, _TABLE_COLUMNS, "table")
    _check_columns(predictions, _PREDICTION_COLUMNS, "predictions")
    _check_columns(rows, _ROW_COLUMNS, "rows")
    _check_development_only(table, predictions, rows, config)
    require(set(table["municipality_code"]) == set(rows["municipality_code"]),
            "The score table and the development rows must cover the same municipalities.")

    families = compare_families(table, alpha=config.comparacion.alfa)
    choices, comparisons = municipality_choices(table, familia_comun=families.familia_comun)
    chosen = chosen_predictions(predictions, choices)
    decision = config.decision
    curve, channel = tau_curve(rows, chosen, observed, tau_grid=decision.tau_grid, k=decision.k,
                               window=decision.W, access=access)
    tau = choose_tau(curve, channel_false_alarms=channel["false_alarms"])
    return Selection(familias=families, elecciones=choices, comparaciones=comparisons, predicciones=chosen,
                     curva=curve, canal=channel, tau=tau)
