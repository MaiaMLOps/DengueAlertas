"""Complementary development analyses of logistic regression (docs/forecasting/analisis_complementario.md).

Nothing here selects or changes the frozen recipe. The functions rebuild, with the code of the pre-registered search,
the single development table of a logistic regression on a given grid of C and feature set, and summarise it with
the out-of-year scheme of the recipe (contrato §6.2; especificación §6.F6):

    examples (build_examples) -> development_trial_data: search._trial_data on splits.yearly_trials of the
        development years, the same trials, B2 and B1 as the search; no trial evaluates a test-period target week
      -> logistic_table(trial_data, grid, features): search._model_unit for logistica_00, logistica_01, ... in the
         order of the grid; the same columns as tabla_unica_desarrollo (with brier), validated
      -> out_of_year_cells(table): comparison.out_of_year_scores (common configuration chosen without year y by
         the mean mejora_b2 over years within each municipality and then over municipalities; ties to the smaller
         grid position), with the log-loss and Brier of the chosen configuration and of B2 in each cell
      -> fixed_cells(table, candidato): the same cells with one configuration fixed in every year
      -> summarize(cells): mean improvement over B2, mean log-loss and Brier, wins against B2 (strictly smaller
         log-loss, comparison.count_wins)
    common_choice(table): the common configuration chosen with all development years (comparison._choose)

The single table and the predictions are development rows only; every metric goes through the test-period barrier.
"""

from collections.abc import Sequence
import json

import numpy as np
import pandas as pd

from src.epidemiology.validation import require

from .comparison import _choose, count_wins, out_of_year_scores
from .config import ForecastConfig
from .guard import RecipeAccess
from .search import Candidate, _development_years, _model_unit, _parameters_json, _trial_data, validate_score_table
from .splits import yearly_trials

FAMILY = "logistica"


def development_trial_data(examples: pd.DataFrame, config: ForecastConfig, access: RecipeAccess) -> dict:
    """Trials of the development years with their B2 and B1 predictions, exactly as in search.run_search."""
    years = _development_years(config)
    trials = yearly_trials(examples, years, first_train_year=config.validacion.primer_anio_entrenamiento)
    first_test = pd.Timestamp(config.validacion.prueba[0], 1, 1).date()
    for trial in trials:
        last = max(examples.loc[trial.evaluate, "target_week"])
        require(last < first_test, f"Trial {trial.year} of {trial.municipality_code} evaluates a test-period target "
                                   f"week ({last}).")
    return _trial_data(examples, trials, config, access)


def grid_candidates(grid: Sequence[float], *, max_iter: int) -> list[Candidate]:
    """logistica_00, logistica_01, ... in the order of the grid (the order breaks exact ties)."""
    values = [float(value) for value in grid]
    require(len(values) > 0 and all(value > 0 for value in values), "The grid of C must hold positive values.")
    require(all(b > a for a, b in zip(values, values[1:])), "The grid of C must be strictly increasing.")
    return [Candidate(f"{FAMILY}_{i:02d}", {"C": value, "max_iter": int(max_iter)}) for i, value in enumerate(values)]


def logistic_table(trial_data: dict, grid: Sequence[float], *, max_iter: int, features: Sequence[str],
                   access: RecipeAccess, clip: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single development table and per-row predictions of the logistic grid (search._model_unit per unit)."""
    scores, predictions = [], []
    for code in sorted(trial_data):
        for position, candidate in enumerate(grid_candidates(grid, max_iter=max_iter)):
            state = {"year": None}
            unit_scores, unit_predictions, _ = _model_unit(code, FAMILY, position, candidate,
                                                           _parameters_json(candidate.params), trial_data[code],
                                                           access, clip, state, features)
            scores.append(unit_scores)
            predictions.append(unit_predictions)
    table = pd.concat(scores, ignore_index=True).drop(columns="segundos")
    validate_score_table(table, municipalities=sorted(trial_data), years=sorted(set(table["anio"])))
    return table, pd.concat(predictions, ignore_index=True)


def candidate_c(table: pd.DataFrame) -> dict[str, float]:
    """C of each candidate of the table."""
    pairs = table[["candidato", "parametros"]].drop_duplicates()
    return {row.candidato: float(json.loads(row.parametros)["C"]) for row in pairs.itertuples(index=False)}


def _cell_metrics(table: pd.DataFrame) -> pd.DataFrame:
    return table.set_index(["municipality_code", "anio", "candidato"])


def out_of_year_cells(table: pd.DataFrame) -> pd.DataFrame:
    """Out-of-year common configuration of each (municipality, year) with its scores and those of B2."""
    cells = out_of_year_scores(table, familia=FAMILY)
    indexed = _cell_metrics(table)
    c_of = candidate_c(table)
    keys = list(zip(cells["municipality_code"], cells["anio"], cells["candidato_comun"]))
    return pd.DataFrame({
        "municipality_code": cells["municipality_code"], "anio": cells["anio"],
        "candidato": cells["candidato_comun"], "C": [c_of[c] for c in cells["candidato_comun"]],
        "logloss": [float(indexed.loc[key, "logloss"]) for key in keys],
        "brier": [float(indexed.loc[key, "brier"]) for key in keys],
        "logloss_b2": [float(indexed.loc[key, "logloss_b2"]) for key in keys],
        "brier_b2": [float(indexed.loc[key, "brier_b2"]) for key in keys],
        "mejora_b2": [float(indexed.loc[key, "mejora_b2"]) for key in keys],
    })


def fixed_cells(table: pd.DataFrame, candidato: str) -> pd.DataFrame:
    """The cells of one configuration fixed in every development year."""
    own = table[table["candidato"] == candidato]
    require(len(own) > 0, f"Candidate {candidato!r} is not in the table.")
    out = own[["municipality_code", "anio", "candidato", "logloss", "brier", "logloss_b2", "brier_b2",
               "mejora_b2"]].copy()
    out.insert(3, "C", candidate_c(table)[candidato])
    return out.sort_values(["municipality_code", "anio"]).reset_index(drop=True)


def summarize(cells: pd.DataFrame) -> dict:
    """Means over the cells and wins against B2 (strictly smaller log-loss)."""
    return {"casillas": int(len(cells)),
            "mejora_b2_media": float(cells["mejora_b2"].mean()),
            "logloss_medio": float(cells["logloss"].mean()),
            "brier_medio": float(cells["brier"].mean()),
            "logloss_b2_medio": float(cells["logloss_b2"].mean()),
            "brier_b2_medio": float(cells["brier_b2"].mean()),
            "victorias_b2": count_wins(cells["logloss"].to_numpy(), cells["logloss_b2"].to_numpy())}


def paired_wins(option: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Cells where the option's log-loss is strictly smaller / larger than the reference's (same cells)."""
    keys = ["municipality_code", "anio"]
    merged = option[keys + ["logloss"]].merge(reference[keys + ["logloss"]], on=keys, suffixes=("", "_ref"),
                                              validate="one_to_one")
    require(len(merged) == len(option) == len(reference), "The two systems must cover the same cells.")
    return {"ganadas": count_wins(merged["logloss"].to_numpy(), merged["logloss_ref"].to_numpy()),
            "perdidas": count_wins(merged["logloss_ref"].to_numpy(), merged["logloss"].to_numpy()),
            "empates": int(np.isclose(merged["logloss"], merged["logloss_ref"], rtol=0, atol=0).sum())}


def common_choice(table: pd.DataFrame) -> str:
    """Common configuration chosen with every development year of the table."""
    return _choose(table[table["familia"] == FAMILY])


def common_by_year(table: pd.DataFrame) -> dict[int, str]:
    """Out-of-year common configuration of each development year (chosen without that year)."""
    own = table[table["familia"] == FAMILY]
    return {int(year): _choose(own[own["anio"] != year]) for year in sorted(set(own["anio"]))}
