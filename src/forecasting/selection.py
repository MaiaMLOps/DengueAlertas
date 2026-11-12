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
      4 alone. hyperparameter_choices(table, familia): the same step 4 with a fixed family, for the climate
         ablation (contrato §4.4, D1 of F7), whose search holds only the family of the recipe
    per-row predictions -> chosen_predictions (the configuration of each municipality)
      -> tau_curve: the development rows of the six municipalities together, through events.evaluate_period
         for the channel (channel_positive) and for every tau of the grid (model_positive), same rows
      -> choose_tau: maximum warned onsets with false alarms <= the channel's; tie -> larger tau; none
         feasible -> fewest false alarms (tie -> larger tau) and restriccion_cumplida = False
    select(...) chains everything and checks that no input has a test-period target week.
    build_recipe(selection, config, provenance) -> the recipe draft of especificación §8 (no approval, no hash)
      -> write_recipe_draft: YAML, never overwriting and never at configs/receta_congelada.yaml. Approval and
         hash are added in a separate unit, with Jesús's approval; only then the test period opens.

Wins are counted on log-loss, the deciding metric in development (contrato §8.2), never on mejora_b2:
1 − a/c can tie in floating point while a < b. The table is put in a canonical order first, so the result
does not depend on the order of its rows.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

from src.epidemiology.validation import require

from .comparison import _choose, count_wins, out_of_year_scores, win_threshold
from .config import ROOT
from .events import channel_positive, evaluate_period, model_positive
from .guard import FROZEN_RECIPE_PATH, HASH_FIELD, RecipeAccess
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
_COMMIT = re.compile(r"[0-9a-f]{40}")
_MD5 = re.compile(r"[0-9a-f]{32}")
CONFIGURATION_SECTIONS = ("canal", "features", "validacion", "lineas_base", "metricas")
_CHOICE_COLUMNS = ["municipality_code", "familia", "candidato", "orden", "parametros", "excepcion_familia",
                   "excepcion_hiperparametros", "excepcion_hiperparametros_nominal"]
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
def _canonical(table: pd.DataFrame, required: tuple[str, ...] = FAMILIES) -> pd.DataFrame:
    """Validate the single table (with every `required` family) and return it in a canonical row order."""
    require(isinstance(table, pd.DataFrame), "table must be the single score table (pandas DataFrame).")
    missing = [column for column in _TABLE_COLUMNS if column not in table.columns]
    require(not missing, f"Score table lacks columns {missing}.")
    absent = [family for family in required if family not in set(table["familia"])]
    require(not absent, f"Score table lacks the families {absent}; selection compares {list(required)}.")
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
        choice, rows = _hyperparameter_step(table, code, final, scores[final].loc[code], common[final], n_years)
        comparisons += rows
        choices.append({**choice, "excepcion_familia": bool(family_exception)})
    return pd.DataFrame(choices)[_CHOICE_COLUMNS], pd.DataFrame(comparisons)


def _hyperparameter_step(table, code, family, own, common_name, n_years) -> tuple[dict, list[dict]]:
    """Step 4 for one municipality within `family`: own candidate chosen without y against the common one chosen
    without y, in every year (out-of-year scores `own` of the municipality)."""
    own_loss = pd.Series(_loss(own, "propio"), index=own.index)
    common_loss = pd.Series(_loss(own, "comun"), index=own.index)
    exception = count_wins(own_loss, common_loss) == n_years
    comparisons = _comparison_rows(code, HYPERPARAMETER_RULE, family, own_loss, own["candidato_propio"],
                                   family, common_loss, own["candidato_comun"])
    rows = table[(table["familia"] == family) & (table["municipality_code"] == code)]
    name = _choose(rows) if exception else common_name
    described = rows[rows["candidato"] == name].iloc[0]
    choice = {"municipality_code": code, "familia": family, "candidato": name, "orden": int(described["orden"]),
              "parametros": str(described["parametros"]), "excepcion_familia": False,
              "excepcion_hiperparametros": bool(exception),
              "excepcion_hiperparametros_nominal": bool(exception and name == common_name)}
    return choice, comparisons


def hyperparameter_choices(table: pd.DataFrame, *, familia: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Only step 4 with a fixed family (climate ablation, contrato §4.4 and D1 of F7): common configuration of
    `familia` and the hyperparameter exception per municipality (4 of 4, out of year). The table may hold only
    that family. Returns (choices, comparisons) with the columns of municipality_choices; excepcion_familia is
    always False and there are no family comparisons."""
    require(familia in FAMILIES, f"Unknown family {familia!r}; expected one of {list(FAMILIES)}.")
    table = _canonical(table, required=(familia,))
    own_family = table[table["familia"] == familia].reset_index(drop=True)
    scores = out_of_year_scores(own_family, familia=familia).set_index(["municipality_code", "anio"])
    common = _choose(own_family)
    n_years = len(set(own_family["anio"]))
    choices, comparisons = [], []
    for code in sorted(set(own_family["municipality_code"])):
        choice, rows = _hyperparameter_step(own_family, code, familia, scores.loc[code], common, n_years)
        choices.append(choice)
        comparisons += rows
    return pd.DataFrame(choices)[_CHOICE_COLUMNS], pd.DataFrame(comparisons)


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


# --------------------------------------------------------------------------- recipe draft (especificación §8)
def _check_provenance(provenance: Mapping, config) -> dict:
    require(isinstance(provenance, Mapping), "provenance must be a mapping.")
    checked = dict(provenance)
    for key in ("commit_codigo", "commit_contrato"):
        value = checked.get(key)
        require(isinstance(value, str) and _COMMIT.fullmatch(value) is not None,
                f"provenance.{key} must be a 40-character lowercase hexadecimal commit; got {value!r}.")
    require(checked.get("version_dvc_canal") == config.datos.version_dvc_canal,
            f"provenance.version_dvc_canal {checked.get('version_dvc_canal')!r} differs from the configuration "
            f"{config.datos.version_dvc_canal!r}.")
    require(isinstance(checked.get("busqueda_id"), str) and checked["busqueda_id"].strip() != "",
            "provenance.busqueda_id must be a non-empty text.")
    digests = checked.get("insumos_md5")
    require(isinstance(digests, Mapping) and len(digests) > 0
            and all(isinstance(value, str) and _MD5.fullmatch(value) for value in digests.values()),
            "provenance.insumos_md5 must map each input file to its md5.")
    checked["insumos_md5"] = {str(name): str(value) for name, value in digests.items()}
    return checked


def _differences(comparisons: pd.DataFrame, code: str, rule: str) -> dict:
    rows = comparisons[(comparisons["municipality_code"] == code) & (comparisons["regla"] == rule)]
    require(len(rows) > 0, f"No {rule} comparison for municipality {code}.")
    rows = rows.sort_values("anio")
    return {str(int(year)): float(value) for year, value in zip(rows["anio"], rows["diferencia"])}


def build_recipe(selection: Selection, *, config, provenance: Mapping) -> dict:
    """Recipe draft with the fields of especificación §8, without aprobacion and hash_sha256."""
    require(isinstance(selection, Selection), "selection must be the result of select().")
    families, comparisons = selection.familias, selection.comparaciones
    municipalities = {}
    for _, row in selection.elecciones.sort_values("municipality_code").iterrows():
        code = str(row["municipality_code"])
        municipalities[code] = {
            "familia": str(row["familia"]), "candidato": str(row["candidato"]), "orden": int(row["orden"]),
            "hiperparametros": json.loads(row["parametros"]),
            "excepcion_familia": {"aplica": bool(row["excepcion_familia"]),
                                  "diferencias_logloss": _differences(comparisons, code, FAMILY_RULE)},
            "excepcion_hiperparametros": {
                "aplica": bool(row["excepcion_hiperparametros"]),
                "nominal": bool(row["excepcion_hiperparametros_nominal"]),
                "diferencias_logloss": _differences(comparisons, code, HYPERPARAMETER_RULE)},
        }
    tau = float(selection.tau.tau)
    require(tau in config.decision.tau_grid, f"tau {tau} is not in the configured grid.")
    points = selection.curva[selection.curva["tau"] == tau]
    require(len(points) == 1, f"The curve must have exactly one row for tau = {tau}.")
    point, channel = points.iloc[0], selection.canal
    decision = {
        "tau": tau, "k": config.decision.k, "W": config.decision.W,
        "restriccion_cumplida": bool(selection.tau.restriccion_cumplida),
        "inicios_evaluables": int(point["evaluable_onsets"]), "inicios_avisados": int(point["warned_onsets"]),
        "falsas_alarmas_modelo": int(point["false_alarms"]), "falsas_alarmas_canal": int(channel["false_alarms"]),
        "inicios_avisados_canal": int(channel["warned_onsets"]),
        "municipio_anios": int(point["municipality_years"]),
    }
    return {
        "familia_comun": str(families.familia_comun),
        "comparacion_familias": {
            "mejora_media_fuera_de_anio": {str(name): float(value) for name, value in families.mejora.items()},
            "victorias": int(families.victorias), "casillas": int(families.casillas),
            "umbral": None if families.umbral is None else int(families.umbral),
            "reemplaza": bool(families.reemplaza)},
        "municipios": municipalities,
        "decision": decision,
        "configuracion": {"h": config.horizonte.h,
                          **{name: getattr(config, name).model_dump(mode="json") for name in CONFIGURATION_SECTIONS}},
        "procedencia": _check_provenance(provenance, config),
    }


class _RecipeDumper(yaml.SafeDumper):
    """Quotes texts made only of digits (DIVIPOLA codes, years), so that no YAML 1.1 or 1.2 reader turns 08001
    into a number."""


def _represent_text(dumper, value: str):
    if value.isdigit():
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="'")
    return dumper.represent_str(value)


_RecipeDumper.add_representer(str, _represent_text)


def write_recipe_draft(recipe: Mapping, path: str | Path) -> Path:
    """Write the draft as YAML; never overwrites and never writes the frozen recipe path."""
    path = Path(path)
    require(isinstance(recipe, Mapping), "recipe must be a mapping.")
    require("aprobacion" not in recipe and HASH_FIELD not in recipe,
            "A draft has no aprobacion or hash_sha256: they are added when Jesús approves the recipe.")
    require(path.resolve() != (ROOT / FROZEN_RECIPE_PATH).resolve(),
            f"A draft is never written at {FROZEN_RECIPE_PATH}; that file is written only when the recipe is frozen.")
    require(not path.exists(), f"{path} exists; a recipe draft is never overwritten.")
    text = yaml.dump(dict(recipe), Dumper=_RecipeDumper, sort_keys=False, allow_unicode=True)
    require(yaml.safe_load(text) == dict(recipe), "The recipe does not round-trip through YAML.")
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return path
