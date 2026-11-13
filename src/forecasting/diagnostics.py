"""Integrated diagnostic analysis (D9.1): regime of each municipality-year and the six inputs of contrato §8.8
(contrato §10.22; especificación §4 and §6.F7). It only reads saved predictions: nothing is refitted.

Flow:
    observed channel product (eligible weeks, target_u1e = 1[C > U1e])
      -> year_regimes / regime_table: episodes = runs of >= EPISODE_MIN_RUN weeks with target_u1e = 1, found on
         the whole eligible series of the municipality; a calendar year (year of the Monday) is "epidemica" if
         one of its weeks belongs to an episode, "post_epidemica" if not and the previous year is epidemic (it does
         not chain), "tranquila" otherwise. It uses weeks after the emission: a diagnostic stratum only, never a
         feature and never a decision.
    development: single score table + saved predictions of F5b
      -> development_predictions(table, predictions, baseline_predictions, familia): in each (municipality,
         year) the common out-of-year configuration of the chosen family (comparison.out_of_year_scores,
         candidato_comun), plus the saved B2 and B1 of that trial; long table with sistema in SYSTEMS.
    test: the predictions of evaluate.run_test (U7.3), with the frozen recipe.
    rows of one period + long predictions + observed product
      -> diagnostic_inputs(...): test-period barrier first, rotulo checked against the period, then
           1. casillas: one row per (municipality, year of the target week);
           2. resumenes: per global / municipio / anio / regimen, each metric aggregated twice (mean of the cell
              improvements and the metric over the rows together; they can disagree in sign);
           3. transicion: strata "fuera" / "dentro" of the epidemic zone at t (contrato §7.2), per year and all;
           4. eventos: onsets and entries of the period (events.evaluate_period for the model at tau and for the
              channel), with warnings, leads and delta_w = lead of the model - lead of the channel (only when the
              event is evaluable and both warn);
           5. alertas: every alert of both systems with continuity, judgeable, false alarm and whether an
              epidemic week follows;
           6. serie: one row per emission with C, central and U1e of the target week, zone at t, y, p of the three
              systems, positive emissions and alert flags (from t_c to the end of each alert).
      -> write_diagnostics(diagnostics, folder): six CSV files, never overwritten.

Differences are the option minus the reference (negative = the model is better); mejora_* and bss_* are
1 - L / L_ref; a win is a strictly smaller loss (a tie is not a win). Log-loss uses the same clip as every
system (metricas.recorte_log_loss).

Command line (development only; nothing in 2023-2025 is read):
    python -B -m src.forecasting.diagnostics --salida reports/forecasting/evidence/diagnostico_desarrollo_<fecha>
"""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

from src.epidemiology.channel_epico import ZONES
from src.epidemiology.validation import require

from .comparison import out_of_year_scores
from .events import _binary, _check_weeks, _runs, channel_positive, evaluate_period, model_positive
from .guard import RecipeAccess, check_test_access
from .metrics import average_precision, brier, improvement, log_loss

# Episode of contrato §8.8 (the notion of run of §2.4): part of the definition, not a tunable parameter.
EPISODE_MIN_RUN = 3
EPIDEMICA, POST_EPIDEMICA, TRANQUILA = "epidemica", "post_epidemica", "tranquila"
DEVELOPMENT_LABEL, TEST_LABEL = "usado_para_elegir", "prueba"
SYSTEMS = ("modelo", "B2", "B1")
INPUTS = ("casillas", "resumenes", "transicion", "eventos", "alertas", "serie")
_KEYS = ["municipality_code", "t"]
_EPIDEMIC_ZONE = ZONES[2]
_ROW_COLUMNS = ("municipality_code", "t", "target_week", "zona_t", "y")
_PREDICTION_COLUMNS = ("municipality_code", "t", "target_week", "y", "sistema", "p")
_OBSERVED_COLUMNS = ("municipality_code", "week_start", "target_u1e", "channel_eligible", "channel_zone",
                     "confirmed_records_screening", "epico_central", "epico_upper")
_TABLE_PREDICTION_COLUMNS = ("municipality_code", "t", "target_week", "familia", "candidato", "anio", "y", "p")

CASILLAS_COLUMNS = [
    "municipality_code", "anio", "regimen", "rotulo", "candidato", "filas", "positivas", "prevalencia",
    "p_medio_modelo", "p_medio_b2", "p_medio_b1", "calibracion_modelo", "calibracion_b2", "calibracion_b1",
    "logloss_modelo", "logloss_b2", "logloss_b1", "brier_modelo", "brier_b2", "brier_b1",
    "dif_logloss_b2", "dif_logloss_b1", "dif_brier_b2", "dif_brier_b1",
    "mejora_logloss_b2", "mejora_logloss_b1", "bss_b2", "bss_b1",
    "gana_logloss_b2", "gana_brier_b2", "gana_logloss_b1", "gana_brier_b1"]
RESUMEN_COLUMNS = [
    "nivel", "grupo", "rotulo", "casillas", "filas", "positivas",
    "victorias_logloss_b2", "victorias_brier_b2", "victorias_logloss_b1", "victorias_brier_b1",
    "mejora_logloss_b2_casillas", "mejora_logloss_b2_filas", "bss_b2_casillas", "bss_b2_filas",
    "mejora_logloss_b1_casillas", "mejora_logloss_b1_filas", "bss_b1_casillas", "bss_b1_filas",
    "calibracion_modelo_casillas", "calibracion_modelo_filas", "calibracion_b2_casillas", "calibracion_b2_filas"]
TRANSICION_COLUMNS = [
    "anio", "estrato", "rotulo", "filas", "positivas", "logloss_modelo", "logloss_b2", "brier_modelo", "brier_b2",
    "mejora_logloss_b2", "bss_b2", "ap_modelo", "ap_b2"]
EVENTOS_COLUMNS = [
    "municipality_code", "tipo", "semana", "anio", "regimen", "rotulo", "evaluable", "censura",
    "avisado_modelo", "avisado_canal", "confirmacion_modelo", "confirmacion_canal",
    "anticipacion_modelo", "anticipacion_canal", "delta_w"]
ALERTAS_COLUMNS = [
    "sistema", "municipality_code", "inicio", "confirmacion", "fin", "continuidad", "juzgable", "falsa_alarma",
    "seguida_de_epidemia", "anio", "regimen", "rotulo"]
SERIE_COLUMNS = [
    "municipality_code", "t", "target_week", "anio", "regimen", "rotulo", "casos_objetivo", "central_objetivo",
    "u1e_objetivo", "zona_t", "y", "p_modelo", "p_b2", "p_b1", "emision_modelo", "emision_canal",
    "alerta_modelo", "alerta_canal"]
_SCORED = (("modelo", "p_modelo"), ("b2", "p_b2"), ("b1", "p_b1"))
_ALERT_NAMES = {"start": "inicio", "confirmation": "confirmacion", "end": "fin", "continuity": "continuidad",
                "judgeable": "juzgable", "false_alarm": "falsa_alarma", "followed_by_epidemic": "seguida_de_epidemia",
                "year": "anio"}


@dataclass(frozen=True)
class Diagnostics:
    """The six inputs of contrato §8.8 for one period."""

    casillas: pd.DataFrame
    resumenes: pd.DataFrame
    transicion: pd.DataFrame
    eventos: pd.DataFrame
    alertas: pd.DataFrame
    serie: pd.DataFrame


def _check_columns(frame, columns, name: str) -> None:
    require(isinstance(frame, pd.DataFrame), f"{name} must be a pandas DataFrame.")
    missing = [column for column in columns if column not in frame.columns]
    require(not missing, f"{name} lacks columns {missing}.")


# --------------------------------------------------------------------------- regimes
def _episode_weeks(weeks: list, values: np.ndarray) -> list:
    return [week for first, length in _runs(values) if length >= EPISODE_MIN_RUN
            for week in weeks[first:first + length]]


def _check_years(years) -> list[int]:
    checked = list(years)
    require(checked, "years must not be empty.")
    for year in checked:
        require(isinstance(year, (int, np.integer)) and not isinstance(year, (bool, np.bool_)),
                f"years must be integers; got {year!r}.")
    return [int(year) for year in checked]


def year_regimes(weeks, exceed, *, years) -> dict[int, str]:
    """Regime of each requested year for one municipality's consecutive eligible weeks (contrato §8.8)."""
    checked = _check_weeks(weeks)
    values = _binary(exceed, "exceed", len(checked))
    covered = {week.year for week in checked}
    epidemic = {week.year for week in _episode_weeks(checked, values)}
    regimes = {}
    for year in _check_years(years):
        require(year in covered and year - 1 in covered,
                f"Year {year} and its previous year must be covered by the eligible series "
                f"({min(covered)}..{max(covered)}).")
        regimes[year] = EPIDEMICA if year in epidemic else POST_EPIDEMICA if year - 1 in epidemic else TRANQUILA
    return regimes


def regime_table(observed: pd.DataFrame, *, years) -> pd.DataFrame:
    """Regime of every (municipality, year) from the channel product, with the episode weeks of the year."""
    _check_columns(observed, ("municipality_code", "week_start", "target_u1e", "channel_eligible"), "observed")
    require(observed["channel_eligible"].dtype == bool, "observed.channel_eligible must be boolean.")
    checked_years = _check_years(years)
    records = []
    for code in sorted(observed["municipality_code"].unique()):
        own = observed[(observed["municipality_code"] == code) & observed["channel_eligible"]]
        own = own.sort_values("week_start")
        weeks = _check_weeks(own["week_start"])
        values = _binary(own["target_u1e"].to_numpy(), "observed.target_u1e", len(weeks))
        regimes = year_regimes(weeks, values, years=checked_years)
        episode_years = pd.Series([week.year for week in _episode_weeks(weeks, values)], dtype="int64")
        for year in checked_years:
            records.append({"municipality_code": code, "anio": year, "regimen": regimes[year],
                            "semanas_en_episodio": int((episode_years == year).sum())})
    return pd.DataFrame(records, columns=["municipality_code", "anio", "regimen", "semanas_en_episodio"])


# --------------------------------------------------------------------------- development predictions
def development_predictions(table: pd.DataFrame, predictions: pd.DataFrame, baseline_predictions: pd.DataFrame, *,
                            familia: str) -> pd.DataFrame:
    """Saved F5b predictions diagnosed in development: the common out-of-year configuration of `familia` in each
    (municipality, year) and the B2/B1 of the same trial. Long table with columns municipality_code, t,
    target_week, anio, y, sistema, candidato and p. The recipe has no exceptions (F6); with exceptions, the
    configuration compared in each case by contrato §8.3 would have to be passed instead."""
    _check_columns(predictions, _TABLE_PREDICTION_COLUMNS, "predictions")
    _check_columns(baseline_predictions, _TABLE_PREDICTION_COLUMNS, "baseline_predictions")
    scores = out_of_year_scores(table, familia=familia)
    frames = []
    for cell in scores.itertuples():
        own = (predictions["municipality_code"] == cell.municipality_code) & (predictions["anio"] == cell.anio)
        model = predictions[own & (predictions["familia"] == familia) & (predictions["candidato"] == cell.candidato_comun)]
        require(len(model) > 0, f"No saved predictions of {cell.candidato_comun} for {cell.municipality_code} "
                                f"in {cell.anio}.")
        frames.append(model.assign(sistema="modelo"))
        keys = set(zip(model["t"], model["target_week"]))
        for system in ("B2", "B1"):
            base = baseline_predictions[(baseline_predictions["municipality_code"] == cell.municipality_code)
                                        & (baseline_predictions["anio"] == cell.anio)
                                        & (baseline_predictions["familia"] == system)]
            require(set(zip(base["t"], base["target_week"])) == keys and len(base) == len(model),
                    f"Saved {system} predictions do not cover the rows of {cell.municipality_code} in {cell.anio}.")
            frames.append(base.assign(sistema=system))
    long = pd.concat(frames, ignore_index=True)
    long = long[["municipality_code", "t", "target_week", "anio", "y", "sistema", "candidato", "p"]]
    return long.sort_values(["sistema", "municipality_code", "t"], kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- helpers of the six inputs
def _aligned(rows: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, np.ndarray]:
    """p of each system in the order of rows; each system must hold exactly the rows' keys, targets and labels."""
    aligned = {}
    for system in SYSTEMS:
        own = predictions[predictions["sistema"] == system]
        require(not own.duplicated(_KEYS).any(), f"Predictions of {system} have duplicated (municipality_code, t).")
        require(len(own) == len(rows),
                f"Predictions of {system} have {len(own)} rows and the period {len(rows)}; they must match.")
        merged = rows[_KEYS + ["target_week", "y"]].merge(own[_KEYS + ["target_week", "y", "p"]], on=_KEYS,
                                                          how="left", suffixes=("", "_prediccion"))
        require(bool(merged["p"].notna().all()), f"Some rows have no {system} prediction.")
        require(bool((merged["target_week"] == merged["target_week_prediccion"]).all()),
                f"A {system} prediction has a target week different from its row.")
        require(bool((merged["y"].astype(int) == merged["y_prediccion"].astype(int)).all()),
                f"A {system} prediction has a label y different from its row.")
        aligned[system] = merged["p"].to_numpy(dtype=float)
    return aligned


def _check_label(rows: pd.DataFrame, access: RecipeAccess, rotulo: str) -> None:
    first, last = access.test_years
    in_test = [first <= week.year <= last for week in rows["target_week"]]
    require(all(in_test) or not any(in_test), "The rows mix development and test target years.")
    expected = TEST_LABEL if all(in_test) else DEVELOPMENT_LABEL
    require(rotulo == expected, f"rotulo must be {expected!r} for these rows; got {rotulo!r}.")


def _losses(frame: pd.DataFrame, column: str, access: RecipeAccess, clip: float) -> tuple[float, float]:
    y, weeks = frame["y"].to_numpy(), list(frame["target_week"])
    p = frame[column].to_numpy()
    return log_loss(y, p, weeks, access=access, clip=clip), brier(y, p, weeks, access=access)


def _cells(base: pd.DataFrame, regime: dict, candidates: dict, rotulo: str, access, clip) -> pd.DataFrame:
    records = []
    for (code, year), cell in base.groupby(["municipality_code", "anio"], sort=True):
        ll, br, mean = {}, {}, {}
        for system, column in _SCORED:
            ll[system], br[system] = _losses(cell, column, access, clip)
            mean[system] = float(cell[column].mean())
        prevalence = float(cell["y"].mean())
        names = sorted({candidates[(code, t)] for t in cell["t"] if (code, t) in candidates})
        records.append({
            "municipality_code": code, "anio": int(year), "regimen": regime[(code, int(year))], "rotulo": rotulo,
            "candidato": "+".join(names) if names else None, "filas": len(cell), "positivas": int(cell["y"].sum()),
            "prevalencia": prevalence,
            "p_medio_modelo": mean["modelo"], "p_medio_b2": mean["b2"], "p_medio_b1": mean["b1"],
            "calibracion_modelo": mean["modelo"] - prevalence, "calibracion_b2": mean["b2"] - prevalence,
            "calibracion_b1": mean["b1"] - prevalence,
            "logloss_modelo": ll["modelo"], "logloss_b2": ll["b2"], "logloss_b1": ll["b1"],
            "brier_modelo": br["modelo"], "brier_b2": br["b2"], "brier_b1": br["b1"],
            "dif_logloss_b2": ll["modelo"] - ll["b2"], "dif_logloss_b1": ll["modelo"] - ll["b1"],
            "dif_brier_b2": br["modelo"] - br["b2"], "dif_brier_b1": br["modelo"] - br["b1"],
            "mejora_logloss_b2": improvement(ll["modelo"], ll["b2"]),
            "mejora_logloss_b1": improvement(ll["modelo"], ll["b1"]),
            "bss_b2": improvement(br["modelo"], br["b2"]), "bss_b1": improvement(br["modelo"], br["b1"]),
            "gana_logloss_b2": ll["modelo"] < ll["b2"], "gana_brier_b2": br["modelo"] < br["b2"],
            "gana_logloss_b1": ll["modelo"] < ll["b1"], "gana_brier_b1": br["modelo"] < br["b1"]})
    return pd.DataFrame(records, columns=CASILLAS_COLUMNS)


def _summaries(base: pd.DataFrame, cells: pd.DataFrame, rotulo: str, access, clip) -> pd.DataFrame:
    records = []
    for level, column in (("global", None), ("municipio", "municipality_code"), ("anio", "anio"),
                          ("regimen", "regimen")):
        if column is None:
            groups = [("todos", cells, base)]
        else:
            groups = [(str(value), cells[cells[column] == value], base[base[column] == value])
                      for value in sorted(cells[column].unique())]
        for name, group_cells, group_rows in groups:
            ll, br = {}, {}
            for system, p_column in _SCORED:
                ll[system], br[system] = _losses(group_rows, p_column, access, clip)
            prevalence = float(group_rows["y"].mean())
            records.append({
                "nivel": level, "grupo": name, "rotulo": rotulo, "casillas": len(group_cells),
                "filas": len(group_rows), "positivas": int(group_rows["y"].sum()),
                **{f"victorias_{metric}": int(group_cells[f"gana_{metric}"].sum())
                   for metric in ("logloss_b2", "brier_b2", "logloss_b1", "brier_b1")},
                "mejora_logloss_b2_casillas": float(group_cells["mejora_logloss_b2"].mean()),
                "mejora_logloss_b2_filas": improvement(ll["modelo"], ll["b2"]),
                "bss_b2_casillas": float(group_cells["bss_b2"].mean()),
                "bss_b2_filas": improvement(br["modelo"], br["b2"]),
                "mejora_logloss_b1_casillas": float(group_cells["mejora_logloss_b1"].mean()),
                "mejora_logloss_b1_filas": improvement(ll["modelo"], ll["b1"]),
                "bss_b1_casillas": float(group_cells["bss_b1"].mean()),
                "bss_b1_filas": improvement(br["modelo"], br["b1"]),
                "calibracion_modelo_casillas": float(group_cells["calibracion_modelo"].mean()),
                "calibracion_modelo_filas": float(group_rows["p_modelo"].mean()) - prevalence,
                "calibracion_b2_casillas": float(group_cells["calibracion_b2"].mean()),
                "calibracion_b2_filas": float(group_rows["p_b2"].mean()) - prevalence})
    return pd.DataFrame(records, columns=RESUMEN_COLUMNS)


def _transition(base: pd.DataFrame, rotulo: str, access, clip) -> pd.DataFrame:
    records = []
    for year in [str(year) for year in sorted(base["anio"].unique())] + ["todos"]:
        year_rows = base if year == "todos" else base[base["anio"] == int(year)]
        inside = year_rows["zona_t"] == _EPIDEMIC_ZONE
        for stratum, part in (("fuera", year_rows[~inside]), ("dentro", year_rows[inside])):
            record = {"anio": year, "estrato": stratum, "rotulo": rotulo, "filas": len(part),
                      "positivas": int(part["y"].sum())}
            if len(part):   # an empty stratum keeps its row, with missing metrics
                ll_model, br_model = _losses(part, "p_modelo", access, clip)
                ll_b2, br_b2 = _losses(part, "p_b2", access, clip)
                y, weeks = part["y"].to_numpy(), list(part["target_week"])
                record.update({
                    "logloss_modelo": ll_model, "logloss_b2": ll_b2, "brier_modelo": br_model, "brier_b2": br_b2,
                    "mejora_logloss_b2": improvement(ll_model, ll_b2), "bss_b2": improvement(br_model, br_b2),
                    "ap_modelo": average_precision(y, part["p_modelo"].to_numpy(), weeks, access=access),
                    "ap_b2": average_precision(y, part["p_b2"].to_numpy(), weeks, access=access)})
            records.append(record)
    return pd.DataFrame(records, columns=TRANSICION_COLUMNS)


def _events(model, channel, regime: dict, rotulo: str) -> pd.DataFrame:
    shared = ["municipality_code", "kind", "week", "year", "evaluable", "censoring"]
    require(model.onsets[shared].equals(channel.onsets[shared]),
            "Model and channel were not evaluated on the same events.")
    records = []
    for m, c in zip(model.onsets.itertuples(), channel.onsets.itertuples()):
        both = bool(m.evaluable and m.warned and c.warned)
        records.append({
            "municipality_code": m.municipality_code, "tipo": m.kind, "semana": m.week, "anio": int(m.year),
            "regimen": regime[(m.municipality_code, int(m.year))], "rotulo": rotulo, "evaluable": bool(m.evaluable),
            "censura": m.censoring, "avisado_modelo": bool(m.warned), "avisado_canal": bool(c.warned),
            "confirmacion_modelo": m.warning_confirmation, "confirmacion_canal": c.warning_confirmation,
            "anticipacion_modelo": m.lead, "anticipacion_canal": c.lead,
            "delta_w": m.lead - c.lead if both else math.nan})
    return pd.DataFrame(records, columns=EVENTOS_COLUMNS)


def _alerts(model, channel, regime: dict, rotulo: str) -> pd.DataFrame:
    frames = []
    for system, period in (("modelo", model), ("canal", channel)):
        table = period.alerts.rename(columns=_ALERT_NAMES).assign(sistema=system, rotulo=rotulo)
        table["regimen"] = [regime[(code, int(year))] for code, year in zip(table["municipality_code"], table["anio"])]
        frames.append(table)
    return pd.concat(frames, ignore_index=True)[ALERTAS_COLUMNS]


def _alert_flags(series: pd.DataFrame, alert_table: pd.DataFrame) -> np.ndarray:
    """True for the emissions from t_c to the end of an alert of the same municipality."""
    flags = np.zeros(len(series), dtype=bool)
    codes, weeks = series["municipality_code"].to_numpy(), series["t"].to_numpy()
    for alert in alert_table.itertuples():
        flags |= (codes == alert.municipality_code) & (weeks >= alert.confirmation) & (weeks <= alert.end)
    return flags


def _series(base: pd.DataFrame, observed: pd.DataFrame, model, channel, tau: float, rotulo: str) -> pd.DataFrame:
    indexed = observed.set_index(["municipality_code", "week_start"])
    require(indexed.index.is_unique, "observed has duplicated (municipality_code, week_start).")
    keys = pd.MultiIndex.from_arrays([base["municipality_code"], base["target_week"]])
    missing = keys.difference(indexed.index)
    require(len(missing) == 0, f"observed lacks target weeks such as {list(missing[:3])}.")
    target = indexed.loc[keys]
    series = base[["municipality_code", "t", "target_week", "anio", "regimen", "zona_t", "y", "p_modelo", "p_b2",
                   "p_b1"]].reset_index(drop=True)
    series["rotulo"] = rotulo
    series["casos_objetivo"] = target["confirmed_records_screening"].to_numpy(dtype=int)
    series["central_objetivo"] = target["epico_central"].to_numpy(dtype=float)
    series["u1e_objetivo"] = target["epico_upper"].to_numpy(dtype=float)
    series["emision_modelo"] = model_positive(series["p_modelo"].to_numpy(), tau=tau)
    series["emision_canal"] = channel_positive(series["zona_t"])
    series["alerta_modelo"] = _alert_flags(series, model.alerts)
    series["alerta_canal"] = _alert_flags(series, channel.alerts)
    return series[SERIE_COLUMNS]


# --------------------------------------------------------------------------- the six inputs
def diagnostic_inputs(rows: pd.DataFrame, predictions: pd.DataFrame, observed: pd.DataFrame, *, tau: float, k: int,
                      window: int, clip: float, access: RecipeAccess, rotulo: str) -> Diagnostics:
    """The six inputs of contrato §8.8 for the rows of one period (development or test)."""
    _check_columns(rows, _ROW_COLUMNS, "rows")
    check_test_access(rows["target_week"], access)
    _check_label(rows, access, rotulo)
    _check_columns(predictions, _PREDICTION_COLUMNS, "predictions")
    _check_columns(observed, _OBSERVED_COLUMNS, "observed")
    require(len(rows) > 0, "rows must not be empty.")
    require(not rows.duplicated(_KEYS).any(), "rows have duplicated (municipality_code, t).")

    base = rows.sort_values(_KEYS, kind="mergesort").copy()
    probabilities = _aligned(base, predictions)
    base["p_modelo"], base["p_b2"], base["p_b1"] = probabilities["modelo"], probabilities["B2"], probabilities["B1"]
    base["anio"] = [week.year for week in base["target_week"]]
    regimes = regime_table(observed[observed["municipality_code"].isin(set(base["municipality_code"]))],
                           years=sorted(set(base["anio"])))
    regime = {(row.municipality_code, row.anio): row.regimen for row in regimes.itertuples()}
    base["regimen"] = [regime[(code, year)] for code, year in zip(base["municipality_code"], base["anio"])]
    candidates = {}
    if "candidato" in predictions.columns:
        own = predictions[predictions["sistema"] == "modelo"]
        candidates = dict(zip(zip(own["municipality_code"], own["t"]), own["candidato"]))

    cells = _cells(base, regime, candidates, rotulo, access, clip)
    model = evaluate_period(base, model_positive(base["p_modelo"].to_numpy(), tau=tau), observed, k=k,
                            window=window, access=access)
    channel = evaluate_period(base, channel_positive(base["zona_t"]), observed, k=k, window=window, access=access)
    return Diagnostics(casillas=cells, resumenes=_summaries(base, cells, rotulo, access, clip),
                       transicion=_transition(base, rotulo, access, clip),
                       eventos=_events(model, channel, regime, rotulo), alertas=_alerts(model, channel, regime, rotulo),
                       serie=_series(base, observed, model, channel, tau, rotulo))


def write_diagnostics(diagnostics: Diagnostics, folder: str | Path) -> list[Path]:
    """Write the six inputs as diag_<input>.csv in folder; existing files are never overwritten."""
    folder = Path(folder)
    paths = [folder / f"diag_{name}.csv" for name in INPUTS]
    existing = [str(path) for path in paths if path.exists()]
    require(not existing, f"Diagnostic files already exist and are never overwritten: {existing}.")
    folder.mkdir(parents=True, exist_ok=True)
    for name, path in zip(INPUTS, paths):
        getattr(diagnostics, name).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return paths


# --------------------------------------------------------------------------- command line (development)
def main(argv: list[str] | None = None) -> int:
    from .config import DEFAULT_CONFIG_PATH, load_config
    from .dataset import build_examples, load_channel_product
    from .guard import load_frozen_recipe, FROZEN_RECIPE_PATH
    from .run_selection import development_rows
    from .search import BASELINE_PREDICTIONS_FILENAME, DEFAULT_OUTPUT_DIR, PREDICTIONS_FILENAME, TABLE_FILENAME

    parser = argparse.ArgumentParser(description="Development diagnostic inputs of contrato §8.8 (no retraining).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--busqueda", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        recipe = load_frozen_recipe(FROZEN_RECIPE_PATH, channel_version=config.datos.version_dvc_canal)
        product = load_channel_product(config.datos.canal, expected_md5=config.datos.version_dvc_canal)
        examples = build_examples(product, h=config.horizonte.h, canal=config.canal, features=config.features)
        long = development_predictions(pd.read_parquet(args.busqueda / TABLE_FILENAME),
                                       pd.read_parquet(args.busqueda / PREDICTIONS_FILENAME),
                                       pd.read_parquet(args.busqueda / BASELINE_PREDICTIONS_FILENAME),
                                       familia=recipe["familia_comun"])
        with tempfile.TemporaryDirectory() as folder:   # development needs no recipe: prove it with none
            access = RecipeAccess.from_config(config, recipe_path=Path(folder) / "sin_receta.yaml")
            diagnostics = diagnostic_inputs(development_rows(examples, config), long, product.to_pandas(),
                                            tau=float(recipe["decision"]["tau"]), k=config.decision.k,
                                            window=config.decision.W, clip=config.metricas.recorte_log_loss,
                                            access=access, rotulo=DEVELOPMENT_LABEL)
        for path in write_diagnostics(diagnostics, args.salida):
            print(f"written {path}")
    except Exception as exc:   # reported with its traceback
        traceback.print_exc()
        print(f"Diagnostics stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
