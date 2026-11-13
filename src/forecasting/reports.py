"""Derivable reports of the forecasting MVP (unit U7.5; contrato §7.2, §7.3, §7.4 and §8.4; especificación §6.F7).
Nothing is refitted and the single test is not run again: every table and figure comes from saved predictions.

Flow:
    development (rotulo "usado_para_elegir"), development_systems(rows, search_dir, sin_clima_dir, familia):
      recipe   = common out-of-year configuration of `familia` in the sin_clima search (as diagnostics.py, decision U7.1-D3);
      ablation = common out-of-year configuration of `familia` in the full-set search (F5b), the cells of step 0;
      B2, B1   = saved baselines of the sin_clima search (identical to F5b).
    test (rotulo "prueba"), test_systems(prueba_dir): the four systems of predicciones_prueba.parquet.
      -> PeriodSystems: rows sorted by (municipality_code, t) and p of modelo, ablacion, B2 and B1 on exactly them
    period_reports(systems, observed, config, tau, access): test-period barrier first, then
      casillas           Brier and log-loss of the four systems per (municipality, year of the target week);
      habilidad          paired_skill: BSS and log-loss improvement over the rows together and as the mean of the
                         cells, and cells won (strictly smaller loss), for modelo and ablacion vs B2, B1 (and vs each
                         other);
      clasificacion_tau  classification_at_tau of modelo and B2 at the recipe tau, global, by year and municipality;
      murphy, confiabilidad  murphy_decomposition and reliability_bins of modelo and ablacion, global and by year;
      calibracion_anual  calibration_by_year of the four systems (mean p − prevalence, rows together);
      eventos_razon      event_report of the model (p >= tau) and the channel (zone "alerta"), global and by municipality;
      duracion_alertas   alert_durations, descriptive and outside the contract;
      curva_amoc         amoc_curve of the model over decision.tau_grid with the channel as one row; in the test
                         period it is descriptive and does not change tau (contrato §7.5).
    The command line adds the development curve of the tau choice (the recipe configuration in the four years,
    logistica_05 in the current recipe;
    development_recipe_probabilities), the table desarrollo_vs_prueba and the figures (PNG and SVG).

Declared conventions (plan approved by the person responsible for the implementation on 2026-11-12; frozen in tests/acceptance/test_f7c_formulas.py):
- Reliability intervals: pd.qcut(p, RELIABILITY_BINS, labels=False, duplicates="drop") on the p of each system over
  the rows of the set (the same call as artifacts._calibration); equal p fall in the same interval; coinciding edges
  merge intervals; by year the deciles are recomputed within the year.
- Murphy (1973): REL = (1/N) Σ n_k (p̄_k − ō_k)², RES = (1/N) Σ n_k (ō_k − ō)², UNC = ō(1 − ō);
  residuo = BS − (REL − RES + UNC) = WBV − 2·WBC (Stephenson, Coelho and Jolliffe, 2008).
- At tau: positive emission p >= tau; sensitivity, specificity and PPV NaN with a zero denominator; F1 = 2TP/(2TP + FP
  + FN), NaN only if TP + FP + FN = 0; MCC NaN when a factor of its denominator is 0; F1 reference 2π/(1 + π).
- Alert ratio of events._summarize (onsets warned by judgeable alerts / judgeable alerts; it may exceed 1); composite
  0.5·detection + 0.5·ratio. Duration of an alert: emissions from t_c to its end, both included.

Command line (refused with uncommitted changes in src/ or configs/ unless --permitir-cambios, recorded in the
manifest; never overwrites):
    python -B -X utf8 -m src.forecasting.reports --salida reports/forecasting/evidence/reportes_<fecha>
        --figuras reports/forecasting/figures/<fecha>
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

from src.epidemiology.validation import require

from .config import DEFAULT_CONFIG_PATH, ROOT, ForecastConfig, load_config
from .diagnostics import DEVELOPMENT_LABEL, TEST_LABEL, development_predictions
from .events import WEEK, PeriodEvents, channel_positive, evaluate_period, model_positive
from .evaluate import aligned_systems
from .guard import RecipeAccess, check_test_access, load_frozen_recipe
from .metrics import _scored_rows, brier, improvement, log_loss

# Number of equal-frequency intervals of the reliability diagram (contrato §7.2): part of the definition, not a
# tunable parameter (the same value as the diagram of the performance card, artifacts._calibration).
RELIABILITY_BINS = 10
SYSTEMS = ("modelo", "ablacion", "B2", "B1")
DEVELOPMENT, TEST = "desarrollo", "prueba"
COMMON_CONFIGURATION, RECIPE_CONFIGURATION, TAU_CHOICE_CONFIGURATION = (
    "comun_fuera_de_anio", "receta", "eleccion_tau_receta")
_KEYS = ["municipality_code", "t"]
_ROW_COLUMNS = ["municipality_code", "t", "target_week", "anio", "zona_t", "y"]
_SKILL_PAIRS = (("modelo", "B2"), ("modelo", "B1"), ("modelo", "ablacion"), ("ablacion", "B2"), ("ablacion", "B1"))
TABLES = ("casillas", "habilidad", "clasificacion_tau", "murphy", "confiabilidad", "calibracion_anual",
          "eventos_razon", "duracion_alertas", "curva_amoc", "desarrollo_vs_prueba")
_BOGOTA = timezone(timedelta(hours=-5))


# =========================================================================== formulas (rows of one set)
def _bins(p: np.ndarray) -> np.ndarray:
    if np.unique(p).size == 1:     # qcut cannot cut a constant vector: one interval
        return np.zeros(len(p), dtype=int)
    return np.asarray(pd.qcut(p, RELIABILITY_BINS, labels=False, duplicates="drop"), dtype=int)


def reliability_bins(y, p, target_weeks, *, access: RecipeAccess) -> pd.DataFrame:
    """Equal-frequency intervals of p with their size, mean p and observed frequency (contrato §7.2)."""
    y, p = _scored_rows(y, p, target_weeks, access)
    labels = _bins(p)
    records = []
    for label in sorted(set(labels)):
        mask = labels == label
        records.append({"intervalo": int(label), "n": int(mask.sum()), "p_media": float(p[mask].mean()),
                        "frecuencia_observada": float(y[mask].mean()), "p_min": float(p[mask].min()),
                        "p_max": float(p[mask].max())})
    return pd.DataFrame(records, columns=["intervalo", "n", "p_media", "frecuencia_observada", "p_min", "p_max"])


def murphy_decomposition(y, p, target_weeks, *, access: RecipeAccess) -> dict:
    """Brier score and its Murphy decomposition with the reliability intervals; residuo = WBV − 2·WBC."""
    y, p = _scored_rows(y, p, target_weeks, access)
    labels, n, base = _bins(p), len(y), float(y.mean())
    reliability = resolution = within_variance = within_covariance = 0.0
    for label in sorted(set(labels)):
        mask = labels == label
        size, p_bar, o_bar = int(mask.sum()), float(p[mask].mean()), float(y[mask].mean())
        reliability += size * (p_bar - o_bar) ** 2
        resolution += size * (o_bar - base) ** 2
        within_variance += float(((p[mask] - p_bar) ** 2).sum())
        within_covariance += float(((p[mask] - p_bar) * (y[mask] - o_bar)).sum())
    score = float(np.mean((p - y) ** 2))
    reliability, resolution = reliability / n, resolution / n
    uncertainty = base * (1 - base)
    return {"filas": n, "intervalos": len(set(labels)), "brier": score, "confiabilidad": reliability,
            "resolucion": resolution, "incertidumbre": uncertainty, "varianza_intra": within_variance / n,
            "covarianza_intra": within_covariance / n, "residuo": score - (reliability - resolution + uncertainty)}


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def classification_at_tau(y, p, target_weeks, *, tau: float, access: RecipeAccess) -> dict:
    """Weekly emission 1[p >= tau] against y (contrato §7.3); NaN where a denominator is 0."""
    y, p = _scored_rows(y, p, target_weeks, access)
    emitted = model_positive(p, tau=tau)
    actual = y == 1
    tp, fp = int((emitted & actual).sum()), int((emitted & ~actual).sum())
    fn, tn = int((~emitted & actual).sum()), int((~emitted & ~actual).sum())
    factors = (tp + fp, tp + fn, tn + fp, tn + fn)
    prevalence = float(actual.mean())
    return {"filas": len(y), "positivas": int(actual.sum()), "prevalencia": prevalence,
            "emisiones_positivas": int(emitted.sum()), "vp": tp, "fp": fp, "fn": fn, "vn": tn,
            "sensibilidad": _ratio(tp, tp + fn), "especificidad": _ratio(tn, tn + fp), "vpp": _ratio(tp, tp + fp),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn), "f1_referencia": 2 * prevalence / (1 + prevalence),
            "mcc": (tp * tn - fp * fn) / math.sqrt(math.prod(factors)) if all(factors) else math.nan}


def _check_rows(rows, columns, name: str = "rows") -> None:
    require(isinstance(rows, pd.DataFrame), f"{name} must be a pandas DataFrame.")
    missing = [column for column in columns if column not in rows.columns]
    require(not missing, f"{name} lacks columns {missing}.")
    require(len(rows) > 0, f"{name} must not be empty.")


def _array(values, length: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    require(array.ndim == 1 and len(array) == length, f"{name} must have one value per row ({length}).")
    return array


def calibration_by_year(rows: pd.DataFrame, probabilities: dict, *, access: RecipeAccess) -> pd.DataFrame:
    """Calibration in the large (mean p − prevalence) per calendar year of the target week and over all the rows,
    with the rows of the municipalities together (contrato §7.2)."""
    _check_rows(rows, ("target_week", "y"))
    check_test_access(rows["target_week"], access)
    require(isinstance(probabilities, dict) and probabilities, "probabilities must be a non-empty mapping.")
    p = {name: _array(values, len(rows), f"probabilities[{name!r}]") for name, values in probabilities.items()}
    y = rows["y"].to_numpy(dtype=float)
    years = np.array([week.year for week in rows["target_week"]])
    records = []
    for year in [str(year) for year in sorted(set(years))] + ["todos"]:
        mask = np.ones(len(rows), dtype=bool) if year == "todos" else years == int(year)
        prevalence = float(y[mask].mean())
        record = {"anio": year, "filas": int(mask.sum()), "positivas": int(y[mask].sum()), "prevalencia": prevalence}
        for name, values in p.items():
            record[f"p_medio_{name}"] = float(values[mask].mean())
            record[f"calibracion_{name}"] = float(values[mask].mean()) - prevalence
        records.append(record)
    return pd.DataFrame(records)


def _cell_masks(rows: pd.DataFrame) -> list[tuple[str, int, np.ndarray]]:
    codes = rows["municipality_code"].to_numpy()
    years = np.array([week.year for week in rows["target_week"]])
    return [(code, int(year), (codes == code) & (years == year))
            for code in sorted(set(codes)) for year in sorted(set(years[codes == code]))]


def paired_skill(rows: pd.DataFrame, p_option, p_reference, *, clip: float, access: RecipeAccess) -> dict:
    """Option against a reference over the same rows (contrato §7.2 and §8.4): improvements over the rows together
    and as the mean of the cell improvements, and cells won (strictly smaller loss)."""
    _check_rows(rows, ("municipality_code", "target_week", "y"))
    check_test_access(rows["target_week"], access)
    option, reference = _array(p_option, len(rows), "p_option"), _array(p_reference, len(rows), "p_reference")
    y, weeks = rows["y"].to_numpy(), list(rows["target_week"])
    losses = {}
    for name, p in (("opcion", option), ("referencia", reference)):
        losses[name] = (brier(y, p, weeks, access=access), log_loss(y, p, weeks, access=access, clip=clip))
    cells = []
    for _, _, mask in _cell_masks(rows):
        cell_weeks = [weeks[i] for i in np.flatnonzero(mask)]
        mine = (brier(y[mask], option[mask], cell_weeks, access=access),
                log_loss(y[mask], option[mask], cell_weeks, access=access, clip=clip))
        theirs = (brier(y[mask], reference[mask], cell_weeks, access=access),
                  log_loss(y[mask], reference[mask], cell_weeks, access=access, clip=clip))
        cells.append((mine, theirs))
    return {"casillas": len(cells),
            "ganadas_brier": sum(mine[0] < theirs[0] for mine, theirs in cells),
            "ganadas_logloss": sum(mine[1] < theirs[1] for mine, theirs in cells),
            "bss_filas": improvement(losses["opcion"][0], losses["referencia"][0]),
            "bss_casillas": float(np.mean([improvement(mine[0], theirs[0]) for mine, theirs in cells])),
            "mejora_logloss_filas": improvement(losses["opcion"][1], losses["referencia"][1]),
            "mejora_logloss_casillas": float(np.mean([improvement(mine[1], theirs[1]) for mine, theirs in cells]))}


# =========================================================================== events (contrato §7.4)
EVENT_COLUMNS = [
    "nivel", "grupo", "inicios_evaluables", "inicios_avisados", "deteccion", "inicios_avisados_por_juzgable",
    "alertas", "alertas_continuidad", "alertas_juzgables", "alertas_censuradas", "alertas_que_avisan",
    "alertas_con_varios_inicios", "razon_alertas", "compuesto", "falsas_alarmas", "proporcion_falsas_alarmas",
    "falsas_seguidas_epidemia", "falsas_no_seguidas", "municipio_anios", "falsas_por_municipio_anio",
    "anticipacion_mediana", "entradas_evaluables", "entradas_avisadas"]
DURATION_COLUMNS = [
    "nivel", "grupo", "alertas", "alertas_no_continuidad", "alertas_truncadas", "duracion_mediana", "duracion_p75",
    "duracion_max", "semanas_emision", "semanas_bajo_alerta", "semanas_bajo_alerta_no_continuidad",
    "fraccion_bajo_alerta"]


def _groups(period: PeriodEvents) -> list[tuple[str, str, dict, pd.DataFrame, pd.DataFrame]]:
    """(nivel, grupo, summary, onsets, alerts) for the whole period and for each municipality."""
    out = [("global", "todos", period.summary, period.onsets, period.alerts)]
    for code, summary in period.by_municipality.iterrows():
        out.append(("municipio", str(code), summary.to_dict(), period.onsets[period.onsets["municipality_code"] == code],
                    period.alerts[period.alerts["municipality_code"] == code]))
    return out


def _warning_counts(onsets: pd.DataFrame, alerts: pd.DataFrame, window: int) -> list[int]:
    """Evaluable onsets in [t_c + 1, t_c + W] of each judgeable non-continuity alert (same municipality)."""
    evaluable = onsets[(onsets["kind"] == "inicio") & onsets["evaluable"]]
    judged = alerts[~alerts["continuity"] & alerts["judgeable"]]
    counts = []
    for alert in judged.itertuples():
        own = evaluable[evaluable["municipality_code"] == alert.municipality_code]
        counts.append(sum(alert.confirmation + WEEK <= week <= alert.confirmation + window * WEEK
                          for week in own["week"]))
    return counts


def _number(value) -> float:
    return math.nan if value is None else float(value)


def event_report(rows: pd.DataFrame, positive, observed: pd.DataFrame, *, k: int, window: int,
                 access: RecipeAccess) -> pd.DataFrame:
    """Event metrics of contrato §7.4 for one system, global and by municipality: detection, alert ratio (of
    events._summarize), composite, false alarms with their breakdown, and the alerts that warn."""
    period = evaluate_period(rows, positive, observed, k=k, window=window, access=access)
    records = []
    for level, group, summary, onsets, alerts in _groups(period):
        counts = _warning_counts(onsets, alerts, window)
        detection, ratio = _number(summary["detection"]), _number(summary["alert_ratio"])
        false_alarms, judgeable = int(summary["false_alarms"]), int(summary["judgeable_alerts"])
        followed = int(summary["false_alarms_followed_by_epidemic"])
        records.append({
            "nivel": level, "grupo": group, "inicios_evaluables": int(summary["evaluable_onsets"]),
            "inicios_avisados": int(summary["warned_onsets"]), "deteccion": detection,
            "inicios_avisados_por_juzgable": int(summary["onsets_warned_by_judgeable"]), "alertas": len(alerts),
            "alertas_continuidad": int(summary["continuity_alerts"]), "alertas_juzgables": judgeable,
            "alertas_censuradas": int(summary["censored_alerts"]),
            "alertas_que_avisan": sum(count >= 1 for count in counts),
            "alertas_con_varios_inicios": sum(count >= 2 for count in counts), "razon_alertas": ratio,
            "compuesto": 0.5 * detection + 0.5 * ratio, "falsas_alarmas": false_alarms,
            "proporcion_falsas_alarmas": _ratio(false_alarms, judgeable), "falsas_seguidas_epidemia": followed,
            "falsas_no_seguidas": false_alarms - followed, "municipio_anios": int(summary["municipality_years"]),
            "falsas_por_municipio_anio": float(summary["false_alarms_per_municipality_year"]),
            "anticipacion_mediana": _number(summary["lead_median"]),
            "entradas_evaluables": int(summary["evaluable_entries"]),
            "entradas_avisadas": int(summary["warned_entries"])})
    return pd.DataFrame(records, columns=EVENT_COLUMNS)


def alert_durations(rows: pd.DataFrame, positive, observed: pd.DataFrame, *, k: int, window: int,
                    access: RecipeAccess) -> pd.DataFrame:
    """Descriptive, outside the contract: duration of the non-continuity alerts (emissions from t_c to the end) and
    the weeks under alert of one system, global and by municipality."""
    period = evaluate_period(rows, positive, observed, k=k, window=window, access=access)
    last = rows.groupby("municipality_code")["t"].max().to_dict()
    emissions = rows.groupby("municipality_code").size().to_dict()
    records = []
    for level, group, _, _, alerts in _groups(period):
        weeks = np.array([(end - confirmation).days // 7 + 1
                          for confirmation, end in zip(alerts["confirmation"], alerts["end"])], dtype=float)
        continuity = alerts["continuity"].to_numpy(dtype=bool)
        own = weeks[~continuity]
        truncated = sum(end == last[code] for code, end in zip(alerts["municipality_code"], alerts["end"]))
        n_emissions = sum(emissions.values()) if level == "global" else emissions[group]
        records.append({
            "nivel": level, "grupo": group, "alertas": len(alerts), "alertas_no_continuidad": int(len(own)),
            "alertas_truncadas": int(truncated),
            "duracion_mediana": float(np.median(own)) if len(own) else math.nan,
            "duracion_p75": float(np.percentile(own, 75)) if len(own) else math.nan,
            "duracion_max": float(own.max()) if len(own) else math.nan, "semanas_emision": int(n_emissions),
            "semanas_bajo_alerta": int(weeks.sum()), "semanas_bajo_alerta_no_continuidad": int(own.sum()),
            "fraccion_bajo_alerta": float(weeks.sum()) / n_emissions})
    return pd.DataFrame(records, columns=DURATION_COLUMNS)


AMOC_COLUMNS = ["sistema", "tau", "inicios_evaluables", "inicios_avisados", "deteccion", "falsas_alarmas",
                "municipio_anios", "falsas_por_municipio_anio", "alertas_juzgables", "alertas_continuidad",
                "anticipacion_mediana"]


def _curve_point(system: str, tau: float, summary: dict) -> dict:
    return {"sistema": system, "tau": tau, "inicios_evaluables": int(summary["evaluable_onsets"]),
            "inicios_avisados": int(summary["warned_onsets"]), "deteccion": _number(summary["detection"]),
            "falsas_alarmas": int(summary["false_alarms"]), "municipio_anios": int(summary["municipality_years"]),
            "falsas_por_municipio_anio": float(summary["false_alarms_per_municipality_year"]),
            "alertas_juzgables": int(summary["judgeable_alerts"]),
            "alertas_continuidad": int(summary["continuity_alerts"]),
            "anticipacion_mediana": _number(summary["lead_median"])}


def amoc_curve(rows: pd.DataFrame, p, observed: pd.DataFrame, *, tau_grid, k: int, window: int,
               access: RecipeAccess) -> pd.DataFrame:
    """AMOC-type curve (contrato §7.4): detection, false alarms and median lead of the model over the tau grid, and
    the channel as one row (tau NaN), on the same rows. Descriptive in the test period: it never changes tau."""
    check_test_access(rows["target_week"], access)
    grid = [float(tau) for tau in tau_grid]
    require(grid, "tau_grid must not be empty.")
    p = _array(p, len(rows), "p")
    records = [_curve_point("modelo", tau, evaluate_period(rows, model_positive(p, tau=tau), observed, k=k,
                                                           window=window, access=access).summary) for tau in grid]
    channel = evaluate_period(rows, channel_positive(rows["zona_t"]), observed, k=k, window=window, access=access)
    records.append(_curve_point("canal", math.nan, channel.summary))
    return pd.DataFrame(records, columns=AMOC_COLUMNS)


# =========================================================================== periods
@dataclass(frozen=True)
class PeriodSystems:
    """Rows of one period and the probabilities of the four systems on exactly those rows."""

    periodo: str
    rotulo: str
    rows: pd.DataFrame          # _ROW_COLUMNS, sorted by (municipality_code, t)
    p: dict                     # system -> np.ndarray in the order of rows
    candidatos: pd.DataFrame    # municipality_code, anio, sistema, candidato (modelo and ablacion)


def _sorted_rows(rows: pd.DataFrame) -> pd.DataFrame:
    _check_rows(rows, ("municipality_code", "t", "target_week", "zona_t", "y"))
    require(not rows.duplicated(_KEYS).any(), "rows have duplicated (municipality_code, t).")
    base = rows.sort_values(_KEYS, kind="mergesort").reset_index(drop=True)
    base = base.assign(anio=[week.year for week in base["target_week"]])
    return base[_ROW_COLUMNS]


def _candidates(long: pd.DataFrame) -> pd.DataFrame:
    own = long[long["sistema"].isin(["modelo", "ablacion"])]
    return (own[["municipality_code", "anio", "sistema", "candidato"]].drop_duplicates()
            .sort_values(["sistema", "municipality_code", "anio"]).reset_index(drop=True))


def _search_files(folder: Path) -> dict:
    folder = Path(folder)
    return {name: pd.read_parquet(folder / f"{name}.parquet")
            for name in ("tabla_unica_desarrollo", "predicciones_desarrollo", "predicciones_lineas_base_desarrollo")}


def development_systems(rows: pd.DataFrame, *, search_dir, sin_clima_dir, familia: str) -> PeriodSystems:
    """Development period «usado para elegir»: the recipe and the ablation with their common out-of-year
    configuration in each (municipality, year), and the saved baselines (decision U7.5-D5, second sense: the inputs
    of each period; glossary of docs/forecasting/README.md)."""
    base = _sorted_rows(rows)
    years = sorted(set(base["anio"]))
    simple, full = _search_files(sin_clima_dir), _search_files(search_dir)
    recipe = development_predictions(simple["tabla_unica_desarrollo"], simple["predicciones_desarrollo"],
                                     simple["predicciones_lineas_base_desarrollo"], familia=familia)
    ablation = development_predictions(full["tabla_unica_desarrollo"], full["predicciones_desarrollo"],
                                       full["predicciones_lineas_base_desarrollo"], familia=familia)
    long = pd.concat([recipe, ablation[ablation["sistema"] == "modelo"].assign(sistema="ablacion")],
                     ignore_index=True)
    long = long[long["anio"].isin(years)]
    p = aligned_systems(base, long, SYSTEMS)
    return PeriodSystems(periodo=DEVELOPMENT, rotulo=DEVELOPMENT_LABEL, rows=base, p=p, candidatos=_candidates(long))


def development_recipe_probabilities(rows: pd.DataFrame, *, sin_clima_dir, familia: str, candidato: str) -> np.ndarray:
    """Saved development predictions of one configuration in every year (the tau choice of contrato §7.5 used the
    recipe configuration, logistica_05 in the current recipe), in the order of the sorted rows."""
    base = _sorted_rows(rows)
    saved = pd.read_parquet(Path(sin_clima_dir) / "predicciones_desarrollo.parquet")
    own = saved[(saved["familia"] == familia) & (saved["candidato"] == candidato)]
    own = own[own["anio"].isin(set(base["anio"]))].assign(sistema="modelo")
    return aligned_systems(base, own, ("modelo",))["modelo"]


def test_systems(prueba_dir) -> PeriodSystems:
    """Test period: the four systems of the single test run (predicciones_prueba.parquet)."""
    predictions = pd.read_parquet(Path(prueba_dir) / "predicciones_prueba.parquet")
    require(set(predictions["sistema"]) == set(SYSTEMS), f"Test predictions must hold the systems {SYSTEMS}.")
    base = _sorted_rows(predictions[predictions["sistema"] == "modelo"])
    p = aligned_systems(base, predictions, SYSTEMS)
    return PeriodSystems(periodo=TEST, rotulo=TEST_LABEL, rows=base, p=p, candidatos=_candidates(predictions))


test_systems.__test__ = False   # a function of this module, not a pytest test


def _tag(frame: pd.DataFrame, systems: PeriodSystems, **columns) -> pd.DataFrame:
    tagged = frame.copy()
    for position, (name, value) in enumerate({"periodo": systems.periodo, "rotulo": systems.rotulo,
                                              **columns}.items()):
        tagged.insert(position, name, value)
    return tagged


def _cells(systems: PeriodSystems, clip: float, access: RecipeAccess) -> pd.DataFrame:
    rows, records = systems.rows, []
    y, weeks = rows["y"].to_numpy(), list(rows["target_week"])
    for code, year, mask in _cell_masks(rows):
        cell_weeks = [weeks[i] for i in np.flatnonzero(mask)]
        record = {"municipality_code": code, "anio": year, "filas": int(mask.sum()), "positivas": int(y[mask].sum())}
        for system in SYSTEMS:
            record[f"brier_{system}"] = brier(y[mask], systems.p[system][mask], cell_weeks, access=access)
            record[f"logloss_{system}"] = log_loss(y[mask], systems.p[system][mask], cell_weeks, access=access,
                                                   clip=clip)
        for system in ("modelo", "ablacion"):
            record[f"bss_{system}_vs_B2"] = improvement(record[f"brier_{system}"], record["brier_B2"])
            record[f"mejora_logloss_{system}_vs_B2"] = improvement(record[f"logloss_{system}"], record["logloss_B2"])
        records.append(record)
    return _tag(pd.DataFrame(records), systems)


def _subsets(rows: pd.DataFrame, by_municipality: bool) -> list[tuple[str, str, np.ndarray]]:
    out = [("global", "todos", np.ones(len(rows), dtype=bool))]
    out += [("anio", str(year), (rows["anio"] == year).to_numpy()) for year in sorted(set(rows["anio"]))]
    if by_municipality:
        out += [("municipio", code, (rows["municipality_code"] == code).to_numpy())
                for code in sorted(set(rows["municipality_code"]))]
    return out


def period_reports(systems: PeriodSystems, observed: pd.DataFrame, *, config: ForecastConfig, tau: float,
                   access: RecipeAccess) -> dict:
    """Every derivable table of one period (see the module docstring). Test-period barrier first."""
    rows = systems.rows
    check_test_access(rows["target_week"], access)
    require(set(systems.p) == set(SYSTEMS), f"systems.p must hold {SYSTEMS}.")
    clip, decision = config.metricas.recorte_log_loss, config.decision
    y, weeks = rows["y"].to_numpy(), list(rows["target_week"])

    skill = pd.DataFrame([{"sistema": option, "referencia": reference,
                           **paired_skill(rows, systems.p[option], systems.p[reference], clip=clip, access=access)}
                          for option, reference in _SKILL_PAIRS])
    classification, murphy, bins = [], [], []
    for level, group, mask in _subsets(rows, by_municipality=True):
        part_weeks = [weeks[i] for i in np.flatnonzero(mask)]
        for system in ("modelo", "B2"):
            classification.append({"sistema": system, "nivel": level, "grupo": group, "tau": float(tau),
                                   **classification_at_tau(y[mask], systems.p[system][mask], part_weeks, tau=tau,
                                                           access=access)})
        if level == "municipio":
            continue
        for system in ("modelo", "ablacion"):
            murphy.append({"sistema": system, "nivel": level, "grupo": group,
                           **murphy_decomposition(y[mask], systems.p[system][mask], part_weeks, access=access)})
            table = reliability_bins(y[mask], systems.p[system][mask], part_weeks, access=access)
            bins.append(table.assign(sistema=system, nivel=level, grupo=group))
    bins_table = pd.concat(bins, ignore_index=True)
    bins_table = bins_table[["sistema", "nivel", "grupo"] + [c for c in bins_table.columns
                                                             if c not in ("sistema", "nivel", "grupo")]]

    positives = {"modelo": model_positive(systems.p["modelo"], tau=tau), "canal": channel_positive(rows["zona_t"])}
    events = pd.concat([event_report(rows, positive, observed, k=decision.k, window=decision.W, access=access)
                        .assign(sistema=system) for system, positive in positives.items()], ignore_index=True)
    durations = pd.concat([alert_durations(rows, positive, observed, k=decision.k, window=decision.W, access=access)
                           .assign(sistema=system) for system, positive in positives.items()], ignore_index=True)
    configuration = COMMON_CONFIGURATION if systems.periodo == DEVELOPMENT else RECIPE_CONFIGURATION
    curve = amoc_curve(rows, systems.p["modelo"], observed, tau_grid=decision.tau_grid, k=decision.k,
                       window=decision.W, access=access)
    first = lambda frame: frame[["sistema"] + [c for c in frame.columns if c != "sistema"]]  # noqa: E731
    return {
        "casillas": _cells(systems, clip, access),
        "habilidad": _tag(skill, systems),
        "clasificacion_tau": _tag(pd.DataFrame(classification), systems),
        "murphy": _tag(pd.DataFrame(murphy), systems),
        "confiabilidad": _tag(bins_table, systems),
        "calibracion_anual": _tag(calibration_by_year(rows, systems.p, access=access), systems),
        "eventos_razon": _tag(first(events), systems),
        "duracion_alertas": _tag(first(durations), systems),
        "curva_amoc": _tag(curve, systems, configuracion=configuration),
    }


def development_vs_test(development: dict, test: dict) -> pd.DataFrame:
    """Skill of the recipe and of the ablation in both periods, with the events of the model and, instead of the
    ablation's (not computed: the contract compares it only weekly, §8.4), those of the channel."""
    frames = []
    for tables in (development, test):
        skill = tables["habilidad"].copy()
        events = tables["eventos_razon"]
        events = events[events["nivel"] == "global"].set_index("sistema")
        for column in ("inicios_evaluables", "inicios_avisados", "falsas_alarmas", "municipio_anios",
                       "falsas_por_municipio_anio"):
            skill[column] = [events.loc["modelo", column] if system == "modelo" else math.nan
                             for system in skill["sistema"]]
        channel = events.loc[["canal"]].reset_index()[["periodo", "rotulo", "sistema", "inicios_evaluables",
                                                       "inicios_avisados", "falsas_alarmas", "municipio_anios",
                                                       "falsas_por_municipio_anio"]]
        frames += [skill, channel]
    return pd.concat(frames, ignore_index=True)


# =========================================================================== command line
def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    """Path relative to the repository when it is inside it (forward slashes), else absolute."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _git(*args: str) -> str:
    result = subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip()


def _check_search(folder: Path, provenance: dict, name: str) -> dict:
    expected = provenance["busquedas"][name]["insumos_md5"]
    found = {}
    for file, md5 in expected.items():
        found[file] = _md5(folder / file)
        require(found[file] == md5, f"{folder / file} is not the file of the recipe provenance ({name}).")
    return found


def build_all(config: ForecastConfig, recipe: dict, *, search_dir: Path, sin_clima_dir: Path, prueba_dir: Path,
              recipe_path: Path) -> tuple[dict, dict, dict]:
    """Tables of both periods (combined), the systems of each period and the observed product."""
    from .dataset import build_examples, load_channel_product
    from .run_selection import development_rows

    canal = Path(config.datos.canal)
    product = load_channel_product(canal if canal.is_absolute() else ROOT / canal,
                                   expected_md5=config.datos.version_dvc_canal)
    observed = product.to_pandas()
    examples = build_examples(product, h=config.horizonte.h, canal=config.canal, features=config.features)
    tau, decision, familia = float(recipe["decision"]["tau"]), config.decision, recipe["familia_comun"]
    candidates = {spec["candidato"] for spec in recipe["municipios"].values()}
    require(len(candidates) == 1, "The tau-choice curve needs one recipe configuration for every municipality.")

    rows = development_rows(examples, config)
    development = development_systems(rows, search_dir=search_dir, sin_clima_dir=sin_clima_dir, familia=familia)
    with tempfile.TemporaryDirectory() as folder:   # development needs no recipe: prove it with none
        access = RecipeAccess.from_config(config, recipe_path=Path(folder) / "sin_receta.yaml")
        dev_tables = period_reports(development, observed, config=config, tau=tau, access=access)
        recipe_p = development_recipe_probabilities(rows, sin_clima_dir=sin_clima_dir, familia=familia,
                                                    candidato=candidates.pop())
        choice = amoc_curve(development.rows, recipe_p, observed, tau_grid=decision.tau_grid, k=decision.k,
                            window=decision.W, access=access)
        events = {DEVELOPMENT: _period_events(development, observed, tau=tau, config=config, access=access)}
    choice = _tag(choice, development, configuracion=TAU_CHOICE_CONFIGURATION)
    dev_tables["curva_amoc"] = pd.concat([choice, dev_tables["curva_amoc"]], ignore_index=True)

    test = test_systems(prueba_dir)
    access = RecipeAccess.from_config(config, recipe_path=recipe_path)
    test_tables = period_reports(test, observed, config=config, tau=tau, access=access)
    events[TEST] = _period_events(test, observed, tau=tau, config=config, access=access)
    combined = {name: pd.concat([dev_tables[name], test_tables[name]], ignore_index=True) for name in dev_tables}
    combined["desarrollo_vs_prueba"] = development_vs_test(dev_tables, test_tables)
    return combined, {DEVELOPMENT: development, TEST: test}, {"observed": observed, "tau": tau,
                                                             "recipe_p": recipe_p, "eventos": events}


def _period_events(systems: PeriodSystems, observed: pd.DataFrame, *, tau: float, config: ForecastConfig,
                   access: RecipeAccess) -> dict:
    """Alerts and onsets of the model and the channel of one period, for the weekly series figures."""
    rows, decision = systems.rows, config.decision
    return {"modelo": evaluate_period(rows, model_positive(systems.p["modelo"], tau=tau), observed, k=decision.k,
                                      window=decision.W, access=access),
            "canal": evaluate_period(rows, channel_positive(rows["zona_t"]), observed, k=decision.k,
                                     window=decision.W, access=access)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derivable reports from saved predictions (U7.5; no retraining).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--busqueda", type=Path, default=ROOT / "data" / "processed" / "forecasting")
    parser.add_argument("--sin-clima", type=Path, default=ROOT / "data" / "processed" / "forecasting" / "sin_clima")
    parser.add_argument("--prueba", type=Path, default=ROOT / "data" / "processed" / "forecasting" / "prueba")
    parser.add_argument("--salida", type=Path, required=True, help="Folder of the CSV tables and the manifest.")
    parser.add_argument("--figuras", type=Path, required=True, help="Folder of the PNG and SVG figures.")
    parser.add_argument("--permitir-cambios", action="store_true",
                        help="Run with uncommitted changes in src/ or configs/ (never for the versioned evidence).")
    args = parser.parse_args(argv)
    try:
        from .figures_reports import FIGURES, write_figures

        dirty = _git("status", "--porcelain=v1", "--untracked-files=all", "--", "src", "configs").splitlines()
        require(not dirty or args.permitir_cambios,
                f"src/ or configs/ have uncommitted changes; the reports must match a commit: {dirty[:10]}")
        outputs = [args.salida / f"{name}.csv" for name in TABLES] + [args.salida / "manifiesto_reportes.json"]
        outputs += [args.figuras / f"{name}.{suffix}" for name in FIGURES for suffix in ("png", "svg")]
        existing = [str(path) for path in outputs if path.exists()]
        require(not existing, f"Reports already exist and are never overwritten: {existing[:5]}.")

        config = load_config(args.config)
        access = RecipeAccess.from_config(config)
        recipe_path = ROOT / access.recipe_path
        recipe = load_frozen_recipe(recipe_path, channel_version=access.channel_version)
        summary = json.loads((args.prueba / "resumen_prueba.json").read_text(encoding="utf-8"))
        require(summary["ejecucion"]["receta_hash"] == recipe["hash_sha256"],
                "The saved test outputs belong to another recipe.")
        md5 = {"completo": _check_search(args.busqueda, recipe["procedencia"], "completo"),
               "sin_clima": _check_search(args.sin_clima, recipe["procedencia"], "sin_clima"),
               "prueba": {"predicciones_prueba.parquet": _md5(args.prueba / "predicciones_prueba.parquet")}}

        tables, systems, extra = build_all(config, recipe, search_dir=args.busqueda, sin_clima_dir=args.sin_clima,
                                           prueba_dir=args.prueba, recipe_path=recipe_path)
        args.salida.mkdir(parents=True, exist_ok=True)
        for name in TABLES:
            tables[name].to_csv(args.salida / f"{name}.csv", index=False, encoding="utf-8", lineterminator="\n")
        figures = write_figures(tables, systems, extra["observed"], extra["eventos"], folder=args.figuras,
                                tau=extra["tau"])
        manifest = {"unidad": "U7.5", "fecha_hora": datetime.now(_BOGOTA).isoformat(timespec="seconds"),
                    "commit": _git("rev-parse", "HEAD") or None, "cambios_sin_commit": bool(dirty),
                    "receta_hash": recipe["hash_sha256"], "ejecucion_prueba": summary["ejecucion"]["ejecucion"],
                    "tau": extra["tau"], "insumos_md5": md5, "tablas": [f"{name}.csv" for name in TABLES],
                    "figuras": [_relative(path) for path in figures]}
        (args.salida / "manifiesto_reportes.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)
                                                              + "\n", encoding="utf-8")
        print(f"{len(TABLES)} tables in {args.salida} and {len(figures)} figure files in {args.figuras}.")
    except Exception as exc:   # reported with its traceback
        traceback.print_exc()
        print(f"Reports stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
