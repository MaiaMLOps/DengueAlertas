"""Precomputed artifacts for the API (contrato §6.1, §8.4, §8.8, §9.4 and §10.18; api_salida_d10.md §2, §6 and §7;
especificación §4 and §6.F7; DV-02): out-of-sample history, performance card and final models. Nothing here reruns
the single test: the card and the history are computed from the saved predictions of evaluate.run_test.

Flow:
    history_table(rows, predictions, observed, *, origen, tau, k, window, access), one period at a time (D4):
      only rolling-origin predictions ("desarrollo_origen_movil" or "prueba_origen_movil"; the final model never
      enters, contrato §9.4) -> alert state of every emission (alert_states, api §2) within the period -> rows of
      api §6 (C, central, U1e and zone of the target week) and the events of the model and the channel
      (events.evaluate_period) as api event types.
    performance_card(rows, predictions, observed, *, config, tau, access), test period only:
      global card from evaluate.score_summary (the figures of resumen_prueba.json) -> levels in the API spelling,
      the lower of Brier and log-loss (§8.4); the reliability bins (10 quantiles of p), the AMOC-type curve over
      the tau grid with the channel as a point, the cells and the note for a municipality without onsets;
      one card per municipality with the descriptive level of §8.8 instead of a statistical conclusion.
    train_final_models(examples, recipe, *, config): per municipality, the recipe's family, hyperparameters and
      feature set, fitted with all its rows (t <= 2025-10-27, contrato §6.1) -> write_final_models: <code>.joblib
      and manifiesto.json, never overwriting.

Command line (from the saved outputs of the single test; refused with uncommitted changes in src/ or configs/):
    python -B -X utf8 -m src.forecasting.artifacts --prueba data/processed/forecasting/prueba
writes ficha_desempeno.json, historial.parquet, historial_eventos.parquet and modelos_finales/ in that folder. The
development part of the history uses the saved F5b-style predictions of the recipe configuration (decision D2,
«usado para elegir»), from the search of its feature set, whose md5 must match the recipe provenance.
"""

import argparse
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd

from src.epidemiology.channel_epico import ZONES
from src.epidemiology.validation import require

from .comparison import CONSISTENT, NONE, TENDENCY
from .config import DEFAULT_CONFIG_PATH, ROOT, ForecastConfig, load_config
from .dataset import FEATURE_SETS, build_examples, load_channel_product
from .events import channel_positive, evaluate_period, model_positive, period_onsets
from .evaluate import (DEFAULT_OUTPUT_DIR as TEST_OUTPUT_DIR, PREDICTIONS_FILENAME as TEST_PREDICTIONS_FILENAME,
                       SYSTEMS, aligned_systems, score_summary)
from .guard import RecipeAccess, check_test_access, load_frozen_recipe
from .models import FAMILIES, fit_model

ORIGINS = {"desarrollo_origen_movil": "desarrollo", "prueba_origen_movil": "prueba"}
FINAL_ORIGIN = "modelo_final"
SIN_ALERTA, SENAL, CONFIRMADA, CONTINUIDAD = ("sin_alerta", "senal_en_observacion", "alerta_confirmada",
                                              "continuidad_epidemica")
API_LEVELS = {CONSISTENT: "mejora_consistente", TENDENCY: "tendencia_favorable", NONE: "sin_mejora"}
FAVORABLE, MIXTO, DESFAVORABLE = "favorable", "mixto", "desfavorable"
_DESCRIPTIVE_RANK = {DESFAVORABLE: 0, MIXTO: 1, FAVORABLE: 2}
HISTORY_COLUMNS = ["municipality_code", "semana_emision", "semana_objetivo", "periodo", "origen", "probabilidad",
                   "b2_estado_canal", "b1_climatologia_estacional", "estado_alerta", "casos_objetivo",
                   "central_objetivo", "u1e_objetivo", "zona_objetivo", "observado_objetivo"]
EVENT_COLUMNS = ["municipality_code", "periodo", "tipo_evento", "sistema", "semana", "semana_alerta",
                 "anticipacion_semanas"]
NO_ONSET_NOTE = "No hubo inicios de brote en 2023–2025: solo se miden falsas alarmas."
CARD_FILENAME = "ficha_desempeno.json"
HISTORY_FILENAME = "historial.parquet"
HISTORY_EVENTS_FILENAME = "historial_eventos.parquet"
FINAL_MODELS_DIRNAME = "modelos_finales"
MANIFEST_FILENAME = "manifiesto.json"
SCHEMA_VERSION = "0.1"
SEARCH_DIRS = {"completo": ROOT / "data" / "processed" / "forecasting",
               "sin_clima": ROOT / "data" / "processed" / "forecasting" / "sin_clima"}
_KEYS = ["municipality_code", "t"]
_REFERENCES = (("vs_b2", "B2"), ("vs_b1", "B1"), ("ablacion_clima", "ablacion"))


@dataclass(frozen=True)
class History:
    """Rows of api §6 and the events of the model and the channel for one period."""

    filas: pd.DataFrame
    eventos: pd.DataFrame


@dataclass(frozen=True)
class FinalModel:
    """Final model of one municipality: the recipe configuration fitted with all its rows."""

    model: object
    municipality_code: str
    familia: str
    candidato: str
    conjunto_variables: str
    hiperparametros: dict
    filas: int
    ultima_emision: date
    ultima_objetivo: date


# --------------------------------------------------------------------------- history (api §2 and §6)
def alert_states(p, zones, *, tau: float, k: int) -> list[str]:
    """State of every emission of one municipality, in order (api §2): sin_alerta below tau; senal_en_observacion
    while the run of emissions >= tau is shorter than k; then alerta_confirmada, or continuidad_epidemica when the
    observed zone at that emission is epidemia."""
    positive = model_positive(p, tau=tau)
    zones = list(zones)
    require(len(zones) == len(positive), "zones and p must have the same length.")
    require(isinstance(k, (int, np.integer)) and not isinstance(k, bool) and k >= 1, f"k must be >= 1; got {k!r}.")
    states, run = [], 0
    for flag, zone in zip(positive, zones):
        run = run + 1 if flag else 0
        if run == 0:
            states.append(SIN_ALERTA)
        elif run < k:
            states.append(SENAL)
        else:
            states.append(CONTINUIDAD if zone == ZONES[2] else CONFIRMADA)
    return states


def _target_lookup(observed: pd.DataFrame, codes, weeks) -> pd.DataFrame:
    indexed = observed.set_index(["municipality_code", "week_start"])
    require(indexed.index.is_unique, "observed has duplicated (municipality_code, week_start).")
    keys = pd.MultiIndex.from_arrays([list(codes), list(weeks)])
    missing = keys.difference(indexed.index)
    require(len(missing) == 0, f"observed lacks target weeks such as {list(missing[:3])}.")
    return indexed.loc[keys]


def history_table(rows: pd.DataFrame, predictions: pd.DataFrame, observed: pd.DataFrame, *, origen: str, tau: float,
                  k: int, window: int, access: RecipeAccess) -> History:
    """History rows and events of one period (decision D4: the alert state never crosses periods)."""
    require(origen in ORIGINS, f"origen must be one of {sorted(ORIGINS)}; got {origen!r}. The final model never "
                               f"enters the history (contrato §9.4).")
    if "origen" in predictions.columns:
        require(not (predictions["origen"] == FINAL_ORIGIN).any(),
                "Final-model predictions never enter the history (contrato §9.4).")
    periodo = ORIGINS[origen]
    first, last = access.test_years
    in_test = [first <= week.year <= last for week in rows["target_week"]]
    require(all(in_test) if periodo == "prueba" else not any(in_test),
            f"The rows do not belong to the {periodo} period of origen {origen!r}.")
    check_test_access(rows["target_week"], access)
    base = rows.sort_values(_KEYS, kind="mergesort").reset_index(drop=True)
    p = aligned_systems(base, predictions, ("modelo", "B2", "B1"))
    states = []
    for code in base["municipality_code"].unique():
        mask = (base["municipality_code"] == code).to_numpy()
        states += alert_states(p["modelo"][mask], base.loc[mask, "zona_t"], tau=tau, k=k)
    target = _target_lookup(observed, base["municipality_code"], base["target_week"])
    filas = pd.DataFrame({
        "municipality_code": base["municipality_code"], "semana_emision": base["t"],
        "semana_objetivo": base["target_week"], "periodo": periodo, "origen": origen, "probabilidad": p["modelo"],
        "b2_estado_canal": p["B2"], "b1_climatologia_estacional": p["B1"], "estado_alerta": states,
        "casos_objetivo": target["confirmed_records_screening"].to_numpy(dtype=int),
        "central_objetivo": target["epico_central"].to_numpy(dtype=float),
        "u1e_objetivo": target["epico_upper"].to_numpy(dtype=float),
        "zona_objetivo": target["channel_zone"].to_numpy(dtype=object),
        "observado_objetivo": base["y"].astype(int).to_numpy()})[HISTORY_COLUMNS]

    events = []
    for system, positive in (("modelo", model_positive(p["modelo"], tau=tau)),
                             ("canal", channel_positive(base["zona_t"]))):
        period = evaluate_period(base, positive, observed, k=k, window=window, access=access)
        for onset in period.onsets[period.onsets["kind"] == "inicio"].itertuples():
            kind = ("inicio_censurado" if not onset.evaluable else "inicio_avisado" if onset.warned
                    else "inicio_no_avisado")
            warned = kind == "inicio_avisado"
            events.append({"municipality_code": onset.municipality_code, "periodo": periodo, "tipo_evento": kind,
                           "sistema": system, "semana": onset.week,
                           "semana_alerta": onset.warning_confirmation if warned else None,
                           "anticipacion_semanas": float(onset.lead) if warned else None})
        for alert in period.alerts.itertuples():
            kind = ("alerta_continuidad" if alert.continuity else "alerta_censurada" if not alert.judgeable
                    else "falsa_alarma" if alert.false_alarm else None)
            if kind is not None:
                events.append({"municipality_code": alert.municipality_code, "periodo": periodo, "tipo_evento": kind,
                               "sistema": system, "semana": alert.confirmation, "semana_alerta": alert.confirmation,
                               "anticipacion_semanas": None})
    return History(filas, pd.DataFrame(events, columns=EVENT_COLUMNS))


# --------------------------------------------------------------------------- performance card (api §7)
def descriptive_level(favorable: bool, wins: int, n: int) -> str:
    """Descriptive level of a municipality (contrato §8.8): favorable if the global metric is favourable and the
    model wins more than half of its cells; mixto if only one holds; desfavorable otherwise."""
    majority = wins * 2 > n
    if favorable and majority:
        return FAVORABLE
    if favorable or majority:
        return MIXTO
    return DESFAVORABLE


def lower_descriptive(first: str, second: str) -> str:
    for level in (first, second):
        require(level in _DESCRIPTIVE_RANK, f"Unknown descriptive level {level!r}.")
    return first if _DESCRIPTIVE_RANK[first] <= _DESCRIPTIVE_RANK[second] else second


def _calibration(y: np.ndarray, p: np.ndarray) -> list[dict]:
    """Reliability bins: 10 quantiles of p within the period (contrato §10.16), ties merged."""
    bins = pd.qcut(p, 10, labels=False, duplicates="drop")
    out = []
    for value in sorted(set(bins)):
        mask = bins == value
        out.append({"probabilidad_media": float(p[mask].mean()), "frecuencia_observada": float(y[mask].mean()),
                    "n": int(mask.sum())})
    return out


def _finite_or_none(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def _card(rows: pd.DataFrame, predictions: pd.DataFrame, observed: pd.DataFrame, *, config: ForecastConfig,
          tau: float, access: RecipeAccess, municipio: str | None) -> dict:
    cells, summary = score_summary(rows, predictions, observed, config=config, tau=tau, access=access)
    estimation = summary["estimacion"]
    n = len(cells)
    comparisons = {}
    for name, reference in _REFERENCES:
        favorable = {metric: bool(estimation[metric]["modelo"] < estimation[metric][reference])
                     for metric in ("brier", "logloss")}
        wins = {metric: int(cells[f"gana_{metric}_vs_{reference}"].sum()) for metric in ("brier", "logloss")}
        conclusion = {"casillas_ganadas": wins["brier"], "casillas_ganadas_logloss": wins["logloss"],
                      "casillas_total": n, "favorable_brier": favorable["brier"],
                      "favorable_logloss": favorable["logloss"]}
        if municipio is None:
            reported = summary["conclusiones"][name]
            conclusion.update({"nivel_conclusion": API_LEVELS[reported["nivel"]],
                               "nivel_brier": API_LEVELS[reported["nivel_brier"]],
                               "nivel_logloss": API_LEVELS[reported["nivel_logloss"]], "nivel_descriptivo": None})
        else:
            level_b = descriptive_level(favorable["brier"], wins["brier"], n)
            level_l = descriptive_level(favorable["logloss"], wins["logloss"], n)
            conclusion.update({"nivel_conclusion": None, "nivel_brier": None, "nivel_logloss": None,
                               "nivel_descriptivo": lower_descriptive(level_b, level_l),
                               "nivel_descriptivo_brier": level_b, "nivel_descriptivo_logloss": level_l})
        comparisons[name] = {"conclusion": conclusion}

    events = summary["eventos"]
    mcnemar = dict(events["mcnemar"])
    mcnemar_level = API_LEVELS[mcnemar.pop("nivel")] if municipio is None else None
    base = rows.sort_values(_KEYS, kind="mergesort").reset_index(drop=True)
    p = aligned_systems(base, predictions, ("modelo",))["modelo"]
    decision = config.decision
    curve = []
    for grid_tau in decision.tau_grid:
        point = evaluate_period(base, model_positive(p, tau=grid_tau), observed, k=decision.k, window=decision.W,
                                access=access).summary
        curve.append({"tau": float(grid_tau), "deteccion": _finite_or_none(point["detection"]),
                      "falsas_alarmas_por_municipio_anio": float(point["false_alarms_per_municipality_year"]),
                      "anticipacion_mediana_semanas": _finite_or_none(point["lead_median"])})
    channel = evaluate_period(base, channel_positive(base["zona_t"]), observed, k=decision.k, window=decision.W,
                              access=access).summary
    onsets = period_onsets(base, observed, window=decision.W)
    onsets = onsets[(onsets["kind"] == "inicio") & onsets["evaluable"]]
    notes = [f"{code}: {NO_ONSET_NOTE}" for code in sorted(base["municipality_code"].unique())
             if not (onsets["municipality_code"] == code).any()]
    card_cells = [{"municipio": row.municipality_code, "anio": int(row.anio),
                   **{f"{metric}_{system.lower()}": float(getattr(row, f"{metric}_{system}"))
                      for system in SYSTEMS for metric in ("brier", "logloss")},
                   "gana_vs_b2": bool(row.gana_brier_vs_B2)} for row in cells.itertuples()]
    return {
        "municipio": municipio, "periodo": "prueba", "tau": float(tau),
        "estimacion": {"bss_vs_b2": estimation["bss_vs_b2"], "bss_vs_b1": estimation["bss_vs_b1"],
                       "mejora_logloss_vs_b2": estimation["mejora_logloss_vs_b2"],
                       "mejora_logloss_vs_b1": estimation["mejora_logloss_vs_b1"],
                       "brier": estimation["brier"], "logloss": estimation["logloss"],
                       "ap": {system: _finite_or_none(value) for system, value in estimation["ap"].items()},
                       "prevalencia": estimation["prevalencia"], **comparisons},
        "eventos": {"inicios_evaluables": events["inicios_evaluables"], "avisados_modelo": events["avisados_modelo"],
                    "avisados_canal": events["avisados_canal"], "municipio_anios": events["municipio_anios"],
                    "mcnemar": {**mcnemar, "nivel_conclusion": mcnemar_level},
                    "falsas_alarmas": events["falsas_alarmas"],
                    "falsas_alarmas_por_municipio_anio": events["falsas_alarmas_por_municipio_anio"],
                    "anticipacion_mediana_semanas": events["anticipacion_mediana_semanas"],
                    "delta_w": events["delta_w"], "alertas_continuidad_modelo": events["alertas_continuidad_modelo"],
                    "entradas": events["entradas"]},
        "calibracion": _calibration(base["y"].to_numpy(dtype=float), p),
        "curva_amoc": curve,
        "punto_canal": {"deteccion": _finite_or_none(channel["detection"]),
                        "falsas_alarmas_por_municipio_anio": float(channel["false_alarms_per_municipality_year"]),
                        "anticipacion_mediana_semanas": _finite_or_none(channel["lead_median"])},
        "casillas": card_cells,
        "notas": notes,
    }


def performance_card(rows: pd.DataFrame, predictions: pd.DataFrame, observed: pd.DataFrame, *, config: ForecastConfig,
                     tau: float, access: RecipeAccess) -> dict:
    """Performance card of api §7 (test period only): global and one per municipality."""
    check_test_access(rows["target_week"], access)
    present = set(predictions["sistema"])
    require(set(SYSTEMS) <= present, f"The card needs the systems {list(SYSTEMS)}; got {sorted(present)}.")
    card = {"version_esquema": SCHEMA_VERSION,
            "global": _card(rows, predictions, observed, config=config, tau=tau, access=access, municipio=None),
            "municipios": {}}
    for code in sorted(rows["municipality_code"].unique()):
        own_rows = rows[rows["municipality_code"] == code]
        own_predictions = predictions[predictions["municipality_code"] == code]
        card["municipios"][code] = _card(own_rows, own_predictions, observed, config=config, tau=tau, access=access,
                                         municipio=code)
    return card


# --------------------------------------------------------------------------- final models (contrato §6.1)
def train_final_models(examples: pd.DataFrame, recipe: dict, *, config: ForecastConfig) -> dict[str, FinalModel]:
    """Per municipality, the recipe configuration fitted with all its rows (target year >= first training year)."""
    first = config.validacion.primer_anio_entrenamiento
    specs = recipe.get("municipios")
    require(isinstance(specs, dict), "The recipe has no municipios.")
    final = {}
    for code in sorted(examples["municipality_code"].unique()):
        spec = specs.get(code)
        require(isinstance(spec, dict), f"The recipe has no configuration for municipality {code}.")
        require(spec.get("familia") in FAMILIES, f"Recipe municipios[{code}] has no valid family.")
        require(spec.get("conjunto_variables") in FEATURE_SETS,
                f"Recipe municipios[{code}] has no valid conjunto_variables (format of D7.2, DV-02).")
        own = examples[(examples["municipality_code"] == code)
                       & (examples["target_week"].map(lambda week: week.year) >= first)]
        require(len(own) > 0, f"Municipality {code} has no rows for its final model.")
        model = fit_model(spec["familia"], spec["hiperparametros"], own,
                          features=FEATURE_SETS[spec["conjunto_variables"]])
        final[code] = FinalModel(model=model, municipality_code=code, familia=spec["familia"],
                                 candidato=spec["candidato"], conjunto_variables=spec["conjunto_variables"],
                                 hiperparametros=dict(spec["hiperparametros"]), filas=len(own),
                                 ultima_emision=max(own["t"]), ultima_objetivo=max(own["target_week"]))
    return final


def write_final_models(final: dict[str, FinalModel], folder: str | Path, *, recipe_hash: str) -> Path:
    """<code>.joblib per municipality and manifiesto.json; nothing is overwritten."""
    import joblib

    folder = Path(folder)
    targets = [folder / MANIFEST_FILENAME] + [folder / f"{code}.joblib" for code in final]
    existing = [str(path) for path in targets if path.exists()]
    require(not existing, f"Final model files already exist and are never overwritten: {existing}.")
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"receta_hash": recipe_hash, "modelos": {}}
    for code, item in sorted(final.items()):
        joblib.dump(item.model, folder / f"{code}.joblib")
        manifest["modelos"][code] = {
            "familia": item.familia, "candidato": item.candidato, "conjunto_variables": item.conjunto_variables,
            "hiperparametros": item.hiperparametros, "filas": item.filas,
            "ultima_emision": item.ultima_emision.isoformat(), "ultima_objetivo": item.ultima_objetivo.isoformat()}
    path = folder / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- command line
def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dirty_paths() -> list[str]:
    result = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain=v1", "--untracked-files=all", "--",
                             "src", "configs"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.splitlines()


def development_history_inputs(recipe: dict, dev_rows: pd.DataFrame, search_dirs: dict = SEARCH_DIRS) -> pd.DataFrame:
    """Saved development predictions of the recipe configuration of each municipality (decision D2) and the B2/B1
    of the same search, as a long table with sistema in {modelo, B2, B1}. The md5 of each search's predictions
    must be the one of the recipe provenance."""
    frames = []
    for code, spec in recipe["municipios"].items():
        feature_set = spec["conjunto_variables"]
        folder = Path(search_dirs[feature_set])
        expected = recipe["procedencia"]["busquedas"][feature_set]["insumos_md5"]["predicciones_desarrollo.parquet"]
        require(_md5(folder / "predicciones_desarrollo.parquet") == expected,
                f"{folder / 'predicciones_desarrollo.parquet'} is not the file of the recipe provenance.")
        model = pd.read_parquet(folder / "predicciones_desarrollo.parquet")
        model = model[(model["municipality_code"] == code) & (model["familia"] == spec["familia"])
                      & (model["candidato"] == spec["candidato"])]
        baselines = pd.read_parquet(folder / "predicciones_lineas_base_desarrollo.parquet")
        baselines = baselines[baselines["municipality_code"] == code]
        frames += [model.assign(sistema="modelo")] + [baselines[baselines["familia"] == system].assign(sistema=system)
                                                      for system in ("B2", "B1")]
    long = pd.concat(frames, ignore_index=True)
    keys = set(zip(dev_rows["municipality_code"], dev_rows["t"]))
    return long[[key in keys for key in zip(long["municipality_code"], long["t"])]]


def main(argv: list[str] | None = None) -> int:
    from .run_selection import development_rows

    parser = argparse.ArgumentParser(description="History, performance card and final models (F7, U7.4).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--prueba", type=Path, default=TEST_OUTPUT_DIR)
    parser.add_argument("--permitir-cambios", action="store_true",
                        help="Run with uncommitted changes in src/ or configs/ (never for the versioned artifacts).")
    args = parser.parse_args(argv)
    try:
        dirty = _dirty_paths()
        require(not dirty or args.permitir_cambios,
                f"src/ or configs/ have uncommitted changes; the artifacts must match a commit: {dirty[:10]}")
        outputs = [args.prueba / CARD_FILENAME, args.prueba / HISTORY_FILENAME, args.prueba / HISTORY_EVENTS_FILENAME,
                   args.prueba / FINAL_MODELS_DIRNAME / MANIFEST_FILENAME]
        existing = [str(path) for path in outputs if path.exists()]
        require(not existing, f"Artifacts already exist and are never overwritten: {existing}.")
        config = load_config(args.config)
        access = RecipeAccess.from_config(config)
        access = RecipeAccess.from_config(config, recipe_path=ROOT / access.recipe_path)
        recipe = load_frozen_recipe(access.recipe_path, channel_version=access.channel_version)
        summary = json.loads((args.prueba / "resumen_prueba.json").read_text(encoding="utf-8"))
        require(summary["ejecucion"]["receta_hash"] == recipe["hash_sha256"],
                "The saved test outputs belong to another recipe.")
        product = load_channel_product(config.datos.canal if Path(config.datos.canal).is_absolute()
                                       else ROOT / config.datos.canal, expected_md5=config.datos.version_dvc_canal)
        observed = product.to_pandas()
        examples = build_examples(product, h=config.horizonte.h, canal=config.canal, features=config.features)
        tau, decision = float(recipe["decision"]["tau"]), config.decision

        predictions = pd.read_parquet(args.prueba / TEST_PREDICTIONS_FILENAME)
        test_rows = predictions[predictions["sistema"] == "modelo"][
            _KEYS + ["target_week", "zona_t", "y"]].reset_index(drop=True)
        card = performance_card(test_rows, predictions, observed, config=config, tau=tau, access=access)
        card["receta"] = {"hash_sha256": recipe["hash_sha256"], "familia_comun": recipe["familia_comun"],
                          "tau": tau, "k": decision.k, "W": decision.W,
                          "ejecucion_prueba": summary["ejecucion"]["ejecucion"]}

        dev_rows = development_rows(examples, config)
        development = history_table(dev_rows, development_history_inputs(recipe, dev_rows), observed,
                                    origen="desarrollo_origen_movil", tau=tau, k=decision.k, window=decision.W,
                                    access=access)
        test = history_table(test_rows, predictions, observed, origen="prueba_origen_movil", tau=tau, k=decision.k,
                             window=decision.W, access=access)
        history = pd.concat([development.filas, test.filas], ignore_index=True)
        history_events = pd.concat([development.eventos, test.eventos], ignore_index=True)

        final = train_final_models(examples, recipe, config=config)
        (args.prueba / CARD_FILENAME).write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")
        history.to_parquet(args.prueba / HISTORY_FILENAME, index=False)
        history_events.to_parquet(args.prueba / HISTORY_EVENTS_FILENAME, index=False)
        write_final_models(final, args.prueba / FINAL_MODELS_DIRNAME, recipe_hash=recipe["hash_sha256"])
        print(f"Artifacts written to {args.prueba}: card, history ({len(history)} rows, "
              f"{len(history_events)} events) and {len(final)} final models.")
    except Exception as exc:   # reported with its traceback
        traceback.print_exc()
        print(f"Artifacts stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
