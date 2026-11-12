"""Hyperparameter search per municipality and the single score table (contrato §6.2, §6.3, §10.9 and §10.12;
especificación §4, §6.F5a and §9).

Flow (rows come from `build_examples`, trials from `splits.yearly_trials`; nothing is rebuilt):
    CLI: load_config -> load_channel_product (md5) -> build_examples -> candidates_from_config(busqueda)
         -> tracking.configure_tracking -> run_search
    run_search(examples, candidates, *, config, output_dir, tracking_uri)
      1. checks before any fit: families and candidate names, parameters (models.make_model), development
         years without test years, trials whose evaluated target weeks are all before the first test year.
         One candidate list serves every municipality by construction (no municipality is passed to it).
      2. manifest busqueda_manifest.json: identity of the search (candidates, examples hash, configuration
         sections, commit). A different manifest in the same folder is an error: runs are never mixed.
      3. units, in order municipality -> baselines (B2, B1) -> family -> candidate (pre-registered order):
           per trial: fit_model(train) -> predict_positive(evaluate) -> log_loss, brier (through the barrier)
           with B2 and B1 of the same trial as references -> one MLflow run (created after computing)
           -> folder partes/<municipality>__<family>__<candidate>/ written as .tmp-* and renamed.
         A unit is complete or absent; an existing folder is skipped when the search is relaunched.
      4. consolidation in canonical order, validation and atomic writing of the final files.

Feature-set variant (contrato §6.6, D7.2): `variante="sin_clima"` fits the same pre-registered candidates of both
families, with the same seeds, without the nine climate anomalies (dataset.FEATURE_SETS["sin_clima"]). The variant
enters the manifest (so the busqueda_id differs) and the run tags; the CLI option --variante sin_clima writes to
data/processed/forecasting/sin_clima/ by default and never into the F5b folder. Without a variant, everything is as
in F5b (the full set). The earlier logistic-only search in ablacion_clima/ (D1 of F7, origin of D7.2) is a record.

Runs are never duplicated. A unit whose run finished but whose folder was not written (a crash in between)
reuses that run: the unit is recomputed and its metrics must match the logged ones (rel = 1e-12, because the
server stores them through JSON and a database). RUNNING runs of the same search (same busqueda_id tag) are
marked FAILED; runs of other searches in the shared experiment are never touched. Run one search per folder
at a time.
"""

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

from src.epidemiology.config import EpidemiologyValidationError
from src.epidemiology.validation import require

from .baselines import b1_predict, b2_predict
from .build_channel import code_state
from .config import DEFAULT_CONFIG_PATH, ROOT, ForecastConfig, SearchSection, load_config
from .dataset import FEATURE_SETS, FEATURES, build_examples, load_channel_product
from .guard import RecipeAccess
from .metrics import brier, improvement, log_loss
from .models import FAMILIES, fit_model, make_model, predict_positive
from .splits import yearly_trials
from .tracking import allow_file_store, configure_tracking

TABLE_FILENAME = "tabla_unica_desarrollo.parquet"
PREDICTIONS_FILENAME = "predicciones_desarrollo.parquet"
BASELINES_FILENAME = "lineas_base_desarrollo.parquet"
BASELINE_PREDICTIONS_FILENAME = "predicciones_lineas_base_desarrollo.parquet"
MANIFEST_FILENAME = "busqueda_manifest.json"
LOG_FILENAME = "busqueda.log"
PARTS_DIRNAME = "partes"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "forecasting"
# Variants of the search (contrato §6.6): name -> features of the models; each writes to the subfolder of its name.
VARIANTS = {"sin_clima": FEATURE_SETS["sin_clima"]}
BASELINE_FAMILIES = ("B2", "B1")
REUSED_RUN_REL = 1e-12          # approved by Jesús for comparing a reused run with the recomputed unit
_NAME_CHARACTERS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
_PHASE = "desarrollo"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """One pre-registered configuration of a family: a name and its hyperparameters."""

    name: str
    params: Mapping[str, Any] = field(hash=False)


# --------------------------------------------------------------------------- candidates from the configuration
def _sample(space, rng: np.random.Generator, upper_cap: float | None):
    low, high = space.rango
    if upper_cap is not None:
        high = min(high, upper_cap)
        require(high >= low, f"num_leaves range [{low}, {space.rango[1]}] is empty with the cap {upper_cap}.")
    if space.escala == "entera":
        return int(rng.integers(int(low), int(high) + 1))
    if space.escala == "real":
        return float(rng.uniform(low, high))
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def candidates_from_config(busqueda: SearchSection) -> dict[str, list[Candidate]]:
    """Expand the pre-registered section into the candidate lists, identical for every municipality.

    Logistic regression: one candidate per C, in the given order. LightGBM: n_candidatos draws from
    numpy default_rng(semilla_muestreo), each parameter drawn in alphabetical order (max_depth before
    num_leaves, so the cap 2^max_depth can apply), then the fixed values and random_state = semilla_modelo.
    """
    require(isinstance(busqueda, SearchSection), "busqueda must be the validated search section.")
    candidates: dict[str, list[Candidate]] = {}
    if busqueda.logistica is not None:
        section = busqueda.logistica
        candidates["logistica"] = [Candidate(f"logistica_{i:02d}", {"C": float(c), "max_iter": section.max_iter})
                                   for i, c in enumerate(section.C)]
    if busqueda.lightgbm is not None:
        section = busqueda.lightgbm
        rng = np.random.default_rng(section.semilla_muestreo)
        drawn = []
        for i in range(section.n_candidatos):
            params: dict[str, Any] = {}
            for name in sorted(section.espacio):
                space = section.espacio[name]
                cap = None
                if space.tope_por_profundidad:
                    cap = 2 ** int(params.get("max_depth", section.fijos.get("max_depth")))
                params[name] = _sample(space, rng, cap)
            params.update(section.fijos)
            params["random_state"] = section.semilla_modelo
            drawn.append(Candidate(f"lightgbm_{i:02d}", params))
        candidates["lightgbm"] = drawn
    return candidates


# --------------------------------------------------------------------------- checks
def _parameters_json(params: Mapping) -> str:
    try:
        return json.dumps(dict(params), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise EpidemiologyValidationError(f"Candidate parameters must be JSON-serializable: {exc}.") from exc


def _check_candidates(candidates) -> list[tuple[str, int, Candidate, str]]:
    """(family, position, candidate, canonical JSON) in pre-registered order; fails before any fit."""
    require(isinstance(candidates, Mapping) and len(candidates) > 0,
            "candidates must be a non-empty mapping family -> list of Candidate.")
    checked = []
    for family, family_candidates in candidates.items():
        require(family in FAMILIES, f"Unknown model family {family!r}; expected one of {FAMILIES}.")
        family_candidates = list(family_candidates)
        require(family_candidates, f"Family {family} has no candidates.")
        names = [candidate.name if isinstance(candidate, Candidate) else None for candidate in family_candidates]
        require(all(isinstance(name, str) and name and set(name) <= _NAME_CHARACTERS for name in names),
                f"Candidates of {family} must be Candidate objects with names made of letters, digits, '_', '.', "
                f"'-'; got {names}.")
        require(len(set(names)) == len(names), f"Repeated candidate names in {family}: {names}.")
        for position, candidate in enumerate(family_candidates):
            make_model(family, candidate.params)
            checked.append((family, position, candidate, _parameters_json(candidate.params)))
    return checked


def _development_years(config: ForecastConfig) -> list[int]:
    first, last = config.validacion.desarrollo
    test_first, test_last = config.validacion.prueba
    years = list(range(first, last + 1))
    overlap = [year for year in years if test_first <= year <= test_last]
    require(not overlap, f"Development years {years} include test years {overlap}; the search never scores the "
                         f"test period (contrato §6.1, CLAUDE.md rule 1).")
    return years


def validate_score_table(table: pd.DataFrame, *, municipalities: Sequence[str], years: Sequence[int]) -> None:
    """The candidate list (family, name, position, parameters) is identical in every municipality and each
    (municipality, family, candidate) has exactly the given years once (contrato §6.2, §10.12)."""
    require(isinstance(table, pd.DataFrame), "table must be a pandas DataFrame.")
    columns = ["municipality_code", "familia", "candidato", "orden", "parametros", "anio"]
    missing = [column for column in columns if column not in table.columns]
    require(not missing, f"Score table lacks columns {missing}.")
    municipalities, years = list(municipalities), sorted(years)
    require(municipalities and years, "municipalities and years must not be empty.")
    unexpected = sorted(set(table["municipality_code"]) - set(municipalities))
    require(not unexpected, f"Score table has unexpected municipalities {unexpected}.")
    duplicated = table.duplicated(["municipality_code", "familia", "candidato", "anio"])
    require(not duplicated.any(), f"Score table has {int(duplicated.sum())} duplicated (municipality, family, "
                                  f"candidate, year) rows.")

    reference = None
    for code in municipalities:
        own = table[table["municipality_code"] == code]
        candidate_list = sorted(set(map(tuple, own[["familia", "candidato", "orden", "parametros"]].values)))
        if reference is None:
            reference = (code, candidate_list)
        require(candidate_list == reference[1],
                f"The candidate list differs between municipalities {reference[0]} and {code}: "
                f"only in {reference[0]}: {sorted(set(reference[1]) - set(candidate_list))}; "
                f"only in {code}: {sorted(set(candidate_list) - set(reference[1]))}.")
        require(len(candidate_list) > 0, f"Municipality {code} has no candidate rows.")
        for (family, name), rows in own.groupby(["familia", "candidato"]):
            require(sorted(rows["anio"]) == years,
                    f"Candidate {family}/{name} of {code} has years {sorted(rows['anio'])}; expected {years}.")
    names = table[["familia", "candidato"]].drop_duplicates()
    described = table[["familia", "candidato", "orden", "parametros"]].drop_duplicates()
    require(len(names) == len(described), "A candidate name has more than one position or parameter set.")


# --------------------------------------------------------------------------- identity of the search
def _examples_hash(examples: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(examples, index=True).to_numpy()
    digest = hashlib.sha256(hashed.tobytes())
    digest.update(json.dumps(list(map(str, examples.columns))).encode("utf-8"))
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, capture_output=True, text=True,
                                check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _manifest_content(examples, checked, config: ForecastConfig, years, codes, variante: str | None = None) -> dict:
    state = code_state()
    content = {
        "anios": years,
        "prueba": list(config.validacion.prueba),
        "primer_anio_entrenamiento": config.validacion.primer_anio_entrenamiento,
        "municipios": codes,
        "configuracion": {
            "h": config.horizonte.h,
            "canal": config.canal.model_dump(mode="json"),
            "features": config.features.model_dump(mode="json"),
            "lineas_base": config.lineas_base.model_dump(mode="json"),
            "metricas": config.metricas.model_dump(mode="json"),
            "version_dvc_canal": config.datos.version_dvc_canal,
        },
        "candidatos": [{"familia": family, "orden": position, "candidato": candidate.name, "parametros": params}
                       for family, position, candidate, params in checked],
        "ejemplos_sha256": _examples_hash(examples),
        "commit": state.get("commit"),
    }
    if variante is not None:   # only added for a variant, so the identity of the F5b search does not change
        content["variante"] = {"nombre": variante, "features": list(VARIANTS[variante])}
    return content


def _search_id(content: dict) -> str:
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _write_atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".tmp-{path.name}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _ensure_manifest(output_dir: Path, content: dict) -> str:
    path = output_dir / MANIFEST_FILENAME
    search_id = _search_id(content)
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        differing = sorted(key for key in set(stored.get("contenido", {})) | set(content)
                           if stored.get("contenido", {}).get(key) != content.get(key))
        require(not differing,
                f"{path} belongs to a different search (differs in {differing}). Use another output folder; "
                f"results of different searches are never mixed.")
        return stored["busqueda_id"]
    payload = {"busqueda_id": search_id, "creado": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "contenido": content}
    _write_atomic_bytes(path, json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"))
    return search_id


# --------------------------------------------------------------------------- MLflow
@dataclass
class _Tracker:
    client: MlflowClient
    experiment_id: str
    search_id: str
    tags: dict
    finished: dict  # clave -> run_id of FINISHED runs of this search


def _tracker(tracking_uri: str, config: ForecastConfig, search_id: str, commit: str | None,
             variante: str | None = None) -> _Tracker:
    allow_file_store(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    name = config.seguimiento.experimentos.desarrollo
    experiment = client.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment is not None else client.create_experiment(name)

    finished: dict[str, str] = {}
    token = None
    while True:
        page = client.search_runs([experiment_id], filter_string=f"tags.busqueda_id = '{search_id}'",
                                  max_results=1000, page_token=token)
        for run in page:
            key = run.data.tags.get("clave")
            if run.info.status == "FINISHED":
                require(key not in finished, f"Search {search_id} has two FINISHED runs for {key}.")
                finished[key] = run.info.run_id
            elif run.info.status == "RUNNING":
                client.set_terminated(run.info.run_id, "FAILED")
                logger.warning("Run %s (%s) of this search was left RUNNING; marked FAILED.", run.info.run_id, key)
        token = page.token
        if not token:
            break
    tags = {"fase": _PHASE, "commit": commit or "desconocido",
            "contrato_commit": _git("log", "-1", "--format=%H", "--", "docs/forecasting/contrato_predictivo.md")
            or "desconocido",
            "datos_dvc": config.datos.version_dvc_canal or "desconocido", "busqueda_id": search_id}
    if variante is not None:
        tags["variante"] = variante
    return _Tracker(client, experiment_id, search_id, tags, finished)


def _log_run(tracker: _Tracker, key: str, tags: dict, params: Mapping, metrics: Mapping) -> str:
    run = tracker.client.create_run(tracker.experiment_id, tags={**tracker.tags, **tags, "clave": key},
                                    run_name=key)
    run_id = run.info.run_id
    try:
        stamp = int(time.time() * 1000)
        tracker.client.log_batch(run_id, metrics=[Metric(name, float(value), stamp, 0)
                                                  for name, value in metrics.items()],
                                 params=[Param(name, str(value)) for name, value in params.items()])
    except BaseException:
        tracker.client.set_terminated(run_id, "FAILED")
        raise
    tracker.client.set_terminated(run_id, "FINISHED")
    return run_id


def _check_reused_run(tracker: _Tracker, run_id: str, key: str, metrics: Mapping) -> None:
    logged = tracker.client.get_run(run_id).data.metrics
    require(set(logged) == set(metrics), f"Reused run {run_id} ({key}) has metrics {sorted(logged)}; "
                                         f"recomputed {sorted(metrics)}.")
    different = {name: (logged[name], value) for name, value in metrics.items()
                 if not math.isclose(logged[name], value, rel_tol=REUSED_RUN_REL, abs_tol=0.0)}
    require(not different, f"Reused run {run_id} ({key}) differs from the recomputed unit (logged, recomputed): "
                           f"{different}.")


# --------------------------------------------------------------------------- units
@dataclass
class _TrialData:
    year: int
    train: pd.DataFrame
    evaluate: pd.DataFrame
    weeks: list
    references: dict    # B2 / B1 -> predictions on evaluate


def _trial_data(examples, trials, config, access) -> dict[str, list[_TrialData]]:
    by_code: dict[str, list[_TrialData]] = {}
    for trial in trials:
        train, evaluate = examples.loc[trial.train], examples.loc[trial.evaluate]
        references = {"B2": b2_predict(train, evaluate, access=access),
                      "B1": b1_predict(train, evaluate, half_window=config.lineas_base.b1_semiventana,
                                       access=access)}
        by_code.setdefault(trial.municipality_code, []).append(
            _TrialData(trial.year, train, evaluate, list(evaluate["target_week"]), references))
    return by_code


def _scores(y, p, weeks, access, clip) -> tuple[float, float]:
    return log_loss(y, p, weeks, access=access, clip=clip), brier(y, p, weeks, access=access)


def _predictions_frame(code, data: _TrialData, family, name, p) -> pd.DataFrame:
    return pd.DataFrame({"municipality_code": code, "t": list(data.evaluate["t"]), "target_week": data.weeks,
                         "familia": family, "candidato": name, "anio": data.year,
                         "y": data.evaluate["y"].to_numpy(dtype="int8"), "p": np.asarray(p, dtype=float)})


def _baseline_unit(code, family, trial_data, access, clip, state):
    scores, predictions, metrics = [], [], {}
    for data in trial_data:
        state["year"] = data.year
        p = data.references[family]
        loss, squared = _scores(data.evaluate["y"].to_numpy(), p, data.weeks, access, clip)
        scores.append({"municipality_code": code, "familia": family, "anio": data.year, "logloss": loss,
                       "brier": squared, "n_filas": len(data.evaluate), "primera_semana_objetivo": min(data.weeks),
                       "ultima_semana_objetivo": max(data.weeks)})
        predictions.append(_predictions_frame(code, data, family, family, p))
        metrics.update({f"logloss_{data.year}": loss, f"brier_{data.year}": squared})
    return pd.DataFrame(scores), pd.concat(predictions, ignore_index=True), metrics


def _model_unit(code, family, position, candidate, params_json, trial_data, access, clip, state, features):
    scores, predictions, metrics = [], [], {}
    for data in trial_data:
        state["year"] = data.year
        start = time.perf_counter()
        model = fit_model(family, candidate.params, data.train, features=features)
        p = predict_positive(model, data.evaluate)
        seconds = time.perf_counter() - start
        y = data.evaluate["y"].to_numpy()
        loss, squared = _scores(y, p, data.weeks, access, clip)
        loss_b2, brier_b2 = _scores(y, data.references["B2"], data.weeks, access, clip)
        loss_b1, brier_b1 = _scores(y, data.references["B1"], data.weeks, access, clip)
        gain = improvement(loss, loss_b2)
        scores.append({"municipality_code": code, "familia": family, "candidato": candidate.name, "orden": position,
                       "parametros": params_json, "anio": data.year, "logloss": loss, "logloss_b2": loss_b2,
                       "mejora_b2": gain, "brier": squared, "brier_b2": brier_b2, "logloss_b1": loss_b1,
                       "brier_b1": brier_b1, "n_filas": len(data.evaluate),
                       "primera_semana_objetivo": min(data.weeks), "ultima_semana_objetivo": max(data.weeks),
                       "segundos": seconds})
        predictions.append(_predictions_frame(code, data, family, candidate.name, p))
        metrics.update({f"logloss_{data.year}": loss, f"brier_{data.year}": squared, f"mejora_b2_{data.year}": gain})
    return pd.DataFrame(scores), pd.concat(predictions, ignore_index=True), metrics


def _write_unit(parts: Path, key: str, scores: pd.DataFrame, predictions: pd.DataFrame, info: dict) -> None:
    temporary = parts / f".tmp-{key}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        scores.to_parquet(temporary / "puntajes.parquet", index=False)
        predictions.to_parquet(temporary / "predicciones.parquet", index=False)
        (temporary / "unidad.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        os.rename(temporary, parts / key)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


# --------------------------------------------------------------------------- consolidation
def _read_units(parts: Path, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = [pd.read_parquet(parts / key / "puntajes.parquet") for key in keys]
    predictions = [pd.read_parquet(parts / key / "predicciones.parquet") for key in keys]
    return pd.concat(scores, ignore_index=True), pd.concat(predictions, ignore_index=True)


def _check_consolidated(table, predictions, codes, years, test_first: int) -> None:
    validate_score_table(table, municipalities=codes, years=years)
    first_test_day = date(test_first, 1, 1)
    require(all(week < first_test_day for week in table["ultima_semana_objetivo"]) and
            all(week < first_test_day for week in predictions["target_week"]),
            f"A scored row has a target week on or after {first_test_day}; the test period is never scored.")
    b2 = table.groupby(["municipality_code", "anio"])["logloss_b2"].nunique()
    require((b2 == 1).all(), "logloss_b2 differs between candidates of the same municipality-year (unpaired).")
    counts = predictions.groupby(["municipality_code", "familia", "candidato", "anio"]).size()
    expected = table.set_index(["municipality_code", "familia", "candidato", "anio"])["n_filas"]
    require(counts.to_dict() == expected.to_dict(), "Predictions do not match n_filas of the score table.")


def _write_final(path: Path, frame: pd.DataFrame, compare_without: Sequence[str] = ()) -> None:
    """Write atomically; an existing file must hold the same content (it is never overwritten)."""
    if path.exists():
        existing = pd.read_parquet(path)
        columns = [column for column in frame.columns if column not in compare_without]
        require(list(existing.columns) == list(frame.columns) and existing[columns].equals(frame[columns]),
                f"{path} exists with different content; it is not overwritten.")
        return
    temporary = path.with_name(f".tmp-{path.name}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


# --------------------------------------------------------------------------- search
def _elapsed(start: float) -> str:
    seconds = int(time.monotonic() - start)
    return f"[{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}]"


def run_search(examples: pd.DataFrame, candidates: Mapping[str, Sequence[Candidate]], *, config: ForecastConfig,
               output_dir: str | Path, tracking_uri: str, variante: str | None = None) -> pd.DataFrame:
    """Score every candidate in every development trial of every municipality; return the single table.
    With `variante` (a key of VARIANTS) the models use that feature set; baselines are unchanged."""
    start = time.monotonic()
    require(variante is None or variante in VARIANTS, f"Unknown variant {variante!r}; expected one of {list(VARIANTS)}.")
    features = FEATURES if variante is None else VARIANTS[variante]
    checked = _check_candidates(candidates)
    years = _development_years(config)
    trials = yearly_trials(examples, years, first_train_year=config.validacion.primer_anio_entrenamiento)
    test_first = config.validacion.prueba[0]
    for trial in trials:
        last = max(examples.loc[trial.evaluate, "target_week"])
        require(last < date(test_first, 1, 1), f"Trial {trial.year} of {trial.municipality_code} evaluates a target "
                                               f"week in the test period ({last}).")
    codes = sorted({trial.municipality_code for trial in trials})

    output_dir = Path(output_dir)
    parts = output_dir / PARTS_DIRNAME
    parts.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(output_dir / LOG_FILENAME, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        content = _manifest_content(examples, checked, config, years, codes, variante)
        search_id = _ensure_manifest(output_dir, content)
        tracker = _tracker(tracking_uri, config, search_id, content["commit"], variante)
        access = RecipeAccess.from_config(config)
        clip = config.metricas.recorte_log_loss
        trial_data = _trial_data(examples, trials, config, access)

        units = []
        for code in codes:
            units += [(code, family, None, None, None) for family in BASELINE_FAMILIES]
            units += [(code, family, position, candidate, params) for family, position, candidate, params in checked]
        total = len(units)
        logger.info("%s Search %s%s: %d municipalities, years %s, %d units (%s); output %s.", _elapsed(start),
                    search_id, "" if variante is None else f" (variant {variante})", len(codes), years, total,
                    ", ".join(f"{family}: {sum(1 for c in checked if c[0] == family)} candidates"
                              for family in dict.fromkeys(c[0] for c in checked)), output_dir)

        computed_seconds: dict[str, list[float]] = {}
        skipped = 0
        for number, (code, family, position, candidate, params_json) in enumerate(units, start=1):
            name = family if candidate is None else candidate.name
            key = f"{code}__{family}__{name}"
            if (parts / key).is_dir():
                skipped += 1
                continue
            state = {"year": None}
            unit_start = time.perf_counter()
            try:
                if candidate is None:
                    scores, predictions, metrics = _baseline_unit(code, family, trial_data[code], access, clip, state)
                    run_params = ({"semiventana": config.lineas_base.b1_semiventana} if family == "B1" else {})
                else:
                    scores, predictions, metrics = _model_unit(code, family, position, candidate, params_json,
                                                               trial_data[code], access, clip, state, features)
                    run_params = dict(candidate.params)
                state["year"] = None
                tags = {"municipio": code, "familia": family, "candidato": name}
                if key in tracker.finished:
                    run_id = tracker.finished[key]
                    _check_reused_run(tracker, run_id, key, metrics)
                    logger.info("%s Unit %s reuses its FINISHED run %s.", _elapsed(start), key, run_id)
                else:
                    run_id = _log_run(tracker, key, tags, run_params, metrics)
                    tracker.finished[key] = run_id
                seconds = time.perf_counter() - unit_start
                _write_unit(parts, key, scores, predictions, {"run_id": run_id, "segundos": seconds})
            except BaseException:
                logger.error("%s Unit %s failed (municipality %s, family %s, candidate %s, year %s); nothing of it "
                             "was kept. Relaunch to continue from this unit.\n%s", _elapsed(start), key, code, family,
                             name, state["year"], traceback.format_exc())
                raise
            computed_seconds.setdefault(family, []).append(seconds)
            logger.info("%s %d/%d %s %s %s  %.1f s  (skipped %d)", _elapsed(start), number, total, code, family,
                        name, seconds, skipped)

        model_keys = [f"{c}__{f}__{cand.name}" for c, f, _, cand, _ in units if cand is not None]
        baseline_keys = [f"{c}__{f}__{f}" for c, f, _, cand, _ in units if cand is None]
        table, predictions = _read_units(parts, model_keys)
        baselines, baseline_predictions = _read_units(parts, baseline_keys)
        _check_consolidated(table, predictions, codes, years, test_first)
        _write_final(output_dir / TABLE_FILENAME, table, compare_without=("segundos",))
        _write_final(output_dir / PREDICTIONS_FILENAME, predictions)
        _write_final(output_dir / BASELINES_FILENAME, baselines)
        _write_final(output_dir / BASELINE_PREDICTIONS_FILENAME, baseline_predictions)

        for family, values in computed_seconds.items():
            logger.info("%s %s: %d units computed in this launch, mean %.2f s per unit (one municipality x one "
                        "candidate, %d trials).", _elapsed(start), family, len(values), sum(values) / len(values),
                        len(years))
        logger.info("%s Search %s complete: %d units (%d skipped); table %s.", _elapsed(start), search_id, total,
                    skipped, output_dir / TABLE_FILENAME)
        return pd.read_parquet(output_dir / TABLE_FILENAME)
    finally:
        logger.removeHandler(handler)
        handler.close()


# --------------------------------------------------------------------------- command line
def _dirty_paths() -> list[str]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all", "--", "src", "configs")
    return status.splitlines() if status else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Development search of the forecasting MVP (F5).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--salida", type=Path, default=None,
                        help=f"Output folder (default {DEFAULT_OUTPUT_DIR}; with a variant, its subfolder named "
                             f"after the variant).")
    parser.add_argument("--variante", choices=sorted(VARIANTS), default=None,
                        help="Feature-set variant of contrato §6.6: both families without the climate anomalies.")
    parser.add_argument("--permitir-cambios", action="store_true",
                        help="Run with uncommitted changes in src/ or configs/ (smoke runs only).")
    args = parser.parse_args(argv)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    try:
        config = load_config(args.config)
        require(config.busqueda is not None,
                f"{args.config} has no busqueda section; it is added with the pre-registration (especificación §7).")
        dirty = _dirty_paths()
        require(not dirty or args.permitir_cambios,
                "src/ or configs/ have uncommitted changes; commit them so the table matches a commit, or pass "
                f"--permitir-cambios for a smoke run. Changes: {dirty[:10]}")
        if dirty:
            logger.warning("Running with uncommitted changes in src/ or configs/ (--permitir-cambios).")
        candidates = candidates_from_config(config.busqueda)
        output_dir = args.salida
        if args.variante is not None:
            output_dir = output_dir or DEFAULT_OUTPUT_DIR / args.variante
            require(Path(output_dir).resolve() != DEFAULT_OUTPUT_DIR.resolve(),
                    "A variant never writes into the folder of the F5b search.")
        output_dir = output_dir or DEFAULT_OUTPUT_DIR
        channel = Path(config.datos.canal)
        product = load_channel_product(channel if channel.is_absolute() else ROOT / channel,
                                       expected_md5=config.datos.version_dvc_canal)
        examples = build_examples(product, h=config.horizonte.h, canal=config.canal, features=config.features)
        uri = configure_tracking(config)
        logger.info("Tracking: %s (mode %s).", uri if config.seguimiento.modo == "local" else "server",
                    config.seguimiento.modo)
        extra = {} if args.variante is None else {"variante": args.variante}
        run_search(examples, candidates, config=config, output_dir=output_dir, tracking_uri=uri, **extra)
    except KeyboardInterrupt:
        print("Search interrupted; completed units are kept. Relaunch the same command to continue.",
              file=sys.stderr)
        return 130
    except Exception as exc:   # reported with its traceback (also in busqueda.log for unit failures)
        traceback.print_exc()
        print(f"Search stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        logger.removeHandler(console)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
