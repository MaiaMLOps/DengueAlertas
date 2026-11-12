"""Single test run with the frozen recipe (contrato §6.1, §6.6, §7, §8.4 and §4.4; especificación §4 «Barrera de
prueba: detalle» and §6.F7; DV-02).

Flow of `run_test(examples, *, config, access, log_path, output_dir, reason)`:
    test rows (target year in validacion.prueba)
      -> guard.check_test_access (barrier) -> guard.load_frozen_recipe
      -> recipe checks per municipality: the model (municipios[code]: familia, candidato, hiperparametros,
         conjunto_variables) and the ablation (ablacion[code]: same family, the OTHER feature set, contrato §4.4/§6.6)
      -> outputs must not exist; the log (JSON Lines) needs an explicit reason after the first execution
      -> log entry written NOW (a failed attempt counts): ejecucion, fecha_hora, commit, receta_hash,
         version_dvc_canal, razon, salida
      -> one trial per (municipality, test year), splits.yearly_trials (training rows with t + h <= t0, D7.1):
           modelo and ablacion: models.fit_model(familia, hiperparametros, train, features=FEATURE_SETS[set])
           B2 and B1: baselines with the same training rows
      -> predicciones_prueba.parquet (four systems on exactly the same rows) and ensayos_prueba.parquet.
Scores of the saved predictions, `score_summary(rows, predictions, observed, *, config, tau, access)`, contrato
§7.2-§7.4 and §8.4 (the card of the API is artifacts.py, U7.4):
    cells per (municipality, year): Brier and log-loss of the four systems and the wins (strictly smaller loss)
    global, over the rows together: Brier, log-loss, BSS and log-loss improvement against B2 and B1, AP
    conclusions of §8.4 (vs_b2, vs_b1, ablacion_clima): level with Brier and with log-loss, the lower one adopted
    events (§7.4, §8.4): onsets of the model at tau and of the channel, McNemar on the discordant onsets, false
    alarms per municipality-year, median lead and delta_w per onset warned by both.

Command line (real data; refused with uncommitted changes in src/ or configs/):
    python -B -X utf8 -m src.forecasting.evaluate --salida data/processed/forecasting/prueba [--razon TEXT]
writes predicciones_prueba.parquet, ensayos_prueba.parquet, casillas_prueba.csv, resumen_prueba.json and the six
diagnostic inputs of §8.8 in diagnostico_prueba/, then logs one MLflow run per municipality and system in the
experiment `seguimiento.experimentos.prueba`. `--sin-seguimiento` skips MLflow; `--solo-seguimiento` logs the
runs from the saved outputs of --salida without running the test again (no log entry).
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

from src.epidemiology.validation import require

from .baselines import b1_predict, b2_predict
from .comparison import combined_level, count_wins, discordant_wins, event_level, weekly_level
from .config import DEFAULT_CONFIG_PATH, ROOT, ForecastConfig, load_config
from .dataset import FEATURE_SETS, build_examples, load_channel_product
from .events import channel_positive, evaluate_period, model_positive
from .guard import RecipeAccess, check_test_access, load_frozen_recipe
from .metrics import average_precision, brier, improvement, log_loss
from .models import FAMILIES, fit_model, predict_positive
from .splits import yearly_trials

SYSTEMS = ("modelo", "ablacion", "B2", "B1")
PREDICTIONS_FILENAME = "predicciones_prueba.parquet"
TRIALS_FILENAME = "ensayos_prueba.parquet"
CELLS_FILENAME = "casillas_prueba.csv"
SUMMARY_FILENAME = "resumen_prueba.json"
DIAGNOSTICS_DIRNAME = "diagnostico_prueba"
DEFAULT_LOG_PATH = ROOT / "reports" / "forecasting" / "bitacora_prueba.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "forecasting" / "prueba"
PREDICTION_COLUMNS = ["municipality_code", "t", "target_week", "anio", "y", "zona_t", "sistema", "candidato",
                      "conjunto_variables", "p"]
_MODEL_SYSTEMS = ("modelo", "ablacion")
_BOGOTA = timezone(timedelta(hours=-5))
_KEYS = ["municipality_code", "t"]


@dataclass(frozen=True)
class TestRun:
    """Predictions of the four systems, the trials and the log entry of one execution."""

    __test__ = False   # not a pytest test class, although its name starts with Test

    predicciones: pd.DataFrame
    ensayos: pd.DataFrame
    entrada: dict


# --------------------------------------------------------------------------- recipe and log
def _system_spec(entry, code: str, where: str) -> dict:
    require(isinstance(entry, dict), f"The recipe lacks {where} for municipality {code} (format of D7.2, DV-02).")
    for key in ("familia", "candidato", "hiperparametros", "conjunto_variables"):
        require(key in entry, f"Recipe {where}[{code}] lacks {key!r} (format of D7.2, DV-02).")
    require(entry["familia"] in FAMILIES, f"Recipe {where}[{code}] has an unknown family {entry['familia']!r}.")
    require(entry["conjunto_variables"] in FEATURE_SETS,
            f"Recipe {where}[{code}] has an unknown feature set {entry['conjunto_variables']!r}.")
    require(isinstance(entry["hiperparametros"], dict), f"Recipe {where}[{code}].hiperparametros must be a mapping.")
    return {key: entry[key] for key in ("familia", "candidato", "hiperparametros", "conjunto_variables")}


def recipe_systems(recipe: dict, codes) -> dict:
    """Model and ablation of every municipality (contrato §4.4 and §6.6): same family, the other feature set."""
    models, ablations = recipe.get("municipios"), recipe.get("ablacion")
    require(isinstance(models, dict) and isinstance(ablations, dict),
            "The recipe must have municipios and ablacion (format of D7.2, DV-02).")
    systems = {}
    for code in codes:
        model = _system_spec(models.get(code), code, "municipios")
        ablation = _system_spec(ablations.get(code), code, "ablacion")
        require(ablation["familia"] == model["familia"],
                f"Municipality {code}: the ablation must use the family of the model ({model['familia']}).")
        require(ablation["conjunto_variables"] != model["conjunto_variables"],
                f"Municipality {code}: the ablation must use the other feature set.")
        systems[code] = {"modelo": model, "ablacion": ablation}
    return systems


def read_log(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _commit() -> str | None:
    try:
        result = subprocess.run(["git", "--no-optional-locks", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


# --------------------------------------------------------------------------- the single run
def run_test(examples: pd.DataFrame, *, config: ForecastConfig, access: RecipeAccess, log_path: str | Path,
             output_dir: str | Path, reason: str | None = None) -> TestRun:
    """Refit the recipe, its ablation and the baselines once per (municipality, test year) and predict the test."""
    require(isinstance(examples, pd.DataFrame) and "target_week" in examples.columns,
            "examples must be the rows of build_examples.")
    first, last = config.validacion.prueba
    target_year = examples["target_week"].map(lambda week: week.year)
    test_rows = examples[(target_year >= first) & (target_year <= last)]
    require(len(test_rows) > 0, f"examples have no rows with target year in {first}..{last}.")
    check_test_access(test_rows["target_week"], access)
    recipe = load_frozen_recipe(access.recipe_path, channel_version=access.channel_version)
    codes = sorted(examples["municipality_code"].unique())
    systems = recipe_systems(recipe, codes)

    output_dir, log_path = Path(output_dir), Path(log_path)
    outputs = [output_dir / PREDICTIONS_FILENAME, output_dir / TRIALS_FILENAME]
    existing = [str(path) for path in outputs if path.exists()]
    require(not existing, f"Test outputs already exist and are never overwritten: {existing}.")
    entries = read_log(log_path)
    if entries:
        require(isinstance(reason, str) and reason.strip() != "",
                f"The test was already run {len(entries)} time(s) ({log_path}); another execution needs an explicit "
                f"reason, which is recorded.")
    entry = {"ejecucion": len(entries) + 1, "fecha_hora": datetime.now(_BOGOTA).isoformat(timespec="seconds"),
             "commit": _commit(), "receta_hash": recipe["hash_sha256"],
             "version_dvc_canal": recipe["procedencia"]["version_dvc_canal"], "razon": reason,
             "salida": str(output_dir)}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:   # written before fitting: an attempt is an execution
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    frames, trial_rows = [], []
    trials = yearly_trials(examples, range(first, last + 1),
                           first_train_year=config.validacion.primer_anio_entrenamiento)
    for trial in trials:
        train, evaluate = examples.loc[trial.train], examples.loc[trial.evaluate]
        code = trial.municipality_code
        p, names, sets = {}, {}, {}
        for system in _MODEL_SYSTEMS:
            spec = systems[code][system]
            model = fit_model(spec["familia"], spec["hiperparametros"], train,
                              features=FEATURE_SETS[spec["conjunto_variables"]])
            p[system] = predict_positive(model, evaluate)
            names[system], sets[system] = spec["candidato"], spec["conjunto_variables"]
        p["B2"] = b2_predict(train, evaluate, access=access)
        p["B1"] = b1_predict(train, evaluate, half_window=config.lineas_base.b1_semiventana, access=access)
        names.update(B2="B2", B1="B1")
        sets.update(B2=None, B1=None)
        for system in SYSTEMS:
            frames.append(pd.DataFrame({
                "municipality_code": code, "t": list(evaluate["t"]), "target_week": list(evaluate["target_week"]),
                "anio": trial.year, "y": evaluate["y"].to_numpy(dtype="int8"), "zona_t": list(evaluate["zona_t"]),
                "sistema": system, "candidato": names[system], "conjunto_variables": sets[system],
                "p": np.asarray(p[system], dtype=float)}))
        trial_rows.append({"municipality_code": code, "anio": trial.year, "t0": trial.t0,
                           "filas_entrenamiento": len(train), "positivas_entrenamiento": int(train["y"].sum()),
                           "primera_objetivo_entrenamiento": min(train["target_week"]),
                           "ultima_objetivo_entrenamiento": max(train["target_week"]),
                           "filas_evaluadas": len(evaluate), "primera_objetivo": min(evaluate["target_week"]),
                           "ultima_objetivo": max(evaluate["target_week"])})
    predictions = pd.concat(frames, ignore_index=True)[PREDICTION_COLUMNS]
    predictions["conjunto_variables"] = predictions["conjunto_variables"].astype(object)
    trials_table = pd.DataFrame(trial_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame, path in ((predictions, outputs[0]), (trials_table, outputs[1])):
        temporary = path.with_name(f".tmp-{path.name}")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    return TestRun(pd.read_parquet(outputs[0]), pd.read_parquet(outputs[1]), entry)


# --------------------------------------------------------------------------- scores (§7 and §8.4)
def aligned_systems(rows: pd.DataFrame, predictions: pd.DataFrame, systems=SYSTEMS) -> dict[str, np.ndarray]:
    """p of every system in the order of rows; each system must hold exactly the rows' keys, targets and labels."""
    out = {}
    for system in systems:
        own = predictions[predictions["sistema"] == system]
        require(len(own) == len(rows) and not own.duplicated(_KEYS).any(),
                f"Predictions of {system} must hold exactly the {len(rows)} rows once.")
        merged = rows[_KEYS + ["target_week", "y"]].merge(own[_KEYS + ["target_week", "y", "p"]], on=_KEYS,
                                                          how="left", suffixes=("", "_p"))
        require(bool(merged["p"].notna().all()), f"Some rows have no {system} prediction.")
        require(bool((merged["target_week"] == merged["target_week_p"]).all()), f"{system}: target week mismatch.")
        require(bool((merged["y"].astype(int) == merged["y_p"].astype(int)).all()), f"{system}: label mismatch.")
        out[system] = merged["p"].to_numpy(dtype=float)
    return out


def _losses(y, p, weeks, access, clip) -> tuple[float, float]:
    return log_loss(y, p, weeks, access=access, clip=clip), brier(y, p, weeks, access=access)


def _median(values) -> float | None:
    clean = [float(value) for value in values if not (isinstance(value, float) and math.isnan(value))]
    return statistics.median(clean) if clean else None


def score_summary(rows: pd.DataFrame, predictions: pd.DataFrame, observed: pd.DataFrame, *, config: ForecastConfig,
                 tau: float, access: RecipeAccess) -> tuple[pd.DataFrame, dict]:
    """Cells and summary of the saved test predictions (contrato §7.2-§7.4 and §8.4). Barrier first."""
    check_test_access(rows["target_week"], access)
    clip, alpha = config.metricas.recorte_log_loss, config.comparacion.alfa
    base = rows.sort_values(_KEYS, kind="mergesort").reset_index(drop=True)
    p = aligned_systems(base, predictions)
    years = np.array([week.year for week in base["target_week"]])
    y, weeks = base["y"].to_numpy(), list(base["target_week"])

    cells = []
    for code in sorted(base["municipality_code"].unique()):
        for year in sorted(set(years)):
            mask = ((base["municipality_code"] == code).to_numpy()) & (years == year)
            record = {"municipality_code": code, "anio": int(year), "filas": int(mask.sum()),
                      "positivas": int(y[mask].sum())}
            cell_weeks = [weeks[i] for i in np.flatnonzero(mask)]
            for system in SYSTEMS:
                record[f"logloss_{system}"], record[f"brier_{system}"] = _losses(y[mask], p[system][mask], cell_weeks,
                                                                                 access, clip)
            for reference in ("B2", "B1", "ablacion"):
                for metric in ("brier", "logloss"):
                    record[f"gana_{metric}_vs_{reference}"] = record[f"{metric}_modelo"] < record[f"{metric}_{reference}"]
            cells.append(record)
    cells = pd.DataFrame(cells)

    pooled = {system: dict(zip(("logloss", "brier"), _losses(y, p[system], weeks, access, clip))) for system in SYSTEMS}
    estimation = {
        "brier": {system: pooled[system]["brier"] for system in SYSTEMS},
        "logloss": {system: pooled[system]["logloss"] for system in SYSTEMS},
        "ap": {system: average_precision(y, p[system], weeks, access=access) for system in SYSTEMS},
        "bss_vs_b2": improvement(pooled["modelo"]["brier"], pooled["B2"]["brier"]),
        "bss_vs_b1": improvement(pooled["modelo"]["brier"], pooled["B1"]["brier"]),
        "mejora_logloss_vs_b2": improvement(pooled["modelo"]["logloss"], pooled["B2"]["logloss"]),
        "mejora_logloss_vs_b1": improvement(pooled["modelo"]["logloss"], pooled["B1"]["logloss"]),
        "prevalencia": float(y.mean()),
    }
    conclusions = {}
    for name, reference in (("vs_b2", "B2"), ("vs_b1", "B1"), ("ablacion_clima", "ablacion")):
        levels, wins = {}, {}
        for metric in ("brier", "logloss"):
            favorable = pooled["modelo"][metric] < pooled[reference][metric]   # BSS > 0 is the same condition
            wins[metric] = count_wins(cells[f"{metric}_modelo"], cells[f"{metric}_{reference}"])
            levels[metric] = weekly_level(bool(favorable), wins[metric], len(cells), alpha=alpha)
        conclusions[name] = {"nivel_brier": levels["brier"], "nivel_logloss": levels["logloss"],
                             "nivel": combined_level(levels["brier"], levels["logloss"]),
                             "casillas_ganadas_brier": wins["brier"], "casillas_ganadas_logloss": wins["logloss"],
                             "casillas": len(cells)}

    decision = config.decision
    model_events = evaluate_period(base, model_positive(p["modelo"], tau=tau), observed, k=decision.k,
                                   window=decision.W, access=access)
    channel_events = evaluate_period(base, channel_positive(base["zona_t"]), observed, k=decision.k,
                                     window=decision.W, access=access)
    onsets_model = model_events.onsets[(model_events.onsets["kind"] == "inicio") & model_events.onsets["evaluable"]]
    onsets_channel = channel_events.onsets[(channel_events.onsets["kind"] == "inicio")
                                           & channel_events.onsets["evaluable"]]
    require(list(onsets_model["week"]) == list(onsets_channel["week"]), "Model and channel onsets differ.")
    warned_model = onsets_model["warned"].to_numpy(dtype=bool)
    warned_channel = onsets_channel["warned"].to_numpy(dtype=bool)
    wins, discordant = discordant_wins(warned_model, warned_channel) if len(onsets_model) else (0, 0)
    both = warned_model & warned_channel
    events = {
        "inicios_evaluables": int(len(onsets_model)), "avisados_modelo": int(warned_model.sum()),
        "avisados_canal": int(warned_channel.sum()),
        "mcnemar": {"discrepantes": discordant, "a_favor_modelo": wins,
                    "nivel": event_level(wins, discordant, alpha=alpha)},
        "falsas_alarmas": {"modelo": model_events.summary["false_alarms"],
                           "canal": channel_events.summary["false_alarms"]},
        "falsas_alarmas_por_municipio_anio": {
            "modelo": model_events.summary["false_alarms_per_municipality_year"],
            "canal": channel_events.summary["false_alarms_per_municipality_year"]},
        "municipio_anios": model_events.summary["municipality_years"],
        "anticipacion_mediana_semanas": {"modelo": _median(onsets_model.loc[onsets_model["warned"], "lead"]),
                                         "canal": _median(onsets_channel.loc[onsets_channel["warned"], "lead"])},
        "delta_w": [{"municipality_code": code, "semana": week.isoformat(), "delta_w": float(lead_m - lead_c)}
                    for code, week, lead_m, lead_c, ok in zip(onsets_model["municipality_code"], onsets_model["week"],
                                                              onsets_model["lead"], onsets_channel["lead"], both)
                    if ok],
        "alertas_continuidad_modelo": model_events.summary["continuity_alerts"],
        "entradas": {"evaluables": model_events.summary["evaluable_entries"],
                     "avisadas_modelo": model_events.summary["warned_entries"],
                     "avisadas_canal": channel_events.summary["warned_entries"]},
    }
    summary = {"tau": float(tau), "k": decision.k, "W": decision.W, "filas": len(base),
               "estimacion": estimation, "conclusiones": conclusions, "eventos": events}
    return cells, summary


# --------------------------------------------------------------------------- MLflow
def log_test_runs(output_dir: Path, *, config: ForecastConfig, recipe: dict, entry: dict) -> int:
    """One run per (municipality, system) in the test experiment, with Brier and log-loss per year, from the saved
    cells; the summary as an artifact of a run per execution. Returns the number of runs."""
    from mlflow.entities import Metric, Param
    from mlflow.tracking import MlflowClient

    from .tracking import allow_file_store, configure_tracking

    cells = pd.read_csv(output_dir / CELLS_FILENAME, dtype={"municipality_code": str})
    predictions = pd.read_parquet(output_dir / PREDICTIONS_FILENAME)
    uri = configure_tracking(config)
    allow_file_store(uri)
    client = MlflowClient(tracking_uri=uri)
    name = config.seguimiento.experimentos.prueba
    experiment = client.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment is not None else client.create_experiment(name)
    common = {"fase": "prueba", "commit": entry.get("commit") or "desconocido", "receta_hash": recipe["hash_sha256"],
              "contrato_commit": recipe["procedencia"]["commit_contrato"],
              "datos_dvc": recipe["procedencia"]["version_dvc_canal"], "ejecucion": str(entry["ejecucion"])}
    count = 0
    for code in sorted(cells["municipality_code"].unique()):
        own = cells[cells["municipality_code"] == code]
        for system in SYSTEMS:
            rows = predictions[(predictions["municipality_code"] == code) & (predictions["sistema"] == system)]
            tags = {**common, "municipio": code, "sistema": system, "candidato": str(rows["candidato"].iloc[0]),
                    "conjunto_variables": str(rows["conjunto_variables"].iloc[0])}
            run = client.create_run(experiment_id, tags=tags, run_name=f"{code}__{system}")
            stamp = int(time.time() * 1000)
            metrics = [Metric(f"{metric}_{int(year)}", float(value), stamp, 0)
                       for year, b, l in zip(own["anio"], own[f"brier_{system}"], own[f"logloss_{system}"])
                       for metric, value in (("brier", b), ("logloss", l))]
            params = []
            if system in _MODEL_SYSTEMS:
                spec = recipe["municipios" if system == "modelo" else "ablacion"][code]
                params = [Param(key, str(value)) for key, value in spec["hiperparametros"].items()]
            client.log_batch(run.info.run_id, metrics=metrics, params=params)
            client.set_terminated(run.info.run_id, "FINISHED")
            count += 1
    run = client.create_run(experiment_id, tags={**common, "sistema": "resumen"}, run_name="resumen_prueba")
    client.log_artifact(run.info.run_id, str(output_dir / SUMMARY_FILENAME))
    client.set_terminated(run.info.run_id, "FINISHED")
    return count + 1


# --------------------------------------------------------------------------- command line
def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"{type(value).__name__} is not JSON-serializable")


def _dirty_paths() -> list[str]:
    result = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain=v1", "--untracked-files=all", "--",
                             "src", "configs"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.splitlines()


def write_outputs(output_dir: Path, cells: pd.DataFrame, summary: dict, run: TestRun) -> None:
    for name in (CELLS_FILENAME, SUMMARY_FILENAME):
        require(not (output_dir / name).exists(), f"{output_dir / name} exists and is never overwritten.")
    cells.to_csv(output_dir / CELLS_FILENAME, index=False, encoding="utf-8", lineterminator="\n")
    payload = {"ejecucion": run.entrada, **summary}
    (output_dir / SUMMARY_FILENAME).write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                                          default=_json_default), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    from .diagnostics import TEST_LABEL, diagnostic_inputs, write_diagnostics

    parser = argparse.ArgumentParser(description="Single test run with the frozen recipe (F7).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bitacora", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--razon", default=None, help="Required for any execution after the first one.")
    parser.add_argument("--sin-seguimiento", action="store_true", help="Do not log runs in MLflow.")
    parser.add_argument("--solo-seguimiento", action="store_true",
                        help="Only log in MLflow the saved outputs of --salida (the test is not run again).")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        access = RecipeAccess.from_config(config)
        recipe = load_frozen_recipe(ROOT / access.recipe_path, channel_version=access.channel_version)
        if args.solo_seguimiento:
            summary = json.loads((args.salida / SUMMARY_FILENAME).read_text(encoding="utf-8"))
            require(summary["ejecucion"]["receta_hash"] == recipe["hash_sha256"],
                    "The saved outputs belong to another recipe.")
            print(f"{log_test_runs(args.salida, config=config, recipe=recipe, entry=summary['ejecucion'])} runs logged.")
            return 0
        dirty = _dirty_paths()
        require(not dirty, f"src/ or configs/ have uncommitted changes; the test runs only from a commit: {dirty[:10]}")
        product = load_channel_product(config.datos.canal if Path(config.datos.canal).is_absolute()
                                       else ROOT / config.datos.canal, expected_md5=config.datos.version_dvc_canal)
        examples = build_examples(product, h=config.horizonte.h, canal=config.canal, features=config.features)
        access = RecipeAccess.from_config(config, recipe_path=ROOT / access.recipe_path)
        run = run_test(examples, config=config, access=access, log_path=args.bitacora, output_dir=args.salida,
                       reason=args.razon)
        print(f"Test run {run.entrada['ejecucion']} logged in {args.bitacora}; predictions in {args.salida}.")
        model = run.predicciones[run.predicciones["sistema"] == "modelo"]
        rows = model[_KEYS + ["target_week", "zona_t", "y"]].reset_index(drop=True)
        observed, tau = product.to_pandas(), float(recipe["decision"]["tau"])
        cells, summary = score_summary(rows, run.predicciones, observed, config=config, tau=tau, access=access)
        write_outputs(args.salida, cells, summary, run)
        diagnostics = diagnostic_inputs(rows, run.predicciones[run.predicciones["sistema"] != "ablacion"], observed,
                                        tau=tau, k=config.decision.k, window=config.decision.W,
                                        clip=config.metricas.recorte_log_loss, access=access, rotulo=TEST_LABEL)
        write_diagnostics(diagnostics, args.salida / DIAGNOSTICS_DIRNAME)
        print(f"Scores and diagnostic inputs written to {args.salida}.")
        if not args.sin_seguimiento:
            print(f"{log_test_runs(args.salida, config=config, recipe=recipe, entry=run.entrada)} runs logged in MLflow.")
    except Exception as exc:   # reported with its traceback
        traceback.print_exc()
        print(f"Evaluation stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
