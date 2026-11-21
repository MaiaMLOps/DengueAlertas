"""Ejecución reproducible del aporte final EWT.

El flujo preserva los tres contratos recibidos (canal, episodios y 66 features),
selecciona modelos sólo con validación rolling-origin 2019-2022 y abre una única
vez la prueba 2023-2025. La purga de h semanas evita que una etiqueta de
entrenamiento utilice resultados del período siguiente.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .episodes_ewt import apply_episode_labels
from .ewt_channel import build_ewt_channel
from .features_ewt import (
    FEATURE_FAMILY_COUNTS,
    FEATURE_SETS,
    materialize_horizon_features,
    build_features,
    usable_columns,
)


@dataclass
class ModelBundle:
    estimator: Any
    calibrator: Any
    columns: list[str]
    threshold: float
    municipality_code: str
    municipality: str
    task: str
    target: str
    horizon: int
    algorithm: str
    feature_set: str
    metadata: dict[str, Any]

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.estimator.predict_proba(frame.reindex(columns=self.columns))[:, 1]
        return apply_platt(self.calibrator, raw)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        probability = self.probability(frame)
        alert = probability >= self.threshold
        return pd.DataFrame(
            {
                "probability": probability,
                "alert": alert.astype(int),
                "threshold": float(self.threshold),
                "task": self.task,
                "horizon_weeks": int(self.horizon),
            },
            index=frame.index,
        )


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    horizon = int(config["label"]["horizon"])
    if horizon < 1:
        raise ValueError("label.horizon debe ser positivo")
    if config["evaluation"].get("purge_weeks", "auto") not in {"auto", horizon}:
        raise ValueError("La purga debe ser 'auto' o igual al horizonte")
    if config["mlflow"].get("transition_stage") is not None:
        raise ValueError("Este aporte no permite transición de etapa en MLflow")
    if config["mlflow"].get("set_alias") is not None:
        raise ValueError("Este aporte no permite alias de ciclo de vida en MLflow")
    return config


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_panel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            frame = pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError("Para leer Parquet instale pyarrow según requirements.txt") from exc
    elif path.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    else:
        raise ValueError("El insumo debe ser .parquet o .csv")
    required = {"municipality_code", "week_start", "confirmed_records_screening"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")
    frame["municipality_code"] = frame["municipality_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    frame["week_start"] = pd.to_datetime(frame["week_start"], errors="raise")
    for column in ("partial_week", "no_eligible_records_observed"):
        if column in frame and frame[column].dtype == object:
            frame[column] = frame[column].astype(str).str.lower().map({"true": True, "false": False})
    return frame.sort_values(["municipality_code", "week_start"]).reset_index(drop=True)


def validate_panel(frame: pd.DataFrame, pilots: dict[str, str]) -> dict[str, Any]:
    selected = frame[frame["municipality_code"].isin(pilots)].copy()
    if selected.empty:
        raise ValueError("No hay municipios piloto en el panel")
    duplicate = int(selected.duplicated(["municipality_code", "week_start"]).sum())
    if duplicate:
        raise ValueError(f"Hay {duplicate} claves municipio-semana duplicadas")
    if (selected["week_start"].dt.weekday != 0).any():
        raise ValueError("week_start debe ser lunes")
    bad_gaps: dict[str, int] = {}
    for code, group in selected.groupby("municipality_code"):
        bad = group["week_start"].sort_values().diff().dropna().ne(pd.Timedelta(days=7)).sum()
        if bad:
            bad_gaps[code] = int(bad)
    if bad_gaps:
        raise ValueError(f"Huecos semanales en el panel: {bad_gaps}")
    missing_pilots = sorted(set(pilots) - set(selected["municipality_code"]))
    if missing_pilots:
        raise ValueError(f"Faltan pilotos: {missing_pilots}")
    return {
        "rows": int(len(selected)),
        "columns": int(selected.shape[1]),
        "municipalities": int(selected["municipality_code"].nunique()),
        "date_min": selected["week_start"].min().date().isoformat(),
        "date_max": selected["week_start"].max().date().isoformat(),
        "duplicate_keys": duplicate,
        "missing_values": {k: int(v) for k, v in selected.isna().sum().items() if v},
    }


def prepare_supervised_panel(source: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pilots = {str(k).zfill(5): str(v) for k, v in config["pilots"].items()}
    source = source[source["municipality_code"].isin(pilots)].copy()
    profile = validate_panel(source, pilots)
    horizon = int(config["label"]["horizon"])
    channel = build_ewt_channel(source)
    labeled, episodes = apply_episode_labels(
        channel,
        horizons=[horizon],
        min_len=int(config["label"]["min_episode_weeks"]),
        merge_gap=int(config["label"]["merge_gap_weeks"]),
        refractory=int(config["label"]["refractory_weeks"]),
        flag_col="excess_flag_ewt",
        known_col="ch_ewt",
    )
    labeled[f"y_zone_h{horizon}"] = (
        labeled.groupby("municipality_code", sort=False)["excess_flag_ewt"]
        .shift(-horizon)
        .astype("Float64")
    )
    features = build_features(labeled, horizons=[horizon], default_horizon=horizon)
    profile.update(
        {
            "ewt_status": {str(k): int(v) for k, v in channel["ewt_status"].value_counts().items()},
            "episodes": int(len(episodes)),
            "feature_family_counts": FEATURE_FAMILY_COUNTS,
            "feature_contract_total": int(sum(FEATURE_FAMILY_COUNTS.values())),
        }
    )
    return features, episodes, profile


def task_target(task: str, horizon: int) -> str:
    mapping = {"zone_future": f"y_zone_h{horizon}", "pre_onset": f"y_pre_onset_h{horizon}"}
    if task not in mapping:
        raise ValueError(f"Tarea desconocida: {task}")
    return mapping[task]


def _positive_weight(y: pd.Series) -> float:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return max(1.0, negatives / max(positives, 1))


def build_estimator(algorithm: str, config: dict[str, Any], y: pd.Series):
    random_state = int(config["random_state"])
    model_cfg = config["modeling"]
    if algorithm == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(model_cfg.get("logistic_c", 0.3)),
                        class_weight="balanced",
                        max_iter=4000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if algorithm == "lightgbm":
        from lightgbm import LGBMClassifier

        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=int(model_cfg.get("tree_estimators", 180)),
                        learning_rate=0.03,
                        num_leaves=7,
                        max_depth=4,
                        min_child_samples=10,
                        reg_lambda=2.0,
                        class_weight="balanced",
                        random_state=random_state,
                        verbosity=-1,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    if algorithm == "xgboost":
        from xgboost import XGBClassifier

        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=int(model_cfg.get("tree_estimators", 180)),
                        learning_rate=0.03,
                        max_depth=3,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=2.0,
                        scale_pos_weight=_positive_weight(y),
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    raise ValueError(f"Algoritmo no soportado: {algorithm}")


def fit_platt(raw_probability: np.ndarray, y_true: np.ndarray):
    p = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return None
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    calibrator.fit(logit, y)
    return calibrator


def apply_platt(calibrator, raw_probability: np.ndarray) -> np.ndarray:
    raw = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    if calibrator is None:
        return raw
    logit = np.log(raw / (1 - raw)).reshape(-1, 1)
    return calibrator.predict_proba(logit)[:, 1]


def choose_cost_threshold(y_true, probabilities, false_negative_cost: float, false_positive_cost: float) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.r_[0.01, np.linspace(0.02, 0.98, 97), p])
    best = (float("inf"), float("inf"), 0.5)
    for threshold in candidates:
        pred = p >= threshold
        fn = int(((y == 1) & ~pred).sum())
        fp = int(((y == 0) & pred).sum())
        cost = false_negative_cost * fn + false_positive_cost * fp
        candidate = (cost, fp, float(threshold))
        if candidate < best:
            best = candidate
    return best[2]


def _runs(dates: pd.Series, values: np.ndarray) -> int:
    ordered = pd.DataFrame({"date": pd.to_datetime(dates), "active": values}).sort_values("date")
    runs = 0
    previous = False
    previous_date = None
    for row in ordered.itertuples(index=False):
        contiguous = previous_date is not None and row.date - previous_date == pd.Timedelta(weeks=1)
        if bool(row.active) and (not previous or not contiguous):
            runs += 1
        previous = bool(row.active)
        previous_date = row.date
    return runs


def classification_metrics(y_true, probability, threshold: float, dates=None) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    pred = p >= float(threshold)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    output: dict[str, Any] = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if dates is not None:
        years = max(pd.Series(pd.to_datetime(dates)).dt.year.nunique(), 1)
        output["alert_runs_per_year"] = float(_runs(pd.Series(dates), pred) / years)
        output["false_alert_runs_per_year"] = float(_runs(pd.Series(dates), pred & (y == 0)) / years)
    return output


def event_metrics(predictions: pd.DataFrame, episodes: pd.DataFrame, code: str, horizon: int) -> dict[str, Any]:
    if predictions.empty:
        return {"n_outbreaks": 0, "outbreaks_anticipated": 0, "event_sensitivity": None, "lead_weeks_mean": None}
    start, end = predictions["week_start"].min(), predictions["week_start"].max()
    local = episodes[episodes["municipality_code"].astype(str).str.zfill(5).eq(code)].copy()
    if not local.empty:
        local["start_week"] = pd.to_datetime(local["start_week"])
        local = local[local["start_week"].between(start, end)]
    anticipated = 0
    leads: list[int] = []
    for episode in local.itertuples(index=False):
        mask = predictions["week_start"].between(
            episode.start_week - pd.Timedelta(weeks=horizon),
            episode.start_week - pd.Timedelta(weeks=1),
        ) & predictions["alert"].astype(bool)
        if mask.any():
            anticipated += 1
            first = predictions.loc[mask, "week_start"].min()
            leads.append(int((episode.start_week - first).days // 7))
    return {
        "n_outbreaks": int(len(local)),
        "outbreaks_anticipated": int(anticipated),
        "event_sensitivity": float(anticipated / len(local)) if len(local) else None,
        "lead_weeks_mean": float(np.mean(leads)) if leads else None,
    }


def rolling_oof(
    municipal: pd.DataFrame,
    target: str,
    algorithm: str,
    feature_set: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    horizon = int(config["label"]["horizon"])
    folds: list[pd.DataFrame] = []
    plans: list[dict[str, Any]] = []
    for year in map(int, config["evaluation"]["selection_years"]):
        valid = municipal[municipal["week_start"].dt.year.eq(year)].dropna(subset=[target]).copy()
        if valid.empty or valid[target].nunique() < 2:
            continue
        valid_start = valid["week_start"].min()
        cutoff = valid_start - pd.Timedelta(weeks=horizon)
        train = municipal[municipal["week_start"].lt(cutoff)].dropna(subset=[target]).copy()
        if train.empty or train[target].nunique() < 2:
            continue
        columns = usable_columns(train, feature_set)
        if not columns:
            continue
        estimator = build_estimator(algorithm, config, train[target].astype(int))
        estimator.fit(train[columns], train[target].astype(int))
        raw = estimator.predict_proba(valid[columns])[:, 1]
        folds.append(
            pd.DataFrame(
                {
                    "week_start": valid["week_start"].to_numpy(),
                    "y_true": valid[target].astype(int).to_numpy(),
                    "raw_probability": raw,
                    "fold_year": year,
                }
            )
        )
        plans.append(
            {
                "validation_year": year,
                "train_start": train["week_start"].min().date().isoformat(),
                "train_end": train["week_start"].max().date().isoformat(),
                "validation_start": valid["week_start"].min().date().isoformat(),
                "validation_end": valid["week_start"].max().date().isoformat(),
                "purge_weeks": horizon,
                "n_train": int(len(train)),
                "n_valid": int(len(valid)),
                "n_features": int(len(columns)),
            }
        )
    # Tarea pre-onset: algunos municipios tienen tan pocos inicios que ningún
    # año aislado contiene ambas clases. En ese caso se usa un único bloque
    # 2019-2022, todavía separado del entrenamiento por la misma purga h y sin
    # tocar la prueba 2023-2025. El plan deja explícito que fue un fallback.
    if not folds:
        selection_years = set(map(int, config["evaluation"]["selection_years"]))
        valid = municipal[municipal["week_start"].dt.year.isin(selection_years)].dropna(subset=[target]).copy()
        if not valid.empty and valid[target].nunique() == 2:
            valid_start = valid["week_start"].min()
            cutoff = valid_start - pd.Timedelta(weeks=horizon)
            train = municipal[municipal["week_start"].lt(cutoff)].dropna(subset=[target]).copy()
            if not train.empty and train[target].nunique() == 2:
                columns = usable_columns(train, feature_set)
                if columns:
                    estimator = build_estimator(algorithm, config, train[target].astype(int))
                    estimator.fit(train[columns], train[target].astype(int))
                    raw = estimator.predict_proba(valid[columns])[:, 1]
                    folds.append(
                        pd.DataFrame(
                            {
                                "week_start": valid["week_start"].to_numpy(),
                                "y_true": valid[target].astype(int).to_numpy(),
                                "raw_probability": raw,
                                "fold_year": "2019-2022-block",
                            }
                        )
                    )
                    plans.append(
                        {
                            "validation_year": "2019-2022-block",
                            "fallback": "pooled_selection_block_due_to_sparse_classes",
                            "train_start": train["week_start"].min().date().isoformat(),
                            "train_end": train["week_start"].max().date().isoformat(),
                            "validation_start": valid["week_start"].min().date().isoformat(),
                            "validation_end": valid["week_start"].max().date().isoformat(),
                            "purge_weeks": horizon,
                            "n_train": int(len(train)),
                            "n_valid": int(len(valid)),
                            "n_features": int(len(columns)),
                        }
                    )
    if not folds:
        return pd.DataFrame(), plans
    return pd.concat(folds, ignore_index=True).sort_values("week_start"), plans


def estimator_importance(estimator, columns: list[str]) -> pd.DataFrame:
    fitted = estimator.named_steps.get("model", estimator)
    if hasattr(fitted, "coef_"):
        signed = np.asarray(fitted.coef_[0], dtype=float)
        values = np.abs(signed)
    elif hasattr(fitted, "feature_importances_"):
        values = np.asarray(fitted.feature_importances_, dtype=float)
        signed = np.full_like(values, np.nan)
    else:
        return pd.DataFrame(columns=["feature", "importance", "coefficient"])
    if len(values) != len(columns):
        raise ValueError("La importancia no coincide con el número de features")
    return pd.DataFrame({"feature": columns, "importance": values, "coefficient": signed}).sort_values("importance", ascending=False)


def _climatology_probability(train: pd.DataFrame, test: pd.DataFrame, target: str, horizon: int) -> np.ndarray:
    train = train.copy()
    test = test.copy()
    train["_target_woy"] = (train["week_start"] + pd.to_timedelta(horizon * 7, unit="D")).dt.isocalendar().week.astype(int)
    test["_target_woy"] = (test["week_start"] + pd.to_timedelta(horizon * 7, unit="D")).dt.isocalendar().week.astype(int)
    prevalence = float(train[target].mean())
    stats = train.groupby("_target_woy")[target].agg(["sum", "count"])
    smooth = (stats["sum"] + 2.0 * prevalence) / (stats["count"] + 2.0)
    return test["_target_woy"].map(smooth).fillna(prevalence).to_numpy(dtype=float)


def train_one(
    panel: pd.DataFrame,
    episodes: pd.DataFrame,
    code: str,
    task: str,
    config: dict[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    horizon = int(config["label"]["horizon"])
    target = task_target(task, horizon)
    municipality = str(config["pilots"][code])
    municipal = materialize_horizon_features(
        panel[panel["municipality_code"].eq(code)].sort_values("week_start").reset_index(drop=True),
        horizon,
    )
    candidates: list[dict[str, Any]] = []
    candidate_objects: dict[tuple[str, str], dict[str, Any]] = {}
    for algorithm in config["modeling"]["algorithms"]:
        for feature_set in config["modeling"]["feature_sets"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oof, fold_plan = rolling_oof(municipal, target, algorithm, feature_set, config)
            if oof.empty or oof["y_true"].nunique() < 2:
                candidates.append(
                    {
                        "task": task,
                        "municipality_code": code,
                        "municipality": municipality,
                        "algorithm": algorithm,
                        "feature_set": feature_set,
                        "status": "skipped",
                        "reason": "sin folds con ambas clases",
                    }
                )
                continue
            calibrator = fit_platt(oof["raw_probability"].to_numpy(), oof["y_true"].to_numpy())
            oof["probability"] = apply_platt(calibrator, oof["raw_probability"].to_numpy())
            threshold = choose_cost_threshold(
                oof["y_true"],
                oof["probability"],
                float(config["evaluation"]["false_negative_cost"]),
                float(config["evaluation"]["false_positive_cost"]),
            )
            oof["threshold"] = threshold
            oof["alert"] = oof["probability"] >= threshold
            metrics = classification_metrics(oof["y_true"], oof["probability"], threshold, oof["week_start"])
            row = {
                "task": task,
                "municipality_code": code,
                "municipality": municipality,
                "algorithm": algorithm,
                "feature_set": feature_set,
                "status": "evaluated",
                "n_folds": int(len(fold_plan)),
                **metrics,
            }
            candidates.append(row)
            candidate_objects[(algorithm, feature_set)] = {
                "row": row,
                "oof": oof,
                "fold_plan": fold_plan,
                "calibrator": calibrator,
                "threshold": threshold,
            }
    evaluated = [row for row in candidates if row.get("status") == "evaluated"]
    if not evaluated:
        raise RuntimeError(f"No hubo candidatos evaluables para {municipality}, {task}")
    eligible = [
        row for row in evaluated
        if float(row.get("false_alert_runs_per_year", math.inf)) <= float(config["evaluation"]["max_false_alert_runs_per_year"])
    ]
    pool = eligible or evaluated
    winner_row = max(
        pool,
        key=lambda row: (
            float(row.get("average_precision") or -1),
            -float(row.get("brier") or math.inf),
            float(row.get("recall") or 0),
            -len(FEATURE_SETS[row["feature_set"]]),
        ),
    )
    selected = candidate_objects[(winner_row["algorithm"], winner_row["feature_set"])]
    test_years = set(map(int, config["evaluation"]["test_years"]))
    test = municipal[municipal["week_start"].dt.year.isin(test_years)].dropna(subset=[target]).copy()
    if test.empty or test[target].nunique() < 2:
        raise RuntimeError(f"Prueba sin ambas clases para {municipality}, {task}")
    test_start = test["week_start"].min()
    train_cutoff = test_start - pd.Timedelta(weeks=horizon)
    final_train = municipal[municipal["week_start"].lt(train_cutoff)].dropna(subset=[target]).copy()
    columns = usable_columns(final_train, winner_row["feature_set"])
    estimator = build_estimator(winner_row["algorithm"], config, final_train[target].astype(int))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(final_train[columns], final_train[target].astype(int))
    raw_test = estimator.predict_proba(test[columns])[:, 1]
    probability = apply_platt(selected["calibrator"], raw_test)
    threshold = float(selected["threshold"])
    predictions = pd.DataFrame(
        {
            "municipality_code": code,
            "municipality": municipality,
            "task": task,
            "week_start": test["week_start"].to_numpy(),
            "y_true": test[target].astype(int).to_numpy(),
            "raw_probability": raw_test,
            "probability": probability,
            "threshold": threshold,
            "alert": probability >= threshold,
        }
    )
    test_metrics = classification_metrics(predictions["y_true"], predictions["probability"], threshold, predictions["week_start"])
    if task == "pre_onset":
        test_metrics.update(event_metrics(predictions, episodes, code, horizon))
    importance = estimator_importance(estimator, columns)
    bundle = ModelBundle(
        estimator=estimator,
        calibrator=selected["calibrator"],
        columns=columns,
        threshold=threshold,
        municipality_code=code,
        municipality=municipality,
        task=task,
        target=target,
        horizon=horizon,
        algorithm=winner_row["algorithm"],
        feature_set=winner_row["feature_set"],
        metadata={"selection_years": config["evaluation"]["selection_years"], "test_years": config["evaluation"]["test_years"]},
    )
    destination = output_root / "models" / task / code
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination / "model.joblib")
    pd.DataFrame(candidates).to_csv(destination / "candidates_validation.csv", index=False)
    selected["oof"].assign(municipality_code=code, task=task).to_csv(destination / "predictions_validation.csv", index=False)
    predictions.to_csv(destination / "predictions_test.csv", index=False)
    importance.to_csv(destination / "feature_importance.csv", index=False)
    (destination / "fold_plan.json").write_text(json.dumps(selected["fold_plan"], ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_rows: list[dict[str, Any]] = []
    prevalence = float(final_train[target].mean())
    persistence = pd.to_numeric(test.get("ch_E_lag0"), errors="coerce").fillna(prevalence).clip(0, 1).to_numpy()
    climatology = _climatology_probability(final_train, test, target, horizon)
    for name, baseline_probability in (("persistence", persistence), ("climatology", climatology)):
        baseline_threshold = choose_cost_threshold(
            selected["oof"]["y_true"],
            np.repeat(prevalence, len(selected["oof"])),
            float(config["evaluation"]["false_negative_cost"]),
            float(config["evaluation"]["false_positive_cost"]),
        ) if name == "climatology" else 0.5
        metrics = classification_metrics(test[target].astype(int), baseline_probability, baseline_threshold, test["week_start"])
        baseline_rows.append(
            {
                "task": task,
                "municipality_code": code,
                "municipality": municipality,
                "baseline": name,
                **metrics,
            }
        )
    result = {
        "task": task,
        "target": target,
        "municipality_code": code,
        "municipality": municipality,
        "horizon": horizon,
        "winner_algorithm": winner_row["algorithm"],
        "winner_feature_set": winner_row["feature_set"],
        "n_features": int(len(columns)),
        "validation_average_precision": winner_row["average_precision"],
        "validation_brier": winner_row["brier"],
        "threshold": threshold,
        "train_start": final_train["week_start"].min().date().isoformat(),
        "train_end": final_train["week_start"].max().date().isoformat(),
        "test_start": test["week_start"].min().date().isoformat(),
        "test_end": test["week_start"].max().date().isoformat(),
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    (destination / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return result, candidates, baseline_rows


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(
    results: pd.DataFrame,
    candidates: pd.DataFrame,
    baselines: pd.DataFrame,
    skipped: pd.DataFrame,
    profile: dict[str, Any],
    mlflow_status: dict[str, Any],
    output_root: Path,
    config: dict[str, Any],
) -> str:
    horizon = int(config["label"]["horizon"])
    lines = [
        "# Reporte final — aporte de ingeniería de variables y modelamiento EWT",
        "",
        f"**Versión metodológica:** {config['methodology_version']}  ",
        f"**Horizonte:** {horizon} semanas  ",
        f"**Selección:** {config['evaluation']['selection_years']} mediante rolling-origin purgado  ",
        f"**Prueba bloqueada:** {config['evaluation']['test_years']}  ",
        "",
        "## Resumen ejecutivo",
        "",
        "Se ejecutó el flujo completo sobre el panel real de seis municipios: canal EWT, episodios, 66 variables sin espaciales, selección cronológica de candidatos y prueba final independiente. La pregunta principal fue si el municipio estará en zona epidémica ocho semanas adelante (`zone_future`); `pre_onset` quedó como análisis secundario porque depende de una regla de inicio sensible a interrupciones semanales.",
        "",
    ]
    primary = results[results["task"].eq("zone_future")]
    if not primary.empty:
        best = primary.sort_values("test_average_precision", ascending=False).iloc[0]
        primary_baselines = baselines[baselines["task"].eq("zone_future")]
        persistence = primary_baselines[primary_baselines["baseline"].eq("persistence")]
        persistence_ap = float(persistence["average_precision"].mean())
        persistence_brier = float(persistence["brier"].mean())
        macro_ap = float(primary["test_average_precision"].mean())
        macro_brier = float(primary["test_brier"].mean())
        macro_prevalence = float(primary["test_prevalence"].mean())
        ap_wins = 0
        brier_wins = 0
        for model_row in primary.itertuples(index=False):
            base = persistence[persistence["municipality_code"].astype(str).str.zfill(5).eq(str(model_row.municipality_code).zfill(5))]
            if not base.empty:
                ap_wins += int(float(model_row.test_average_precision) > float(base.iloc[0]["average_precision"]))
                brier_wins += int(float(model_row.test_brier) < float(base.iloc[0]["brier"]))
        lines.append(
            f"El mayor AUC-PR de prueba para la tarea principal fue **{_fmt(best.test_average_precision)}** en **{best.municipality}**, con {best.winner_algorithm} y `{best.winner_feature_set}`. La media macro fue **{_fmt(macro_ap)}**, pero debe leerse junto con una prevalencia media muy alta (**{_fmt(macro_prevalence)}**)."
        )
        lines.append(
            f"Frente a persistencia, el modelo elevó el AUC-PR medio sólo de **{_fmt(persistence_ap)}** a **{_fmt(macro_ap)}** y ganó en {ap_wins}/6 municipios; en calibración ocurrió lo contrario: el Brier medio empeoró de **{_fmt(persistence_brier)}** a **{_fmt(macro_brier)}** y sólo ganó en {brier_wins}/6. Por tanto, el resultado no demuestra todavía una mejora operativa general sobre la referencia."
        )
        lines.append("")
    lines += [
        "## Datos y controles",
        "",
        f"- Panel: {profile['rows']:,} filas, {profile['municipalities']} municipios, {profile['date_min']} a {profile['date_max']}.",
        f"- Canal: {profile['ewt_status']}; episodios derivados: {profile['episodes']}.",
        f"- Contrato de variables: {profile['feature_contract_total']} features = {profile['feature_family_counts']}.",
        "- Causalidad temporal: casos y clima rezagados una semana; canal anual sólo con los cinco años previos; purga de ocho semanas en cada fold y antes de la prueba.",
        "- La prueba 2023–2025 no intervino en selección, calibración ni elección del umbral.",
        "",
        "## Resultados por municipio",
        "",
        "| Tarea | Municipio | Modelo | Features | N | AUC-PR | ROC-AUC | Brier | Recall | Especificidad | Umbral |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results.sort_values(["task", "municipality"]).itertuples(index=False):
        lines.append(
            f"| {row.task} | {row.municipality} | {row.winner_algorithm} | {row.winner_feature_set} | {row.test_n} | {_fmt(row.test_average_precision)} | {_fmt(row.test_roc_auc)} | {_fmt(row.test_brier)} | {_fmt(row.test_recall)} | {_fmt(row.test_specificity)} | {_fmt(row.threshold)} |"
        )
    if not skipped.empty:
        lines += ["", "### Combinaciones no evaluables", ""]
        for row in skipped.itertuples(index=False):
            lines.append(f"- `{row.task}` / **{row.municipality}**: {row.reason}")
    lines += ["", "## Comparación con referencias", "", "| Tarea | Municipio | Referencia | AUC-PR | Brier | Recall | Especificidad |", "|---|---|---|---:|---:|---:|---:|"]
    for row in baselines.sort_values(["task", "municipality", "baseline"]).itertuples(index=False):
        lines.append(
            f"| {row.task} | {row.municipality} | {row.baseline} | {_fmt(row.average_precision)} | {_fmt(row.brier)} | {_fmt(row.recall)} | {_fmt(row.specificity)} |"
        )
    lines += ["", "## Aporte de ingeniería de variables", ""]
    win_counts = results.groupby(["task", "winner_feature_set"]).size().reset_index(name="wins")
    for row in win_counts.itertuples(index=False):
        lines.append(f"- `{row.task}`: `{row.winner_feature_set}` ganó en {row.wins} municipio(s).")
    family_rows = []
    for path in output_root.glob("models/*/*/feature_importance.csv"):
        data = pd.read_csv(path).head(15)
        task, code = path.parts[-3], path.parts[-2]
        for feature in data["feature"]:
            family_rows.append({"task": task, "code": code, "feature": feature, "family": feature.split("_", 1)[0]})
    if family_rows:
        family = pd.DataFrame(family_rows).groupby(["task", "family"]).size().reset_index(name="top15_appearances")
        family.to_csv(output_root / "feature_family_importance.csv", index=False)
        lines.append("")
        lines.append("Apariciones de cada familia en el top 15 de los modelos ganadores:")
        for row in family.itertuples(index=False):
            lines.append(f"- `{row.task}` / `{row.family}_`: {row.top15_appearances}.")
    lines += [
        "",
        "## Hallazgos metodológicos",
        "",
        "1. La tarea `zone_future` conserva muchas más semanas evaluables que `pre_onset`; por ello es la base más estable para comparar modelos y referencias.",
        "2. `pre_onset` enmascara brote y período refractario. Cuando el EWT permanece excedido durante rachas largas, quedan pocas negativas/positivas independientes; sus métricas deben acompañarse siempre del tamaño de prueba y número de episodios.",
        "3. La comparación entre `epi`, `epi_clima` y `epi_clima_demo` es una ablación directa del aporte incremental. Ganar con más variables no demuestra causalidad climática o demográfica.",
        "4. La calibración Platt usa únicamente predicciones fuera de muestra de 2019–2022. El umbral minimiza costo 5:1 de falso negativo frente a falso positivo y nunca se recalibra con la prueba.",
        "5. La selección territorial fue retrospectiva y los confirmados son estados finales; el desempeño no equivale todavía a una validación prospectiva con retrasos de notificación reales.",
        "6. En `zone_future`, la especificidad de prueba quedó entre 0 y 0,30 y fue cero en tres municipios. El costo 5:1 más la alta prevalencia empujó al sistema a alertar casi siempre; antes de uso operativo debe redefinirse el costo o imponerse una restricción mínima de especificidad/falsas semanas de alerta.",
        "7. Barranquilla es un fallo externo claro: AUC-PR 0,719, por debajo de su prevalencia 0,764 y de persistencia 0,845; ROC-AUC 0,353. Villavicencio sólo tuvo una semana negativa en prueba, por lo que su AUC-PR 1,000 no prueba discriminación estable.",
        "",
        "## MLflow",
        "",
        f"Estado de esta ejecución: `{mlflow_status.get('status')}`. {mlflow_status.get('message', '')}",
        "Los candidatos se registran como runs y sólo los ganadores se crean como versiones de modelo. El código no ejecuta transición de etapa ni asigna alias de ciclo de vida.",
        "",
        "## Archivos para auditoría",
        "",
        "- `results.csv`: ganadores y métricas de prueba.",
        "- `candidates_validation.csv`: todos los candidatos evaluados en selección.",
        "- `baselines_test.csv`: persistencia y climatología.",
        "- `models/<tarea>/<DIVIPOLA>/`: bundle, predicciones, importancias, folds y métricas.",
        "- `run_manifest.json`: huella del insumo, configuración y versiones.",
        "",
        "## Recomendación para el equipo",
        "",
        "Tomar `zone_future` a ocho semanas como resultado principal de investigación y `pre_onset` sólo como sensibilidad. **No promover todavía los modelos a despliegue**: primero exigir mejora frente a persistencia en AUC-PR y Brier, una especificidad mínima acordada, y una simulación prospectiva con disponibilidad real de SIVIGILA/clima. Después, congelar parámetros en Git/DVC y repetir la prueba sin reabrir 2023–2025.",
    ]
    report = "\n".join(lines) + "\n"
    (output_root / "REPORTE_FINAL.md").write_text(report, encoding="utf-8")
    return report


def run_pipeline(input_path: str | Path, config_path: str | Path, output_dir: str | Path, enable_mlflow: bool = True) -> dict[str, Any]:
    from .mlflow_io import publish_to_mlflow

    config = load_config(config_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    source = read_panel(input_path)
    panel, episodes, profile = prepare_supervised_panel(source, config)
    panel.to_csv(output_root / "prepared_supervised_panel.csv.gz", index=False, compression="gzip")
    episodes.to_csv(output_root / "episodes.csv", index=False)
    tasks = [config["label"]["primary_task"], *config["label"].get("secondary_tasks", [])]
    results: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for task in tasks:
        for code in config["pilots"]:
            code = str(code).zfill(5)
            try:
                result, local_candidates, local_baselines = train_one(panel, episodes, code, task, config, output_root)
            except RuntimeError as exc:
                skipped.append(
                    {
                        "task": task,
                        "municipality_code": code,
                        "municipality": config["pilots"][code],
                        "reason": str(exc),
                    }
                )
                continue
            results.append(result)
            candidates.extend(local_candidates)
            baselines.extend(local_baselines)
    results_df = pd.DataFrame(results)
    candidates_df = pd.DataFrame(candidates)
    baselines_df = pd.DataFrame(baselines)
    skipped_df = pd.DataFrame(skipped)
    results_df.to_csv(output_root / "results.csv", index=False)
    candidates_df.to_csv(output_root / "candidates_validation.csv", index=False)
    baselines_df.to_csv(output_root / "baselines_test.csv", index=False)
    skipped_df.to_csv(output_root / "skipped_models.csv", index=False)
    mlflow_status = publish_to_mlflow(output_root, config) if enable_mlflow else {"status": "disabled", "message": "Deshabilitado por argumento de ejecución."}
    (output_root / "mlflow_status.json").write_text(json.dumps(mlflow_status, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "input_path": str(Path(input_path).resolve()),
        "input_sha256": sha256(input_path),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256(config_path),
        "profile": profile,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "mlflow": mlflow_status,
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    build_report(results_df, candidates_df, baselines_df, skipped_df, profile, mlflow_status, output_root, config)
    return {"results": results, "profile": profile, "mlflow": mlflow_status, "output_dir": str(output_root)}
