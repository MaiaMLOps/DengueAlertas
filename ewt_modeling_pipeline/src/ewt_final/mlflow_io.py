"""Publicación opcional en MLflow sin transición de etapa ni alias."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os

import joblib
import numpy as np
import pandas as pd


def _numeric_metrics(row: dict[str, Any], prefixes: tuple[str, ...] = ("",)) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in row.items():
        if not any(key.startswith(prefix) for prefix in prefixes):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
            output[str(key)[:250]] = float(value)
    return output


def publish_to_mlflow(output_root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("mlflow", {})
    if not cfg.get("enabled", True):
        return {"status": "disabled", "message": "MLflow está deshabilitado en configuración."}
    if cfg.get("transition_stage") is not None or cfg.get("set_alias") is not None:
        raise ValueError("No se permiten etapa ni alias en este aporte")
    try:
        import mlflow
        import mlflow.pyfunc
    except ImportError:
        return {
            "status": "skipped_dependency",
            "message": "No se publicó porque MLflow no está instalado; ejecute pip install -r requirements.txt.",
        }
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.get("tracking_uri", ""))
    if tracking_uri.startswith("http") and not (
        os.getenv("CF_ACCESS_CLIENT_ID") and os.getenv("CF_ACCESS_CLIENT_SECRET")
    ):
        return {
            "status": "skipped_credentials",
            "tracking_uri": tracking_uri,
            "message": "No se publicó: faltan CF_ACCESS_CLIENT_ID/CF_ACCESS_CLIENT_SECRET en el entorno.",
        }
    from .mlflow_auth import configure_cloudflare_access

    configure_cloudflare_access()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", cfg.get("experiment", "dengue-ews-ewt-h8-final"))
    mlflow.set_experiment(experiment)
    root = Path(output_root)
    candidates = pd.read_csv(root / "candidates_validation.csv")
    published_candidates = 0
    registered_winners: list[dict[str, Any]] = []
    try:
        for row in candidates[candidates["status"].eq("evaluated")].to_dict("records"):
            name = f"candidate-{row['task']}-{row['municipality_code']}-{row['algorithm']}-{row['feature_set']}"
            with mlflow.start_run(run_name=name):
                mlflow.set_tags(
                    {
                        "lifecycle": "candidate",
                        "task": row["task"],
                        "municipality_code": str(row["municipality_code"]).zfill(5),
                        "methodology_version": config["methodology_version"],
                        "stage_policy": "none",
                    }
                )
                mlflow.log_params(
                    {
                        "algorithm": row["algorithm"],
                        "feature_set": row["feature_set"],
                        "horizon_weeks": int(config["label"]["horizon"]),
                        "purge_weeks": int(config["label"]["horizon"]),
                        "calibration": config["modeling"].get("calibrate", "platt_oof"),
                    }
                )
                mlflow.log_metrics(_numeric_metrics(row))
                published_candidates += 1

        results = pd.read_csv(root / "results.csv")
        for row in results.to_dict("records"):
            code = str(row["municipality_code"]).split(".")[0].zfill(5)
            task = row["task"]
            model_dir = root / "models" / task / code
            bundle = joblib.load(model_dir / "model.joblib")
            registered_name = f"{cfg.get('registered_model_prefix', 'DengueAlertas_EWT')}_{task}_{code}_h{int(row['horizon'])}"

            class RuntimeModel(mlflow.pyfunc.PythonModel):
                def __init__(self, local_bundle):
                    self.local_bundle = local_bundle

                def predict(self, context, model_input, params=None):
                    return self.local_bundle.predict(pd.DataFrame(model_input))

            with mlflow.start_run(run_name=f"winner-{task}-{code}") as run:
                mlflow.set_tags(
                    {
                        "lifecycle": "selected_winner",
                        "task": task,
                        "municipality_code": code,
                        "methodology_version": config["methodology_version"],
                        "stage_policy": "none",
                    }
                )
                mlflow.log_params(
                    {
                        "algorithm": row["winner_algorithm"],
                        "feature_set": row["winner_feature_set"],
                        "horizon_weeks": int(row["horizon"]),
                        "n_features": int(row["n_features"]),
                        "threshold": float(row["threshold"]),
                    }
                )
                mlflow.log_metrics(_numeric_metrics(row, prefixes=("test_", "validation_")))
                for artifact in (
                    "metrics.json",
                    "candidates_validation.csv",
                    "predictions_validation.csv",
                    "predictions_test.csv",
                    "feature_importance.csv",
                    "fold_plan.json",
                ):
                    if (model_dir / artifact).exists():
                        mlflow.log_artifact(str(model_dir / artifact), artifact_path="evidence")
                model_info = mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=RuntimeModel(bundle),
                    registered_model_name=registered_name if cfg.get("register_winners", True) else None,
                    code_paths=[str(Path(__file__).resolve().parents[1])],
                    pip_requirements=[
                        "pandas>=2,<3",
                        "numpy>=1.24,<3",
                        "scikit-learn>=1.3,<2",
                        "lightgbm>=4,<5",
                        "xgboost>=2,<4",
                        "joblib>=1.3,<2",
                    ],
                )
                registered_winners.append(
                    {
                        "task": task,
                        "municipality_code": code,
                        "run_id": run.info.run_id,
                        "registered_model_name": registered_name,
                        "model_uri": model_info.model_uri,
                        "stage": "None",
                    }
                )
    except Exception as exc:
        return {
            "status": "failed",
            "tracking_uri": tracking_uri,
            "experiment": experiment,
            "published_candidates": published_candidates,
            "registered_winners": registered_winners,
            "message": f"Publicación parcial/fallida: {type(exc).__name__}: {exc}",
        }
    return {
        "status": "published",
        "tracking_uri": tracking_uri,
        "experiment": experiment,
        "published_candidates": published_candidates,
        "registered_winners": registered_winners,
        "message": "Runs y ganadores publicados sin etapa ni alias.",
    }
