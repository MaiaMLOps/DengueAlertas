import argparse
from pathlib import Path
import warnings
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent if script_dir.name == "experimentos_municipios" else script_dir
    while not (repo_root / ".git").exists() and repo_root.parent != repo_root:
        repo_root = repo_root.parent

    mlflow.set_tracking_uri(f"sqlite:///{repo_root / 'mlflow.db'}")
    default_data = repo_root / "data" / "processed" / "epidemiology" / "pilot_weekly_labels.parquet"
    if not default_data.exists():
        default_data = repo_root / "pilot_weekly_labels.parquet"

    parser = argparse.ArgumentParser(description="Baseline N1 por Municipio (Regresión Logística)")
    parser.add_argument("--input_data", type=str, default=str(default_data))
    parser.add_argument("--municipality_code", type=str, default="54001")
    parser.add_argument("--horizonte", type=int, default=4)
    parser.add_argument("--target_type", type=str, default="exceso_simple", choices=["exceso_simple", "racha_brote"])
    args = parser.parse_args()

    muni_code = args.municipality_code
    h = args.horizonte
    target_type = args.target_type

    with mlflow.start_run(run_name=f"BASELINE_{muni_code}_{target_type}_h{h}"):
        mlflow.log_params({
            "municipality_code": muni_code,
            "model_type": "logistic_regression",
            "horizonte_h": h,
            "target_type": target_type,
            "model_family": "N1_Autoregressive_Baseline",
        })

        df = pd.read_parquet(args.input_data)
        df_muni = df[df["municipality_code"] == muni_code].sort_values("week_start").reset_index(drop=True)
        
        if target_type == "exceso_simple":
            df_muni["y_eval"] = df_muni["excess_label"]
        else:
            A = df_muni["excess_label"]
            df_muni["y_eval"] = ((A == 1) & ((A.shift(1) == 1) | (A.shift(-1) == 1))).astype(int)

        df_muni[f"target_h{h}"] = df_muni["y_eval"].shift(-h)
        df_muni[f"target_eligible_h{h}"] = df_muni["label_eligible"].shift(-h)

        for lag in range(1, 13):
            df_muni[f"cases_lag{lag}"] = df_muni["confirmed_records_screening"].shift(lag)

        df_model = df_muni.dropna(subset=[f"target_h{h}", "cases_lag12"]).copy()
        df_model = df_model[df_model[f"target_eligible_h{h}"] == True].reset_index(drop=True)

        features = [col for col in df_model.columns if col.startswith("cases_lag")]
        X = df_model[features]
        y = df_model[f"target_h{h}"].astype(int)

        tscv = TimeSeriesSplit(n_splits=3)
        auc_pr_scores, f1_scores = [], []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
            ])
            pipeline.fit(X_train, y_train)

            y_prob = pipeline.predict_proba(X_test)[:, 1]
            y_pred = pipeline.predict(X_test)

            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            auc_pr_scores.append(auc(recall, precision))
            f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

        mean_auc_pr = float(np.mean(auc_pr_scores))
        mean_f1 = float(np.mean(f1_scores))
        
        mlflow.log_metric("mean_auc_pr", mean_auc_pr)
        mlflow.log_metric("mean_f1_score", mean_f1)

        oot_prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        oot_rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        conf_mat = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = conf_mat.ravel() if conf_mat.size == 4 else (0, 0, 0, 0)

        print(f"METRICAS_SUMMARY|{mean_auc_pr:.4f}|{mean_f1:.4f}|{oot_prec:.4f}|{oot_rec:.4f}|{tn}|{fp}|{fn}|{tp}")
        print("TOP_FEATURES|N/A (Línea Base)")

        pipeline.fit(X, y)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path=f"baseline_{muni_code}_{target_type}",
            skops_trusted_types=[
                "sklearn.pipeline.Pipeline",
                "sklearn.preprocessing._data.StandardScaler",
                "sklearn.linear_model._logistic.LogisticRegression",
            ],
        )