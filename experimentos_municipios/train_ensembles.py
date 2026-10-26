import argparse
from pathlib import Path
import warnings
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

def get_model(model_name: str, ratio_desbalance: float):
    if model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=5, scale_pos_weight=ratio_desbalance, random_state=42, verbose=-1, n_jobs=-1)
    elif model_name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=5, scale_pos_weight=ratio_desbalance, random_state=42, n_jobs=-1, eval_metric="logloss")
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)
    elif model_name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=200, learning_rate=0.03, depth=5, scale_pos_weight=ratio_desbalance, random_seed=42, verbose=0, thread_count=-1)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

def log_model_artifact(model, model_type: str, artifact_name: str):
    if model_type == "lightgbm":
        import mlflow.lightgbm
        mlflow.lightgbm.log_model(model, artifact_path=artifact_name)
    elif model_type == "xgboost":
        import mlflow.xgboost
        mlflow.xgboost.log_model(model, artifact_path=artifact_name)
    elif model_type == "catboost":
        import mlflow.catboost
        mlflow.catboost.log_model(model, artifact_path=artifact_name)
    else:
        import mlflow.sklearn
        mlflow.sklearn.log_model(model, artifact_path=artifact_name, skops_trusted_types=["sklearn.tree._tree.Tree"])

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent if script_dir.name == "experimentos_municipios" else script_dir
    while not (repo_root / ".git").exists() and repo_root.parent != repo_root:
        repo_root = repo_root.parent

    mlflow.set_tracking_uri(f"sqlite:///{repo_root / 'mlflow.db'}")
    default_data = repo_root / "data" / "processed" / "epidemiology" / "pilot_weekly_labels.parquet"
    if not default_data.exists():
        default_data = repo_root / "pilot_weekly_labels.parquet"

    parser = argparse.ArgumentParser(description="Modelos de Ensambles N2")
    parser.add_argument("--input_data", type=str, default=str(default_data))
    parser.add_argument("--municipality_code", type=str, default="54001")
    parser.add_argument("--model", type=str, default="lightgbm", choices=["lightgbm", "xgboost", "random_forest", "catboost"])
    parser.add_argument("--horizonte", type=int, default=4)
    parser.add_argument("--target_type", type=str, default="exceso_simple", choices=["exceso_simple", "racha_brote", "inicio_racha_masking"])
    args = parser.parse_args()

    muni_code = args.municipality_code
    model_type = args.model
    h = args.horizonte
    target_type = args.target_type

    with mlflow.start_run(run_name=f"{model_type.upper()}_{muni_code}_{target_type}_h{h}"):
        df = pd.read_parquet(args.input_data)
        df_muni = df[df["municipality_code"] == muni_code].sort_values("week_start").reset_index(drop=True)

        # LÓGICA UNIFICADA: SOPORTA TARGETS NORMALES Y ENMASCARADOS
        if target_type == "exceso_simple":
            df_muni["y_eval"] = df_muni["excess_label"]
            df_muni["brote_activo"] = 0
        elif target_type == "racha_brote":
            A = df_muni["excess_label"]
            df_muni["y_eval"] = ((A == 1) & ((A.shift(1) == 1) | (A.shift(-1) == 1))).astype(int)
            df_muni["brote_activo"] = 0
        elif target_type == "inicio_racha_masking":
            A = df_muni["excess_label"]
            A_sostenido = ((A == 1) & ((A.shift(1) == 1) | (A.shift(-1) == 1))).astype(int)
            df_muni["y_eval"] = ((A_sostenido == 1) & (A_sostenido.shift(1) == 0)).astype(int)
            df_muni["brote_activo"] = A_sostenido

        df_muni[f"target_h{h}"] = df_muni["y_eval"].shift(-h)
        df_muni[f"activo_h{h}"] = df_muni["brote_activo"].shift(-h)
        df_muni[f"target_eligible_h{h}"] = df_muni["label_eligible"].shift(-h)

        feature_cols = []
        variables_base = ["confirmed_records_screening", "precipitation_mm", "temperature_c", "relative_humidity_pct"]
        for lag in range(1, 13):
            for col in variables_base:
                if col in df_muni.columns:
                    lag_name = f"{col}_lag{lag}"
                    df_muni[lag_name] = df_muni[col].shift(lag)
                    feature_cols.append(lag_name)

        if "ewt" in df_muni.columns:
            df_muni["ewt_t"] = df_muni["ewt"]
            df_muni["ratio_casos_ewt"] = df_muni["confirmed_records_screening"].shift(1) / (df_muni["ewt"] + 0.001)
            feature_cols.extend(["ewt_t", "ratio_casos_ewt"])

        if "population_density_km2" in df_muni.columns:
            feature_cols.append("population_density_km2")

        subset_cols = [f"target_h{h}", f"activo_h{h}", "confirmed_records_screening_lag12"]
        if "ewt_t" in df_muni.columns:
            subset_cols.append("ewt_t")

        df_model = df_muni.dropna(subset=subset_cols).copy()
        df_model = df_model[df_model[f"target_eligible_h{h}"] == True].reset_index(drop=True)

        # CREACIÓN DE LA MÁSCARA
        mask = (df_model[f"target_h{h}"] == 0) & (df_model[f"activo_h{h}"] == 1)

        X = df_model[feature_cols]
        y = df_model[f"target_h{h}"].astype(int)

        tscv = TimeSeriesSplit(n_splits=3)
        auc_pr_scores, f1_scores = [], []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            valid_train_mask = ~mask.iloc[train_idx]
            X_train = X.iloc[train_idx][valid_train_mask]
            y_train = y.iloc[train_idx][valid_train_mask]

            valid_test_mask = ~mask.iloc[test_idx]
            X_test = X.iloc[test_idx][valid_test_mask]
            y_test = y.iloc[test_idx][valid_test_mask]
            
            if y_train.sum() == 0 or len(y_train) == 0:
                continue

            ratio_desbalance = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)
            clf = get_model(model_type, ratio_desbalance)
            clf.fit(X_train, y_train)

            y_prob = clf.predict_proba(X_test)[:, 1]
            y_pred = clf.predict(X_test)

            if y_test.sum() > 0:
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                auc_pr_scores.append(auc(recall, precision))
            else:
                auc_pr_scores.append(0.0)
                
            f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

        mean_auc_pr = float(np.mean(auc_pr_scores)) if auc_pr_scores else 0.0
        mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

        mlflow.log_metric("mean_auc_pr", mean_auc_pr)
        mlflow.log_metric("mean_f1_score", mean_f1)

        oot_prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        oot_rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        conf_mat = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = conf_mat.ravel() if conf_mat.size == 4 else (0, 0, 0, 0)

        print(f"METRICAS_SUMMARY|{mean_auc_pr:.4f}|{mean_f1:.4f}|{oot_prec:.4f}|{oot_rec:.4f}|{tn}|{fp}|{fn}|{tp}")

        valid_global_mask = ~mask
        X_global = X[valid_global_mask]
        y_global = y[valid_global_mask]

        ratio_final = (len(y_global) - y_global.sum()) / (y_global.sum() + 1e-9)
        final_model = get_model(model_type, ratio_final)
        final_model.fit(X_global, y_global)
        
        # --- EXTRACCIÓN DE TOP FEATURES ---
        if hasattr(final_model, "feature_importances_"):
            importances = final_model.feature_importances_
        elif hasattr(final_model, "get_feature_importance"):
            importances = final_model.get_feature_importance()
        else:
            importances = None

        if importances is not None:
            top_features = pd.DataFrame({"Variable": feature_cols, "Importancia": importances}).sort_values("Importancia", ascending=False)
            top3 = top_features.head(3)
            feats_str = " ; ".join([f"{row['Variable']} ({row['Importancia']:.4f})" for _, row in top3.iterrows()])
            print(f"TOP_FEATURES|{feats_str}")
        else:
            print("TOP_FEATURES|N/A")

        log_model_artifact(final_model, model_type, f"{model_type}_{muni_code}_{target_type}")