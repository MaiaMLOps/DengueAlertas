import argparse
from pathlib import Path
import warnings
from catboost import CatBoostClassifier
import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
from sklearn.metrics import auc, classification_report, confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # 1. Obtener la raíz del repositorio (sube un nivel desde modelos_enmascarados/)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent  # Llega a DengueAlertas/

    # 2. Conectar a MLflow en la base de datos de la raíz
    mlflow.set_tracking_uri(f"sqlite:///{repo_root / 'mlflow.db'}")

    # 3. Ruta dinámica al dataset (buscando en data/processed/)
    default_data = repo_root / "data" / "processed" / "muy_alta_transmision_labeled.parquet"
    if not default_data.exists():
        # Fallback si estuviera en epidemiology o en la raíz
        if (repo_root / "data" / "processed" / "epidemiology" / "muy_alta_transmision_labeled.parquet").exists():
            default_data = repo_root / "data" / "processed" / "epidemiology" / "muy_alta_transmision_labeled.parquet"
        elif (repo_root / "muy_alta_transmision_labeled.parquet").exists():
            default_data = repo_root / "muy_alta_transmision_labeled.parquet"
        else:
            default_data = script_dir / "muy_alta_transmision_labeled.parquet"

    parser = argparse.ArgumentParser(description="CatBoost N2 con Masking")
    parser.add_argument("--input_data", type=str, default=str(default_data))
    parser.add_argument("--horizonte", type=int, default=4)
    args = parser.parse_args()

    horizonte_h = args.horizonte

    with mlflow.start_run(run_name="CatBoost_N2_Masking"):
        df = pd.read_parquet(args.input_data)
        
        df[f"target_h{horizonte_h}"] = df.groupby("municipality_code")["y_it"].shift(-horizonte_h)
        df[f"exceso_futuro_h{horizonte_h}"] = df.groupby("municipality_code")["A_it"].shift(-horizonte_h)

        df_model = df.dropna(subset=[f"target_h{horizonte_h}", f"exceso_futuro_h{horizonte_h}"]).copy()
        df_model = df_model.sort_values("week_start").reset_index(drop=True)

        mask = (df_model[f"target_h{horizonte_h}"] == 0) & (df_model[f"exceso_futuro_h{horizonte_h}"] == 1)
        
        features_n2 = [col for col in df_model.columns if ("lag" in col) and ("A_lag" not in col)]
        df_model["umbral_ewt_actual"] = df_model["EWT_5y"]
        df_model["indice_exceso_t"] = df_model["cases_lag1"] / (df_model["EWT_5y"] + 0.001)
        
        # En CatBoost la dejamos como string y le pasamos el nombre a cat_features
        df_model["municipality_code"] = df_model["municipality_code"].astype(str)
        features_n2.extend(["municipality_code", "umbral_ewt_actual", "indice_exceso_t"])

        X = df_model[features_n2]
        y = df_model[f"target_h{horizonte_h}"]

        tscv = TimeSeriesSplit(n_splits=3)
        auc_pr_scores, f1_scores = [], []

        print("\n=======================================================")
        print(f"--- ENTRENAMIENTO CATBOOST N2 CON MASKING ---")
        print(f"Horizonte predictivo (h): {horizonte_h} semanas")
        print(f"Dataset: {args.input_data}")
        print(f"=======================================================")

        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            train_idx, test_idx = df_model.iloc[train_index].index, df_model.iloc[test_index].index

            valid_train_mask = ~mask.iloc[train_idx]
            X_train, y_train = X.iloc[train_idx][valid_train_mask], y.iloc[train_idx][valid_train_mask]

            valid_test_mask = ~mask.iloc[test_idx]
            X_test, y_test = X.iloc[test_idx][valid_test_mask], y.iloc[test_idx][valid_test_mask]

            ratio_desbalance = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)

            model_fold = CatBoostClassifier(
                iterations=200, learning_rate=0.03, depth=5,
                scale_pos_weight=ratio_desbalance, cat_features=["municipality_code"],
                random_seed=42, verbose=0, thread_count=-1
            )
            model_fold.fit(X_train, y_train)

            y_prob = model_fold.predict_proba(X_test)[:, 1]
            y_pred = model_fold.predict(X_test)

            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            auc_pr_scores.append(auc(recall, precision))
            f1_scores.append(f1_score(y_test, y_pred))

        mlflow.log_metric("mean_auc_pr", np.mean(auc_pr_scores))
        mlflow.log_metric("mean_f1_score", np.mean(f1_scores))

        print("\n--- RESULTADOS CATBOOST CON MASKING (ROLLING-ORIGIN CV) ---")
        print(f"Mean AUC-PR: {np.mean(auc_pr_scores):.4f}")
        print(f"Mean F1-Score: {np.mean(f1_scores):.4f}")

        # Reporte Final (OOT enmascarado)
        print("\nReporte detallado de clasificación (OOT sin semanas epidémicas activas):")
        print(classification_report(y_test, y_pred, target_names=["Control (0)", "Inicio Brote (1)"]))
        
        conf_mat = confusion_matrix(y_test, y_pred)
        print("Matriz de Confusión (Enmascarada):")
        print(f"[{conf_mat[0][0]:>5}  {conf_mat[0][1]:>5}]")
        print(f"[{conf_mat[1][0]:>5}  {conf_mat[1][1]:>5}]")
        
        valid_global_mask = ~mask
        X_global, y_global = X[valid_global_mask], y[valid_global_mask]
        ratio_final = (len(y_global) - y_global.sum()) / (y_global.sum() + 1e-9)
        
        model_final = CatBoostClassifier(
            iterations=200, learning_rate=0.03, depth=5, scale_pos_weight=ratio_final,
            cat_features=["municipality_code"], random_seed=42, verbose=0, thread_count=-1
        )
        model_final.fit(X_global, y_global)
        mlflow.catboost.log_model(model_final, name="modelo_cb_masking")

        importances = pd.DataFrame({
            "Variable": features_n2,
            "Importancia": model_final.get_feature_importance(),
        }).sort_values(by="Importancia", ascending=False)

        print("\nTop 10 Variables (Modelo con Masking):")
        print(importances.head(10).to_string(index=False))