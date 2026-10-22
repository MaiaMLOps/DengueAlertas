import argparse
from pathlib import Path
import warnings
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
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

    parser = argparse.ArgumentParser(description="Entrenar LightGBM N2 con Masking Epidemiológico")
    parser.add_argument("--input_data", type=str, default=str(default_data))
    parser.add_argument("--horizonte", type=int, default=4)
    args = parser.parse_args()

    path = args.input_data
    horizonte_h = args.horizonte

    print(f"\n=======================================================")
    print(f"--- ENTRENAMIENTO LIGHTGBM N2 CON MASKING ---")
    print(f"Horizonte predictivo (h): {horizonte_h} semanas")
    print(f"Dataset: {path}")
    print(f"=======================================================")

    # 2. Iniciar MLflow
    with mlflow.start_run(run_name="LightGBM_N2_Masking"):
        mlflow.log_param("horizonte_semanas", horizonte_h)

        # 3. Carga y Preparación de Datos
        df = pd.read_parquet(path)
        
        # Desplazamiento del target original de inicio de brote
        df[f"target_h{horizonte_h}"] = df.groupby("municipality_code")["y_it"].shift(-horizonte_h)
        
        # NUEVO: Desplazamiento de la bandera de Exceso Sostenido para Masking
        # Identificamos las semanas que son de epidemia ACTIVA (ya pasó el inicio).
        # Esto ocurre cuando A_it = 1 y A_it-1 = 1, pero NO es un inicio nuevo.
        # Una forma robusta es identificar las rachas. 
        # Si A_it está disponible, la semana activa se define cuando el target es 0, pero A_it es 1.
        df[f"exceso_futuro_h{horizonte_h}"] = df.groupby("municipality_code")["A_it"].shift(-horizonte_h)

        df_model = df.dropna(subset=[f"target_h{horizonte_h}", f"exceso_futuro_h{horizonte_h}"]).copy()
        df_model = df_model.sort_values("week_start").reset_index(drop=True)

        # 4. Creación de la Máscara de Entrenamiento y Evaluación
        # Queremos IGNORAR (enmascarar) las semanas donde el brote ya está activo pero no es inicio.
        # Condición de enmascaramiento: Target de inicio es 0, PERO el municipio está en exceso (A_it = 1).
        mask = (df_model[f"target_h{horizonte_h}"] == 0) & (df_model[f"exceso_futuro_h{horizonte_h}"] == 1)
        
        # Seleccionamos las predictores
        features_n2 = [col for col in df_model.columns if ("lag" in col) and ("A_lag" not in col)]
        df_model["umbral_ewt_actual"] = df_model["EWT_5y"]
        df_model["indice_exceso_t"] = df_model["cases_lag1"] / (df_model["EWT_5y"] + 0.001)
        
        # Soporte Categórico explícito (importante para LightGBM)
        df_model["municipality_code"] = df_model["municipality_code"].astype("category")
        features_n2.extend(["municipality_code", "umbral_ewt_actual", "indice_exceso_t"])

        X = df_model[features_n2]
        y = df_model[f"target_h{horizonte_h}"]

        # 5. Validación Cronológica Estricta (Rolling-Origin CV)
        tscv = TimeSeriesSplit(n_splits=3)
        
        auc_pr_scores = []
        f1_scores = []

        print("\nAplicando validación Rolling-Origin con Enmascaramiento de Período Activo...")

        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            # Extraemos los índices
            train_idx = df_model.iloc[train_index].index
            test_idx = df_model.iloc[test_index].index

            # APLICAMOS LA MÁSCARA AL CONJUNTO DE ENTRENAMIENTO
            # Filtramos para quedarnos solo con las filas que NO están enmascaradas
            valid_train_mask = ~mask.iloc[train_idx]
            X_train = X.iloc[train_idx][valid_train_mask]
            y_train = y.iloc[train_idx][valid_train_mask]

            # APLICAMOS LA MÁSCARA AL CONJUNTO DE PRUEBA
            # Evaluaremos el modelo solo en las semanas de control o en los inicios reales.
            valid_test_mask = ~mask.iloc[test_idx]
            X_test = X.iloc[test_idx][valid_test_mask]
            y_test = y.iloc[test_idx][valid_test_mask]

            # Penalización por desbalance local (con los datos limpios)
            suma_positivos = y_train.sum()
            ratio_desbalance = (len(y_train) - suma_positivos) / (suma_positivos + 1e-9)

            model_fold = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=5,
                scale_pos_weight=ratio_desbalance,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )

            model_fold.fit(X_train, y_train)

            y_prob = model_fold.predict_proba(X_test)[:, 1]
            y_pred = model_fold.predict(X_test)

            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            fold_auc_pr = auc(recall, precision)
            fold_f1 = f1_score(y_test, y_pred)

            auc_pr_scores.append(fold_auc_pr)
            f1_scores.append(fold_f1)

            mlflow.log_metric(f"fold_{fold}_auc_pr", fold_auc_pr)
            mlflow.log_metric(f"fold_{fold}_f1_score", fold_f1)

        mean_auc_pr = np.mean(auc_pr_scores)
        mean_f1 = np.mean(f1_scores)

        mlflow.log_metric("mean_auc_pr", mean_auc_pr)
        mlflow.log_metric("mean_f1_score", mean_f1)

        print("\n--- RESULTADOS LIGHTGBM CON MASKING (ROLLING-ORIGIN CV) ---")
        print(f"Mean AUC-PR: {mean_auc_pr:.4f}")
        print(f"Mean F1-Score: {mean_f1:.4f}")

        # 6. Reporte Final (Último Fold)
        print("\nReporte detallado de clasificación (OOT sin semanas epidémicas activas):")
        print(classification_report(y_test, y_pred, target_names=["Control (0)", "Inicio Brote (1)"]))
        
        conf_mat = confusion_matrix(y_test, y_pred)
        print("Matriz de Confusión (Enmascarada):")
        print(f"[{conf_mat[0][0]:>5}  {conf_mat[0][1]:>5}]")
        print(f"[{conf_mat[1][0]:>5}  {conf_mat[1][1]:>5}]")

        # 7. Entrenamiento del modelo final aplicando la máscara global
        valid_global_mask = ~mask
        X_global = X[valid_global_mask]
        y_global = y[valid_global_mask]

        ratio_final = (len(y_global) - y_global.sum()) / (y_global.sum() + 1e-9)
        model_final = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=5,
            scale_pos_weight=ratio_final,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        model_final.fit(X_global, y_global)

        mlflow.lightgbm.log_model(model_final, "modelo_lgbm_masking")

        importances = pd.DataFrame({
            "Variable": features_n2,
            "Importancia": model_final.feature_importances_,
        }).sort_values(by="Importancia", ascending=False)

        print("\nTop 10 Variables (Modelo con Masking):")
        print(importances.head(10).to_string(index=False))