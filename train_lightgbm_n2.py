import argparse
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, auc, f1_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import mlflow
import mlflow.lightgbm

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # 1. Configuración de argumentos para DVC / Orquestación
    parser = argparse.ArgumentParser(
        description="Entrenar Linea Base N1 (Regresión Logística Autorregresiva)"
    )
    parser.add_argument(
        "--input_data",
        type=str,
        default=r"data\processed\muy_alta_transmision_labeled.parquet",  # <--- Ruta por defecto
        help="Ruta al dataset parquet preprocesado",
    )
    parser.add_argument(
        "--horizonte",
        type=int,
        default=4,  # <--- Horizonte por defecto
        help="Horizonte predictivo en semanas",
    )
    args = parser.parse_args()

    # Asignar los argumentos a variables utilizables
    path = args.input_data
    horizonte_h = args.horizonte

    print(f"\n=======================================================")
    print(f"--- ENTRENAMIENTO LÍNEA BASE N1 (Logistic Regression) ---")
    print(f"Horizonte predictivo (h): {horizonte_h} semanas")
    print(f"Dataset de entrada: {path}")
    print(f"=======================================================")

    # 2. Iniciar el registro centralizado en MLflow
    with mlflow.start_run(run_name="LightGBM_N2_RollingOrigin"):
        mlflow.set_tag("status", "staging")
        mlflow.set_tag("ui_integration", "enabled")
        mlflow.set_tag("risk_stratum", "Muy Alta Transmision")
        mlflow.log_param("horizonte_semanas", horizonte_h)

        # 3. Carga y preparación del dataset
        df = pd.read_parquet(path)
        df[f'target_h{horizonte_h}'] = df['y_it'].shift(-horizonte_h)
        df_model = df.dropna(subset=[f'target_h{horizonte_h}']).reset_index(drop=True)

        # 4. Feature Engineering (Inteligencia Epidemiológica)
        features_n2 = [col for col in df_model.columns if ('lag' in col) and ('A_lag' not in col)]
        df_model['umbral_ewt_actual'] = df_model['EWT_5y']
        df_model['indice_exceso_t'] = df_model['cases_lag1'] / (df_model['EWT_5y'] + 0.001)
        
        df_model['municipality_code'] = df_model['municipality_code'].astype('category')
        features_n2.extend(['municipality_code', 'umbral_ewt_actual', 'indice_exceso_t'])
        
        print(f"Predictores N2: {len(features_n2)} variables (Espacio + Clima + Índices Epidemiológicos)")

        # 5. Validación Cronológica Estricta (Rolling-Origin)
        print("\nAplicando validación Rolling-Origin (TimeSeriesSplit)...")
        X = df_model[features_n2]
        y = df_model[f'target_h{horizonte_h}']

        tscv = TimeSeriesSplit(n_splits=3)
        mlflow.log_param("n_splits_cv", tscv.n_splits)

        auc_pr_scores = []
        f1_scores = []

        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Control de división por cero en desbalance
            suma_positivos = y_train.sum()
            ratio_desbalance = (len(y_train) - suma_positivos) / (suma_positivos + 1e-9)
            
            model_fold = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=6,
                scale_pos_weight=ratio_desbalance,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            
            model_fold.fit(X_train, y_train, categorical_feature=['municipality_code'])
            
            y_prob = model_fold.predict_proba(X_test)[:, 1]
            y_pred = model_fold.predict(X_test)
            
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            fold_auc_pr = auc(recall, precision)
            fold_f1 = f1_score(y_test, y_pred)
            
            auc_pr_scores.append(fold_auc_pr)
            f1_scores.append(fold_f1)
            
            mlflow.log_metric(f"fold_{fold}_auc_pr", fold_auc_pr)
            mlflow.log_metric(f"fold_{fold}_f1_score", fold_f1)

        # 6. Resultados Finales y Modelo Definitivo
        mean_auc_pr = np.mean(auc_pr_scores)
        mean_f1 = np.mean(f1_scores)
        
        mlflow.log_metric("mean_auc_pr", mean_auc_pr)
        mlflow.log_metric("mean_f1_score", mean_f1)

        print("\n--- RESULTADOS MODELO N2 (ROLLING-ORIGIN CV) ---")
        print(f"Mean AUC-PR: {mean_auc_pr:.4f}")
        print(f"Mean F1-Score: {mean_f1:.4f}")

        # Entrenamos un modelo final con el 100% de los datos para guardarlo como el artefacto de producción
        print("\nEntrenando modelo final con el 100% del dataset para artefacto MLflow...")
        ratio_final = (len(y) - y.sum()) / (y.sum() + 1e-9)
        model_final = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=6,
            scale_pos_weight=ratio_final, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, n_jobs=-1
        )
        model_final.fit(X, y, categorical_feature=['municipality_code'])
        
        # Opcional: Para imprimir la matriz de confusión, predecimos sobre el último fold (test_index)
        y_pred_final = model_fold.predict(X.iloc[test_index])
        y_test_final = y.iloc[test_index]
        
        from sklearn.metrics import confusion_matrix
        print("\nReporte detallado de clasificación (último fold de validación):")
        print(classification_report(y_test_final, y_pred_final, target_names=["Control/Epidemia (0)", "Inicio Brote (1)"]))
        
        print("\nMatriz de Confusión (último fold):")
        conf_mat = confusion_matrix(y_test_final, y_pred_final)
        print(f"[{conf_mat[0][0]}  {conf_mat[0][1]}]")
        print(f"[{conf_mat[1][0]}   {conf_mat[1][1]}]")

        mlflow.lightgbm.log_model(model_final, "modelo_lgbm_final")

        importances = pd.DataFrame({
            'Variable': features_n2,
            'Importancia': model_final.feature_importances_
        }).sort_values(by='Importancia', ascending=False)

        print("\nTop 10 Variables más importantes del modelo final:")
        print(importances.head(10).to_string(index=False))