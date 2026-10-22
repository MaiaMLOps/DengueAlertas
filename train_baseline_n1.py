import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, auc, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import mlflow
import mlflow.sklearn

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
    with mlflow.start_run(run_name="Baseline_N1_LogisticRegression"):
        mlflow.set_tag("status", "staging")
        mlflow.set_tag("ui_integration", "enabled")
        mlflow.set_tag("model_family", "N1_Autoregressive_Baseline")
        mlflow.set_tag("risk_stratum", "Muy Alta Transmision")
        mlflow.log_param("horizonte_semanas", horizonte_h)

        # 3. Carga y alineación temporal del panel
        df = pd.read_parquet(path)
        
        # Desplazamiento del target agrupado por municipio para no contaminar entre ciudades
        df[f'target_h{horizonte_h}'] = df.groupby('municipality_code')['y_it'].shift(-horizonte_h)
        df_model = df.dropna(subset=[f'target_h{horizonte_h}']).copy()

        # Orden cronológico estricto para que Rolling-Origin entrene en pasado y evalúe en futuro
        df_model = df_model.sort_values("week_start").reset_index(drop=True)

        # 4. Selección estricta de variables para N1 (Solo historial de casos, sin clima ni espacio)
        features_n1 = [col for col in df_model.columns if col.startswith('cases_lag')]
        print(f"Predictores N1: {len(features_n1)} variables autorregresivas ({features_n1[0]} a {features_n1[-1]})")

        X = df_model[features_n1]
        y = df_model[f'target_h{horizonte_h}']

        # 5. Validación Cronológica Estricta (Rolling-Origin CV)
        print("\nAplicando validación Rolling-Origin (TimeSeriesSplit)...")
        tscv = TimeSeriesSplit(n_splits=3)
        mlflow.log_param("n_splits_cv", tscv.n_splits)

        auc_pr_scores = []
        f1_scores = []

        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Pipeline con StandardScaler para convergencia estable del modelo lineal
            pipeline_fold = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
            ])

            pipeline_fold.fit(X_train, y_train)

            y_prob = pipeline_fold.predict_proba(X_test)[:, 1]
            y_pred = pipeline_fold.predict(X_test)

            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            fold_auc_pr = auc(recall, precision)
            fold_f1 = f1_score(y_test, y_pred)

            auc_pr_scores.append(fold_auc_pr)
            f1_scores.append(fold_f1)

            mlflow.log_metric(f"fold_{fold}_auc_pr", fold_auc_pr)
            mlflow.log_metric(f"fold_{fold}_f1_score", fold_f1)

        # 6. Resultados Promedio Globales
        mean_auc_pr = np.mean(auc_pr_scores)
        mean_f1 = np.mean(f1_scores)

        mlflow.log_metric("mean_auc_pr", mean_auc_pr)
        mlflow.log_metric("mean_f1_score", mean_f1)

        print("\n--- RESULTADOS LÍNEA BASE N1 (ROLLING-ORIGIN CV) ---")
        print(f"Mean AUC-PR: {mean_auc_pr:.4f}")
        print(f"Mean F1-Score: {mean_f1:.4f}")

        # 7. Evaluación Detallada en el último fold (Período OOT más reciente)
        y_test_final = y.iloc[test_index]
        y_pred_final = pipeline_fold.predict(X.iloc[test_index])

        print("\nReporte detallado de clasificación (último fold de validación):")
        print(classification_report(y_test_final, y_pred_final, target_names=["Control/Epidemia (0)", "Inicio Brote (1)"]))

        print("Matriz de Confusión (último fold):")
        conf_mat = confusion_matrix(y_test_final, y_pred_final)
        print(f"[{conf_mat[0][0]:>5}  {conf_mat[0][1]:>5}]")
        print(f"[{conf_mat[1][0]:>5}  {conf_mat[1][1]:>5}]")

        # 8. Entrenamiento del modelo final con el 100% de los datos y guardado en MLflow
        print("\nEntrenando modelo final con el 100% del dataset para artefacto MLflow...")
        pipeline_final = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ])
        pipeline_final.fit(X, y)

        mlflow.sklearn.log_model(pipeline_final, "modelo_lr_n1_final")

        # 9. Coeficientes del modelo lineal (Impacto de cada rezago de casos)
        coeficientes = pd.DataFrame({
            'Rezagos_Casos': features_n1,
            'Peso_Coeficiente': pipeline_final.named_steps['clf'].coef_[0]
        }).sort_values(by='Peso_Coeficiente', ascending=False)

        print("\nPesos de la Regresión Logística (Impacto por rezago):")
        print(coeficientes.to_string(index=False))