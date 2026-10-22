import argparse
from pathlib import Path
import warnings
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import auc, classification_report, confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

warnings.filterwarnings('ignore')

if __name__ == "__main__":
  # 1. Definición dinámica de rutas y argumentos para DVC / VS Code Play ▶️
  BASE_DIR = Path(__file__).resolve().parent
  default_data = (
      BASE_DIR
      / "data"
      / "processed"
      / "muy_alta_transmision_labeled.parquet"
  )

  parser = argparse.ArgumentParser(
      description=(
          "Entrenar XGBoost N2 Global (Panel 56 Municipios Muy Alta"
          " Transmision)"
      )
  )
  parser.add_argument(
      "--input_data",
      type=str,
      default=str(default_data),
      help="Ruta al dataset parquet consolidado",
  )
  parser.add_argument(
      "--horizonte",
      type=int,
      default=4,
      help="Horizonte predictivo (h semanas)",
  )
  args = parser.parse_args()

  path = args.input_data
  horizonte_h = args.horizonte

  print(f"\n=======================================================")
  print(f"--- ENTRENAMIENTO MACHINE LEARNING N2 (XGBoost Global) ---")
  print(f"Horizonte predictivo (h): {horizonte_h} semanas")
  print(f"Dataset de entrada: {path}")
  print(f"=======================================================")

  # 2. Iniciar el registro centralizado en MLflow
  with mlflow.start_run(run_name="XGBoost_N2_RollingOrigin"):
    mlflow.set_tag("status", "staging")
    mlflow.set_tag("ui_integration", "enabled")
    mlflow.set_tag("model_family", "N2_Gradient_Boosting_XGBoost")
    mlflow.set_tag("risk_stratum", "Muy Alta Transmision")
    mlflow.log_param("horizonte_semanas", horizonte_h)

    # 3. Cargar y ordenar temporalmente el panel
    df = pd.read_parquet(path)

    # Desplazamiento del target por municipio para no contaminar entre ciudades
    df[f"target_h{horizonte_h}"] = df.groupby("municipality_code")[
        "y_it"
    ].shift(-horizonte_h)
    df_model = df.dropna(subset=[f"target_h{horizonte_h}"]).copy()

    # Orden cronológico estricto para la validación Rolling-Origin
    df_model = df_model.sort_values("week_start").reset_index(drop=True)

    # 4. Feature Engineering e Inteligencia Epidemiológica
    features_n2 = [
        col
        for col in df_model.columns
        if ("lag" in col) and ("A_lag" not in col)
    ]

    # Indicadores del canal endémico
    df_model["umbral_ewt_actual"] = df_model["EWT_5y"]
    df_model["indice_exceso_t"] = df_model["cases_lag1"] / (
        df_model["EWT_5y"] + 0.001
    )

    # En XGBoost se requiere tipo category explícito con enable_categorical=True
    df_model["municipality_code"] = df_model["municipality_code"].astype(
        "category"
    )
    features_n2.extend(
        ["municipality_code", "umbral_ewt_actual", "indice_exceso_t"]
    )

    print(
        f"Predictores evaluados: {len(features_n2)} (Clima + Casos +"
        " Indicadores EWT + Espacio)"
    )

    X = df_model[features_n2]
    y = df_model[f"target_h{horizonte_h}"]

    # 5. Validación Cronológica Estricta (Rolling-Origin CV)
    tscv = TimeSeriesSplit(n_splits=3)
    mlflow.log_param("n_splits_cv", tscv.n_splits)

    auc_pr_scores = []
    f1_scores = []

    print(
        "\nAplicando validación Rolling-Origin (3 pliegues cronológicos en el"
        " panel global)..."
    )

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]

      # Penalización del desbalance de clases (Ceros vs Unos)
      suma_positivos = y_train.sum()
      ratio_desbalance = (len(y_train) - suma_positivos) / (
          suma_positivos + 1e-9
      )

      # Configuración del clasificador XGBoost con soporte categórico
      model_fold = xgb.XGBClassifier(
          n_estimators=300,
          learning_rate=0.03,
          max_depth=6,
          scale_pos_weight=ratio_desbalance,
          subsample=0.8,
          colsample_bytree=0.8,
          random_state=42,
          tree_method="hist",
          enable_categorical=True,
          eval_metric="logloss",
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

    # 6. Resultados Promedio Globales
    mean_auc_pr = np.mean(auc_pr_scores)
    mean_f1 = np.mean(f1_scores)

    mlflow.log_metric("mean_auc_pr", mean_auc_pr)
    mlflow.log_metric("mean_f1_score", mean_f1)

    print("\n--- RESULTADOS MODELO XGBOOST N2 (ROLLING-ORIGIN CV) ---")
    print(f"Mean AUC-PR: {mean_auc_pr:.4f}")
    print(f"Mean F1-Score: {mean_f1:.4f}")

    # 7. Evaluación Detallada en el último fold (Período OOT más reciente)
    X_test_last = X.iloc[test_index]
    y_test_last = y.iloc[test_index]
    y_pred_last = model_fold.predict(X_test_last)

    print("\nReporte detallado de clasificación (último fold de validación):")
    print(
        classification_report(
            y_test_last,
            y_pred_last,
            target_names=["Control/Epidemia (0)", "Inicio Brote (1)"],
        )
    )

    print("Matriz de Confusión (último fold):")
    conf_mat = confusion_matrix(y_test_last, y_pred_last)
    print(f"[{conf_mat[0][0]:>5}  {conf_mat[0][1]:>5}]")
    print(f"[{conf_mat[1][0]:>5}  {conf_mat[1][1]:>5}]")

    # 8. Entrenamiento del modelo final con el 100% de los datos y guardado en MLflow
    print(
        "\nEntrenando modelo final con el 100% del dataset para artefacto"
        " MLflow..."
    )
    ratio_final = (len(y) - y.sum()) / (y.sum() + 1e-9)
    model_final = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        scale_pos_weight=ratio_final,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model_final.fit(X, y)

    mlflow.xgboost.log_model(model_final, "modelo_xgb_final")

    # 9. Importancia de Variables (Feature Importance por ganancia o peso)
    importances = pd.DataFrame({
        "Variable": features_n2,
        "Importancia": model_final.feature_importances_,
    }).sort_values(by="Importancia", ascending=False)

    print("\nTop 10 Variables más importantes del modelo final XGBoost:")
    print(importances.head(10).to_string(index=False))