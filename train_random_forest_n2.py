import argparse
from pathlib import Path
import warnings
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

if __name__ == "__main__":
  # 1. Detección dinámica de la ruta del dataset para ejecución con botón Run ▶️
  current_dir = Path(__file__).resolve().parent

  if (current_dir / "muy_alta_transmision_labeled.parquet").exists():
    default_data = current_dir / "muy_alta_transmision_labeled.parquet"
  elif (
      current_dir / "data" / "processed" / "muy_alta_transmision_labeled.parquet"
  ).exists():
    default_data = (
        current_dir
        / "data"
        / "processed"
        / "muy_alta_transmision_labeled.parquet"
    )
  else:
    default_data = (
        current_dir.parents[1]
        / "data"
        / "processed"
        / "muy_alta_transmision_labeled.parquet"
    )

  parser = argparse.ArgumentParser(
      description=(
          "Entrenar Random Forest N2 Global (Panel 56 Municipios Muy Alta"
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

  print(f"\n===========================================================")
  print(f"--- ENTRENAMIENTO MACHINE LEARNING N2 (Random Forest) ---")
  print(f"Horizonte predictivo (h): {horizonte_h} semanas")
  print(f"Dataset de entrada: {path}")
  print(f"===========================================================")

  # 2. Iniciar el registro centralizado en MLflow
  with mlflow.start_run(run_name="RandomForest_N2_RollingOrigin"):
    mlflow.set_tag("status", "staging")
    mlflow.set_tag("ui_integration", "enabled")
    mlflow.set_tag("model_family", "N2_Bagging_Random_Forest")
    mlflow.set_tag("risk_stratum", "Muy Alta Transmision")
    mlflow.log_param("horizonte_semanas", horizonte_h)

    # 3. Cargar y ordenar temporalmente el panel
    df = pd.read_parquet(path)

    # Desplazamiento del target por municipio para no cruzar series entre ciudades
    df[f"target_h{horizonte_h}"] = df.groupby("municipality_code")[
        "y_it"
    ].shift(-horizonte_h)
    df_model = df.dropna(subset=[f"target_h{horizonte_h}"]).copy()

    # Orden cronológico estricto para validación temporal Rolling-Origin
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

    # Scikit-Learn requiere valores numéricos para RandomForestClassifier
    # Convertimos el código municipal a entero o índice categórico
    df_model["municipality_code_encoded"] = df_model[
        "municipality_code"
    ].astype("category").cat.codes
    features_n2.extend(
        ["municipality_code_encoded", "umbral_ewt_actual", "indice_exceso_t"]
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

      # Random Forest con balanceo de pesos para penalizar ceros vs unos
      model_fold = RandomForestClassifier(
          n_estimators=200,
          max_depth=12,
          min_samples_split=5,
          class_weight="balanced",
          random_state=42,
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

    print("\n--- RESULTADOS MODELO RANDOM FOREST N2 (ROLLING-ORIGIN CV) ---")
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
    model_final = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model_final.fit(X, y)

    # Guardado en MLflow usando cloudpickle (evita el bloqueo de skops)
    mlflow.sklearn.log_model(
        sk_model=model_final, 
        artifact_path="modelo_rf_final", 
        serialization_format="cloudpickle"
    )

    # 9. Importancia de Variables (Gini Impurity)
    importances = pd.DataFrame({
        "Variable": features_n2,
        "Importancia": model_final.feature_importances_,
    }).sort_values(by="Importancia", ascending=False)

    print("\nTop 10 Variables más importantes del modelo final Random Forest:")
    print(importances.head(10).to_string(index=False))