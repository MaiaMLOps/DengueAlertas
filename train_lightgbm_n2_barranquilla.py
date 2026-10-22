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
  BASE_DIR = Path(__file__).resolve().parent
  default_data = (
      BASE_DIR / "data" / "processed" / "barranquilla_labeled.parquet"
  )

  # Configuración de argumentos con defaults para ejecución con botón Run ▶️
  parser = argparse.ArgumentParser(
      description="Entrenar LightGBM N2 Piloto Barranquilla"
  )
  parser.add_argument(
      "--input_data",
      type=str,
      default=str(default_data),
      help="Ruta al dataset de Barranquilla",
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
  print(f"--- ENTRENAMIENTO LIGHTGBM N2 (Piloto Barranquilla) ---")
  print(f"Horizonte predictivo (h): {horizonte_h} semanas")
  print(f"Dataset: {path}")
  print(f"=======================================================")

  with mlflow.start_run(run_name="LightGBM_N2_Barranquilla_Pilot"):
    mlflow.set_tag("status", "pilot_test")
    mlflow.set_tag("municipality", "Barranquilla (08001)")
    mlflow.log_param("horizonte_semanas", horizonte_h)

    # 1. Cargar y alinear target
    df = pd.read_parquet(path)
    df[f"target_h{horizonte_h}"] = df["y_it"].shift(-horizonte_h)
    df_model = df.dropna(subset=[f"target_h{horizonte_h}"]).reset_index(
        drop=True
    )

    # 2. Selección de predictores e Ingeniería Epidemiológica
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
    features_n2.extend(["umbral_ewt_actual", "indice_exceso_t"])

    print(
        f"Predictores evaluados: {len(features_n2)} (Clima + Casos +"
        " Indicadores EWT)"
    )

    X = df_model[features_n2]
    y = df_model[f"target_h{horizonte_h}"]

    # 3. Validación Cronológica Estricta (Rolling-Origin CV)
    tscv = TimeSeriesSplit(n_splits=3)
    mlflow.log_param("n_splits_cv", tscv.n_splits)

    auc_pr_scores = []
    f1_scores = []

    print("\nAplicando validación Rolling-Origin (3 pliegues cronológicos)...")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]

      # Ajuste de penalización por desbalance local
      suma_positivos = y_train.sum()
      ratio_desbalance = (len(y_train) - suma_positivos) / (
          suma_positivos + 1e-9
      )

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

    print("\n--- RESULTADOS MODELO N2 EN BARRANQUILLA (ROLLING-ORIGIN CV) ---")
    print(f"Mean AUC-PR: {mean_auc_pr:.4f}")
    print(f"Mean F1-Score: {mean_f1:.4f}")

    # 4. Evaluación en el último fold (Período más reciente)
    X_test_last = X.iloc[test_index]
    y_test_last = y.iloc[test_index]
    y_pred_last = model_fold.predict(X_test_last)

    print("\nReporte detallado de clasificación (último período OOT):")
    print(
        classification_report(
            y_test_last,
            y_pred_last,
            target_names=["Control/Epidemia (0)", "Inicio Brote (1)"],
        )
    )

    print("Matriz de Confusión (último período):")
    conf_mat = confusion_matrix(y_test_last, y_pred_last)
    print(f"[{conf_mat[0][0]:>4}  {conf_mat[0][1]:>4}]")
    print(f"[{conf_mat[1][0]:>4}  {conf_mat[1][1]:>4}]")

    # 5. Modelo final con 100% de la serie de Barranquilla
    ratio_final = (len(y) - y.sum()) / (y.sum() + 1e-9)
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
    model_final.fit(X, y)

    mlflow.lightgbm.log_model(model_final, "modelo_lgbm_barranquilla_final")

    importances = pd.DataFrame({
        "Variable": features_n2,
        "Importancia": model_final.feature_importances_,
    }).sort_values(by="Importancia", ascending=False)

    print("\nTop 10 Variables más importantes para Barranquilla:")
    print(importances.head(10).to_string(index=False))