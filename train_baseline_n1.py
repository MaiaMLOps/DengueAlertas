import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 1. Cargar el dataset etiquetado de Villavicencio (generado en el paso anterior)
# Asegúrate de ejecutar este script en la raíz de tu proyecto DVC
path = r"data\processed\villavicencio_pilot_labeled.parquet"
try:
    df = pd.read_parquet(path)
    print("Dataset cargado correctamente.")
except FileNotFoundError:
    print("Error: No se encontró el dataset. Debes correr primero preprocess_pilot_barranquilla.py")
    exit()

# Asegurar orden temporal
df = df.sort_values("week_start").reset_index(drop=True)

# 2. Configuración del experimento
horizonte_h = 4  # Predicción a 4 semanas (1 mes)
print(f"\n--- ENTRENAMIENTO LÍNEA BASE N1 (Regresión Logística Autorregresiva) ---")
print(f"Horizonte predictivo (h): {horizonte_h} semanas")

# Desplazar la etiqueta objetivo hacia atrás para alinear la semana actual (t) con la predicción (t+h)
df[f'target_h{horizonte_h}'] = df['y_it'].shift(-horizonte_h)

# Eliminar las últimas 'h' semanas porque no conocemos su futuro
df_model = df.dropna(subset=[f'target_h{horizonte_h}']).reset_index(drop=True)

# 3. Selección estricta de variables para N1 (Solo información histórica de casos)
# Aislamos el modelo N1: NO SE USA CLIMA, solo los casos observados en semanas previas (lags 1 a 12)
features_n1 = [col for col in df_model.columns if col.startswith('cases_lag')]
print(f"Predictores N1 ({len(features_n1)} variables):", features_n1)

# 4. Partición Cronológica Estricta (Rolling-Origin / Sin Fuga de Datos)
# Entrenamos con el 75% inicial de la serie y evaluamos en el 25% final (Ej. 2019-2025)
split_idx = int(len(df_model) * 0.75)

train_df = df_model.iloc[:split_idx]
test_df = df_model.iloc[split_idx:]

X_train = train_df[features_n1]
y_train = train_df[f'target_h{horizonte_h}']

X_test = test_df[features_n1]
y_test = test_df[f'target_h{horizonte_h}']

print(f"\nPeríodo de Entrenamiento: {train_df['week_start'].min().date()} a {train_df['week_start'].max().date()} ({len(train_df)} semanas)")
print(f"Período de Prueba (OOT): {test_df['week_start'].min().date()} a {test_df['week_start'].max().date()} ({len(test_df)} semanas)")

# 5. Inicialización y Entrenamiento del Modelo Estadístico Base
# Usamos class_weight='balanced' para penalizar el desbalance de ceros (no brotes) vs unos (brotes)
model_n1 = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model_n1.fit(X_train, y_train)

# 6. Predicción Probabilística y Binaria
# y_prob es la probabilidad P(y_it=1 | casos_pasados)
y_prob = model_n1.predict_proba(X_test)[:, 1]
y_pred = model_n1.predict(X_test)

# 7. Evaluación de Clasificación (AUC-PR y F1-Score)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
auc_pr = auc(recall, precision)
f1 = f1_score(y_test, y_pred)

print("\n--- RESULTADOS LÍNEA BASE N1 (TEST) ---")
print(f"AUC-PR (Área bajo la curva Precisión-Cobertura): {auc_pr:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nReporte detallado de clasificación:")
print(classification_report(y_test, y_pred, target_names=["Control/Epidemia (0)", "Inicio Brote (1)"]))

# 8. Análisis de coeficientes (Opcional, para explicabilidad)
coeficientes = pd.DataFrame({
    'Lag_Epidemiológico': features_n1,
    'Peso (Coeficiente)': model_n1.coef_[0]
}).sort_values(by='Peso (Coeficiente)', ascending=False)

print("\nPesos de la regresión (Impacto de semanas pasadas):")
print(coeficientes.head())