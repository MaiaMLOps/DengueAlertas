import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, auc, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 1. Cargar el dataset
path = r"data\processed\muy_alta_transmision_labeled.parquet"
df = pd.read_parquet(path)
df = df.sort_values("week_start").reset_index(drop=True)

# 2. Configurar el horizonte predictivo
horizonte_h = 4
print(f"\n--- ENTRENAMIENTO MACHINE LEARNING N2 (LightGBM) ---")
print(f"Horizonte predictivo (h): {horizonte_h} semanas")

df[f'target_h{horizonte_h}'] = df['y_it'].shift(-horizonte_h)
df_model = df.dropna(subset=[f'target_h{horizonte_h}']).reset_index(drop=True)

# 3. Selección de variables y FEATURE ENGINEERING (El secreto epidemiológico)
# Extraemos los rezagos normales
features_n2 = [col for col in df_model.columns if ('lag' in col) and ('A_lag' not in col)]

# INGENIERÍA DE CARACTERÍSTICAS: Le enseñamos al modelo a leer el Canal Endémico
# 1. El límite esperado actual de la ciudad
df_model['umbral_ewt_actual'] = df_model['EWT_5y']

# 2. El Índice Epidémico: ¿Qué tan cerca estamos del límite hoy (t)?
# Si esto es > 1, la ciudad ya cruzó la línea de alerta hoy.
df_model['indice_exceso_t'] = df_model['cases_lag1'] / (df_model['EWT_5y'] + 0.001)

# Agregamos las variables clave y la ciudad
df_model['municipality_code'] = df_model['municipality_code'].astype('category')
features_n2.extend(['municipality_code', 'umbral_ewt_actual', 'indice_exceso_t'])

print(f"Predictores N2 ({len(features_n2)} variables: Espacio + Clima + Índices Epidemiológicos)")

# 4. Partición Cronológica Estricta (Rolling-Origin Global)
split_idx = int(len(df_model) * 0.75)
train_df = df_model.iloc[:split_idx]
test_df = df_model.iloc[split_idx:]

X_train, y_train = train_df[features_n2], train_df[f'target_h{horizonte_h}']
X_test, y_test = test_df[features_n2], test_df[f'target_h{horizonte_h}']

# Calcular el ratio exacto de desbalance para calibrar probabilidades
ratio_desbalance = (len(y_train) - y_train.sum()) / y_train.sum()

# 5. Entrenamiento del Modelo Global LightGBM (Optimizado para AUC-PR)
model_n2 = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    scale_pos_weight=ratio_desbalance, # Mejor que 'balanced' para métricas PR
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

model_n2.fit(
    X_train, y_train,
    categorical_feature=['municipality_code']
)

# 6. Predicción Probabilística
y_prob = model_n2.predict_proba(X_test)[:, 1]
y_pred = model_n2.predict(X_test)

# 7. Evaluación (El momento de la verdad frente a N1)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
auc_pr = auc(recall, precision)
f1 = f1_score(y_test, y_pred)

print("\n--- RESULTADOS MODELO N2 (TEST) ---")
print(f"AUC-PR (Área bajo la curva Precisión-Cobertura): {auc_pr:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"(AUC-PR de la línea base N1 fue: 0.0082)")

print("\nReporte detallado de clasificación:")
print(classification_report(y_test, y_pred, target_names=["Control/Epidemia (0)", "Inicio Brote (1)"]))

# 8. Explicabilidad: Importancia de las variables (Feature Importance)
# Esto validará la relevancia del clima
importances = pd.DataFrame({
    'Variable': features_n2,
    'Importancia': model_n2.feature_importances_
}).sort_values(by='Importancia', ascending=False)

print("\nTop 10 Variables más importantes para predecir el brote:")
print(importances.head(10).to_string(index=False))