import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 1. Definición dinámica de rutas
BASE_DIR = Path(__file__).resolve().parent
input_path = BASE_DIR / "data" / "processed" / "master_weekly_screening.parquet"
output_path = BASE_DIR / "data" / "processed" / "barranquilla_labeled.parquet"

print("==============================================================")
print("1. Cargando panel maestro para BARRANQUILLA (DIVIPOLA: 08001)")
print("==============================================================")

if not input_path.exists():
  raise FileNotFoundError(f"No se encontró el archivo maestro en: {input_path}")

df_all = pd.read_parquet(input_path)

# Filtrar estrictamente Barranquilla (08001)
COD_BARRANQUILLA = "08001"
df_baq = df_all[
    df_all["municipality_code"].astype(str).str.zfill(5) == COD_BARRANQUILLA
].copy()

# Preparación temporal y control de semana parcial
df_baq["week_start"] = pd.to_datetime(df_baq["week_start"])
df_baq["cases"] = (
    df_baq["confirmed_records_screening"].fillna(0).astype(float)
)
df_baq["year"] = df_baq["week_start"].dt.year
df_baq["epi_week"] = df_baq["week_start"].dt.isocalendar().week.astype(int)

# Excluir semana parcial y ordenar cronológicamente
df_baq = (
    df_baq[df_baq["partial_week"] == False]
    .sort_values("week_start")
    .reset_index(drop=True)
)
print(f"Total semanas analíticas cargadas: {len(df_baq)}")

print(
    "\n2. Calculando Canal Endémico (n=5 años, epsilon=0.001) y Etiqueta"
    " y_it..."
)

n_years = 5
epsilon = 0.001
ewt_list = []

for idx, row in df_baq.iterrows():
  curr_year = row["year"]
  curr_week = row["epi_week"]

  # Tomar exactamente las mismas semanas de los 5 años anteriores
  past_years = range(curr_year - n_years, curr_year)
  historical_matches = df_baq[
      (df_baq["year"].isin(past_years)) & (df_baq["epi_week"] == curr_week)
  ]["cases"].values

  if len(historical_matches) == n_years:
    log_vals = np.log(historical_matches + epsilon)
    mean_log = np.mean(log_vals)
    ewt = np.exp(mean_log) - epsilon
    ewt_list.append(max(0.0, ewt))
  else:
    ewt_list.append(np.nan)

df_baq["EWT_5y"] = ewt_list

# Etiqueta A_it (semana sobre el umbral)
df_baq["A_it"] = np.where(
    df_baq["EWT_5y"].isna(),
    np.nan,
    (df_baq["cases"] >= df_baq["EWT_5y"]).astype(int),
)

# Etiqueta y_it: 2 semanas consecutivas sobre el EWT tras 1 semana bajo control
df_baq["A_lag1"] = df_baq["A_it"].shift(1)
df_baq["A_lag2"] = df_baq["A_it"].shift(2)
df_baq["y_it"] = df_baq["A_it"] * df_baq["A_lag1"] * (1 - df_baq["A_lag2"])

# 3. Construcción de rezagos (1 a 12 semanas)
climate_cols = ["temperature_c", "precipitation_mm", "relative_humidity_pct"]
for col in climate_cols:
  for lag in range(1, 13):
    df_baq[f"{col}_lag{lag}"] = df_baq[col].shift(lag)

for lag in range(1, 13):
  df_baq[f"cases_lag{lag}"] = df_baq["cases"].shift(lag)

# 4. Limpieza de semanas de inicialización (burn-in de 5 años + 12 semanas de rezagos)
df_baq = df_baq.dropna(subset=["y_it", "cases_lag12"]).reset_index(drop=True)

# 5. Guardado del dataset piloto
output_path.parent.mkdir(parents=True, exist_ok=True)
df_baq.to_parquet(output_path, index=False)

print("\n[OK] Preprocesamiento de Barranquilla finalizado.")
print(f"Semanas útiles tras periodo de calentamiento: {len(df_baq)}")
print(f"Total de Inicios de Brote reales detectados: {int(df_baq['y_it'].sum())}")
print(f"Dataset guardado en: {output_path}")