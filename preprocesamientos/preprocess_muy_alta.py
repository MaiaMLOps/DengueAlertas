import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("1. Cargando panel maestro de 78 municipios...")
# Carga directa del archivo de DVC
df_all = pd.read_parquet(r"data\processed\master_weekly_screening.parquet")

# -------------------------------------------------------------------------
# FILTRO ESTRATÉGICO: Solo Municipios de MUY ALTA TRANSMISIÓN (57 municipios)
# -------------------------------------------------------------------------
muy_alta_divipola = [
    "05001", "05088", "05147", "05154", "05360", "05579", # Antioquia
    "08001", "08078", "08296", "08433", "08638", "08758", # Atlántico
    "76001", "76109", "76147", "76318", "76520", "76834", "76892", # Valle del Cauca
    "68001", "68081", "68276", "68307", "68547", "68679", # Santander
    "54001", "54261", "54405", "54498", "54810", "54874", # Norte de Santander
    "41001", "41132", "41298", "41396", "41551", # Huila
    "50001", "50006", "50313", "50573", # Meta
    "73001", "73268", "73349", "73449", # Tolima
    "13001", "13244", "13430", # Bolívar
    "20001", "20011", "20710", # Cesar
    "66001", "66170", # Risaralda
    "85001", "18753", "95001", "86001", "63001" # Otros (Yopal, San Vicente, etc)
]

# Aplicamos el filtro y preparamos fechas
df_all = df_all[df_all["municipality_code"].astype(str).str.zfill(5).isin(muy_alta_divipola)].copy()
print(f"Filtro aplicado. Evaluando {df_all['municipality_code'].nunique()} municipios de Muy Alta Transmisión.")

df_all["week_start"] = pd.to_datetime(df_all["week_start"])
df_all["cases"] = df_all["confirmed_records_screening"].fillna(0).astype(float)
df_all["year"] = df_all["week_start"].dt.year
df_all["epi_week"] = df_all["week_start"].dt.isocalendar().week.astype(int)

# Excluir semana parcial y ordenar para la lógica secuencial
df_all = df_all[df_all["partial_week"] == False].sort_values(["municipality_code", "week_start"]).reset_index(drop=True)

print("\n2. Calculando Canal Endémico y Etiquetas (n=5, epsilon=0.001) agrupado por municipio...")

def process_municipality(df):
    n_years = 5
    epsilon = 0.001
    ewt_list = []
    
    for idx, row in df.iterrows():
        curr_year = row["year"]
        curr_week = row["epi_week"]
        
        # Filtro de n-1 años históricos
        past_years = range(curr_year - n_years, curr_year)
        historical_matches = df[(df["year"].isin(past_years)) & (df["epi_week"] == curr_week)]["cases"].values
        
        if len(historical_matches) == n_years:
            log_vals = np.log(historical_matches + epsilon)
            mean_log = np.mean(log_vals)
            ewt = np.exp(mean_log) - epsilon
            ewt_list.append(max(0.0, ewt))
        else:
            ewt_list.append(np.nan)
            
    df["EWT_5y"] = ewt_list
    
    # Etiqueta predictiva: 2 semanas de exceso tras 1 semana de control
    df["A_it"] = np.where(df["EWT_5y"].isna(), np.nan, (df["cases"] >= df["EWT_5y"]).astype(int))
    df["A_lag1"] = df["A_it"].shift(1)
    df["A_lag2"] = df["A_it"].shift(2)
    df["y_it"] = df["A_it"] * df["A_lag1"] * (1 - df["A_lag2"])
    
    # Construcción de Rezagos (Lags) Climáticos y Epidemiológicos (1 a 12 semanas)
    climate_cols = ["temperature_c", "precipitation_mm", "relative_humidity_pct"]
    for col in climate_cols:
        for lag in range(1, 13):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
            
    for lag in range(1, 13):
        df[f"cases_lag{lag}"] = df["cases"].shift(lag)
        
    return df

# Aplicamos la función a cada municipio por separado y reensamblamos
df_model = df_all.groupby("municipality_code", group_keys=False).apply(process_municipality)

# Limpieza final
df_model = df_model.dropna(subset=["y_it", "cases_lag12"]).reset_index(drop=True)

# -------------------------------------------------------------------------
# GUARDADO DE DATOS (NUEVA RUTA)
# -------------------------------------------------------------------------
output_path = r"data\processed\muy_alta_transmision_labeled.parquet"
df_model.to_parquet(output_path, index=False)

print(f"\n[OK] Pipeline finalizado exitosamente.")
print(f"Total de registros globales procesados: {len(df_model)}")
print(f"Total de Inicios de Brote reales detectados: {int(df_model['y_it'].sum())}")
print(f"Dataset consolidado guardado en: {output_path}")