import numpy as np
import pandas as pd
from scipy import stats

# 1. Cargar el panel procesado por DVC
df_all = pd.read_parquet(r"data\processed\master_weekly_screening.parquet")

# 2. Filtrar el municipio piloto: Villavicencio (Código 50001)
baq = (
    df_all[df_all["municipality_code"].astype(str).str.zfill(5) == "50001"]
    .copy()
    .sort_values("week_start")
    .reset_index(drop=True)
)

# Convertir tipos y extraer componentes de fecha
baq["week_start"] = pd.to_datetime(baq["week_start"])
baq["cases"] = baq["confirmed_records_screening"].fillna(0).astype(float)
baq["year"] = baq["week_start"].dt.year
# Semana epidemiológica continua y semana del año (1 a 53)
baq["epi_week"] = baq["week_start"].dt.isocalendar().week.astype(int)

# Excluir la última semana si está marcada como parcial según el diseño
baq_full = baq[baq["partial_week"] == False].copy().reset_index(drop=True)

# 3. Función matemática del Canal Endémico (Ecuación 1 y 2 con desplazamiento mínimo)
def compute_endemic_channel(df, n_years=5, epsilon=0.001):
    """Calcula EWT y Límite Superior (LS) mediante media geométrica móvil de n años previos."""
    ewt_series = pd.Series(index=df.index, dtype=float)
    ls_series = pd.Series(index=df.index, dtype=float)

    # Identificador de semana del año para cruces rápidos
    for idx, row in df.iterrows():
        curr_year = row["year"]
        curr_week = row["epi_week"]

        # Años históricos estrictamente previos: [curr_year - n_years, curr_year - 1]
        # Al ir hasta curr_year (sin incluirlo) garantizamos evaluar exactamente n-1 años previos.
        past_years = range(curr_year - n_years, curr_year)

        # Filtrar los casos de la misma semana en los n años anteriores (C_{w, y-j})
        historical_matches = df[
            (df["year"].isin(past_years)) & (df["epi_week"] == curr_week)
        ]["cases"].values

        # Requiere historial completo de los n años para estimar el umbral
        if len(historical_matches) == n_years:
            # Transformación logarítmica con desplazamiento mínimo epsilon (Umaña et al. 2026)
            log_vals = np.log(historical_matches + epsilon)
            mean_log = np.mean(log_vals)
            std_log = np.std(log_vals, ddof=1)

            # EWT: Media geométrica destransformada
            ewt = np.exp(mean_log) - epsilon

            # Límite Superior: Intervalo de confianza t-student al 95%
            t_crit = stats.t.ppf(0.975, df=n_years - 1)
            ls_log = mean_log + t_crit * (std_log / np.sqrt(n_years))
            ls = np.exp(ls_log) - epsilon

            ewt_series.loc[idx] = max(0.0, ewt)
            ls_series.loc[idx] = max(0.0, ls)

    return ewt_series, ls_series


# 4. Generar canales para n = 5 (recomendación empírica) y n = 7 (control histórico)
print("Calculando canales endémicos para Villavicencio...")
baq_full["EWT_5y"], baq_full["LS_5y"] = compute_endemic_channel(
    baq_full, n_years=5
)
baq_full["EWT_7y"], baq_full["LS_7y"] = compute_endemic_channel(
    baq_full, n_years=7
)

# 5. Construcción de etiquetas lógicas (Ecuación formal de exceso y racha)
# A_it = 1 si C_it >= EWT_it (cubre zona de alerta y de epidemia)
# Se usa np.where para manejar correctamente las primeras semanas (sin historial) donde EWT es NaN
baq_full["A_it"] = np.where(
    baq_full["EWT_5y"].isna(),
    np.nan,
    (baq_full["cases"] >= baq_full["EWT_5y"]).astype(int),
)

# y_it = A_it * A_{i, t-1} * (1 - A_{i, t-2}) -> Inicio sostenido de brote
baq_full["A_lag1"] = baq_full["A_it"].shift(1)
baq_full["A_lag2"] = baq_full["A_it"].shift(2)
baq_full["y_it"] = baq_full["A_it"] * baq_full["A_lag1"] * (1 - baq_full["A_lag2"])

# 6. Ingeniería de rezagos climáticos (Lags de 1 a 12 semanas)
climate_cols = ["temperature_c", "precipitation_mm", "relative_humidity_pct"]
for col in climate_cols:
    for lag in range(1, 13):
        baq_full[f"{col}_lag{lag}"] = baq_full[col].shift(lag)

# Rezagos epidemiológicos de casos observados
for lag in range(1, 13):
    baq_full[f"cases_lag{lag}"] = baq_full["cases"].shift(lag)

# 7. Filtrar periodo analítico válido (después del periodo de inicialización/burn-in)
# Con n=5 años (2007-2011), las predicciones válidas inician desde 2012
df_pilot_ready = baq_full[baq_full["EWT_5y"].notna()].dropna(
    subset=["y_it", "cases_lag12"]
)

# Guardar dataset procesado para modelado
output_path = r"data\processed\Villavicencio_pilot_labeled.parquet"
df_pilot_ready.to_parquet(output_path, index=False)

print(f"\n[OK] Pipeline finalizado exitosamente.")
print(f"Dimensiones finales: {df_pilot_ready.shape}")
print(f"Total semanas analizadas: {len(df_pilot_ready)}")
print(f"Casos totales registrados en el periodo: {df_pilot_ready['cases'].sum()}")
print(
    f"Inicios de brote (y_it = 1) detectados: {int(df_pilot_ready['y_it'].sum())}"
)
print(f"Archivo guardado en: {output_path}")