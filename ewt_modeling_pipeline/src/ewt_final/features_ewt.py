"""Construcción point-in-time de las 66 variables de dengue-ews v0.4 con canal EWT.

Versión de ``features.py`` donde la familia ``ch_`` se deriva del canal EWT
(:mod:`ewt_channel`: ``ch_ewt``, ``ch_central``, ``ch_gsd``, ``ch_interval``,
``ch_upper``, ``excess_flag_ewt``) y se eliminan las variables espaciales.
Las familias epi_, cal_, clim_ y demo_ se mantienen idénticas.

Cada fila es un municipio en una semana de emisión ``t``. Casos y clima se rezagan
una semana; los parámetros del canal en ``t`` solo usan los cinco años previos.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

COUNT = "confirmed_records_screening"
REPORTING_LAG_WEEKS = 1
CLIMATE_LAG_WEEKS = 1

EWT_SHIFT = 0.001   # mismo desplazamiento del EWT
GSD_FLOOR = 0.25    # piso de la desviación geométrica para el z (historias casi constantes)

EPI_FEATURES = [
    *(f"epi_logc_lag{lag}" for lag in (0, 1, 2, 3, 4, 8, 12, 26, 52)),
    *(f"epi_logc_mean{window}" for window in (4, 8, 12)),
    *(f"epi_logc_std{window}" for window in (4, 8, 12)),
    "epi_growth_1w",
    "epi_growth_4w",
    "epi_yoy_4w",
    "epi_active_frac12",
]

CHANNEL_FEATURES = [
    "ch_logratio_ewt_lag0",
    "ch_logratio_ewt_lag1",
    "ch_logratio_ewt_lag2",
    "ch_E_lag0",
    "ch_E_lag1",
    "ch_E_lag2",
    "ch_z_ewt_lag0",
    "ch_z_ewt_lag1",
    "ch_z_ewt_max4",
    "ch_ratio_ub_lag0",
    "ch_ratio_ub_lag1",
    "ch_episode_active",
    "ch_log_ewt_target",
    "ch_log_ub_target",
    "ch_gap_to_target_ub",
    "ch_log_gsd",
]

CALENDAR_FEATURES = ["cal_sin_woy", "cal_cos_woy"]

CLIMATE_SOURCES = ("temperature_c", "precipitation_mm", "relative_humidity_pct")
CLIMATE_FEATURES = [
    f"clim_{source}_{suffix}"
    for source in CLIMATE_SOURCES
    for suffix in (
        "lag0", "lag2", "lag4", "lag6", "lag8", "lag12",
        "mean4", "mean8", "anom4",
    )
]

DEMOGRAPHY_FEATURES = ["demo_log_population", "demo_log_density"]

ALL_FEATURES = (
    EPI_FEATURES
    + CHANNEL_FEATURES
    + CALENDAR_FEATURES
    + CLIMATE_FEATURES
    + DEMOGRAPHY_FEATURES
)

FEATURE_SETS = {
    "epi": EPI_FEATURES + CHANNEL_FEATURES + CALENDAR_FEATURES,
    "epi_clima": EPI_FEATURES + CHANNEL_FEATURES + CALENDAR_FEATURES + CLIMATE_FEATURES,
    "epi_clima_demo": EPI_FEATURES + CHANNEL_FEATURES + CALENDAR_FEATURES + CLIMATE_FEATURES + DEMOGRAPHY_FEATURES,
    "full": ALL_FEATURES,
}

FEATURE_FAMILY_COUNTS = {
    "epi": len(EPI_FEATURES),
    "ch": len(CHANNEL_FEATURES),
    "cal": len(CALENDAR_FEATURES),
    "clim": len(CLIMATE_FEATURES),
    "demo": len(DEMOGRAPHY_FEATURES),
}


def monday_ordinal(week_start: pd.Series) -> pd.Series:
    """Ordinal (desde 1) del lunes dentro de su año; igual que en ewt_channel."""
    year = week_start.dt.year
    jan1 = pd.to_datetime(year.astype(str) + "-01-01")
    first_monday = jan1 + pd.to_timedelta((7 - jan1.dt.weekday) % 7, unit="D")
    return ((week_start - first_monday).dt.days // 7 + 1).astype(int)


def _shifted_rolling(series: pd.Series, lag: int, window: int, operation: str,
                     min_periods: int | None = None) -> pd.Series:
    shifted = series.shift(lag)
    minimum = min_periods if min_periods is not None else max(2, window // 2)
    rolling = shifted.rolling(window, min_periods=minimum)
    return getattr(rolling, operation)()


def _channel_lookup(out: pd.DataFrame, target_ordinal: np.ndarray, column: str) -> pd.Series:
    """Valor del canal en (municipio, año de t, ordinal objetivo).

    El canal de un año solo usa los cinco años previos, así que para un objetivo
    dentro del mismo año ya es conocido en t. Si ``t+h`` cruza de año, su ordinal
    (1..h) se busca en el año en curso: el canal del año siguiente incorporaría
    semanas posteriores a ``t``.
    """
    key = pd.MultiIndex.from_arrays([out["municipality_code"], out["_year"], target_ordinal])
    table = out.set_index(["municipality_code", "_year", "_ordinal"])[column]
    if not table.index.is_unique:
        raise ValueError("Panel con (municipio, año, ordinal) duplicado")
    return pd.Series(table.reindex(key).to_numpy(), index=out.index)


def _add_horizon_helpers(out: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    """Precalcula canal/calendario objetivo por horizonte sin usar casos futuros."""
    for horizon in sorted({int(value) for value in horizons}):
        if horizon < 1:
            raise ValueError("Los horizontes deben ser positivos")
        target_date = out["week_start"] + pd.to_timedelta(horizon * 7, unit="D")
        target_ordinal = monday_ordinal(target_date).to_numpy()

        for source, stem in (("ch_ewt", "ewt"), ("ch_upper", "ub")):
            if source not in out:
                target_value = pd.Series(np.nan, index=out.index, dtype=float)
            else:
                target_value = _channel_lookup(out, target_ordinal, source)
            out[f"_h{horizon}_ch_log_{stem}_target"] = np.log1p(
                pd.to_numeric(target_value, errors="coerce").clip(lower=0)
            )

        week = ((target_date.dt.dayofyear - 1) // 7 + 1).astype(float)
        angle = 2.0 * np.pi * week / 52.18
        out[f"_h{horizon}_cal_sin_woy"] = np.sin(angle)
        out[f"_h{horizon}_cal_cos_woy"] = np.cos(angle)
    return out


def materialize_horizon_features(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Devuelve una copia con las cinco variables dependientes de ``h`` activas."""
    horizon = int(horizon)
    out = frame.copy()
    mapping = {
        "ch_log_ewt_target": f"_h{horizon}_ch_log_ewt_target",
        "ch_log_ub_target": f"_h{horizon}_ch_log_ub_target",
        "cal_sin_woy": f"_h{horizon}_cal_sin_woy",
        "cal_cos_woy": f"_h{horizon}_cal_cos_woy",
    }
    missing = [source for source in mapping.values() if source not in out]
    if missing:
        generic_missing = [target for target in mapping if target not in out]
        if generic_missing:
            raise ValueError(
                f"Faltan variables dependientes del horizonte {horizon}: {generic_missing}"
            )
        return out
    for target, source in mapping.items():
        out[target] = out[source]
    out["ch_gap_to_target_ub"] = out["epi_logc_lag0"] - out["ch_log_ub_target"]
    return out


def build_features(frame: pd.DataFrame, horizons: Iterable[int] = (1, 2, 4),
                   default_horizon: int = 4) -> pd.DataFrame:
    """Construye las 66 variables y auxiliares por horizonte (canal EWT, sin espaciales)."""
    required = {
        "municipality_code", "week_start", COUNT,
        "ch_ewt", "ch_upper", "ch_gsd", "excess_flag_ewt", "episode_active",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas de entrada: {sorted(missing)}")

    out = frame.copy()
    out["municipality_code"] = out["municipality_code"].astype(str).str.zfill(5)
    out["week_start"] = pd.to_datetime(out["week_start"])
    out = out.sort_values(["municipality_code", "week_start"]).reset_index(drop=True)
    out[COUNT] = pd.to_numeric(out[COUNT], errors="coerce")
    out["_year"] = out["week_start"].dt.year
    out["_ordinal"] = monday_ordinal(out["week_start"])
    out["_log_cases"] = np.log1p(out[COUNT].clip(lower=0))
    out["_excess_numeric"] = out["excess_flag_ewt"].astype("Float64").astype(float)
    grouped = out.groupby("municipality_code", sort=False)

    # Historia epidemiológica (19).
    for lag in (0, 1, 2, 3, 4, 8, 12, 26, 52):
        out[f"epi_logc_lag{lag}"] = grouped["_log_cases"].shift(REPORTING_LAG_WEEKS + lag)
    for window in (4, 8, 12):
        out[f"epi_logc_mean{window}"] = grouped["_log_cases"].transform(
            lambda series, w=window: _shifted_rolling(series, REPORTING_LAG_WEEKS, w, "mean")
        )
        out[f"epi_logc_std{window}"] = grouped["_log_cases"].transform(
            lambda series, w=window: _shifted_rolling(series, REPORTING_LAG_WEEKS, w, "std")
        )
    out["epi_growth_1w"] = out["epi_logc_lag0"] - out["epi_logc_lag1"]
    out["epi_growth_4w"] = grouped["_log_cases"].transform(
        lambda series: _shifted_rolling(series, 1, 4, "mean")
        - _shifted_rolling(series, 5, 4, "mean")
    )
    out["epi_yoy_4w"] = grouped["_log_cases"].transform(
        lambda series: _shifted_rolling(series, 1, 4, "mean")
        - _shifted_rolling(series, 53, 4, "mean")
    )
    out["epi_active_frac12"] = grouped[COUNT].transform(
        lambda series: series.gt(0).astype(float).shift(1).rolling(12, min_periods=6).mean()
    )

    # Canal EWT y detector (16).
    count = out[COUNT].clip(lower=0)
    ewt = pd.to_numeric(out["ch_ewt"], errors="coerce").clip(lower=0)
    upper = pd.to_numeric(out["ch_upper"], errors="coerce").clip(lower=0)
    gsd = pd.to_numeric(out["ch_gsd"], errors="coerce").clip(lower=0)
    out["_logratio_ewt"] = np.log((count + 1.0) / (ewt + 1.0))
    # z geométrico: (log(C+s) - mean(log(x+s))) / gsd; mean(log(x+s)) = log(EWT+s) por definición.
    out["_z_ewt"] = (np.log(count + EWT_SHIFT) - np.log(ewt + EWT_SHIFT)) / gsd.clip(lower=GSD_FLOOR)
    out["_ratio_ub"] = count / upper.clip(lower=1.0)
    grouped = out.groupby("municipality_code", sort=False)
    for lag in (0, 1, 2):
        out[f"ch_logratio_ewt_lag{lag}"] = grouped["_logratio_ewt"].shift(1 + lag)
        out[f"ch_E_lag{lag}"] = grouped["_excess_numeric"].shift(1 + lag)
    out["ch_z_ewt_lag0"] = grouped["_z_ewt"].shift(1)
    out["ch_z_ewt_lag1"] = grouped["_z_ewt"].shift(2)
    out["ch_z_ewt_max4"] = grouped["_z_ewt"].transform(
        lambda series: series.shift(1).rolling(4, min_periods=2).max()
    )
    out["ch_ratio_ub_lag0"] = grouped["_ratio_ub"].shift(1)
    out["ch_ratio_ub_lag1"] = grouped["_ratio_ub"].shift(2)
    out["ch_episode_active"] = grouped["episode_active"].shift(1).astype("boolean").fillna(False).astype(float)
    out["ch_log_gsd"] = np.log1p(gsd)

    # Calendario objetivo y canal objetivo dependientes del horizonte.
    horizons = tuple(sorted({int(value) for value in horizons} | {int(default_horizon)}))
    _add_horizon_helpers(out, horizons)

    # Clima (27): 9 transformaciones para cada una de tres variables.
    for source in CLIMATE_SOURCES:
        if source not in out:
            out[source] = np.nan
        out[source] = pd.to_numeric(out[source], errors="coerce")
        grouped = out.groupby("municipality_code", sort=False)
        for lag in (0, 2, 4, 6, 8, 12):
            out[f"clim_{source}_lag{lag}"] = grouped[source].shift(CLIMATE_LAG_WEEKS + lag)
        out[f"clim_{source}_mean4"] = grouped[source].transform(
            lambda series: series.shift(CLIMATE_LAG_WEEKS).rolling(4, min_periods=2).mean()
        )
        out[f"clim_{source}_mean8"] = grouped[source].transform(
            lambda series: series.shift(CLIMATE_LAG_WEEKS).rolling(8, min_periods=4).mean()
        )
        out[f"clim_{source}_anom4"] = grouped[source].transform(
            lambda series: (
                series.shift(CLIMATE_LAG_WEEKS).rolling(4, min_periods=2).mean()
                - series.shift(CLIMATE_LAG_WEEKS).rolling(52, min_periods=26).mean()
            )
        )

    # Demografía (2), con arrastre únicamente dentro del municipio.
    grouped = out.groupby("municipality_code", sort=False)
    for source in ("population", "population_density_km2"):
        if source not in out:
            out[source] = np.nan
        out[source] = grouped[source].ffill()
    out["demo_log_population"] = np.log(pd.to_numeric(out["population"], errors="coerce").clip(lower=1))
    out["demo_log_density"] = np.log1p(pd.to_numeric(out["population_density_km2"], errors="coerce").clip(lower=0))

    # Materialización por defecto para inspección, gráficos y compatibilidad.
    out = materialize_horizon_features(out, default_horizon)

    # Contrato explícito: si se altera el esquema, fallar en desarrollo/CI.
    if len(ALL_FEATURES) != 66 or FEATURE_FAMILY_COUNTS != {
        "epi": 19, "ch": 16, "cal": 2, "clim": 27, "demo": 2,
    }:
        raise AssertionError("El contrato de variables v0.4-EWT debe contener 66 columnas")
    missing_output = [column for column in ALL_FEATURES if column not in out]
    if missing_output:
        raise AssertionError(f"No se construyeron variables v0.4-EWT: {missing_output}")
    return out.drop(
        columns=["_log_cases", "_excess_numeric", "_logratio_ewt", "_z_ewt", "_ratio_ub",
                 "_year", "_ordinal"],
        errors="ignore",
    )


def usable_columns(frame: pd.DataFrame, feature_set: str) -> list[str]:
    """Filtra vacías/constantes dentro del tramo de ajuste, nunca fuera de él."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Conjunto de variables desconocido: {feature_set}")
    return [
        column for column in FEATURE_SETS[feature_set]
        if column in frame
        and frame[column].notna().any()
        and frame[column].nunique(dropna=True) > 1
    ]
