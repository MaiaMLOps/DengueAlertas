"""Canal EWT (media geométrica con desplazamiento 0,001) para dengue-ews v0.4.

Complementa a ``endemic_channel.py`` (NB-GLM) con el mismo contrato de entrada:
columnas ``municipality_code``, ``week_start`` y ``confirmed_records_screening``.

Reglas (resumen):
  * Cada semana pertenece al año de su lunes ``week_start``.
  * La semana ordinal ``w`` es el número de lunes del año, empezando en 1
    (no es ISO ni se parten semanas entre años).
  * Referencia: exactamente los años y-5..y-1, con todos sus conteos
    (incluidos años epidémicos). Si w=53 y el año histórico no tiene lunes 53,
    se usa su ordinal 52; ninguna otra sustitución.
  * EWT = exp(mean(log(C+0.001))) - 0.001, evaluado como
    0.001 * expm1(mean(log1p(C/0.001))).
    - Historia constante -> devuelve exactamente ese conteo.
    - Media exactamente entera -> identidad entera prod(1000*C+1) = (1000*k+1)^5.
    - Sin tolerancias ni umbrales epidemiológicos mínimos.
    - Una anomalía numérica detiene el cálculo.
  * Etiqueta = 1 si conteo > EWT; igualdad = 0.
  * 2007-2011 inicializan; se etiqueta desde 2012 con cinco referencias válidas.
  * Semana parcial 2025-12-29: se conserva sin etiqueta.
  * Semana no elegible: ``ch_ewt`` y ``excess_flag_ewt`` nulos, con motivo en
    ``ewt_status``. Huecos estructurales del panel: se rechazan (ValueError).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

COUNT = "confirmed_records_screening"
OFFSET = 1#0.001
ROUND_DIGITS = 5
SCALE = 1000  # 1 / OFFSET, entero para la identidad exacta

STATUS_LABELED = "etiquetada"
STATUS_INIT = "inicializacion"
STATUS_PARTIAL = "semana_parcial"
STATUS_NO_HISTORY = "sin_historia"


@dataclass(frozen=True)
class EwtConfig:
    window_years: int = 5
    first_label_year: int = 2012
    partial_week_starts: tuple[str, ...] = ("2025-12-29",)
    # qt(alpha; n-1) en valor absoluto: 0.05 -> t unilateral 95 % (2.132 con n=5);
    # use 0.025 si se quiere el cuantil de un IC bilateral del 95 % (2.776).
    alpha: float = 0.05


# --------------------------------------------------------------------------- #
# Calendario por lunes
# --------------------------------------------------------------------------- #
def monday_ordinal(week_start: pd.Series) -> pd.Series:
    """Ordinal (desde 1) del lunes dentro de su año calendario."""
    year = week_start.dt.year
    jan1 = pd.to_datetime(year.astype(str) + "-01-01")
    first_monday = jan1 + pd.to_timedelta((7 - jan1.dt.weekday) % 7, unit="D")
    return ((week_start - first_monday).dt.days // 7 + 1).astype(int)


def mondays_in_year(year: int) -> int:
    """Número de lunes (52 o 53) del año calendario."""
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    offset = (7 - jan1.weekday()) % 7
    span = (pd.Timestamp(year=year, month=12, day=31) - jan1).days
    return (span - offset) // 7 + 1


# --------------------------------------------------------------------------- #
# EWT exacto
# --------------------------------------------------------------------------- #
def _iroot(n: int, k: int) -> int:
    """Raíz k-ésima entera (piso) por Newton, sin aritmética flotante."""
    if n < 2:
        return n
    x = 1 << ((n.bit_length() + k - 1) // k)
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            return x
        x = y


@lru_cache(maxsize=None)
def ewt(counts: tuple[int, ...]) -> float:
    """EWT de una historia de conteos enteros no negativos."""
    n = len(counts)
    if n == 0:
        raise ValueError("EWT requiere al menos un conteo")
    first = counts[0]
    if all(c == first for c in counts):
        return float(first)

    # Media exactamente entera: prod(1000*C+1) == (1000*k+1)^n. Solo igualdad exacta.
    product = 1
    for c in counts:
        product *= SCALE * c + 1
    root = _iroot(product, n)
    if root ** n == product and root % SCALE == 1:
        return float((root - 1) // SCALE)

    logs = [math.log1p(c / OFFSET) for c in counts]
    if not all(math.isfinite(v) for v in logs):
        raise FloatingPointError(f"log1p no finito para la historia {counts}")
    value = OFFSET * math.expm1(math.fsum(logs) / n)
    if not math.isfinite(value) or value < 0.0:
        raise FloatingPointError(f"EWT numéricamente anómalo ({value}) para {counts}")
    return value


@lru_cache(maxsize=None)
def ewt_band(counts: tuple[int, ...], alpha: float) -> tuple[float, float, float, float, float]:
    """Devuelve (ewt_exacto, central, gsd, interval, upper).

    central  = round(EWT, 5)
    gsd      = round(sd(log(x + 0.001)), 5), ddof = 1
    interval = |qt(alpha; n-1)| * gsd / sqrt(n)
    upper    = central * exp(interval)   (0 si central = 0)
    """
    n = len(counts)
    if n < 2:
        raise ValueError("La desviación con ddof=1 requiere al menos dos conteos")
    exact = ewt(counts)
    central = round(exact, ROUND_DIGITS)

    if all(c == counts[0] for c in counts):
        gsd = 0.0  # historia constante: desviación exactamente cero
    else:
        logs = [math.log(c + OFFSET) for c in counts]
        mean = math.fsum(logs) / n
        gsd = round(math.sqrt(math.fsum((v - mean) ** 2 for v in logs) / (n - 1)), ROUND_DIGITS)

    interval = abs(float(student_t.ppf(alpha, n - 1))) * gsd / math.sqrt(n)
    upper = central * math.exp(interval) if central != 0.0 else 0.0
    if not all(math.isfinite(v) for v in (central, gsd, interval, upper)):
        raise FloatingPointError(f"Banda EWT numéricamente anómala para {counts}")
    return exact, central, gsd, interval, upper


# --------------------------------------------------------------------------- #
# Validación del panel
# --------------------------------------------------------------------------- #
def _validate_panel(result: pd.DataFrame) -> None:
    if result[COUNT].isna().any():
        raise ValueError("Conteos nulos: el panel no se rellena ni se corrige")
    counts = result[COUNT].to_numpy(dtype=float)
    if not np.all(np.isfinite(counts)) or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("Los conteos deben ser enteros no negativos")
    if result["week_start"].isna().any():
        raise ValueError("week_start con valores nulos")
    if (result["week_start"].dt.weekday != 0).any():
        raise ValueError("week_start debe ser lunes")
    gaps = result.groupby("municipality_code", sort=False)["week_start"].diff().dropna()
    bad = gaps[gaps != pd.Timedelta(days=7)]
    if not bad.empty:
        codes = sorted(result.loc[bad.index, "municipality_code"].unique())[:10]
        raise ValueError(f"Huecos o duplicados estructurales en el panel (municipios: {codes})")


# --------------------------------------------------------------------------- #
# Canal
# --------------------------------------------------------------------------- #
def build_ewt_channel(frame: pd.DataFrame, cfg: EwtConfig = EwtConfig()) -> pd.DataFrame:
    """Calcula ``ch_ewt``, ``excess_flag_ewt`` y ``ewt_status`` por municipio y semana."""
    required = {"municipality_code", "week_start", COUNT}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas para el canal: {sorted(missing)}")

    result = frame.copy()
    result["municipality_code"] = result["municipality_code"].astype(str).str.zfill(5)
    result["week_start"] = pd.to_datetime(result["week_start"])
    result = result.sort_values(["municipality_code", "week_start"]).reset_index(drop=True)
    _validate_panel(result)

    result["year"] = result["week_start"].dt.year
    result["ewt_week_ordinal"] = monday_ordinal(result["week_start"])

    partial = {pd.Timestamp(d) for d in cfg.partial_week_starts}
    n_mondays = {int(y): mondays_in_year(int(y)) for y in result["year"].unique()}
    for y in range(int(result["year"].min()) - cfg.window_years, int(result["year"].max()) + 1):
        n_mondays.setdefault(y, mondays_in_year(y))

    ewt_values = np.full(len(result), np.nan)
    central_values = np.full(len(result), np.nan)
    gsd_values = np.full(len(result), np.nan)
    interval_values = np.full(len(result), np.nan)
    upper_values = np.full(len(result), np.nan)
    flags = np.full(len(result), pd.NA, dtype=object)
    status = np.full(len(result), STATUS_NO_HISTORY, dtype=object)

    for _, idx in result.groupby("municipality_code", sort=False).indices.items():
        sub = result.iloc[idx]
        years = sub["year"].to_numpy()
        ords = sub["ewt_week_ordinal"].to_numpy()
        dates = sub["week_start"].to_numpy()
        counts = sub[COUNT].to_numpy(dtype=float).astype(np.int64)
        lookup = {(int(y), int(o)): int(c) for y, o, c in zip(years, ords, counts)}

        for pos in range(len(sub)):
            row = idx[pos]
            year, week, count = int(years[pos]), int(ords[pos]), int(counts[pos])
            if pd.Timestamp(dates[pos]) in partial:
                status[row] = STATUS_PARTIAL
                continue
            if year < cfg.first_label_year:
                status[row] = STATUS_INIT
                continue
            history = []
            for hist_year in range(year - cfg.window_years, year):
                ordinal = week
                if week == 53 and n_mondays[hist_year] < 53:
                    ordinal = 52  # única sustitución permitida
                value = lookup.get((hist_year, ordinal))
                if value is None:
                    break
                history.append(value)
            if len(history) != cfg.window_years:
                status[row] = STATUS_NO_HISTORY
                continue
            reference, central, gsd, interval, upper = ewt_band(tuple(history), cfg.alpha)
            ewt_values[row] = reference
            central_values[row] = central
            gsd_values[row] = gsd
            interval_values[row] = interval
            upper_values[row] = upper
            flags[row] = count > upper#reference  # igualdad -> False; compara contra el EWT exacto
            status[row] = STATUS_LABELED

    result["ch_ewt"] = ewt_values              # EWT exacto (el que usa la etiqueta)
    result["ch_central"] = central_values      # round(EWT, 5)
    result["ch_gsd"] = gsd_values              # round(sd(log(x+0.001)), 5), ddof=1
    result["ch_interval"] = interval_values    # |qt(alpha; n-1)| * gsd / sqrt(n)
    result["ch_upper"] = upper_values          # central * exp(interval); 0 si central = 0
    result["excess_flag_ewt"] = pd.array(flags, dtype="boolean")
    result["ewt_status"] = status
    return result


def build_channel(frame: pd.DataFrame, cfg) -> pd.DataFrame:
    """Despachador para añadir a endemic_channel.py (detector='ewt' vs NB-GLM)."""
    from endemic_channel import build_endemic_channel

    if getattr(cfg, "detector", None) == "ewt":
        out = build_ewt_channel(frame)
        out["excess_flag_v04"] = out["excess_flag_ewt"]
        return out
    return build_endemic_channel(frame, cfg)
