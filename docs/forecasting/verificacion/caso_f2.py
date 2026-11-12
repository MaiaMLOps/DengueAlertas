"""Cálculo independiente de las 15 features (contrato §5.2) para un (municipio, t), h = 8.

Implementación propia, sin código del repositorio. Convenciones de descriptivos_d3.py:
week_ordinal por lunes, semana 53 -> 52, epiCo geométrico con shift 1 y n = 5.
Uso: python -I caso_f2.py <pilot_weekly_labels.parquet> <municipio> <t YYYY-MM-DD>
"""
from __future__ import annotations

import math
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import stats

N, SHIFT, H = 5, 1.0, 8
T_STAR = abs(float(stats.t.ppf(0.025, df=N - 1)))
CLIM = ["temperature_c", "precipitation_mm", "relative_humidity_pct"]
COUNT = "confirmed_records_screening"


def week_ordinal(d: date) -> int:
    first = date(d.year, 1, 1)
    first_monday = first + timedelta(days=(-first.weekday()) % 7)
    return (d - first_monday).days // 7 + 1


def history(series: dict, s: date) -> list[float]:
    """Valores de la misma semana ordinal en los N años previos al año de s (53 -> 52)."""
    w, out = week_ordinal(s), []
    for y in range(s.year - N, s.year):
        v = series.get((y, w))
        if v is None and w == 53:
            v = series.get((y, 52))
        if v is None:
            raise KeyError(f"falta historia {y}-{w} para {s}")
        out.append(v)
    return out


def channel(series: dict, s: date) -> tuple[float, float, list]:
    hist = history(series, s)
    x = np.asarray(hist, dtype=float) + SHIFT
    central = round(float(np.exp(np.log(x).mean())) - SHIFT, 5)
    gsd = round(float(np.std(np.log(x), ddof=1)), 5) if len(set(hist)) > 1 else 0.0
    upper = 0.0 if central <= 0 else float(central * math.exp(T_STAR * gsd / math.sqrt(N)))
    return central, upper, hist


def features(df: pd.DataFrame, t: date, verbose: bool = False) -> dict:
    """df: panel de UN municipio. Solo se usan filas con week_start <= t."""
    d = df[df["ws"] <= t]
    by = {c: {(s.year, week_ordinal(s)): v for s, v in zip(d["ws"], d[c])} for c in [COUNT] + CLIM}
    val = {c: dict(zip(d["ws"], d[c])) for c in [COUNT] + CLIM}
    weeks = [t - timedelta(weeks=k) for k in range(11, -1, -1)]  # t-11 ... t
    C = {s: val[COUNT][s] for s in weeks}
    ch = {s: channel(by[COUNT], s) for s in weeks}
    tgt = t + timedelta(weeks=H)
    cen_h, u_h, hist_h = channel(by[COUNT], tgt)
    pos = {s: math.log(C[s] + 1) - math.log(ch[s][1] + 1) for s in weeks}
    zone_t = "seguridad" if C[t] <= ch[t][0] else ("alerta" if C[t] <= ch[t][1] else "epidemia")
    anom = {c: {s: val[c][s] - float(np.mean(history(by[c], s))) for s in weeks} for c in CLIM}
    blocks = {"m1": weeks[8:12], "m2": weeks[4:8], "m3": weeks[0:4]}
    w_h = week_ordinal(tgt)
    f = {
        "zona_t": zone_t,
        "posicion_4s": float(np.mean([pos[s] for s in weeks[8:12]])),
        "crecimiento_4s": math.log(C[t] + 1) - math.log(C[weeks[7]] + 1),
        "cambio_umbral": math.log(u_h + 1) - math.log(ch[t][1] + 1),
        "sem_sin": math.sin(2 * math.pi * (w_h - 1) / 52.18),
        "sem_cos": math.cos(2 * math.pi * (w_h - 1) / 52.18),
    }
    for b, ws in blocks.items():
        f[f"anom_temp_{b}"] = float(np.mean([anom["temperature_c"][s] for s in ws]))
        f[f"anom_prec_{b}"] = float(np.sum([anom["precipitation_mm"][s] for s in ws]))
        f[f"anom_hum_{b}"] = float(np.mean([anom["relative_humidity_pct"][s] for s in ws]))
    if verbose:
        print(f"t = {t} (semana ordinal {week_ordinal(t)}), t+h = {tgt} (ordinal {w_h})")
        print(f"U1e_t+h: historia {hist_h} -> central {cen_h}, U1e {u_h:.6f}")
        print(f"U1e_t:   historia {ch[t][2]} -> central {ch[t][0]}, U1e {ch[t][1]:.6f}; C_t = {C[t]}")
        print("s, C_s, central_s, U1e_s, pos_s:")
        for s in weeks:
            print(f"  {s} {C[s]:4d} {ch[s][0]:9.5f} {ch[s][1]:10.5f} {pos[s]: .6f}")
    return f


def main() -> int:
    path, code, t = sys.argv[1], sys.argv[2], date.fromisoformat(sys.argv[3])
    df = pd.read_parquet(path)
    df["ws"] = pd.to_datetime(df["week_start"]).dt.date
    g = df[df["municipality_code"] == code].sort_values("ws").reset_index(drop=True)
    f = features(g, t, verbose=True)  # la función solo ve filas con week_start <= t
    print("\nFeatures:")
    for k, v in f.items():
        print(f"  {k:16s} {v if isinstance(v, str) else f'{v: .6f}'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
