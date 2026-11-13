#!/usr/bin/env python3
"""Evidencia de las decisiones D3 (target) y D1.1 (parametrización única del canal) — DengueAlertas.
 
Análisis descriptivo de etiquetas. No entrena ni evalúa modelos.
 
Calcula, con el canal de epiCo (method="geometric", shift=1, n=5):
  1. Reproducción de U1e frente al estudio de umbrales (945 positivas, 175 rachas).
  2. Descriptivos del target A_s = 1[C_s > U1e_s] y de las alternativas descartadas en D3
     (ventana de 4 semanas, episodio sostenido, inicio de brote al estilo de Umaña et al.).
  3. Autocorrelación del estado: cambio entre la última semana observada s y s+k, entradas y salidas.
  4. Comparación de la línea central de D1 (EWT, eps=0,001) con la central de epiCo (shift=1).
  5. Zonas del canal epiCo y matriz de transición de zona(s) -> zona(s+5).
  6. (D4) Información del estado observado en t sobre el estado en t+h: kappa, P(1|0), P(1|1) y
     entradas para h = 1…52.
 
Uso:
    python -I descriptivos_d3.py <pilot_weekly_labels.parquet>
 
Entrada esperada: SHA-256 b3b151a54b2dc7e8c18eb78757ad6a40093626bb807757bbccb6045b354f2c7f.
"""
from __future__ import annotations
 
import hashlib
import math
import sys
from datetime import date, timedelta
from pathlib import Path
 
import numpy as np
import pandas as pd
from scipy import stats
 
EXPECTED_SHA256 = "b3b151a54b2dc7e8c18eb78757ad6a40093626bb807757bbccb6045b354f2c7f"
COUNT = "confirmed_records_screening"
N_YEARS = 5
SHIFT = 1.0
T_STAR = abs(float(stats.t.ppf(0.025, df=N_YEARS - 1)))
K_PERSIST = 5  # h = 4 contado desde la emisión con último dato en t-1 (convención del avance)
 
 
# --------------------------------------------------------------------------- canal epiCo
def week_ordinal(d: date) -> int:
    first = date(d.year, 1, 1)
    first_monday = first + timedelta(days=(-first.weekday()) % 7)
    return (d - first_monday).days // 7 + 1
 
 
def epico_channel(history: tuple[int, ...]) -> tuple[float, float]:
    """(central, upper) de epiCo::endemic_channel con redondeo a 5 decimales."""
    x = np.asarray(history, dtype=float) + SHIFT
    central = round(float(np.exp(np.log(x).mean())) - SHIFT, 5)
    gsd = round(float(np.std(np.log(x), ddof=1)), 5) if len(set(history)) > 1 else 0.0
    if central <= 0:
        return central, 0.0
    return central, float(math.exp(math.log(central) + T_STAR * gsd / math.sqrt(N_YEARS)))
 
 
def add_channel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["municipality_code", "week_start"]).reset_index(drop=True)
    starts = [pd.Timestamp(s).date() for s in df["week_start"]]
    counts = {(c, s.year, week_ordinal(s)): int(n)
              for c, s, n in zip(df["municipality_code"], starts, df[COUNT])}
    central, upper, hist_zero = [], [], []
    for code, s, eligible in zip(df["municipality_code"], starts, df["label_eligible"]):
        if not eligible:
            central.append(np.nan); upper.append(np.nan); hist_zero.append(False); continue
        w = week_ordinal(s)
        hist = []
        for y in range(s.year - N_YEARS, s.year):
            v = counts.get((code, y, w))
            if v is None and w == 53:
                v = counts.get((code, y, 52))
            if v is None:
                raise ValueError(f"Historia incompleta en semana elegible {code} {s}")
            hist.append(v)
        c, u = epico_channel(tuple(hist))
        central.append(c); upper.append(u); hist_zero.append(0 in hist)
    df["epico_central"] = central
    df["epico_upper"] = upper
    df["hist_has_zero"] = hist_zero
    return df
 
 
# --------------------------------------------------------------------------- utilidades
def runs(a: np.ndarray) -> list[tuple[int, int]]:
    """Rachas maximales de unos: (inicio, longitud)."""
    out, i = [], 0
    while i < len(a):
        if a[i] == 1:
            j = i
            while j < len(a) and a[j] == 1:
                j += 1
            out.append((i, j - i)); i = j
        else:
            i += 1
    return out
 
 
def per_municipality(e: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, g in e.groupby("municipality_code", sort=True):
        a = g["A"].to_numpy()
        m = len(a)
        rs = runs(a)
        sustained = np.zeros(m, dtype=int)
        for st, ln in rs:
            if ln >= 3:
                sustained[st:st + ln] = 1
        onsets = [st for st, ln in rs if ln >= 3 and st >= 2 and a[st - 1] == 0 and a[st - 2] == 0]
        window4 = np.array([a[s + 1:s + 5].max() for s in range(m - 4)])
        x, y = a[:-K_PERSIST], a[K_PERSIST:]
        rows.append({
            "municipio": f"{code} {g['municipality'].iloc[0]}",
            "semanas": m, "positivas": int(a.sum()), "prevalencia": a.mean(),
            "rachas": len(rs), "rachas_1sem": sum(1 for _, ln in rs if ln == 1),
            "entradas_0a1": sum(1 for st, _ in rs if st >= 1),
            "positivas_sostenidas": int(sustained.sum()),
            "prev_ventana4": window4.mean(),
            "inicios_umana": len(onsets),
            "anios_inicios": ",".join(str(v) for v in g["year"].to_numpy()[onsets]),
            "pares_k5": len(x), "desde_0": int((x == 0).sum()), "desde_1": int((x == 1).sum()),
            "entradas_k5": int(((x == 0) & (y == 1)).sum()), "salidas_k5": int(((x == 1) & (y == 0)).sum()),
        })
    return pd.DataFrame(rows)
 
 
def main() -> int:
    path = Path(sys.argv[1])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"SHA-256 inesperado: {digest}")
    df = pd.read_parquet(path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = add_channel(df)
    e = df[df["label_eligible"]].copy()
    e["year"] = e["week_start"].dt.year
    C = e[COUNT].to_numpy()
    e["A"] = (C > e["epico_upper"].to_numpy()).astype(int)
    e["above_central"] = (C > e["epico_central"].to_numpy()).astype(int)
    e["zone"] = e["above_central"] + e["A"]  # 0 seguridad, 1 alerta, 2 epidemia
 
    pd.set_option("display.width", 220, "display.max_columns", 30)
    tab = per_municipality(e)
 
    print("== 1. Reproducción de U1e ==")
    print(f"semanas elegibles={len(e)}  positivas={int(e.A.sum())}  rachas={int(tab.rachas.sum())}  "
          f"(esperado 4380 / 945 / 175)")
    assert (len(e), int(e.A.sum()), int(tab.rachas.sum())) == (4380, 945, 175)
    assert bool((e["epico_upper"] >= e["epico_central"] - 1e-9).all())
 
    print("\n== 2. Target y alternativas de D3, por municipio ==")
    print(tab.drop(columns=["pares_k5", "desde_0", "desde_1", "entradas_k5", "salidas_k5"]).to_string(index=False))
    tot = tab.sum(numeric_only=True)
    print(f"\nGlobal: prevalencia A={tot.positivas / tot.semanas:.3f}; rachas de 1 semana={int(tot.rachas_1sem)} "
          f"({tot.rachas_1sem / tot.rachas:.3f} de las rachas; {tot.rachas_1sem / tot.positivas:.3f} de las positivas); "
          f"positivas en rachas >= 3={int(tot.positivas_sostenidas)} ({tot.positivas_sostenidas / tot.semanas:.3f} de las semanas); "
          f"ventana4 media={tab.prev_ventana4.mean():.3f}; entradas 0->1={int(tot.entradas_0a1)}; "
          f"inicios Umaña={int(tot.inicios_umana)}")
 
    print(f"\n== 3. Cambio de estado entre s y s+{K_PERSIST} ==")
    print(f"pares={int(tot.pares_k5)}  cambio={ (tot.entradas_k5 + tot.salidas_k5) / tot.pares_k5:.3f}  "
          f"entradas={int(tot.entradas_k5)} (P(1|0)={tot.entradas_k5 / tot.desde_0:.3f})  "
          f"salidas={int(tot.salidas_k5)} (P(0|1)={tot.salidas_k5 / tot.desde_1:.3f})")
    print("entradas por municipio:", dict(zip(tab.municipio, tab.entradas_k5)))
 
    print("\n== 4. EWT (D1, eps=0,001) frente a central de epiCo (shift=1) ==")
    ewt, cen = e["ewt"].to_numpy(), e["epico_central"].to_numpy()
    print(f"central > EWT en {int((cen > ewt + 1e-9).sum())} de {len(e)} semanas; central < EWT en {int((cen < ewt - 1e-9).sum())}")
    for name, mask in [("historia sin ceros", ~e.hist_has_zero.to_numpy()), ("historia con algún cero", e.hist_has_zero.to_numpy())]:
        ratio = cen[mask] / np.where(ewt[mask] > 0, ewt[mask], np.nan)
        print(f"  {name:24s} n={int(mask.sum()):5d}  mediana central/EWT={np.nanmedian(ratio):.3f}")
    print(f"P(C>EWT)={np.mean(C > ewt):.3f}  P(C>central)={np.mean(C > cen):.3f}  "
          f"etiquetas discrepantes={np.mean((C > ewt) != (C > cen)):.3f}")
 
    print("\n== 5. Zonas del canal epiCo ==")
    print("seguridad / alerta / epidemia:", [round(float((e.zone == z).mean()), 3) for z in (0, 1, 2)])
    for name, col in [("C > central", "above_central"), ("C > U1e", "A")]:
        ch = n = mono = 0
        for _, g in e.groupby("municipality_code"):
            a = g[col].to_numpy()
            ch += int((a[:-K_PERSIST] != a[K_PERSIST:]).sum()); n += len(a) - K_PERSIST
            yr = g.groupby("year")[col].mean()
            mono += int(((yr >= 0.95) | (yr <= 0.05)).sum())
        print(f"  {name:12s} prevalencia={e[col].mean():.3f}  cambio_k5={ch / n:.3f}  años casi monoclase={mono}/84")
    M = np.zeros((3, 3), dtype=int)
    for _, g in e.groupby("municipality_code"):
        z = g["zone"].to_numpy()
        np.add.at(M, (z[:-K_PERSIST], z[K_PERSIST:]), 1)
    print("matriz zona(s) -> zona(s+5) [filas: seguridad, alerta, epidemia]:\n", M)
    print(f"acuerdo de la persistencia de zona={np.trace(M) / M.sum():.3f};  "
          f"P(epidemia | alerta)={M[1, 2] / M[1].sum():.3f};  P(epidemia | seguridad)={M[0, 2] / M[0].sum():.3f}")
 
    print("\n== 6. (D4) Información del estado observado en t sobre el estado en t+h ==")
    print(" h   kappa  P(1|0)  P(1|1)  entradas salidas | kappa por municipio")
    for h in (1, 2, 4, 6, 8, 10, 12, 16, 26, 52):
        xs, ys, per = [], [], []
        for _, g in e.groupby("municipality_code", sort=True):
            a = g["A"].to_numpy()
            xs.append(a[:-h]); ys.append(a[h:]); per.append(cohen_kappa(a[:-h], a[h:]))
        x, y = np.concatenate(xs), np.concatenate(ys)
        print(f"{h:2d}  {cohen_kappa(x, y):6.3f}  {y[x == 0].mean():.3f}   {y[x == 1].mean():.3f}   "
              f"{int(((x == 0) & (y == 1)).sum()):5d}   {int(((x == 1) & (y == 0)).sum()):5d}   | "
              + " ".join(f"{v:.2f}" for v in per))
    return 0
 
 
def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float(np.mean(a == b))
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe)
 
 
if __name__ == "__main__":
    sys.exit(main())
 