"""Filas de entrenamiento por ensayo y municipio con el corte anterior (a) y con D7.1 (b), h = 8.

Implementación propia, sin código del repositorio; reutiliza caso_f2.py (canal epiCo, week_ordinal).
(a) entrenamiento con año(t + 8) < Y; (b) D7.1: t + 8 <= t0, con t0 la primera emisión evaluada de Y.
Uso: python -I tamanos_d71.py <pilot_weekly_labels.parquet>
"""
import sys
from datetime import date, timedelta
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # con -I (Python 3.11) la carpeta del script no está en sys.path
from caso_f2 import COUNT, channel, week_ordinal
H = 8
df = pd.read_parquet(sys.argv[1]); df["ws"] = pd.to_datetime(df["week_start"]).dt.date
out = []
for code, g in df.groupby("municipality_code"):
    g = g.sort_values("ws")
    by = {(s.year, week_ordinal(s)): v for s, v in zip(g.ws, g[COUNT])}
    cnt = dict(zip(g.ws, g[COUNT]))
    rows, t = [], date(2012, 3, 19)
    while t <= date(2025, 10, 27):
        tg = t + timedelta(weeks=H)
        rows.append((t, tg, int(cnt[tg] > channel(by, tg)[1])))
        t += timedelta(weeks=1)
    r = pd.DataFrame(rows, columns=["t", "tg", "y"]); r["Y"] = r.tg.map(lambda d: d.year)
    for Y in range(2019, 2026):
        ev = r[r.Y == Y]; t0 = ev.t.min()
        a = r[r.Y < Y]; b = r[r.tg <= t0]
        out.append((code, Y, t0, len(ev), len(a), len(b), len(a) - len(b), a.y.sum() - b.y.sum()))
o = pd.DataFrame(out, columns=["mun", "Y", "t0", "eval", "train_a", "train_b", "quitadas", "pos_quitadas"])
print(o.to_string(index=False))
print("quitadas únicas:", sorted(o.quitadas.unique()), "| t0 por año:", o.groupby("Y").t0.unique().to_dict())
print("positivas quitadas por año:", o.groupby("Y").pos_quitadas.sum().to_dict(), "total", o.pos_quitadas.sum(), "de", o.quitadas.sum())
