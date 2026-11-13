"""B1 y B2 de una fila con dos reglas de entrenamiento (contrato §4.1, §6.1), h = 8.

(a) contrato: entrenamiento con año(t + 8) < Y.
(b) hueco point-in-time: entrenamiento con t + 8 <= t0, donde t0 es la primera emisión
    evaluada del ensayo (primer t con año(t + 8) = Y). Las etiquetas de entrenamiento son
    las conocidas al emitir t0.
Uso: python -I caso_f3_hueco.py <parquet> <municipio> <Y> <t evaluada>
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # con -I (Python 3.11) la carpeta del script no está en sys.path
from caso_f2 import COUNT, channel, week_ordinal

H = 8
T_FIRST = date(2012, 3, 19)


def main() -> int:
    path, code, year, t_eval = sys.argv[1], sys.argv[2], int(sys.argv[3]), date.fromisoformat(sys.argv[4])
    df = pd.read_parquet(path)
    df["ws"] = pd.to_datetime(df["week_start"]).dt.date
    g = df[df["municipality_code"] == code].sort_values("ws")
    by = {(s.year, week_ordinal(s)): v for s, v in zip(g["ws"], g[COUNT])}
    cnt = dict(zip(g["ws"], g[COUNT]))

    def zone(s: date) -> str:
        c, u, _ = channel(by, s)
        return "seguridad" if cnt[s] <= c else ("alerta" if cnt[s] <= u else "epidemia")

    rows, t = [], T_FIRST
    while (t + timedelta(weeks=H)).year <= year:
        tg = t + timedelta(weeks=H)
        rows.append((t, tg, zone(t), int(cnt[tg] > channel(by, tg)[1]), week_ordinal(tg)))
        t += timedelta(weeks=1)
    r = pd.DataFrame(rows, columns=["t", "tg", "zona", "y", "w"])
    t0 = r.loc[r.tg.map(lambda d: d.year) == year, "t"].min()
    print(f"primera emisión evaluada t0 = {t0} (objetivo {t0 + timedelta(weeks=H)})")
    z = zone(t_eval)
    w = week_ordinal(t_eval + timedelta(weeks=H))
    window = [w + k for k in range(-2, 3)]
    assert min(window) >= 1 and max(window) <= 52, window
    for name, mask in [("(a) año(t+8) < Y", r.tg.map(lambda d: d.year) < year),
                       ("(b) t+8 <= t0", r.tg <= t0)]:
        tr = r[mask]
        sub, b = tr[tr.zona == z], tr[tr.w.isin(window)]
        print(f"\n{name}: {len(tr)} filas, {tr.y.sum()} positivas, t de {tr.t.min()} a {tr.t.max()}")
        print(tr.groupby("zona")["y"].agg(["sum", "count"]).to_string())
        print(f"B2 (zona {z}) = ({sub.y.sum()}+1)/({len(sub)}+2) = {(sub.y.sum() + 1) / (len(sub) + 2)!r}")
        print(f"B1 ordinales {window} = ({b.y.sum()}+1)/({len(b)}+2) = {(b.y.sum() + 1) / (len(b) + 2)!r}")
    gap = r[(r.tg > t0) & (r.tg.map(lambda d: d.year) < year)]
    print(f"\nfilas que quita (b): {len(gap)} (objetivos {gap.tg.min()} a {gap.tg.max()}), positivas {gap.y.sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
