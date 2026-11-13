"""Inicios de brote evaluables por periodo bajo la regla de eventos de D8 (contrato_predictivo.md §7.4).
 
Solo usa la etiqueta observada (canal de epiCo, D1.1): no entrena ni evalúa modelos.
 
Un inicio en la semana s (primera de >= 3 semanas con C > U1e tras >= 2 con C <= U1e)
es evaluable en un periodo si:
  - s cae en el periodo (año de la semana objetivo, D7);
  - la ventana de emparejamiento [s - 12, s - 1] y la confirmación de una alerta en
    s - 12 (que exige la emisión de s - 13) están cubiertas por emisiones del periodo.
    La primera emisión del periodo es la primera semana objetivo menos h = 8;
  - el inicio se confirma dentro del periodo (s + 2 <= última semana objetivo).
 
Resultado esperado: desarrollo 10 de 11 (se censura Villavicencio 2019-01-28);
prueba 10 de 10.
 
Uso: python -I eventos_d8.py <pilot_weekly_labels.parquet>
"""
from __future__ import annotations
 
import hashlib
import sys
from pathlib import Path
 
import pandas as pd
 
sys.path.insert(0, str(Path(__file__).resolve().parent))
from descriptivos_d3 import COUNT, EXPECTED_SHA256, add_channel, runs  # noqa: E402
 
H = 8
W = 12
PERIODS = {"desarrollo": (2019, 2022), "prueba": (2023, 2025)}
 
 
def onsets(a) -> list[int]:
    return [st for st, ln in runs(a) if ln >= 3 and st >= 2 and a[st - 1] == 0 and a[st - 2] == 0]
 
 
def main() -> int:
    path = Path(sys.argv[1])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"SHA-256 inesperado: {digest}")
    df = pd.read_parquet(path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    e = add_channel(df)
    e = e[e["label_eligible"]].copy()
    e["A"] = (e[COUNT] > e["epico_upper"]).astype(int)
    wk = pd.Timedelta(weeks=1)
 
    rows = []
    for _, g in e.groupby("municipality_code", sort=True):
        g = g.sort_values("week_start").reset_index(drop=True)
        a, ws = g["A"].to_numpy(), g["week_start"]
        for name, (y0, y1) in PERIODS.items():
            tgt = ws[(ws.dt.year >= y0) & (ws.dt.year <= y1)]
            t_min = tgt.min() - H * wk
            for st in onsets(a):
                s = ws[st]
                if not (y0 <= s.year <= y1):
                    continue
                covered = (s - (W + 1) * wk) >= t_min
                confirmed = (s + 2 * wk) <= tgt.max()
                rows.append({"municipio": g["municipality"].iloc[0], "periodo": name,
                             "inicio": s.date(), "ventana_cubierta": covered,
                             "confirmado_en_periodo": confirmed,
                             "evaluable": bool(covered and confirmed)})
    r = pd.DataFrame(rows)
    print(r.to_string(index=False))
    print()
    print(r.groupby("periodo")["evaluable"].agg(evaluables="sum", inicios="count"))
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())