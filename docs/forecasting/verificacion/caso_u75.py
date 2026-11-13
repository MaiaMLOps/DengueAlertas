"""caso_u75.py: independent check of the derivable reports of U7.5 (contract §7.2, §7.3, §7.4, §8.4).

Written on 2026-11-12 as the independent check of U7.5, before its figures were frozen in
tests/acceptance/test_f7c_reportes_real.py and test_f7c_identidades_real.py. It imports nothing from src/: it reuses
only caso_f7_diagnostico.py (independent script of F7), which rebuilds the out-of-year common configuration, the cell
table and the event rule from the contract.

Usage (from the repository root):
    .venv/Scripts/python.exe -X utf8 -I docs/forecasting/verificacion/caso_u75.py
        --canal <pilot_weekly_channel.parquet>
        --busqueda-sin-clima <data/processed/forecasting/sin_clima>
        --busqueda-completo <data/processed/forecasting>
        --predicciones-prueba <prueba/predicciones_prueba.parquet>
        --serie-prueba <prueba diag_serie.csv (only for p_b2, p_b1 of the test period)>
        --salida <new dir>

Definitions (as read from the contract, to contrast with the conventions declared in
tests/acceptance/test_f7c_formulas.py):
  §7.3  yhat = 1[p >= tau]; sensitivity TP/(TP+FN); specificity TN/(TN+FP); PPV TP/(TP+FP);
        F1 2TP/(2TP+FP+FN); MCC (TP·TN − FP·FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)), NaN if a factor is 0;
        reference F1 of "all positive" 2π/(1+π).
  §7.2  Murphy with 10 equal-frequency bins of p (pd.qcut, as the reliability diagram of the card):
        REL = Σ n_k (p̄_k − ō_k)² / N, RES = Σ n_k (ō_k − ō)² / N, UNC = ō(1 − ō);
        residue = Brier − (REL − RES + UNC) = within-bin variance of p − 2·within-bin covariance(p, y).
        Calibration in the large per year with pooled rows: mean p − prevalence.
  §7.4  alert ratio = onsets warned / judgeable non-continuity alerts; composite = 0.5·detection + 0.5·ratio;
        false alarms split into followed / not followed by some epidemic-zone week in [t_c+1, t_c+12].
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import caso_f7_diagnostico as f7  # noqa: E402  (independent script, not src/)

# Adapter for a period without onsets. caso_f7_diagnostico.py is frozen (its md5 is cited by frozen acceptance tests)
# and its events() fails when the table of onsets is empty: it sorts the empty DataFrame by "municipality_code" and
# "semana", columns that the empty table does not have (KeyError). That happens with the per-municipality blocks of a
# municipality without onsets in the period (Cali in the test period). events_safe() calls f7.events() unchanged and,
# only while it runs, makes that one sort of an empty table return an empty table with the expected columns; the
# totals it computes are untouched.
_ONSET_COLUMNS = ["municipality_code", "semana", "evaluable", "antic_modelo", "antic_canal"]


def events_safe(rows, series, tau):
    original_sort = pd.DataFrame.sort_values

    def sort_values(self, by, *args, **kwargs):
        if self.empty and list(by) == _ONSET_COLUMNS[:2]:
            return pd.DataFrame(columns=_ONSET_COLUMNS).astype(
                {"evaluable": bool, "antic_modelo": float, "antic_canal": float})
        return original_sort(self, by, *args, **kwargs)

    pd.DataFrame.sort_values = sort_values
    try:
        return f7.events(rows, series, tau)
    finally:
        pd.DataFrame.sort_values = original_sort


TAU = 0.05
GRID = [round(0.05 * i, 2) for i in range(1, 20)]


# ------------------------------------------------------------------------------------------- loading

def load_development(search: Path) -> pd.DataFrame:
    rows = f7.development_predictions(search, "logistica")
    rows["municipality_code"] = rows["municipality_code"].astype(str)
    return rows


def load_test(predictions: Path, series: Path) -> pd.DataFrame:
    p = pd.read_parquet(predictions)
    p["municipality_code"] = p["municipality_code"].astype(str)
    wide = p.pivot_table(index=["municipality_code", "t", "target_week", "anio", "y"], columns="sistema",
                         values="p").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"modelo": "p_modelo", "ablacion": "p_ablacion"})
    s = pd.read_csv(series, dtype={"municipality_code": str})
    s["municipality_code"] = s["municipality_code"].str.zfill(5)
    wide["t"] = pd.to_datetime(wide["t"]).dt.strftime("%Y-%m-%d")
    s["t"] = pd.to_datetime(s["t"]).dt.strftime("%Y-%m-%d")
    out = wide.merge(s[["municipality_code", "t", "p_b2", "p_b1", "y"]].rename(columns={"y": "y_serie"}),
                     on=["municipality_code", "t"], how="left", validate="one_to_one")
    if len(out) != 936 or out[["p_b2", "p_b1"]].isna().any().any() or (out["y"] != out["y_serie"]).any():
        f7.fail("test rows do not align")
    return out.drop(columns="y_serie")


# ------------------------------------------------------------------------------------------- metrics

def classification(y, p, tau=TAU) -> dict:
    y = np.asarray(y, int)
    yhat = (np.asarray(p, float) >= tau).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else float("nan")
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    pi = y.mean()
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "positivas_emitidas": tp + fp,
            "sensibilidad": div(tp, tp + fn), "especificidad": div(tn, tn + fp), "vpp": div(tp, tp + fp),
            "f1": div(2 * tp, 2 * tp + fp + fn),
            "mcc": (tp * tn - fp * fn) / math.sqrt(denom) if denom else float("nan"),
            "prevalencia": float(pi), "f1_referencia": float(2 * pi / (1 + pi))}


def murphy(y, p, bins=10) -> dict:
    y = np.asarray(y, float); p = np.asarray(p, float)
    labels = pd.qcut(p, bins, labels=False)
    n, obar = len(y), y.mean()
    rel = res = wbv = wbc = 0.0
    counts = []
    for k in range(bins):
        m = labels == k
        nk, pk, ok = m.sum(), p[m].mean(), y[m].mean()
        counts.append(int(nk))
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - obar) ** 2
        wbv += ((p[m] - pk) ** 2).sum()
        wbc += ((p[m] - pk) * (y[m] - ok)).sum()
    rel, res, wbv, wbc = rel / n, res / n, wbv / n, wbc / n
    unc = obar * (1 - obar)
    bs = float(np.mean((p - y) ** 2))
    residue = bs - (rel - res + unc)
    if abs(residue - (wbv - 2 * wbc)) > 1e-12:
        f7.fail("Murphy identity with within-bin terms does not close")
    return {"brier": bs, "confiabilidad": rel, "resolucion": res, "incertidumbre": unc,
            "brier_aprox": rel - res + unc, "residuo": residue, "n_por_intervalo": counts}


def calibration_by_year(rows: pd.DataFrame, systems) -> pd.DataFrame:
    out = []
    for year, g in list(rows.groupby("anio")) + [("todos", rows)]:
        rec = {"anio": year, "filas": len(g), "prevalencia": g["y"].mean()}
        for s in systems:
            rec[f"p_medio_{s}"] = g[f"p_{s}"].mean()
            rec[f"calib_{s}"] = g[f"p_{s}"].mean() - g["y"].mean()
        out.append(rec)
    return pd.DataFrame(out)


def scores(rows: pd.DataFrame, model: str, regime) -> dict:
    r = rows.drop(columns="p_modelo").rename(columns={f"p_{model}": "p_modelo"}) if model != "modelo" else rows
    cells = f7.cell_table(r, regime)
    y = r["y"]
    out = {"casillas": len(cells)}
    for ref in ("b2", "b1"):
        out[f"bss_{ref}_filas"] = 1 - f7.brier(y, r["p_modelo"]) / f7.brier(y, r[f"p_{ref}"])
        out[f"mejora_ll_{ref}_filas"] = 1 - f7.logloss(y, r["p_modelo"]) / f7.logloss(y, r[f"p_{ref}"])
        out[f"bss_{ref}_prom_casillas"] = cells[f"bss_{ref}"].mean()
        out[f"mejora_ll_{ref}_prom_casillas"] = cells[f"mejora_ll_{ref}"].mean()
        out[f"gana_brier_{ref}"] = int(cells[f"gana_brier_{ref}"].sum())
        out[f"gana_ll_{ref}"] = int(cells[f"gana_ll_{ref}"].sum())
    return out, cells


def event_block(rows, series, tau=TAU) -> dict:
    table, tot = events_safe(rows, series, tau)
    out = {}
    for s in ("modelo", "canal"):
        t = tot[s]
        det = t["inicios_avisados"] / t["inicios_evaluables"] if t["inicios_evaluables"] else float("nan")
        ratio = t["inicios_avisados"] / t["juzgables"] if t["juzgables"] else float("nan")
        verdaderas = t["juzgables"] - t["falsas_alarmas"]
        out[s] = {**t, "deteccion": det, "razon_alertas": ratio,
                  "alertas_con_inicio": verdaderas,
                  "razon_alternativa_alertas_con_inicio": verdaderas / t["juzgables"] if t["juzgables"] else float("nan"),
                  "compuesto": 0.5 * det + 0.5 * ratio if not (math.isnan(det) or math.isnan(ratio)) else float("nan"),
                  "fa_no_seguidas": t["falsas_alarmas"] - t["fa_seguidas_de_epidemia"]}
        ev = table[table["evaluable"]]
        leads = ev[f"antic_{s}"].dropna()
        out[s]["anticipacion_mediana"] = float(leads.median()) if len(leads) else None
    return out


def amoc(rows, series) -> pd.DataFrame:
    out = []
    for tau in GRID:
        table, tot = events_safe(rows, series, tau)
        m = tot["modelo"]
        ev = table[table["evaluable"]]
        leads = ev["antic_modelo"].dropna()
        out.append({"tau": tau, "avisados": m["inicios_avisados"], "evaluables": m["inicios_evaluables"],
                    "falsas_alarmas": m["falsas_alarmas"], "antic_mediana": leads.median() if len(leads) else None})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------- main

def by_group(rows, key, fn):
    return {str(k): fn(g) for k, g in rows.groupby(key)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canal", type=Path, required=True)
    ap.add_argument("--busqueda-sin-clima", type=Path, required=True)
    ap.add_argument("--busqueda-completo", type=Path, required=True)
    ap.add_argument("--predicciones-prueba", type=Path, required=True)
    ap.add_argument("--serie-prueba", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    a = ap.parse_args(argv)
    a.salida.mkdir(parents=True, exist_ok=True)

    regime = f7.regimes(a.canal)
    series = f7.observed_series(a.canal)
    dev = load_development(a.busqueda_sin_clima)
    dev_abl = load_development(a.busqueda_completo)
    dev = dev.merge(dev_abl[["municipality_code", "t", "p_modelo", "configuracion"]]
                    .rename(columns={"p_modelo": "p_ablacion", "configuracion": "config_ablacion"}),
                    on=["municipality_code", "t"], validate="one_to_one")
    test = load_test(a.predicciones_prueba, a.serie_prueba)
    periods = {"desarrollo": dev, "prueba": test}
    result = {"configuracion_desarrollo": dev.groupby("anio")["configuracion"].first().to_dict(),
              "configuracion_ablacion_desarrollo": dev.groupby("anio")["config_ablacion"].first().to_dict()}

    for name, rows in periods.items():
        block = {}
        # §7.3
        block["clasificacion"] = {
            "global": {s: classification(rows["y"], rows[f"p_{s}"]) for s in ("modelo", "b2")},
            "por_anio": by_group(rows, "anio", lambda g: {s: classification(g["y"], g[f"p_{s}"]) for s in ("modelo", "b2")}),
            "por_municipio": by_group(rows, "municipality_code", lambda g: {s: classification(g["y"], g[f"p_{s}"]) for s in ("modelo", "b2")}),
        }
        # §7.2
        block["murphy"] = {"global": {s: murphy(rows["y"], rows[f"p_{s}"]) for s in ("modelo", "ablacion")},
                           "por_anio": by_group(rows, "anio", lambda g: murphy(g["y"], g["p_modelo"]))}
        block["calibracion_por_anio"] = calibration_by_year(rows, ("modelo", "ablacion", "b2", "b1")).to_dict(orient="records")
        # tabla desarrollo frente a prueba
        block["receta"], cells = scores(rows, "modelo", regime)
        block["ablacion"], _ = scores(rows, "ablacion", regime)
        cells.to_csv(a.salida / f"casillas_{name}.csv", index=False)
        # §7.4
        block["eventos"] = {"global": event_block(rows, series),
                            "por_municipio": by_group(rows, "municipality_code", lambda g: event_block(g, series))}
        block["amoc"] = amoc(rows, series).to_dict(orient="records")
        result[name] = block

    (a.salida / "u75_independiente.json").write_text(json.dumps(result, indent=1, default=float, ensure_ascii=False),
                                                     encoding="utf-8")
    # compact console report
    pd.set_option("display.width", 220)
    for name in periods:
        b = result[name]
        print(f"\n===== {name.upper()} =====")
        print("§7.3 global:")
        print(pd.DataFrame(b["clasificacion"]["global"]).T.round(4).to_string())
        print("§7.3 por año (modelo | b2):")
        print(pd.DataFrame({k: {**{f"m_{x}": v["modelo"][x] for x in ("sensibilidad", "especificidad", "vpp", "f1", "mcc", "f1_referencia")},
                                **{f"b2_{x}": v["b2"][x] for x in ("sensibilidad", "especificidad", "vpp", "f1", "mcc")}}
                            for k, v in b["clasificacion"]["por_anio"].items()}).T.round(3).to_string())
        print("§7.3 por municipio:")
        print(pd.DataFrame({k: {**{f"m_{x}": v["modelo"][x] for x in ("sensibilidad", "especificidad", "vpp", "f1", "mcc", "f1_referencia")},
                                **{f"b2_{x}": v["b2"][x] for x in ("sensibilidad", "especificidad", "vpp", "f1", "mcc")}}
                            for k, v in b["clasificacion"]["por_municipio"].items()}).T.round(3).to_string())
        print("§7.2 Murphy global:")
        print(pd.DataFrame(b["murphy"]["global"]).T.drop(columns="n_por_intervalo").round(5).to_string())
        print("n por intervalo (modelo):", b["murphy"]["global"]["modelo"]["n_por_intervalo"])
        print("§7.2 Murphy por año (modelo):")
        print(pd.DataFrame(b["murphy"]["por_anio"]).T.drop(columns="n_por_intervalo").round(5).to_string())
        print("§7.2 calibración en grande por año:")
        print(pd.DataFrame(b["calibracion_por_anio"]).round(4).to_string(index=False))
        print("Receta / ablación:")
        print(pd.DataFrame({"receta": b["receta"], "ablacion": b["ablacion"]}).round(4).to_string())
        print("§7.4 eventos global:")
        print(pd.DataFrame(b["eventos"]["global"]).round(3).to_string())
        print("§7.4 por municipio (modelo / canal):")
        rows_ = []
        for code, e in b["eventos"]["por_municipio"].items():
            for s in ("modelo", "canal"):
                x = e[s]
                rows_.append({"mun": code, "sis": s, "eval": x["inicios_evaluables"], "avis": x["inicios_avisados"],
                              "juzg": x["juzgables"], "fa": x["falsas_alarmas"], "fa_seg": x["fa_seguidas_de_epidemia"],
                              "cont": x["continuidad"], "det": x["deteccion"], "razon": x["razon_alertas"],
                              "comp": x["compuesto"]})
        print(pd.DataFrame(rows_).round(3).to_string(index=False))
        print("AMOC:")
        print(pd.DataFrame(b["amoc"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
