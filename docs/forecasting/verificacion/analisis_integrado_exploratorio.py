r"""analisis_integrado_exploratorio.py: exploratory calculations [X] of the integrated analysis (contract §8.8, D9.1)
and an independent cross-check of the derivable reports of U7.5.

Scope. The document docs/forecasting/analisis_resultados.md cites two kinds of figures: those of the frozen evidence
(reports/forecasting/evidence/, reports/forecasting/figures/20261112/) and exploratory ones, marked [X], that were
not declared before the single test run and use information posterior to the emission (phases relative to episodes, weeks since the end
of an episode, alert state at an onset). This script produces every [X] figure and table of that document and checks
that the U7.5 tables it reads agree with an independent recomputation. It changes nothing in the recipe, retrains
nothing, adds no statistical test (§8.5) and does not redefine the event rule of §7.4.

It imports nothing from src/. It reuses caso_f7_diagnostico.py (same folder; its md5 is cited by frozen tests and it
is not modified), which rebuilds the out-of-year common configuration of development, the regimes, the channel
series and the event rule of §7.4.

Usage (from the repository root, with the DVC data pulled):

    .venv/Scripts/python.exe -X utf8 -I docs/forecasting/verificacion/analisis_integrado_exploratorio.py \
        --canal data/processed/epidemiology/pilot_weekly_channel.parquet \
        --busqueda-sin-clima data/processed/forecasting/sin_clima \
        --predicciones-prueba data/processed/forecasting/prueba/predicciones_prueba.parquet \
        --evidencia reports/forecasting/evidence \
        --salida-tablas reports/forecasting/evidence/exploratorio_20261112 \
        --salida-figuras reports/forecasting/figures/exploratorio_20261112

Outputs are never overwritten: the run fails if an output file already exists. The exit code is 1 if a check of the
U7.5 cross-check fails. The cross-check covers the quantities the document cites (cells, skill, classification at
tau, Murphy of the model, calibration per year, global events and weeks under alert); it does not cover the ablation,
the alert ratio (inicios_avisados_por_juzgable / alertas_juzgables) nor «alertas_que_avisan». The channel's alert ratio
in development follows convention U7.5-D4 (9/63 instead of the 10/63 of an independent count: Villavicencio
2022-10-31 is warned only by a non-judgeable alert). The full independent check of U7.5 is caso_u75.py.

Definitions (all diagnostic strata; none is information available at emission):
  episode      run of >= 3 consecutive eligible weeks with target_u1e = 1 on the whole eligible series (as §8.8).
  phase        of the TARGET week w, in this order of precedence: «durante» if w lies inside an episode; «antes»
               if an episode starts 1-8 weeks after w; «después» if an episode ended 1-13 weeks before w; «resto».
  net gain     per row, (p_B2 - y)^2 - (p_model - y)^2; the share of a phase is its sum over the total sum, so
               shares add to 100 % and can be negative.
  stratum      «dentro» if the channel zone at the emission week t is epidemic, «fuera» otherwise (as §8.8).
  weeks since  number of eligible weeks between the end of the last episode and the target week (0 inside one).
  alert run    >= 2 consecutive positive emissions (model: p >= tau; channel: zone «alerta»), confirmed at the
               second week t_c; continuity if the zone at t_c is epidemic (as §7.4). «Under alert at an onset»
               means t_c <= onset week <= last week of the run (continuity runs included); the base rate is the
               share of emission weeks that lie in [t_c, end] of some run.
  persistence  positive if the channel zone at the emission week t is epidemic (y_hat = zone at t). It coincides with
               B2 for tau between 0.25 and 0.50 in test and for tau 0.25 and 0.30 in development, because only the
               epidemic zone gives B2 a high value there.
  tau of max F1 the grid tau with the largest F1 of the model in development (ties to the larger tau); used only to
               describe the label at a weekly-optimal point chosen without the test period. 0.50 is reported as a
               threshold without selection. None of them changes the recipe tau = 0.05.
  channel zone the zone observed at t (seguridad, alerta, epidemia): information available when emitting. Reading bands
               of p̂ within each zone are descriptive and were set after seeing the finer intervals.
  AUC          Mann-Whitney with ties at half; descriptive, not a metric of the contract.
  Murphy of B2 B2 takes few distinct values; its decomposition is reported with its exact groups (one per distinct
               value, residue 0) and with 10 equal-frequency bins collapsing ties (the procedure of the model).
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import caso_f7_diagnostico as f7  # noqa: E402  (independent script, not src/)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

TAU = 0.05
W = 12
WEEK = pd.Timedelta(weeks=1)
PHASES = ("antes", "durante", "después", "resto")
PHASE_LABEL = {"antes": "Antes (1–8 sem previas)", "durante": "Durante el episodio",
               "después": "Después (1–13 sem)", "resto": "Resto (tranquilas)"}
NAMES = f7.NAMES
ORDER = ("41001", "50001", "76001", "08758", "54001", "08001")  # Neiva, Villavicencio, Cali, Soledad, Cúcuta, Barranquilla
PERIODS = ("desarrollo", "prueba")
PERIOD_TITLE = {"desarrollo": "Desarrollo 2019–2022 (usado para elegir)", "prueba": "Prueba 2023–2025"}
LABEL = {"desarrollo": "usado_para_elegir", "prueba": "prueba"}
TRAMOS = ((0, 0, "en episodio"), (1, 13, "1–13"), (14, 26, "14–26"), (27, 52, "27–52"), (53, 10**6, ">52 o nunca"))

# Palette of the U7.5 figures (src/forecasting/figures_reports.py): model blue, channel orange, B2 muted grey.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, NEUTRAL, RED = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f0efec", "#e34948"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "svg.hashsalt": "analisis-integrado-x", "lines.linewidth": 2.0,
})


# ------------------------------------------------------------------------------------------------ loading

def load_development(search: Path) -> pd.DataFrame:
    rows = f7.development_predictions(search, "logistica")
    rows["municipality_code"] = rows["municipality_code"].astype(str).str.zfill(5)
    rows["t"] = pd.to_datetime(rows["t"])
    rows["target_week"] = pd.to_datetime(rows["target_week"])
    return rows.reset_index(drop=True)


def load_test(predictions: Path, series_csv: Path) -> pd.DataFrame:
    """Test predictions of the model (single run) joined with B2 and B1 of the frozen test series."""
    p = pd.read_parquet(predictions)
    p["municipality_code"] = p["municipality_code"].astype(str).str.zfill(5)
    wide = p.pivot_table(index=["municipality_code", "t", "target_week", "anio", "y"], columns="sistema",
                         values="p").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"modelo": "p_modelo", "ablacion": "p_ablacion"})
    s = pd.read_csv(series_csv, dtype={"municipality_code": str})
    s["municipality_code"] = s["municipality_code"].str.zfill(5)
    for frame in (wide, s):
        frame["t"] = pd.to_datetime(frame["t"])
    out = wide.merge(s[["municipality_code", "t", "p_b2", "p_b1", "y"]].rename(columns={"y": "y_serie"}),
                     on=["municipality_code", "t"], how="left", validate="one_to_one")
    if len(out) != 936 or out[["p_b2", "p_b1"]].isna().any().any() or (out["y"] != out["y_serie"]).any():
        f7.fail("test rows do not align with diag_serie.csv")
    out["target_week"] = pd.to_datetime(out["target_week"])
    return out.drop(columns="y_serie").reset_index(drop=True)


# ----------------------------------------------------------------------------------- episodes and phases

def episodes(series: dict) -> dict:
    out = {}
    for code, s in series.items():
        ex, wk, found, i = s["exceed"], s["weeks"], [], 0
        while i < len(ex):
            if ex[i] != 1:
                i += 1
                continue
            j = i
            while j + 1 < len(ex) and ex[j + 1] == 1:
                j += 1
            if j - i + 1 >= f7.EPISODE:
                found.append((wk[i], wk[j]))
            i = j + 1
        out[code] = found
    return out


def phase(eps: list, w: pd.Timestamp) -> str:
    if any(s <= w <= e for s, e in eps):
        return "durante"
    if any(0 < (s - w).days // 7 <= 8 for s, e in eps):
        return "antes"
    if any(0 < (w - e).days // 7 <= 13 for s, e in eps):
        return "después"
    return "resto"


def weeks_since(series: dict, eps: dict) -> dict:
    """Eligible weeks elapsed since the end of the last episode, per (code, week); 0 inside an episode."""
    out = {}
    for code, s in series.items():
        inside = {w for a, b in eps[code] for w in s["weeks"] if a <= w <= b}
        last = None
        for k, w in enumerate(s["weeks"]):
            if w in inside:
                out[(code, w)] = 0
                last = k
            else:
                out[(code, w)] = k - last if last is not None else 10**6
    return out


def annotate(rows: pd.DataFrame, series: dict, eps: dict, since: dict) -> pd.DataFrame:
    rows = rows.copy()
    rows["fase"] = [phase(eps[c], w) for c, w in zip(rows["municipality_code"], rows["target_week"])]
    rows["zona_t"] = [series[c]["zone"][t] for c, t in zip(rows["municipality_code"], rows["t"])]
    rows["estrato"] = np.where(rows["zona_t"] == "epidemia", "dentro", "fuera")
    rows["semanas_desde"] = [since[(c, w)] for c, w in zip(rows["municipality_code"], rows["target_week"])]
    rows["ganancia_brier"] = (rows["p_b2"] - rows["y"]) ** 2 - (rows["p_modelo"] - rows["y"]) ** 2
    return rows


# --------------------------------------------------------------------------------------------- metrics

def skill(g: pd.DataFrame) -> dict:
    y = g["y"].to_numpy(float)
    if len(y) == 0:
        return {"filas": 0}
    return {"filas": len(y), "positivas": int(y.sum()), "prevalencia": float(y.mean()),
            "p_medio_modelo": float(g["p_modelo"].mean()), "p_medio_b2": float(g["p_b2"].mean()),
            "mejora_ll_b2": 1 - f7.logloss(y, g["p_modelo"]) / f7.logloss(y, g["p_b2"]),
            "bss_b2": 1 - f7.brier(y, g["p_modelo"]) / f7.brier(y, g["p_b2"])}


def phase_table(rows: pd.DataFrame, periodo: str) -> pd.DataFrame:
    total = rows["ganancia_brier"].sum()
    out = []
    for f in PHASES:
        g = rows[rows["fase"] == f]
        wins = sum(f7.logloss(h["y"], h["p_modelo"]) < f7.logloss(h["y"], h["p_b2"])
                   for _, h in g.groupby("municipality_code"))
        out.append({"periodo": periodo, "rotulo": LABEL[periodo], "fase": f, **skill(g),
                    "ganancia_brier": float(g["ganancia_brier"].sum()),
                    "porcentaje_mejora_neta": float(100 * g["ganancia_brier"].sum() / total),
                    "municipios_ganan_ll": int(wins), "municipios": int(g["municipality_code"].nunique())})
    return pd.DataFrame(out)


def phase_by_municipality(rows: pd.DataFrame, periodo: str) -> pd.DataFrame:
    out = []
    for code in ORDER:
        for f in PHASES:
            g = rows[(rows["municipality_code"] == code) & (rows["fase"] == f)]
            out.append({"periodo": periodo, "rotulo": LABEL[periodo], "municipality_code": code,
                        "municipio": NAMES[code], "fase": f, **skill(g)})
    return pd.DataFrame(out)


def strata_by_municipality(rows: pd.DataFrame, periodo: str) -> pd.DataFrame:
    out = []
    for code in ORDER:
        r = rows[rows["municipality_code"] == code]
        for name, mask in (("fuera", r["estrato"] == "fuera"), ("dentro", r["estrato"] == "dentro"),
                           ("despues_1_13", r["semanas_desde"].between(1, 13))):
            out.append({"periodo": periodo, "rotulo": LABEL[periodo], "municipality_code": code,
                        "municipio": NAMES[code], "estrato": name, **skill(r[mask])})
    return pd.DataFrame(out)


def since_table(rows: pd.DataFrame, periodo: str) -> pd.DataFrame:
    out = []
    for lo, hi, name in TRAMOS:
        g = rows[rows["semanas_desde"].between(lo, hi)]
        out.append({"periodo": periodo, "rotulo": LABEL[periodo], "tramo": name, **skill(g)})
    return pd.DataFrame(out)


def murphy_groups(y, p, labels) -> dict:
    y, p, labels = np.asarray(y, float), np.asarray(p, float), np.asarray(labels)
    n, obar, rel, res = len(y), y.mean(), 0.0, 0.0
    for k in np.unique(labels):
        m = labels == k
        rel += m.sum() * (p[m].mean() - y[m].mean()) ** 2
        res += m.sum() * (y[m].mean() - obar) ** 2
    bs = float(np.mean((p - y) ** 2))
    unc = obar * (1 - obar)
    return {"grupos": int(len(np.unique(labels))), "brier": bs, "confiabilidad": rel / n, "resolucion": res / n,
            "incertidumbre": unc, "residuo": bs - (rel / n - res / n + unc)}


def murphy_table(rows: pd.DataFrame, periodo: str) -> pd.DataFrame:
    y = rows["y"]
    out = [{"sistema": "modelo", "agrupamiento": "10 intervalos de igual frecuencia",
            **murphy_groups(y, rows["p_modelo"], pd.qcut(rows["p_modelo"], 10, labels=False))},
           {"sistema": "B2", "agrupamiento": "un grupo por valor distinto",
            **murphy_groups(y, rows["p_b2"], rows["p_b2"].round(12))},
           {"sistema": "B2", "agrupamiento": "10 intervalos de igual frecuencia, empates unidos",
            **murphy_groups(y, rows["p_b2"], pd.qcut(rows["p_b2"], 10, labels=False, duplicates="drop"))}]
    return pd.DataFrame(out).assign(periodo=periodo, rotulo=LABEL[periodo])


def runs_share(rows: pd.DataFrame, periodo: str) -> pd.DataFrame:
    """Share of positive target weeks that lie in runs of >= 3 consecutive positive target weeks."""
    return _runs_share([g["y"].to_numpy(int) for _, g in rows.sort_values("target_week").groupby("municipality_code")],
                       periodo, LABEL[periodo])


def _runs_share(sequences: list, periodo: str, rotulo: str) -> pd.DataFrame:
    total = in_runs = 0
    for y in sequences:
        i = 0
        while i < len(y):
            if y[i] != 1:
                i += 1
                continue
            j = i
            while j + 1 < len(y) and y[j + 1] == 1:
                j += 1
            total += j - i + 1
            in_runs += (j - i + 1) if j - i + 1 >= 3 else 0
            i = j + 1
    return pd.DataFrame([{"periodo": periodo, "rotulo": rotulo, "positivas": total,
                          "positivas_en_rachas_3": in_runs, "proporcion": in_runs / total}])


def training_episodes(rows: pd.DataFrame, eps: dict, periodo: str) -> pd.DataFrame:
    """Episodes available before the period (ended before its first emission) and episodes inside the period."""
    out = []
    for code in ORDER:
        r = rows[rows["municipality_code"] == code]
        first_t, first_w, last_w = r["t"].min(), r["target_week"].min(), r["target_week"].max()
        before = [(a, b) for a, b in eps[code] if b < first_t]
        inside = [(a, b) for a, b in eps[code] if first_w <= b and a <= last_w]
        out.append({"periodo": periodo, "rotulo": LABEL[periodo], "municipality_code": code, "municipio": NAMES[code],
                    "primera_emision": first_t.date(), "episodios_antes_del_periodo": len(before),
                    "semanas_en_episodio_antes": int(sum((b - a).days // 7 + 1 for a, b in before)),
                    "episodios_en_el_periodo": len(inside)})
    return pd.DataFrame(out)


def ap_grouped_ties(y, p) -> float:
    """Average precision with tied scores grouped in one threshold (the convention of average_precision_score)."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    if y.sum() == 0:
        return float("nan")
    ap, prev_recall = 0.0, 0.0
    for threshold in np.unique(p)[::-1]:
        m = p >= threshold
        recall = y[m].sum() / y.sum()
        ap += (recall - prev_recall) * y[m].mean()
        prev_recall = recall
    return float(ap)


def ap_row_order(y, p) -> float:
    """Average precision with ties broken by row order (stable sort): shows the sensitivity of AP to ties."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    if y.sum() == 0:
        return float("nan")
    y = y[np.argsort(-p, kind="mergesort")]
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def ap_ties(rows: pd.DataFrame, periodo: str, ficha: dict | None) -> pd.DataFrame:
    out = []
    groups = [("todos", rows)] + [(c, rows[rows["municipality_code"] == c]) for c in ORDER]
    for code, g in groups:
        g = g.sort_values(["municipality_code", "t"])
        for s, col in (("modelo", "modelo"), ("b2", "B2")):
            rec = {"periodo": periodo, "rotulo": LABEL[periodo], "grupo": code,
                   "municipio": NAMES.get(code, "todos"), "sistema": col, "valores_distintos": g[f"p_{s}"].nunique(),
                   "ap_empates_agrupados": ap_grouped_ties(g["y"], g[f"p_{s}"]),
                   "ap_orden_de_filas": ap_row_order(g["y"], g[f"p_{s}"])}
            if ficha is not None:
                block = ficha["global"] if code == "todos" else ficha["municipios"][code]
                rec["ap_ficha"] = block["estimacion"]["ap"][col]
            out.append(rec)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ weekly classification of the label

GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))
ZONES = ("seguridad", "alerta", "epidemia")
ZONE_LABEL = {"seguridad": "Seguridad", "alerta": "Alerta", "epidemia": "Epidemia"}
P_BINS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.0000001)


def confusion(y, positive) -> dict:
    """Metrics of a weekly hard classification (contract §7.3 definitions) for any decision vector."""
    y, h = np.asarray(y, int), np.asarray(positive, int)
    tp, fp = int(((h == 1) & (y == 1)).sum()), int(((h == 1) & (y == 0)).sum())
    tn, fn = int(((h == 0) & (y == 0)).sum()), int(((h == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    pi = float(y.mean()) if len(y) else float("nan")
    return {"filas": len(y), "positivas": int(y.sum()), "emisiones_positivas": tp + fp,
            "fraccion_emitida": div(tp + fp, len(y)), "vp": tp, "fp": fp, "vn": tn, "fn": fn,
            "sensibilidad": div(tp, tp + fn), "especificidad": div(tn, tn + fp), "vpp": div(tp, tp + fp),
            "vpn": div(tn, tn + fn), "f1": div(2 * tp, 2 * tp + fp + fn),
            "mcc": (tp * tn - fp * fn) / math.sqrt(den) if den else float("nan"),
            "f1_referencia": 2 * pi / (1 + pi) if len(y) else float("nan")}


def tau_max_f1(rows: pd.DataFrame) -> float:
    """Threshold of the grid with the largest F1 of the model in DEVELOPMENT; ties go to the larger tau (as §7.5).
    It is chosen without looking at the test period and only serves to describe the label at a weekly-optimal point."""
    scores = [(confusion(rows["y"], rows["p_modelo"] >= t)["f1"], t) for t in GRID]
    best = max(f for f, _ in scores)
    return max(t for f, t in scores if f == best)


def classification_tables(rows: pd.DataFrame, periodo: str, tau_f1: float) -> pd.DataFrame:
    """Model and B2 at tau = 0.05 (recipe), tau_f1 (max F1 in development) and 0.50 (no selection), and persistence
    (positive if the channel zone at t is epidemic). Global, per year, per municipality and per transition stratum."""
    thresholds = {"receta_0.05": TAU, "max_f1_desarrollo": tau_f1, "sin_seleccion_0.50": 0.50}
    groups = [("global", "todos", rows)]
    groups += [("anio", str(y), g) for y, g in rows.groupby("anio")]
    groups += [("municipio", c, rows[rows["municipality_code"] == c]) for c in ORDER]
    groups += [("estrato", s, rows[rows["estrato"] == s]) for s in ("fuera", "dentro")]
    out = []
    for nivel, grupo, g in groups:
        base = {"periodo": periodo, "rotulo": LABEL[periodo], "nivel": nivel, "grupo": grupo,
                "municipio": NAMES.get(grupo, "")}
        for name, t in thresholds.items():
            for system, col in (("modelo", "p_modelo"), ("B2", "p_b2")):
                out.append({**base, "sistema": system, "umbral": name, "tau": t, **confusion(g["y"], g[col] >= t)})
        out.append({**base, "sistema": "persistencia", "umbral": "zona epidemia en t", "tau": float("nan"),
                    **confusion(g["y"], g["zona_t"] == "epidemia")})
    return pd.DataFrame(out)


def threshold_curve(rows: pd.DataFrame, periodo: str) -> pd.DataFrame:
    out = []
    for t in GRID:
        for system, col in (("modelo", "p_modelo"), ("B2", "p_b2")):
            out.append({"periodo": periodo, "rotulo": LABEL[periodo], "sistema": system, "tau": t,
                        **confusion(rows["y"], rows[col] >= t)})
    out.append({"periodo": periodo, "rotulo": LABEL[periodo], "sistema": "persistencia", "tau": float("nan"),
                **confusion(rows["y"], rows["zona_t"] == "epidemia")})
    return pd.DataFrame(out)


def auc(y, p) -> float:
    """Area under the ROC curve (Mann-Whitney, ties at half). Descriptive: not a metric of the contract."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(p).rank().to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def zone_summary(rows: pd.DataFrame, periodo: str) -> pd.DataFrame:
    """What the probability adds to the channel zone observed at emission t (information available when emitting)."""
    out = []
    groups = [("todas", rows)] + [(z, rows[rows["zona_t"] == z]) for z in ZONES]
    for zone, g in groups:
        y = g["y"].to_numpy(int)
        rec = {"periodo": periodo, "rotulo": LABEL[periodo], "zona_t": zone, **skill(g),
               "desv_p_modelo": float(g["p_modelo"].std()), "desv_p_b2": float(g["p_b2"].std()),
               "ap_modelo": ap_grouped_ties(y, g["p_modelo"]), "ap_b2": ap_grouped_ties(y, g["p_b2"]),
               "auc_modelo": auc(y, g["p_modelo"]), "auc_b2": auc(y, g["p_b2"])}
        if zone == "todas":
            rec["ap_persistencia"] = ap_grouped_ties(y, (g["zona_t"] == "epidemia").astype(float))
            rec["auc_persistencia"] = auc(y, (g["zona_t"] == "epidemia").astype(float))
        rec["municipios_ganan_ll"] = int(sum(f7.logloss(h["y"], h["p_modelo"]) < f7.logloss(h["y"], h["p_b2"])
                                             for _, h in g.groupby("municipality_code")))
        rec["auc_modelo_mayor_municipios"] = int(sum(auc(h["y"], h["p_modelo"]) > auc(h["y"], h["p_b2"])
                                                     for _, h in g.groupby("municipality_code")
                                                     if 0 < h["y"].sum() < len(h)))
        rec["municipios_con_ambas_clases"] = int(sum(0 < h["y"].sum() < len(h)
                                                     for _, h in g.groupby("municipality_code")))
        out.append(rec)
    return pd.DataFrame(out)


def zone_bins(rows: pd.DataFrame, periodo: str) -> pd.DataFrame:
    """Observed frequency of an epidemic-zone target week by interval of p̂, within each channel zone at t."""
    out = []
    for zone in ZONES:
        g = rows[rows["zona_t"] == zone]
        cut = pd.cut(g["p_modelo"], P_BINS, right=False)
        for interval, h in g.groupby(cut, observed=True):
            out.append({"periodo": periodo, "rotulo": LABEL[periodo], "zona_t": zone,
                        "intervalo_p": f"[{interval.left:.2f}, {min(interval.right, 1.0):.2f})",
                        "p_min": interval.left, "filas": len(h), "positivas": int(h["y"].sum()),
                        "frecuencia_observada": float(h["y"].mean()), "p_medio_modelo": float(h["p_modelo"].mean()),
                        "p_medio_b2": float(h["p_b2"].mean())})
    return pd.DataFrame(out)


READING_BANDS = {"seguridad": (0.0, 0.05, 1.0000001), "alerta": (0.0, 0.05, 0.20, 1.0000001),
                 "epidemia": (0.0, 0.30, 0.70, 1.0000001)}


def zone_reading(rows: pd.DataFrame, periodo: str) -> pd.DataFrame:
    """Coarse bands of p̂ within each channel zone, for a reading guide. The bands are descriptive and were set after
    seeing the finer intervals of zone_bins; they are not thresholds of any rule."""
    out = []
    for zone, edges in READING_BANDS.items():
        g = rows[rows["zona_t"] == zone]
        cut = pd.cut(g["p_modelo"], edges, right=False)
        for interval, h in g.groupby(cut, observed=False):
            out.append({"periodo": periodo, "rotulo": LABEL[periodo], "zona_t": zone,
                        "banda_p": f"[{interval.left:.2f}, {min(interval.right, 1.0):.2f})",
                        "filas": len(h), "fraccion_de_la_zona": len(h) / len(g), "positivas": int(h["y"].sum()),
                        "frecuencia_observada": float(h["y"].mean()) if len(h) else float("nan"),
                        "p_medio_modelo": float(h["p_modelo"].mean()) if len(h) else float("nan"),
                        "p_medio_b2": float(h["p_b2"].mean()) if len(h) else float("nan"),
                        "frecuencia_de_la_zona": float(g["y"].mean())})
    return pd.DataFrame(out)


def pr_points(y, p) -> pd.DataFrame:
    """Precision-recall points with tied scores grouped (one point per distinct value)."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    pts = []
    for threshold in np.unique(p)[::-1]:
        m = p >= threshold
        pts.append({"umbral": float(threshold), "recall": y[m].sum() / y.sum(), "precision": float(y[m].mean())})
    return pd.DataFrame(pts)


# -------------------------------------------------------------------------------- alerts at each onset

def alert_runs(r: pd.DataFrame, series: dict, code: str, system: str) -> tuple[list, int]:
    ts = list(r["t"])
    zones = [series[code]["zone"][t] for t in ts]
    pos = (r["p_modelo"].to_numpy() >= TAU) if system == "modelo" else np.array([z == "alerta" for z in zones])
    runs, i, under = [], 0, 0
    while i < len(ts):
        if not pos[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ts) and pos[j + 1] and ts[j + 1] - ts[j] == WEEK:
            j += 1
        if j - i + 1 >= f7.K:
            runs.append({"tc": ts[i + 1], "fin": ts[j], "semanas_desde_tc": j - i,
                         "continuidad": zones[i + 1] == "epidemia"})
            under += j - i
        i = j + 1
    return runs, under


def onset_detail(rows: pd.DataFrame, series: dict, periodo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    table, _ = f7.events(rows, series, TAU)
    table = table[table["evaluable"]]
    detail, base = [], []
    for system in ("modelo", "canal"):
        weeks = under_total = 0
        runs_by_code = {}
        for code, r in rows.sort_values("t").groupby("municipality_code"):
            runs, under = alert_runs(r, series, code, system)
            runs_by_code[code] = runs
            weeks += len(r)
            under_total += under
        hits = 0
        for _, o in table.iterrows():
            code, s = o["municipality_code"], pd.Timestamp(o["semana"])
            runs = runs_by_code[code]
            warn = [a for a in runs if not a["continuidad"] and s - W * WEEK <= a["tc"] <= s - WEEK]
            cover = [a for a in runs if a["tc"] <= s <= a["fin"]]
            before = [a for a in runs if a["fin"] < s and not a["continuidad"]]
            if warn:
                a, estado = max(warn, key=lambda x: x["tc"]), "avisado"
            elif cover:
                a = cover[0]
                estado = ("en continuidad" if a["continuidad"] else
                          "alerta confirmada más de 12 semanas antes")
            elif before and (s - max(before, key=lambda x: x["fin"])["fin"]).days // 7 <= W:
                a, estado = max(before, key=lambda x: x["fin"]), "alerta apagada antes del inicio"
            else:
                a, estado = None, "sin alerta"
            hits += bool(cover)
            detail.append({"periodo": periodo, "rotulo": LABEL[periodo], "municipality_code": code,
                           "municipio": NAMES[code], "inicio": s.date(), "sistema": system, "estado": estado,
                           "avisado_7_4": o[f"antic_{system}"] is not None and not pd.isna(o[f"antic_{system}"]),
                           "anticipacion": o[f"antic_{system}"], "bajo_alerta_en_inicio": bool(cover),
                           "t_c": a["tc"].date() if a else None, "fin": a["fin"].date() if a else None,
                           "semanas_desde_tc": a["semanas_desde_tc"] if a else None,
                           "semanas_tc_a_inicio": (s - a["tc"]).days // 7 if a else None})
        frac = under_total / weeks
        base.append({"periodo": periodo, "rotulo": LABEL[periodo], "sistema": system, "inicios_evaluables": len(table),
                     "inicios_bajo_alerta": hits, "semanas_emision": weeks, "semanas_bajo_alerta": under_total,
                     "fraccion_bajo_alerta": frac, "esperado_por_tasa_base": frac * len(table)})
    det = pd.DataFrame(detail)
    warned = det.groupby("sistema")["avisado_7_4"].sum()
    if not (warned == (det["estado"] == "avisado").groupby(det["sistema"]).sum()).all():
        f7.fail("onset states disagree with the §7.4 event table")
    return det, pd.DataFrame(base)


# --------------------------------------------------------------------------------- U7.5 cross-check

def cross_check(periods: dict, cells: dict, series: dict, reports: Path) -> pd.DataFrame:
    """Recomputes the U7.5 quantities cited by the document and compares them with the frozen tables."""
    checks = []

    def add(periodo, tabla, clave, independiente, repositorio, tol=1e-9):
        ok = (pd.isna(independiente) and pd.isna(repositorio)) or (
            not pd.isna(independiente) and not pd.isna(repositorio)
            and abs(float(independiente) - float(repositorio)) <= tol)
        checks.append({"periodo": periodo, "tabla": tabla, "clave": clave, "independiente": independiente,
                       "repositorio": repositorio, "coincide": bool(ok)})

    read = {name: pd.read_csv(reports / f"{name}.csv", dtype={"grupo": str, "municipality_code": str})
            for name in ("casillas", "habilidad", "clasificacion_tau", "murphy", "calibracion_anual",
                         "eventos_razon", "duracion_alertas")}
    for periodo, rows in periods.items():
        # casillas: Brier and log-loss of model, B2, B1 per cell
        rep = read["casillas"][read["casillas"]["periodo"] == periodo]
        for _, c in cells[periodo].iterrows():
            rr = rep[(rep["municipality_code"].str.zfill(5) == c["municipality_code"]) & (rep["anio"] == c["anio"])]
            for s, col in (("modelo", "modelo"), ("b2", "B2"), ("b1", "B1")):
                add(periodo, "casillas", f"{c['municipality_code']}-{c['anio']}|brier_{col}",
                    c[f"brier_{s}"], rr[f"brier_{col}"].iloc[0])
                add(periodo, "casillas", f"{c['municipality_code']}-{c['anio']}|logloss_{col}",
                    c[f"ll_{s}"], rr[f"logloss_{col}"].iloc[0])
        # habilidad: wins and both aggregations vs B2 and B1
        rep = read["habilidad"][(read["habilidad"]["periodo"] == periodo) & (read["habilidad"]["sistema"] == "modelo")]
        y = rows["y"]
        for ref in ("b2", "b1"):
            rr = rep[rep["referencia"] == ref.upper()].iloc[0]
            c = cells[periodo]
            add(periodo, "habilidad", f"{ref}|ganadas_brier", int(c[f"gana_brier_{ref}"].sum()), rr["ganadas_brier"])
            add(periodo, "habilidad", f"{ref}|ganadas_logloss", int(c[f"gana_ll_{ref}"].sum()), rr["ganadas_logloss"])
            add(periodo, "habilidad", f"{ref}|bss_filas",
                1 - f7.brier(y, rows["p_modelo"]) / f7.brier(y, rows[f"p_{ref}"]), rr["bss_filas"])
            add(periodo, "habilidad", f"{ref}|bss_casillas", c[f"bss_{ref}"].mean(), rr["bss_casillas"])
            add(periodo, "habilidad", f"{ref}|mejora_logloss_filas",
                1 - f7.logloss(y, rows["p_modelo"]) / f7.logloss(y, rows[f"p_{ref}"]), rr["mejora_logloss_filas"])
            add(periodo, "habilidad", f"{ref}|mejora_logloss_casillas", c[f"mejora_ll_{ref}"].mean(),
                rr["mejora_logloss_casillas"])
        # clasificación en tau: global and per municipality, model and B2
        rep = read["clasificacion_tau"][read["clasificacion_tau"]["periodo"] == periodo]
        groups = [("global", "todos", rows)] + [("municipio", c, g) for c, g in rows.groupby("municipality_code")]
        for nivel, grupo, g in groups:
            for s, col in (("modelo", "modelo"), ("b2", "B2")):
                yy = g["y"].to_numpy(int)
                yh = (g[f"p_{s}"].to_numpy() >= TAU).astype(int)
                tp, fp = int(((yh == 1) & (yy == 1)).sum()), int(((yh == 1) & (yy == 0)).sum())
                tn, fn = int(((yh == 0) & (yy == 0)).sum()), int(((yh == 0) & (yy == 1)).sum())
                den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
                mcc = (tp * tn - fp * fn) / math.sqrt(den) if den else float("nan")
                width = 5 if nivel == "municipio" else 0
                rr = rep[(rep["sistema"] == col) & (rep["nivel"] == nivel) & (rep["grupo"].str.zfill(width) == grupo)]
                for k, v in (("vp", tp), ("fp", fp), ("vn", tn), ("fn", fn), ("mcc", mcc)):
                    add(periodo, "clasificacion_tau", f"{col}|{nivel}|{grupo}|{k}", v, rr[k].iloc[0])
        # Murphy of the model, global
        rep = read["murphy"]
        rr = rep[(rep["periodo"] == periodo) & (rep["sistema"] == "modelo") & (rep["nivel"] == "global")].iloc[0]
        m = murphy_groups(rows["y"], rows["p_modelo"], pd.qcut(rows["p_modelo"], 10, labels=False))
        for k in ("brier", "confiabilidad", "resolucion", "incertidumbre", "residuo"):
            add(periodo, "murphy", f"modelo|global|{k}", m[k], rr[k])
        # calibration in the large per year (model, B2)
        rep = read["calibracion_anual"][read["calibracion_anual"]["periodo"] == periodo]
        for year, g in rows.groupby("anio"):
            rr = rep[rep["anio"].astype(str) == str(year)].iloc[0]
            add(periodo, "calibracion_anual", f"{year}|calibracion_modelo", g["p_modelo"].mean() - g["y"].mean(),
                rr["calibracion_modelo"])
            add(periodo, "calibracion_anual", f"{year}|calibracion_B2", g["p_b2"].mean() - g["y"].mean(),
                rr["calibracion_B2"])
        # events (§7.4), global
        _, tot = f7.events(rows, series, TAU)
        rep = read["eventos_razon"]
        rep = rep[(rep["periodo"] == periodo) & (rep["nivel"] == "global")]
        for s in ("modelo", "canal"):
            rr = rep[rep["sistema"] == s].iloc[0]
            for mine, theirs in (("inicios_evaluables", "inicios_evaluables"), ("inicios_avisados", "inicios_avisados"),
                                 ("alertas", "alertas"), ("continuidad", "alertas_continuidad"),
                                 ("juzgables", "alertas_juzgables"), ("falsas_alarmas", "falsas_alarmas"),
                                 ("fa_seguidas_de_epidemia", "falsas_seguidas_epidemia"),
                                 ("entradas_evaluables", "entradas_evaluables"),
                                 ("entradas_avisadas", "entradas_avisadas")):
                add(periodo, "eventos_razon", f"{s}|{theirs}", tot[s][mine], rr[theirs])
        # share of weeks under alert (duration table)
        rep = read["duracion_alertas"][(read["duracion_alertas"]["periodo"] == periodo)
                                       & (read["duracion_alertas"]["nivel"] == "global")]
        for s in ("modelo", "canal"):
            under = sum(alert_runs(r, series, c, s)[1] for c, r in rows.sort_values("t").groupby("municipality_code"))
            add(periodo, "duracion_alertas", f"{s}|semanas_bajo_alerta", under,
                rep[rep["sistema"] == s]["semanas_bajo_alerta"].iloc[0])
    return pd.DataFrame(checks)


# ---------------------------------------------------------------------------------------------- figures

def save(fig, folder: Path, name: str) -> list:
    paths = []
    for suffix, metadata in (("png", {"Software": None}), ("svg", {"Date": None, "Creator": None})):
        path = folder / f"{name}.{suffix}"
        if path.exists():
            f7.fail(f"{path} exists and is never overwritten")
        fig.savefig(path, dpi=200 if suffix == "png" else None, bbox_inches="tight", metadata=metadata)
        paths.append(path)
    plt.close(fig)
    return paths


def figure_phase_share(phases: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    for ax, periodo in zip(axes, PERIODS):
        t = phases[phases["periodo"] == periodo].set_index("fase").loc[list(PHASES)]
        ypos = np.arange(len(PHASES))[::-1]
        colors = [BLUE if v >= 0 else RED for v in t["porcentaje_mejora_neta"]]
        ax.barh(ypos, t["porcentaje_mejora_neta"], color=colors, height=0.55, edgecolor=SURFACE, linewidth=2)
        ax.axvline(0, color=INK_2, linewidth=0.8)
        for yv, v in zip(ypos, t["porcentaje_mejora_neta"]):
            ax.text(v + (3 if v >= 0 else -3), yv, f"{v:+.0f} %", va="center", ha="left" if v >= 0 else "right",
                    color=INK, fontsize=9)
        ax.set_yticks(ypos, [f"{PHASE_LABEL[f]}\n{n} semanas objetivo" for f, n in zip(PHASES, t["filas"])])
        if periodo == "prueba":
            ax.tick_params(axis="y", labelleft=True)
        ax.set_xlim(-80, 165)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Porcentaje de la mejora neta del Brier frente a B2")
        ax.set_title(PERIOD_TITLE[periodo])
    fig.suptitle("Aporte de cada fase de la semana objetivo a la mejora del Brier frente a B2 [X]", y=1.02)
    fig.text(0.5, -0.06, "Fases definidas con información posterior a la emisión (estrato de diagnóstico). "
             "Los porcentajes suman 100 % y pueden ser negativos.", ha="center", color=INK_2, fontsize=8)
    fig.subplots_adjust(wspace=0.55)
    return fig


def figure_phase_map(by_mun: pd.DataFrame):
    cmap = LinearSegmentedColormap.from_list("bss", [RED, NEUTRAL, BLUE])
    norm = TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    for ax, periodo in zip(axes, PERIODS):
        t = by_mun[by_mun["periodo"] == periodo]
        grid = np.full((len(ORDER), len(PHASES)), np.nan)
        for i, code in enumerate(ORDER):
            for j, f in enumerate(PHASES):
                v = t[(t["municipality_code"] == code) & (t["fase"] == f)]["mejora_ll_b2"]
                grid[i, j] = v.iloc[0] if len(v) and not pd.isna(v.iloc[0]) else np.nan
        ax.imshow(np.clip(grid, -0.6, 0.6), cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(ORDER)):
            for j in range(len(PHASES)):
                n = t[(t["municipality_code"] == ORDER[i]) & (t["fase"] == PHASES[j])]["filas"].iloc[0]
                txt = "—" if np.isnan(grid[i, j]) else f"{grid[i, j]:+.2f}".replace(".", ",")
                ax.text(j, i - 0.12, txt, ha="center", va="center", fontsize=9,
                        color=INK if np.isnan(grid[i, j]) or abs(grid[i, j]) < 0.35 else "#ffffff")
                ax.text(j, i + 0.25, f"n={n}", ha="center", va="center", fontsize=7,
                        color=INK_2 if np.isnan(grid[i, j]) or abs(grid[i, j]) < 0.35 else "#ffffff")
        ax.set_xticks(range(len(PHASES)), ["Antes", "Durante", "Después", "Resto"])
        ax.set_yticks(range(len(ORDER)), [NAMES[c] for c in ORDER])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(PERIOD_TITLE[periodo])
    fig.suptitle("Mejora relativa de log-loss frente a B2 por municipio y fase [X]", y=1.0)
    fig.text(0.5, -0.04, "Azul: el modelo mejora a B2; rojo: B2 mejora al modelo (escala recortada en ±0,6). "
             "n = semanas objetivo de la casilla municipio-fase.", ha="center", color=INK_2, fontsize=8)
    fig.subplots_adjust(wspace=0.08)
    return fig


def figure_since(since: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), sharey=True)
    for ax, periodo in zip(axes, PERIODS):
        t = since[since["periodo"] == periodo].reset_index(drop=True)
        x = np.arange(len(t))
        ax.plot(x, t["prevalencia"], color=INK, marker="s", markersize=6, linewidth=1.4, label="Prevalencia observada")
        ax.plot(x, t["p_medio_modelo"], color=BLUE, marker="o", markersize=7, markeredgecolor=SURFACE,
                label=r"$\hat{p}$ medio del modelo")
        ax.plot(x, t["p_medio_b2"], color=MUTED, marker="D", markersize=6, linestyle=":", label="B2 medio")
        for xi, v, n in zip(x, t["mejora_ll_b2"], t["filas"]):
            ax.text(xi, 1.02, f"{100 * v:+.0f} %", ha="center", va="bottom", fontsize=8, color=BLUE if v >= 0 else RED,
                    transform=ax.get_xaxis_transform())
        ax.set_xticks(x, [f"{name}\nn={n}" for name, n in zip(t["tramo"], t["filas"])])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Semanas desde el fin del último episodio (semana objetivo)")
        ax.set_title(PERIOD_TITLE[periodo], pad=18)
    axes[0].set_ylabel("Probabilidad o frecuencia")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Probabilidad media frente a prevalencia según el tiempo desde el último episodio [X]", y=1.05)
    fig.text(0.5, -0.12, "Sobre cada tramo: mejora relativa de log-loss del modelo frente a B2.",
             ha="center", color=INK_2, fontsize=8)
    return fig


def figure_onsets(detail: pd.DataFrame, rows: pd.DataFrame, series: dict):
    ev = detail[detail["sistema"] == "modelo"][["municipality_code", "inicio"]].drop_duplicates()
    ev = ev.sort_values(["inicio", "municipality_code"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.4, 0.42 * len(ev) + 1.4))
    lo, hi = -30, 6
    for i, o in ev.iterrows():
        code, s = o["municipality_code"], pd.Timestamp(o["inicio"])
        r = rows[rows["municipality_code"] == code].sort_values("t")
        yv = len(ev) - 1 - i
        for k, (system, color, dy) in enumerate((("modelo", BLUE, 0.16), ("canal", ORANGE, -0.16))):
            runs, _ = alert_runs(r, series, code, system)
            for a in runs:
                x0, x1 = (a["tc"] - s).days / 7, (a["fin"] - s).days / 7 + 1
                if x1 < lo or x0 > hi:
                    continue
                ax.barh(yv + dy, min(x1, hi) - max(x0, lo), left=max(x0, lo), height=0.26, color=color,
                        alpha=0.45 if a["continuidad"] else 1.0, edgecolor=SURFACE, linewidth=1)
        state = detail[(detail["municipality_code"] == code) & (detail["inicio"] == o["inicio"])]
        sm = state[state["sistema"] == "modelo"].iloc[0]
        sc = state[state["sistema"] == "canal"].iloc[0]
        ax.text(hi + 0.6, yv, f"modelo: {'avisa' if sm['avisado_7_4'] else 'no avisa'} · "
                f"canal: {'avisa' if sc['avisado_7_4'] else 'no avisa'}", va="center", fontsize=8, color=INK_2)
    ax.axvspan(-W, 0, color=NEUTRAL, zorder=0)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(range(len(ev))[::-1], [f"{NAMES[c]} {d}" for c, d in zip(ev["municipality_code"], ev["inicio"])])
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Semanas relativas al inicio de brote (0 = semana del inicio)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE),
               plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.45), plt.Rectangle((0, 0), 1, 1, color=NEUTRAL)]
    ax.legend(handles, ["Alerta del modelo (de t_c al fin)", "Alerta del canal (de t_c al fin)",
                        "Alerta en continuidad (no avisa)", "Ventana de aviso de §7.4 (12 semanas)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=8)
    ax.set_title("Alertas alrededor de los 10 inicios de brote evaluables de prueba [X]")
    return fig


def figure_regimes(cells: dict):
    regs = ("epidemica", "post-epidemica", "tranquila")
    reg_label = {"epidemica": "Epidémica", "post-epidemica": "Post-epidémica", "tranquila": "Tranquila"}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    rng = np.random.default_rng(20261112)
    for ax, (col, xlabel) in zip(axes, (("mejora_ll_b2", "Mejora relativa de log-loss frente a B2"),
                                         ("dif_calib", "Calibración en grande: modelo − B2"))):
        for k, periodo in enumerate(PERIODS):
            c = cells[periodo].copy()
            c["dif_calib"] = c["calib_modelo"] - c["calib_b2"]
            for j, reg in enumerate(regs):
                v = c[c["regimen"] == reg][col].to_numpy()
                yv = len(regs) - 1 - j + (0.15 if periodo == "prueba" else -0.15)
                jitter = rng.uniform(-0.05, 0.05, len(v))
                filled = periodo == "prueba"
                ax.scatter(np.clip(v, -1, 1), yv + jitter, s=34, color=BLUE if filled else SURFACE,
                           edgecolor=BLUE, linewidth=1.4, zorder=3,
                           label=("Prueba" if filled else "Desarrollo (usado para elegir)") if j == 0 else None)
                if len(v):
                    ax.text(1.02, yv, f"n={len(v)}", transform=ax.get_yaxis_transform(), va="center", fontsize=7,
                            color=MUTED)
        ax.axvline(0, color=INK_2, linewidth=0.8)
        ax.set_yticks(range(len(regs))[::-1], [reg_label[r] for r in regs])
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(-1.05, 1.05)
    axes[0].set_title("Habilidad por casilla (municipio-año)")
    axes[1].set_title("Sobreestimación relativa a B2 por casilla")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Casillas por régimen (estrato de diagnóstico de §8.8)", y=1.02)
    fig.text(0.5, -0.05, "Valores de log-loss recortados en ±1. A la derecha de cero: el modelo mejora a B2 "
             "(izquierda) "
             "o sobreestima más que B2 (derecha).", ha="center", color=INK_2, fontsize=8)
    fig.subplots_adjust(wspace=0.35)
    return fig


def _comma(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def figure_pr(periods: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    for ax, periodo in zip(axes, PERIODS):
        r = periods[periodo]
        y = r["y"].to_numpy(int)
        model, b2 = pr_points(y, r["p_modelo"]), pr_points(y, r["p_b2"])
        pers = confusion(y, r["zona_t"] == "epidemia")
        ax.axhline(y.mean(), color=AXIS, linestyle="--", linewidth=1,
                   label=f"Sin información (prevalencia {_comma(y.mean())})")
        ax.step(model["recall"], model["precision"], where="post", color=BLUE,
                label=f"Modelo (AP {_comma(ap_grouped_ties(y, r['p_modelo']))})")
        ax.plot(b2["recall"], b2["precision"], color=MUTED, linestyle=":", marker="D", markersize=5,
                label=f"B2 (AP {_comma(ap_grouped_ties(y, r['p_b2']))}; un punto por valor distinto)")
        ax.plot(pers["sensibilidad"], pers["vpp"], marker="s", markersize=9, color=INK, linestyle="none",
                markeredgecolor=SURFACE, label="Persistencia (zona epidémica en t)")
        ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="Recall (sensibilidad)", title=PERIOD_TITLE[periodo])
        ax.legend(loc="lower left", fontsize=7.5)
    axes[0].set_ylabel("Precisión (VPP)")
    fig.suptitle("Curva precisión-recall de la etiqueta semanal y = 1[C > U1e] a 8 semanas [X]", y=1.0)
    return fig


def figure_thresholds(curve: pd.DataFrame, tau_f1: float):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.2), sharex=True, sharey="row")
    for j, periodo in enumerate(PERIODS):
        c = curve[curve["periodo"] == periodo]
        for i, (metric, label) in enumerate((("f1", "F1"), ("mcc", "MCC"))):
            ax = axes[i, j]
            m, b = c[c["sistema"] == "modelo"], c[c["sistema"] == "B2"]
            pers = c[c["sistema"] == "persistencia"][metric].iloc[0]
            ax.plot(m["tau"], m[metric], color=BLUE, marker="o", markersize=4, label="Modelo")
            ax.plot(b["tau"], b[metric], color=MUTED, linestyle=":", marker="D", markersize=4, label="B2")
            ax.axhline(pers, color=INK, linestyle="--", linewidth=1.2, label="Persistencia")
            if metric == "f1":
                ax.axhline(c["f1_referencia"].iloc[0], color=AXIS, linewidth=1, label="Marcar todo")
            ax.axvline(TAU, color=ORANGE, linewidth=1)
            ax.axvline(tau_f1, color=INK_2, linewidth=1, linestyle="-.")
            ax.set_ylim(0, 0.85)
            if i == 0:
                ax.set_title(PERIOD_TITLE[periodo])
            if j == 0:
                ax.set_ylabel(label)
            if i == 1:
                ax.set_xlabel("Umbral τ")
    axes[0, 0].legend(loc="lower left", fontsize=8, ncol=2)
    axes[0, 1].text(TAU + 0.01, 0.05, "τ de la receta", color=ORANGE, fontsize=8)
    axes[0, 1].text(tau_f1 + 0.01, 0.05, "τ de F1 máximo en desarrollo", color=INK_2, fontsize=8)
    fig.suptitle("Clasificación semanal de la etiqueta según el umbral [X]", y=1.0)
    fig.text(0.5, -0.02, "B2 toma pocos valores: entre τ ≈ 0,25 y 0,50 coincide con la persistencia en prueba. "
             "La curva es descriptiva y no elige τ.", ha="center", color=INK_2, fontsize=8)
    return fig


def figure_zones(bins: pd.DataFrame, summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.2), sharey=True)
    for ax, zone in zip(axes, ZONES):
        ax.plot([0, 1], [0, 1], color=AXIS, linestyle="--", linewidth=1)
        for periodo in PERIODS:
            b = bins[(bins["periodo"] == periodo) & (bins["zona_t"] == zone) & (bins["filas"] >= 10)]
            s = summary[(summary["periodo"] == periodo) & (summary["zona_t"] == zone)].iloc[0]
            filled = periodo == "prueba"
            ax.plot(b["p_medio_modelo"], b["frecuencia_observada"], color=BLUE, linewidth=1 if filled else 0.8,
                    linestyle="-" if filled else ":")
            ax.scatter(b["p_medio_modelo"], b["frecuencia_observada"], s=12 + 1.6 * b["filas"],
                       color=BLUE if filled else SURFACE, edgecolor=BLUE, linewidth=1.3, zorder=3,
                       label=f"Modelo por intervalo de p̂, {periodo}")
            ax.scatter([s["p_medio_b2"]], [s["prevalencia"]], s=90, marker="D",
                       color=MUTED if filled else SURFACE, edgecolor=MUTED, linewidth=1.5, zorder=4,
                       label=f"B2, media de la zona, {periodo}")
        st = summary[(summary["periodo"] == "prueba") & (summary["zona_t"] == zone)].iloc[0]
        sd = summary[(summary["periodo"] == "desarrollo") & (summary["zona_t"] == zone)].iloc[0]
        txt = (f"Prueba: {st['filas']} semanas, {int(st['positivas'])} positivas\n"
               f"Desarrollo: {sd['filas']} semanas, {int(sd['positivas'])} positivas")
        if not pd.isna(st["auc_modelo"]):
            txt += f"\nAUC prueba: modelo {_comma(st['auc_modelo'])}, B2 {_comma(st['auc_b2'])}"
        corner = (0.97, 0.03, "right", "bottom") if zone == "epidemia" else (0.03, 0.97, "left", "top")
        ax.text(corner[0], corner[1], txt, transform=ax.transAxes, ha=corner[2], va=corner[3], fontsize=7.5,
                color=INK_2)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=r"$\hat{p}$ medio del intervalo",
               title=f"Zona del canal en t: {ZONE_LABEL[zone]}")
        ax.set_aspect("equal")
    axes[0].set_ylabel("Frecuencia observada de zona epidémica en t + 8")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8)
    fig.suptitle("Lo que la probabilidad añade al estado actual del canal [X]", y=1.0)
    fig.text(0.5, -0.16, "Intervalos de p̂: [0; 0,05), [0,05; 0,10), [0,10; 0,20), [0,20; 0,30), [0,30; 0,50), "
             "[0,50; 0,70), [0,70; 0,90), [0,90; 1]. Solo intervalos con ≥ 10 semanas; "
             "tamaño proporcional a las semanas.",
             ha="center", color=INK_2, fontsize=8)
    return fig


# ------------------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--canal", type=Path, required=True)
    ap.add_argument("--busqueda-sin-clima", type=Path, required=True)
    ap.add_argument("--predicciones-prueba", type=Path, required=True)
    ap.add_argument("--evidencia", type=Path, required=True)
    ap.add_argument("--salida-tablas", type=Path, required=True)
    ap.add_argument("--salida-figuras", type=Path, required=True)
    a = ap.parse_args(argv)
    for folder in (a.salida_tablas, a.salida_figuras):
        folder.mkdir(parents=True, exist_ok=True)

    series = f7.observed_series(a.canal)
    eps = episodes(series)
    since = weeks_since(series, eps)
    regime = f7.regimes(a.canal)
    regime["municipality_code"] = regime["municipality_code"].astype(str).str.zfill(5)
    periods = {"desarrollo": load_development(a.busqueda_sin_clima),
               "prueba": load_test(a.predicciones_prueba, a.evidencia / "prueba_20261112" / "diag_serie.csv")}
    periods = {k: annotate(v, series, eps, since) for k, v in periods.items()}
    cells = {k: f7.cell_table(v, regime) for k, v in periods.items()}

    tables = {
        "fases": pd.concat([phase_table(v, k) for k, v in periods.items()]),
        "fases_municipio": pd.concat([phase_by_municipality(v, k) for k, v in periods.items()]),
        "estratos_municipio": pd.concat([strata_by_municipality(v, k) for k, v in periods.items()]),
        "semanas_desde_episodio": pd.concat([since_table(v, k) for k, v in periods.items()]),
        "murphy_b2": pd.concat([murphy_table(v, k) for k, v in periods.items()]),
        "zona_canal": pd.concat([zone_summary(v, k) for k, v in periods.items()]),
        "zona_canal_intervalos": pd.concat([zone_bins(v, k) for k, v in periods.items()]),
        "zona_canal_lectura": pd.concat([zone_reading(v, k) for k, v in periods.items()]),
        "episodios_entrenamiento": pd.concat([training_episodes(v, eps, k) for k, v in periods.items()]),
        "rachas_positivas": pd.concat([runs_share(v, k) for k, v in periods.items()] + [_runs_share(
            [np.asarray(s["exceed"], int) for s in series.values()], "serie_elegible_completa", "descriptivo")]),
    }
    tau_f1 = tau_max_f1(periods["desarrollo"])
    tables["clasificacion_umbrales"] = pd.concat([classification_tables(v, k, tau_f1) for k, v in periods.items()])
    tables["clasificacion_curva_tau"] = pd.concat([threshold_curve(v, k) for k, v in periods.items()])
    ficha_path = a.evidencia / "prueba_20261112" / "ficha_desempeno.json"
    ficha = json.loads(ficha_path.read_text(encoding="utf-8"))
    tables["ap_empates"] = pd.concat([ap_ties(periods["desarrollo"], "desarrollo", None),
                                      ap_ties(periods["prueba"], "prueba", ficha)])
    t = tables["ap_empates"].query("periodo == 'prueba'")
    if (abs(t["ap_empates_agrupados"] - t["ap_ficha"]) > 1e-9).any():
        f7.fail("grouped-ties AP does not reproduce the AP of the test card")
    details, bases = zip(*(onset_detail(v, series, k) for k, v in periods.items()))
    tables["inicios_detalle"] = pd.concat(details)
    tables["tasa_base_alerta"] = pd.concat(bases)
    check = cross_check(periods, cells, series, a.evidencia / "reportes_20261112")
    tables["contraste_u75"] = check

    written = []
    for name, frame in tables.items():
        path = a.salida_tablas / f"{name}.csv"
        if path.exists():
            f7.fail(f"{path} exists and is never overwritten")
        frame.to_csv(path, index=False, encoding="utf-8")
        written.append(path)
    figures = {"fases_aporte": figure_phase_share(tables["fases"]),
               "fases_municipio": figure_phase_map(tables["fases_municipio"]),
               "semanas_desde_episodio": figure_since(tables["semanas_desde_episodio"]),
               "inicios_prueba": figure_onsets(tables["inicios_detalle"].query("periodo == 'prueba'"),
                                               periods["prueba"], series),
               "regimen_casillas": figure_regimes(cells),
               "clasificacion_pr": figure_pr(periods),
               "clasificacion_umbral": figure_thresholds(tables["clasificacion_curva_tau"], tau_f1),
               "zona_canal": figure_zones(tables["zona_canal_intervalos"], tables["zona_canal"])}
    for name, fig in figures.items():
        written += save(fig, a.salida_figuras, name)

    failed = check[~check["coincide"]]

    def md5(path) -> str:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()

    inputs = {"canal": a.canal, "predicciones_prueba": a.predicciones_prueba,
              "diag_serie_prueba": a.evidencia / "prueba_20261112" / "diag_serie.csv"}
    inputs.update({f"sin_clima/{n}": a.busqueda_sin_clima / n for n in
                   ("tabla_unica_desarrollo.parquet", "predicciones_desarrollo.parquet",
                    "predicciones_lineas_base_desarrollo.parquet")})
    manifest = {"script": "docs/forecasting/verificacion/analisis_integrado_exploratorio.py", "tau": TAU,
                "tau_max_f1_desarrollo": tau_f1,
                "insumos_md5": {k: md5(v) for k, v in inputs.items()},
                "comprobaciones_u75": int(len(check)), "coinciden": int(check["coincide"].sum()),
                "no_coinciden": int(len(failed)),
                "salidas": [p.as_posix() for p in written]}
    path = a.salida_tablas / "manifiesto_exploratorio.json"
    if path.exists():
        f7.fail(f"{path} exists and is never overwritten")
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(f"== tau de F1 máximo en desarrollo: {tau_f1}")
    for name in ("fases", "semanas_desde_episodio", "murphy_b2", "rachas_positivas", "tasa_base_alerta", "zona_canal",
                 "zona_canal_lectura"):
        print(f"== {name}")
        print(tables[name].round(4).to_string(index=False))
    print("== inicios (prueba)")
    detail = tables["inicios_detalle"].query("periodo == 'prueba'").drop(columns=["periodo", "rotulo"])
    print(detail.to_string(index=False))
    print(f"== contraste U7.5: {manifest['coinciden']}/{manifest['comprobaciones_u75']} coinciden; "
          f"no coinciden: {manifest['no_coinciden']}")
    if len(failed):
        print(failed.to_string(index=False))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
