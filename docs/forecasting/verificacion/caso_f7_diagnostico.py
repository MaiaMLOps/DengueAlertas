"""caso_f7_diagnostico.py: independent check of the integrated diagnosis of contract §8.8 (D9.1).

Written in the chat «Desarrollo guiado con Claude Code» on 2026-11-12. It imports nothing from src/: it
rebuilds from the contract the regime classification, the cell table and the summaries of §8.8, so that
diagnostics.py (F7) can be compared with it before its literals are frozen.

Usage, development (configuration chosen out of year, common, of the recipe's family):

    python -X utf8 -I caso_f7_diagnostico.py \
        --canal data/processed/epidemiology/pilot_weekly_channel.parquet \
        --busqueda data/processed/forecasting --familia logistica --csv diagnostico_dev.csv

Usage, test (after the single run of F7): pass the test predictions with --predicciones-prueba, a parquet
with columns municipality_code, t, target_week, anio, y, p_modelo, p_b2, p_b1 (names configurable below).

Definitions (contract §8.8):
  episode       run of >= 3 consecutive eligible weeks with target_u1e = 1, on the whole eligible series;
  epidemic      a (municipality, calendar year) with at least one week of the year inside an episode;
  post-epidemic not epidemic, and the previous calendar year is epidemic;
  quiet         the rest.
  The year of a cell is the calendar year of its target weeks. Regime is a diagnosis stratum only.
Cell metrics: log-loss (natural log, clip 1e-15), Brier, mean p, calibration in the large = mean p -
prevalence (signed); absolute differences model - reference (negative = model better); relative
improvement 1 - L_model / L_ref; a win is a strictly smaller loss.
Summaries by municipality, year and regime, with the two aggregations of §8.8: mean of per-cell relative
improvements, and the metric over the pooled rows.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-15
EPISODE = 3
DEV_YEARS = (2019, 2020, 2021, 2022)
NAMES = {"08001": "Barranquilla", "08758": "Soledad", "41001": "Neiva", "50001": "Villavicencio",
         "54001": "Cúcuta", "76001": "Cali"}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


# --------------------------------------------------------------------------------------------- regime

def regimes(channel_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(channel_path, columns=["municipality_code", "week_start", "channel_eligible", "target_u1e"])
    frame = frame[frame["channel_eligible"].astype(bool)].copy()
    frame["week"] = pd.to_datetime(frame["week_start"])
    records = []
    for code, group in frame.groupby("municipality_code"):
        group = group.sort_values("week")
        weeks = list(group["week"])
        if any((b - a).days != 7 for a, b in zip(weeks, weeks[1:])):
            fail(f"{code}: eligible weeks are not consecutive")
        exceed = group["target_u1e"].astype(int).to_numpy()
        in_episode = np.zeros(len(exceed), bool)
        i = 0
        while i < len(exceed):
            if exceed[i] != 1:
                i += 1
                continue
            j = i
            while j + 1 < len(exceed) and exceed[j + 1] == 1:
                j += 1
            if j - i + 1 >= EPISODE:
                in_episode[i:j + 1] = True
            i = j + 1
        years = np.array([w.year for w in weeks])
        epidemic = {int(y): bool(in_episode[years == y].any()) for y in sorted(set(years))}
        for year in sorted(epidemic):
            if epidemic[year]:
                regime = "epidemica"
            elif epidemic.get(year - 1, False):
                regime = "post-epidemica"
            else:
                regime = "tranquila"
            records.append({"municipality_code": str(code), "anio": year, "regimen": regime,
                            "episodio_en_anio": epidemic[year]})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------- out-of-year common configuration

def choose(cells: pd.DataFrame) -> str:
    scores = []
    for (name, position), group in cells.groupby(["candidato", "orden"]):
        per_code = [math.fsum(g["mejora_b2"]) / len(g) for _, g in group.groupby("municipality_code")]
        scores.append((math.fsum(per_code) / len(per_code), int(position), str(name)))
    best = max(score for score, _, _ in scores)
    return sorted((position, name) for score, position, name in scores if score == best)[0][1]


def development_predictions(search: Path, family: str) -> pd.DataFrame:
    table = pd.read_parquet(search / "tabla_unica_desarrollo.parquet")
    table = table[table["familia"] == family]
    predictions = pd.read_parquet(search / "predicciones_desarrollo.parquet")
    baselines = pd.read_parquet(search / "predicciones_lineas_base_desarrollo.parquet")
    pieces = []
    for year in DEV_YEARS:
        candidate = choose(table[table["anio"] != year])
        model = predictions[(predictions["familia"] == family) & (predictions["candidato"] == candidate)
                            & (predictions["anio"] == year)]
        pieces.append(model.assign(configuracion=candidate))
    model = pd.concat(pieces)[["municipality_code", "t", "target_week", "anio", "y", "p", "configuracion"]]
    model = model.rename(columns={"p": "p_modelo"})
    for name in ("B2", "B1"):
        ref = baselines[baselines["familia"] == name][["municipality_code", "t", "p"]].rename(columns={"p": f"p_{name.lower()}"})
        model = model.merge(ref, on=["municipality_code", "t"], how="left", validate="one_to_one")
    if model[["p_b2", "p_b1"]].isna().any().any() or len(model) != 1248:
        fail(f"development rows do not align: {len(model)} rows")
    return model


# ------------------------------------------------------------------------------------------- metrics

def logloss(y, p) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def cell_table(rows: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (code, year), g in rows.groupby(["municipality_code", "anio"]):
        y = g["y"].to_numpy(float)
        rec = {"municipality_code": str(code), "municipio": NAMES.get(str(code), str(code)), "anio": int(year),
               "filas": len(g), "positivas": int(y.sum()), "prevalencia": float(y.mean())}
        for s in ("modelo", "b2", "b1"):
            p = g[f"p_{s}"].to_numpy(float)
            rec[f"p_medio_{s}"] = float(p.mean())
            rec[f"calib_{s}"] = float(p.mean() - y.mean())
            rec[f"ll_{s}"] = logloss(y, p)
            rec[f"brier_{s}"] = brier(y, p)
        for ref in ("b2", "b1"):
            rec[f"dif_ll_{ref}"] = rec["ll_modelo"] - rec[f"ll_{ref}"]
            rec[f"dif_brier_{ref}"] = rec["brier_modelo"] - rec[f"brier_{ref}"]
            rec[f"mejora_ll_{ref}"] = 1 - rec["ll_modelo"] / rec[f"ll_{ref}"]
            rec[f"bss_{ref}"] = 1 - rec["brier_modelo"] / rec[f"brier_{ref}"]
            rec[f"gana_ll_{ref}"] = rec["ll_modelo"] < rec[f"ll_{ref}"]
            rec[f"gana_brier_{ref}"] = rec["brier_modelo"] < rec[f"brier_{ref}"]
        records.append(rec)
    cells = pd.DataFrame(records).merge(regime, on=["municipality_code", "anio"], how="left", validate="one_to_one")
    if cells["regimen"].isna().any():
        fail("some cells have no regime")
    return cells


def summary(rows: pd.DataFrame, cells: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = rows.merge(cells[["municipality_code", "anio", "regimen", "municipio"]], on=["municipality_code", "anio"])
    out = []
    for value, c in cells.groupby(key, sort=False):
        r = rows[rows[key] == value]
        y = r["y"].to_numpy(float)
        rec = {key: value, "casillas": len(c), "filas": len(r), "positivas": int(y.sum())}
        for ref in ("b2", "b1"):
            rec[f"gana_ll_{ref}"] = int(c[f"gana_ll_{ref}"].sum())
            rec[f"gana_brier_{ref}"] = int(c[f"gana_brier_{ref}"].sum())
            rec[f"mejora_ll_{ref}_promedio_casillas"] = float(c[f"mejora_ll_{ref}"].mean())
            rec[f"bss_{ref}_promedio_casillas"] = float(c[f"bss_{ref}"].mean())
            rec[f"mejora_ll_{ref}_filas_juntas"] = 1 - logloss(y, r["p_modelo"]) / logloss(y, r[f"p_{ref}"])
            rec[f"bss_{ref}_filas_juntas"] = 1 - brier(y, r["p_modelo"]) / brier(y, r[f"p_{ref}"])
            rec[f"dif_ll_{ref}_media"] = float(c[f"dif_ll_{ref}"].mean())
        rec["calib_modelo_media"] = float(c["calib_modelo"].mean())
        rec["calib_b2_media"] = float(c["calib_b2"].mean())
        out.append(rec)
    return pd.DataFrame(out)


def hypothesis_r(cells: pd.DataFrame) -> dict:
    result = {}
    for part, regime, column, wins_needed in (("a", "epidemica", "gana_ll_b2", True),
                                              ("b", "post-epidemica", "gana_ll_b2", False)):
        c = cells[cells["regimen"] == regime]
        if len(c) < 2:
            result[part] = {"estado": "no evaluable", "casillas": len(c)}
            continue
        wins = int(c[column].sum())
        holds = wins > len(c) / 2 if wins_needed else (len(c) - wins) > len(c) / 2
        result[part] = {"estado": "sostenida" if holds else "no sostenida", "casillas": len(c), "victorias_modelo": wins}
    c = cells[cells["regimen"] == "post-epidemica"]
    if len(c) < 2:
        result["c"] = {"estado": "no evaluable", "casillas": len(c)}
    else:
        more = int((c["calib_modelo"] > c["calib_b2"]).sum())
        result["c"] = {"estado": "sostenida" if more > len(c) / 2 else "no sostenida", "casillas": len(c),
                       "modelo_sobreestima_mas": more}
    return result



# ------------------------------------------------------------------- transition strata and events

def observed_series(channel_path: Path) -> dict:
    frame = pd.read_parquet(channel_path, columns=["municipality_code", "week_start", "channel_eligible",
                                                   "target_u1e", "channel_zone"])
    frame = frame[frame["channel_eligible"].astype(bool)].copy()
    frame["week"] = pd.to_datetime(frame["week_start"])
    out = {}
    for code, g in frame.groupby("municipality_code"):
        g = g.sort_values("week")
        out[str(code)] = {"weeks": list(g["week"]), "exceed": [int(v) for v in g["target_u1e"]],
                          "zone": dict(zip(g["week"], map(str, g["channel_zone"])))}
        out[str(code)]["exceed_at"] = dict(zip(out[str(code)]["weeks"], out[str(code)]["exceed"]))
    return out


def transition(rows: pd.DataFrame, series: dict) -> pd.DataFrame:
    rows = rows.copy()
    rows["zona_t"] = [series[c]["zone"][pd.Timestamp(t)] for c, t in zip(rows["municipality_code"], rows["t"])]
    rows["estrato"] = np.where(rows["zona_t"] == "epidemia", "dentro", "fuera")
    out = []
    for year in sorted(rows["anio"].unique()) + ["todos"]:
        for stratum in ("fuera", "dentro"):
            r = rows[(rows["estrato"] == stratum) & ((rows["anio"] == year) if year != "todos" else True)]
            y = r["y"].to_numpy(float)
            out.append({"anio": year, "estrato": stratum, "filas": len(r), "positivas": int(y.sum()),
                        "mejora_ll_b2": 1 - logloss(y, r["p_modelo"]) / logloss(y, r["p_b2"]),
                        "bss_b2": 1 - brier(y, r["p_modelo"]) / brier(y, r["p_b2"])})
    return pd.DataFrame(out)


W, K = 12, 2
WEEK = pd.Timedelta(weeks=1)


def _onsets(weeks, exceed):
    return [weeks[i] for i in range(2, len(weeks) - 2)
            if exceed[i - 2] == 0 and exceed[i - 1] == 0 and exceed[i] == exceed[i + 1] == exceed[i + 2] == 1]


def _entries(weeks, exceed):
    return [weeks[i] for i in range(1, len(weeks)) if exceed[i] == 1 and exceed[i - 1] == 0]


def _alerts(ts, positive, zones):
    found, i = [], 0
    while i < len(ts):
        if not positive[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ts) and positive[j + 1] and ts[j + 1] - ts[j] == WEEK:
            j += 1
        if j - i + 1 >= K:
            found.append({"tc": ts[i + K - 1], "continuity": zones[i + K - 1] == "epidemia"})
        i = j + 1
    return found


def events(rows: pd.DataFrame, series: dict, tau: float) -> tuple[pd.DataFrame, dict]:
    onset_rows, totals = [], {}
    for system in ("modelo", "canal"):
        totals[system] = {"alertas": 0, "continuidad": 0, "juzgables": 0, "falsas_alarmas": 0,
                          "fa_seguidas_de_epidemia": 0, "censuradas": 0, "entradas_evaluables": 0,
                          "entradas_avisadas": 0}
    per_onset = {}
    for code, r in rows.groupby("municipality_code"):
        r = r.sort_values("t")
        ts = [pd.Timestamp(t) for t in r["t"]]
        targets = [pd.Timestamp(t) for t in r["target_week"]]
        s = series[code]
        zones = [s["zone"][t] for t in ts]
        first_emission, first_target, last_target = ts[0], targets[0], targets[-1]
        all_onsets = _onsets(s["weeks"], s["exceed"])
        positives = {"modelo": (r["p_modelo"].to_numpy() >= tau), "canal": np.array([z == "alerta" for z in zones])}
        for system, pos in positives.items():
            found = _alerts(ts, pos, zones)
            tot = totals[system]
            tot["alertas"] += len(found)
            tot["continuidad"] += sum(a["continuity"] for a in found)
            for a in found:
                a["judgeable"] = a["tc"] + (W + 2) * WEEK <= last_target
                if not a["continuity"] and not a["judgeable"]:
                    tot["censuradas"] += 1
                if a["judgeable"] and not a["continuity"]:
                    tot["juzgables"] += 1
                    window = [a["tc"] + j * WEEK for j in range(1, W + 1)]
                    if not any(w in all_onsets for w in window):
                        tot["falsas_alarmas"] += 1
                        tot["fa_seguidas_de_epidemia"] += any(s["exceed_at"].get(w, 0) == 1 for w in window)
            useful = [a for a in found if not a["continuity"]]

            def lead(week):
                m = [a["tc"] for a in useful if week - W * WEEK <= a["tc"] <= week - WEEK]
                return (week - max(m)).days // 7 if m else None

            for week in all_onsets:
                if first_target <= week <= last_target:
                    evaluable = not (week - (W + 1) * WEEK < first_emission or week + 2 * WEEK > last_target)
                    per_onset.setdefault((code, week), {"municipality_code": code, "semana": week.date(),
                                                        "evaluable": evaluable})[f"antic_{system}"] = lead(week)
            for week in _entries(s["weeks"], s["exceed"]):
                if first_target <= week <= last_target and week - (W + 1) * WEEK >= first_emission:
                    tot["entradas_evaluables"] += 1
                    tot["entradas_avisadas"] += lead(week) is not None
    table = pd.DataFrame(per_onset.values()).sort_values(["municipality_code", "semana"])
    both = table["antic_modelo"].notna() & table["antic_canal"].notna()
    table["delta_w"] = np.where(both, table["antic_modelo"] - table["antic_canal"], np.nan)
    for system in ("modelo", "canal"):
        ev = table[table["evaluable"]]
        totals[system]["inicios_evaluables"] = int(len(ev))
        totals[system]["inicios_avisados"] = int(ev[f"antic_{system}"].notna().sum())
    return table, totals


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--canal", type=Path, required=True)
    parser.add_argument("--busqueda", type=Path)
    parser.add_argument("--familia", default="logistica")
    parser.add_argument("--predicciones-prueba", type=Path)
    parser.add_argument("--tau", type=float, default=0.10, help="tau of the frozen recipe (events)")
    parser.add_argument("--csv", type=Path, help="write the cell table here")
    parser.add_argument("--json", type=Path, help="write summaries and hypotheses here")
    args = parser.parse_args(argv)
    if (args.busqueda is None) == (args.predicciones_prueba is None):
        parser.error("give exactly one of --busqueda (development) or --predicciones-prueba (test)")

    regime = regimes(args.canal)
    if args.busqueda is not None:
        rows = development_predictions(args.busqueda, args.familia)
        label = "DESARROLLO (usado para elegir; configuración común fuera de año)"
    else:
        rows = pd.read_parquet(args.predicciones_prueba)
        label = "PRUEBA (receta congelada)"
    rows["municipality_code"] = rows["municipality_code"].astype(str)
    cells = cell_table(rows, regime)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print(label)
    print("\nRégimen por municipio y año (serie elegible completa):")
    print(regime.pivot(index="municipality_code", columns="anio", values="regimen").to_string())
    print("\nTabla de casillas:")
    show = ["municipio", "anio", "regimen", "positivas", "prevalencia", "p_medio_modelo", "p_medio_b2",
            "calib_modelo", "calib_b2", "ll_modelo", "ll_b2", "ll_b1", "dif_ll_b2", "dif_ll_b1",
            "mejora_ll_b2", "bss_b2", "gana_ll_b2", "gana_brier_b2", "gana_ll_b1"]
    print(cells[show].round(4).to_string(index=False))
    summaries = {}
    for key in ("municipio", "anio", "regimen"):
        s = summary(rows, cells, key)
        summaries[key] = s
        print(f"\nResumen por {key}:")
        print(s.round(4).to_string(index=False))
    totals = {"casillas": len(cells), "gana_ll_b2": int(cells["gana_ll_b2"].sum()),
              "gana_brier_b2": int(cells["gana_brier_b2"].sum()), "gana_ll_b1": int(cells["gana_ll_b1"].sum()),
              "gana_brier_b1": int(cells["gana_brier_b1"].sum()),
              "ll_filas_juntas": {s: logloss(rows["y"], rows[f"p_{s}"]) for s in ("modelo", "b2", "b1")},
              "brier_filas_juntas": {s: brier(rows["y"], rows[f"p_{s}"]) for s in ("modelo", "b2", "b1")}}
    hr = hypothesis_r(cells)
    series = observed_series(args.canal)
    trans = transition(rows, series)
    print("\nTransición (dentro = zona epidemia en t):")
    print(trans.to_string(index=False))
    onsets_table, event_totals = events(rows, series, args.tau)
    print(f"\nEventos con tau = {args.tau}, k = {K}, W = {W}:")
    print(onsets_table.to_string(index=False))
    print(json.dumps(event_totals, ensure_ascii=False))
    print("\nTotales:", json.dumps(totals, ensure_ascii=False))
    print("H-R (descriptiva):", json.dumps(hr, ensure_ascii=False))
    if args.csv:
        cells.to_csv(args.csv, index=False)
        print(f"\nTabla de casillas en {args.csv}")
    if args.json:
        payload = {"etiqueta": label, "totales": totals, "h_r": hr,
                   "resumenes": {k: v.to_dict(orient="records") for k, v in summaries.items()}}
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
