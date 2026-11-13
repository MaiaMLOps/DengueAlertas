"""caso_f6.py: independent check of F6 (selection of hyperparameters and family, then tau).

Written in the chat «Desarrollo guiado con Claude Code» on 2026-11-12, before looking at any F5b result.
It does not import anything from src/: it rebuilds the rules from the contract (§4.3, §6.2, §7.4, §7.5,
§8.2, §8.3), the pre-registration in especificación §7 and the F4 conventions approved on 2026-11-11.
The goal is to compare its output with selection.py in U6.3. The comparison is meaningful only if the two
implementations do not share code.

Usage (from the repository root, with the virtual environment active):

    # Only the channel in development (no F5b scores are read). Can be run right away.
    python -I caso_f6.py --canal data/processed/epidemiology/pilot_weekly_channel.parquet --solo-canal

    # Full selection and tau (U6.3, after selection.py has run).
    python -I caso_f6.py --canal data/processed/epidemiology/pilot_weekly_channel.parquet \
        --busqueda data/processed/forecasting --json caso_f6_resultado.json

What it computes:
  0. md5 of every input file read (compare with the .dvc files).
  1. Development rows rebuilt from the channel product: target week t + 8 eligible and in 2019-2022,
     t = target - 8 weeks. Checks 208 per municipality and, in the full mode, that the predictions have
     exactly those (t, target, y).
  2. Table integrity: recomputes each logloss of the single table from the predictions (clip 1e-15), the
     B2 logloss from the baseline predictions, and mejora_b2 = 1 - LL / LL_B2.
  3. Selection in the pre-registered order: common configuration per family (mean over years, then over
     municipalities; exact tie -> smallest `orden`); family comparison with out-of-year common
     configurations (strictly larger mean and >= 17 of 24 wins on logloss); family exception per
     municipality (4 of 4, common configurations); hyperparameter exception inside the final family
     (own chosen without y beats common chosen without y in the 4 years -> own chosen with the 4 years).
  4. Events (contract §7.4 with the F4 conventions) for the channel and for the chosen configuration on
     every tau of the grid, and the tau rule of §7.5.

Reference values of the channel in development (especificación §6.F4, computed with events.py, F4):
detection 10/10, median lead 3 (range 1-7), 63 judgeable alerts, 48 false alarms (12 followed by an
epidemic week), alert ratio 9/63, 0 continuity alerts, entries 36/44.
"""

import argparse
import hashlib
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

H = 8
K = 2
W = 12
ONSET_RUN = 3          # >= 3 weeks above U1e ...
ONSET_PRIOR = 2        # ... after >= 2 weeks at or below U1e
DEV_YEARS = (2019, 2020, 2021, 2022)
TAU_GRID = tuple(round(i * 0.05, 2) for i in range(1, 20))
ALPHA = 0.05
EPS = 1e-15
FAMILIES = ("logistica", "lightgbm")   # parsimony order
WEEK = timedelta(weeks=1)
REL, ABS = 1e-9, 1e-12

TABLE = "tabla_unica_desarrollo.parquet"
PREDICTIONS = "predicciones_desarrollo.parquet"
BASELINES = "lineas_base_desarrollo.parquet"
BASELINE_PREDICTIONS = "predicciones_lineas_base_desarrollo.parquet"
MANIFEST = "busqueda_manifest.json"


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def as_dates(values) -> list[date]:
    return [ts.date() for ts in pd.to_datetime(pd.Series(values))]


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL, abs_tol=ABS)


# ---------------------------------------------------------------------------------------------------
# Channel product and development rows
# ---------------------------------------------------------------------------------------------------

def load_channel(path: Path) -> dict:
    frame = pd.read_parquet(path, columns=["municipality_code", "week_start", "channel_eligible",
                                           "target_u1e", "channel_zone"])
    frame = frame[frame["channel_eligible"].astype(bool)].copy()
    frame["week"] = as_dates(frame["week_start"])
    series = {}
    for code, group in frame.groupby("municipality_code"):
        group = group.sort_values("week")
        weeks = list(group["week"])
        if any(later - earlier != WEEK for earlier, later in zip(weeks, weeks[1:])):
            fail(f"eligible weeks of {code} are not consecutive")
        exceed = [int(v) for v in group["target_u1e"]]
        series[str(code)] = {
            "weeks": weeks,
            "exceed": exceed,
            "exceed_at": dict(zip(weeks, exceed)),
            "zone_at": dict(zip(weeks, map(str, group["channel_zone"]))),
        }
    return series


def development_rows(series: dict) -> dict:
    """Per municipality, emissions sorted by t: (t, target, y, zone at t)."""
    rows = {}
    for code, s in series.items():
        targets = [w for w in s["weeks"] if w.year in DEV_YEARS]
        ts = [w - H * WEEK for w in targets]
        missing = [t for t in ts if t not in s["zone_at"]]
        if missing:
            fail(f"{code}: emission weeks without channel zone: {missing[:3]}")
        if any(later - earlier != WEEK for earlier, later in zip(ts, ts[1:])):
            fail(f"{code}: development emissions are not consecutive weeks")
        rows[code] = pd.DataFrame({"t": ts, "target": targets,
                                   "y": [s["exceed_at"][w] for w in targets],
                                   "zone": [s["zone_at"][t] for t in ts]})
    sizes = {code: len(r) for code, r in rows.items()}
    if set(sizes.values()) != {208}:
        fail(f"expected 208 development rows per municipality; got {sizes}")
    return rows


# ---------------------------------------------------------------------------------------------------
# Events (contract §7.4 with the F4 conventions)
# ---------------------------------------------------------------------------------------------------

def onsets(weeks: list[date], exceed: list[int]) -> list[date]:
    found = []
    for i in range(ONSET_PRIOR, len(weeks) - ONSET_RUN + 1):
        before = exceed[i - ONSET_PRIOR:i]
        run = exceed[i:i + ONSET_RUN]
        if all(v == 0 for v in before) and all(v == 1 for v in run):
            found.append(weeks[i])
    return found


def entries(weeks: list[date], exceed: list[int]) -> list[date]:
    return [weeks[i] for i in range(1, len(weeks)) if exceed[i] == 1 and exceed[i - 1] == 0]


def alerts_of(rows: pd.DataFrame, positive: np.ndarray) -> list[dict]:
    """One alert per maximal run of >= K positive emissions, dated at the K-th week of the run."""
    ts = list(rows["t"])
    zones = list(rows["zone"])
    found, i = [], 0
    while i < len(ts):
        if not positive[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ts) and positive[j + 1] and ts[j + 1] - ts[j] == WEEK:
            j += 1
        if j - i + 1 >= K:
            c = i + K - 1
            found.append({"tc": ts[c], "continuity": zones[c] == "epidemia"})
        i = j + 1
    return found


def evaluate(rows_by_code: dict, positive_by_code: dict, series: dict) -> dict:
    totals = {"evaluable_onsets": 0, "warned_onsets": 0, "warned_by_judgeable": 0, "judgeable": 0,
              "false_alarms": 0, "fa_followed_by_epidemic": 0, "continuity": 0, "evaluable_entries": 0,
              "warned_entries": 0, "censored_onsets": 0, "leads": [], "municipality_years": 0}
    per_code = {}
    for code, rows in rows_by_code.items():
        s = series[code]
        first_emission, first_target, last_target = rows["t"].iloc[0], rows["target"].iloc[0], rows["target"].iloc[-1]
        all_onsets = onsets(s["weeks"], s["exceed"])
        found = alerts_of(rows, positive_by_code[code])
        for alert in found:
            alert["judgeable"] = alert["tc"] + (W + 2) * WEEK <= last_target
            alert["false_alarm"] = False
            if alert["judgeable"] and not alert["continuity"]:
                window = [alert["tc"] + j * WEEK for j in range(1, W + 1)]
                alert["false_alarm"] = not any(w in all_onsets for w in window)
                alert["followed_by_epidemic"] = any(s["exceed_at"].get(w, 0) == 1 for w in window)
        useful = [a for a in found if not a["continuity"]]

        def warning(week):
            matches = [a for a in useful if week - W * WEEK <= a["tc"] <= week - WEEK]
            if not matches:
                return False, False, None
            latest = max(a["tc"] for a in matches)
            return True, any(a["judgeable"] for a in matches), (week - latest).days // 7

        mine = {"evaluable_onsets": 0, "warned_onsets": 0, "false_alarms": 0}
        for week in all_onsets:
            if not first_target <= week <= last_target:
                continue
            censored = week - (W + 1) * WEEK < first_emission or week + 2 * WEEK > last_target
            if censored:
                totals["censored_onsets"] += 1
                continue
            warned, by_judgeable, lead = warning(week)
            totals["evaluable_onsets"] += 1
            mine["evaluable_onsets"] += 1
            if warned:
                totals["warned_onsets"] += 1
                mine["warned_onsets"] += 1
                totals["leads"].append(lead)
            totals["warned_by_judgeable"] += int(by_judgeable)
        for week in entries(s["weeks"], s["exceed"]):
            if first_target <= week <= last_target and week - (W + 1) * WEEK >= first_emission:
                totals["evaluable_entries"] += 1
                totals["warned_entries"] += int(warning(week)[0])
        judgeable = [a for a in useful if a["judgeable"]]
        false_alarms = [a for a in judgeable if a["false_alarm"]]
        totals["judgeable"] += len(judgeable)
        totals["false_alarms"] += len(false_alarms)
        totals["fa_followed_by_epidemic"] += sum(a["followed_by_epidemic"] for a in false_alarms)
        totals["continuity"] += sum(a["continuity"] for a in found)
        totals["municipality_years"] += len({w.year for w in rows["target"]})
        mine["false_alarms"] = len(false_alarms)
        per_code[code] = mine
    leads = totals.pop("leads")
    totals["lead_median"] = float(np.median(leads)) if leads else math.nan
    totals["lead_range"] = (min(leads), max(leads)) if leads else None
    totals["per_municipality"] = per_code
    return totals


def channel_events(rows: dict, series: dict) -> dict:
    return evaluate(rows, {c: (r["zone"] == "alerta").to_numpy() for c, r in rows.items()}, series)


def print_events(title: str, ev: dict) -> None:
    print(f"\n{title}")
    print(f"  inicios evaluables {ev['evaluable_onsets']} (censurados {ev['censored_onsets']}), "
          f"avisados {ev['warned_onsets']}; anticipación mediana {ev['lead_median']}, rango {ev['lead_range']}")
    print(f"  alertas juzgables {ev['judgeable']}, falsas alarmas {ev['false_alarms']} "
          f"({ev['false_alarms'] / ev['municipality_years']:.4f} por municipio-año, "
          f"{ev['municipality_years']} municipio-años), {ev['fa_followed_by_epidemic']} seguidas de epidemia")
    print(f"  razón de alertas {ev['warned_by_judgeable']}/{ev['judgeable']}; continuidad {ev['continuity']}; "
          f"entradas {ev['warned_entries']}/{ev['evaluable_entries']}")


# ---------------------------------------------------------------------------------------------------
# Selection (contract §4.3, §6.2, §8.3; especificación §7)
# ---------------------------------------------------------------------------------------------------

def binomial_threshold(n: int, alpha: float) -> int | None:
    for k in range(n + 1):
        tail = math.fsum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n
        if tail < alpha:
            return k
    return None


def choose(cells: pd.DataFrame) -> tuple[str, float]:
    """Largest mean mejora_b2 (years, then municipalities); exact tie -> smallest orden.
    Returns the candidate and the gap to the runner-up (to flag near ties)."""
    scores = []
    for (name, position), group in cells.groupby(["candidato", "orden"]):
        per_code = [math.fsum(g["mejora_b2"]) / len(g) for _, g in group.groupby("municipality_code")]
        scores.append((math.fsum(per_code) / len(per_code), int(position), str(name)))
    best = max(score for score, _, _ in scores)
    winners = sorted((position, name) for score, position, name in scores if score == best)
    others = [score for score, _, _ in scores if score != best]
    gap = best - max(others) if others else math.inf
    return winners[0][1], gap


def out_of_year(table: pd.DataFrame, family: str) -> pd.DataFrame:
    own = table[table["familia"] == family]
    codes, years = sorted(own["municipality_code"].unique()), sorted(own["anio"].unique())
    cell = own.set_index(["municipality_code", "anio", "candidato"])
    records = []
    for year in years:
        others = own[own["anio"] != year]
        common, common_gap = choose(others)
        for code in codes:
            mine, mine_gap = choose(others[others["municipality_code"] == code])
            records.append({
                "municipality_code": code, "anio": int(year), "comun": common, "propio": mine,
                "ll_comun": float(cell.loc[(code, year, common), "logloss"]),
                "ll_propio": float(cell.loc[(code, year, mine), "logloss"]),
                "mejora_comun": float(cell.loc[(code, year, common), "mejora_b2"]),
                "brecha_comun": common_gap, "brecha_propio": mine_gap})
    return pd.DataFrame(records)


def check_table(table: pd.DataFrame, predictions: pd.DataFrame, baseline_predictions: pd.DataFrame) -> None:
    def logloss(group):
        y = group["y"].to_numpy(dtype=float)
        p = np.clip(group["p"].to_numpy(dtype=float), EPS, 1 - EPS)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    recomputed = {key: logloss(g) for key, g in predictions.groupby(["municipality_code", "familia", "candidato", "anio"])}
    b2 = baseline_predictions[baseline_predictions["familia"] == "B2"]
    recomputed_b2 = {key: logloss(g) for key, g in b2.groupby(["municipality_code", "anio"])}
    bad = 0
    for row in table.itertuples(index=False):
        ll = recomputed[(row.municipality_code, row.familia, row.candidato, row.anio)]
        ll_b2 = recomputed_b2[(row.municipality_code, row.anio)]
        ok = close(ll, row.logloss) and close(ll_b2, row.logloss_b2) and close(1 - ll / ll_b2, row.mejora_b2)
        bad += not ok
    print(f"  logloss, logloss_b2 y mejora_b2 recalculados desde las predicciones: "
          f"{len(table) - bad} de {len(table)} coinciden (rel {REL}, abs {ABS})")
    if bad:
        fail("the single table does not match the predictions")


def select(table: pd.DataFrame) -> dict:
    result = {}
    common = {}
    for family in FAMILIES:
        name, gap = choose(table[table["familia"] == family])
        common[family] = name
        print(f"  Paso 1. Configuración común de {family}: {name} (ventaja sobre el segundo: {gap:.3e})")
    result["comun_4_anios"] = common

    oy = {family: out_of_year(table, family) for family in FAMILIES}
    for family in FAMILIES:
        near = oy[family][["brecha_comun", "brecha_propio"]].min().min()
        if near < 1e-12:
            print(f"  AVISO: casi empate (< 1e-12) en una elección fuera de año de {family}: {near:.3e}")
    simple, complex_ = oy[FAMILIES[0]], oy[FAMILIES[1]]
    pair = simple.merge(complex_, on=["municipality_code", "anio"], suffixes=("_s", "_c"))
    if len(pair) != 24:
        fail(f"expected 24 cells; got {len(pair)}")
    mean_simple = math.fsum(pair.groupby("municipality_code")["mejora_comun_s"].mean()) / pair["municipality_code"].nunique()
    mean_complex = math.fsum(pair.groupby("municipality_code")["mejora_comun_c"].mean()) / pair["municipality_code"].nunique()
    wins = int((pair["ll_comun_c"] < pair["ll_comun_s"]).sum())
    ties = int((pair["ll_comun_c"] == pair["ll_comun_s"]).sum())
    threshold = binomial_threshold(24, ALPHA)
    replaces = mean_complex > mean_simple and wins >= threshold
    family = FAMILIES[1] if replaces else FAMILIES[0]
    print(f"  Paso 2. Mejora media fuera de año: {FAMILIES[0]} {mean_simple:.6f}, {FAMILIES[1]} {mean_complex:.6f}; "
          f"victorias de {FAMILIES[1]} {wins} de 24 (empates {ties}, umbral {threshold}) -> familia común {family}")
    result["familias"] = {"mejora_fuera_de_anio": {FAMILIES[0]: mean_simple, FAMILIES[1]: mean_complex},
                          "victorias": wins, "empates": ties, "umbral": threshold, "familia_comun": family}

    alternative = FAMILIES[0] if family == FAMILIES[1] else FAMILIES[1]
    final, comparisons = {}, []
    for code in sorted(pair["municipality_code"].unique()):
        cells = pair[pair["municipality_code"] == code].sort_values("anio")
        ll_common_family = cells["ll_comun_s" if family == FAMILIES[0] else "ll_comun_c"].to_numpy()
        ll_alternative = cells["ll_comun_c" if family == FAMILIES[0] else "ll_comun_s"].to_numpy()
        family_exception = bool((ll_alternative < ll_common_family).all())
        own_family = alternative if family_exception else family
        diffs_family = (ll_alternative - ll_common_family).tolist()

        cells_own = oy[own_family][oy[own_family]["municipality_code"] == code].sort_values("anio")
        hyper_exception = bool((cells_own["ll_propio"] < cells_own["ll_comun"]).all())
        diffs_hyper = (cells_own["ll_propio"] - cells_own["ll_comun"]).tolist()
        if hyper_exception:
            name, _ = choose(table[(table["familia"] == own_family) & (table["municipality_code"] == code)])
        else:
            name = common[own_family]
        nominal = hyper_exception and name == common[own_family]
        final[code] = {"familia": own_family, "candidato": name, "excepcion_familia": family_exception,
                       "excepcion_hiperparametros": hyper_exception, "excepcion_nominal": nominal,
                       "dif_familia": diffs_family, "dif_hiperparametros": diffs_hyper}
        comparisons.append(code)
        print(f"  {code}: familia {own_family}{' (excepción)' if family_exception else ''}, candidato {name}"
              f"{' (excepción' + (', nominal)' if nominal else ')') if hyper_exception else ''}")
        print(f"      Δlogloss alternativa − común, años {list(cells['anio'])}: "
              + ", ".join(f"{d:+.5f}" for d in diffs_family))
        print(f"      Δlogloss propio − común dentro de {own_family}: " + ", ".join(f"{d:+.5f}" for d in diffs_hyper))
    result["municipios"] = final
    return result


# ---------------------------------------------------------------------------------------------------
# tau (contract §7.5)
# ---------------------------------------------------------------------------------------------------

def choose_tau(curve: list[dict], channel_false_alarms: int) -> tuple[float, bool]:
    feasible = [c for c in curve if c["false_alarms"] <= channel_false_alarms]
    if feasible:
        best = max(c["warned_onsets"] for c in feasible)
        return max(c["tau"] for c in feasible if c["warned_onsets"] == best), True
    fewest = min(c["false_alarms"] for c in curve)
    return max(c["tau"] for c in curve if c["false_alarms"] == fewest), False


def tau_step(rows: dict, series: dict, predictions: pd.DataFrame, final: dict, channel: dict) -> dict:
    probabilities = {}
    for code, rows_code in rows.items():
        choice = final[code]
        own = predictions[(predictions["municipality_code"] == code) & (predictions["familia"] == choice["familia"])
                          & (predictions["candidato"] == choice["candidato"])].copy()
        own["t"] = as_dates(own["t"])
        own["target_week"] = as_dates(own["target_week"])
        own = own.sort_values("t").reset_index(drop=True)
        same = (len(own) == len(rows_code) and list(own["t"]) == list(rows_code["t"])
                and list(own["target_week"]) == list(rows_code["target"])
                and list(own["y"].astype(int)) == list(rows_code["y"]))
        if not same:
            fail(f"{code}: predictions of the chosen configuration do not match the development rows")
        probabilities[code] = own["p"].to_numpy(dtype=float)
    curve = []
    for tau in TAU_GRID:
        ev = evaluate(rows, {c: p >= tau for c, p in probabilities.items()}, series)
        curve.append({"tau": tau, "warned_onsets": ev["warned_onsets"], "evaluable_onsets": ev["evaluable_onsets"],
                      "false_alarms": ev["false_alarms"], "judgeable": ev["judgeable"],
                      "continuity": ev["continuity"], "lead_median": ev["lead_median"]})
    tau, met = choose_tau(curve, channel["false_alarms"])
    print(f"\n  {'τ':>5} {'avisados':>9} {'FA':>4} {'juzgables':>10} {'continuidad':>12} {'anticipación':>13}")
    for c in curve:
        print(f"  {c['tau']:>5.2f} {c['warned_onsets']:>4}/{c['evaluable_onsets']:<4} {c['false_alarms']:>4} "
              f"{c['judgeable']:>10} {c['continuity']:>12} {c['lead_median']:>13}")
    print(f"  Canal: {channel['warned_onsets']}/{channel['evaluable_onsets']} avisados, {channel['false_alarms']} FA")
    print(f"  τ elegido: {tau}; restricción de falsas alarmas cumplida: {met}")
    return {"curva": curve, "tau": tau, "restriccion_cumplida": met}


# ---------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--canal", type=Path, required=True)
    parser.add_argument("--busqueda", type=Path)
    parser.add_argument("--solo-canal", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if not args.solo_canal and args.busqueda is None:
        parser.error("--busqueda is required unless --solo-canal")

    print("Insumos (md5):")
    print(f"  {args.canal.name}: {md5(args.canal)}")
    series = load_channel(args.canal)
    rows = development_rows(series)
    print(f"Filas de desarrollo: {sum(len(r) for r in rows.values())} "
          f"({len(rows)} municipios × 208); positivas {sum(int(r['y'].sum()) for r in rows.values())}")
    channel = channel_events(rows, series)
    print_events("Canal en desarrollo (referencia F4: 10/10, mediana 3 (1-7), 63 juzgables, 48 FA, 12 seguidas de "
                 "epidemia, razón 9/63, continuidad 0, entradas 36/44):", channel)
    output = {"canal": {k: v for k, v in channel.items() if k != "per_municipality"}}
    if not args.solo_canal:
        for name in (TABLE, PREDICTIONS, BASELINES, BASELINE_PREDICTIONS, MANIFEST):
            print(f"  {name}: {md5(args.busqueda / name)}")
        manifest = json.loads((args.busqueda / MANIFEST).read_text(encoding="utf-8"))
        print(f"  busqueda_id en el manifiesto: {manifest.get('busqueda_id', '(no está)')}")
        table = pd.read_parquet(args.busqueda / TABLE)
        predictions = pd.read_parquet(args.busqueda / PREDICTIONS)
        baseline_predictions = pd.read_parquet(args.busqueda / BASELINE_PREDICTIONS)
        if set(table["anio"]) != set(DEV_YEARS):
            fail(f"table years are {sorted(set(table['anio']))}, expected {DEV_YEARS}")
        if max(as_dates(predictions["target_week"])).year > max(DEV_YEARS):
            fail("predictions contain target weeks after development")
        print("\nIntegridad de la tabla única:")
        check_table(table, predictions, baseline_predictions)
        print("\nSelección:")
        selection = select(table)
        print("\nτ (contrato §7.5):")
        output.update({"seleccion": selection, "tau": tau_step(rows, series, predictions, selection["municipios"],
                                                                channel)})
    if args.json:
        args.json.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\nResultado escrito en {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
