"""Alternative emission rule B01 (post-hoc analysis declared in docs/forecasting/analisis_complementario.md).

B01: the emission of week t is positive if p >= tau AND the observed channel zone at t is not "epidemia"
(events.model_positive_outside_epidemic). Same probabilities, the tau of the frozen recipe, k and W of the
configuration; events.alerts and events.evaluate_period are unchanged.

Inputs are the saved per-row predictions of the frozen recipe (diag_serie.csv of the single test run and of the
out-of-year development diagnosis); nothing is refitted and no parameter is chosen. Systems: the contract rule
(p >= tau), B01 and the channel alert (channel_positive), each through evaluate_period on the same rows.

Outputs (new folder): per period, municipality and year, warned onsets and entries, alerts, false alarms, the
fraction of weeks under alert and the lead (median and mean) measured from the most recent alert in the window
(the contract measure) and from the first one; discordant onsets and McNemar (comparison module); the composition
of B01 emissions by zone and the alert-zone weeks that the probability filter discards.

Run from the repository root:
    .venv/Scripts/python.exe -B scripts/analisis_b01.py --salida reports/forecasting/evidence/complementario_20261120
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.epidemiology.validation import require  # noqa: E402
from src.forecasting.comparison import discordant_wins, event_level  # noqa: E402
from src.forecasting.complementary import first_alert_leads, weeks_under_alert  # noqa: E402
from src.forecasting.config import DEFAULT_CONFIG_PATH, load_config, resolve_path  # noqa: E402
from src.forecasting.dataset import load_channel_product  # noqa: E402
from src.forecasting.events import (  # noqa: E402
    ONSET,
    channel_positive,
    emission_positive,
    evaluate_period,
)
from src.forecasting.guard import RecipeAccess, load_frozen_recipe  # noqa: E402

PERIODS = {
    "prueba": Path("reports/forecasting/evidence/prueba_20261112/diag_serie.csv"),
    "desarrollo": Path("reports/forecasting/evidence/diagnostico_desarrollo_sin_clima_20261112/diag_serie.csv"),
}
SYSTEMS = ("regla_actual", "B01", "canal")
_ALERT_ZONE, _EPIDEMIC_ZONE = "alerta", "epidemia"


def _read_series(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(resolve_path(path), dtype={"municipality_code": str})
    for column in ("t", "target_week"):
        rows[column] = pd.to_datetime(rows[column]).dt.date
    require(not rows.duplicated(["municipality_code", "t"]).any(), f"{path}: duplicated rows.")
    return rows.sort_values(["municipality_code", "t"]).reset_index(drop=True)


def _median(values) -> float | None:
    clean = [float(v) for v in values if not math.isnan(v)]
    return statistics.median(clean) if clean else None


def _mean(values) -> float | None:
    clean = [float(v) for v in values if not math.isnan(v)]
    return statistics.fmean(clean) if clean else None


def _metrics(onsets: pd.DataFrame, alerts: pd.DataFrame, covered: pd.Series, n_rows: int) -> dict:
    evaluable = onsets[onsets["evaluable"]]
    starts = evaluable[evaluable["kind"] == ONSET]
    entries = evaluable[evaluable["kind"] != ONSET]
    judgeable = alerts[~alerts["continuity"] & alerts["judgeable"]]
    warned = starts[starts["warned"]]
    return {"inicios_avisados": int(starts["warned"].sum()), "inicios_evaluables": len(starts),
            "entradas_avisadas": int(entries["warned"].sum()), "entradas_evaluables": len(entries),
            "alertas": len(alerts), "alertas_continuidad": int(alerts["continuity"].sum()),
            "falsas_alarmas": int(judgeable["false_alarm"].sum()),
            "semanas_bajo_alerta": int(covered.sum()), "semanas": int(n_rows),
            "fraccion_bajo_alerta": float(covered.sum() / n_rows) if n_rows else math.nan,
            "anticipacion_mediana_contrato": _median(warned["lead"]),
            "anticipacion_media_contrato": _mean(warned["lead"]),
            "anticipacion_mediana_primera": _median(warned["lead_primera"]),
            "anticipacion_media_primera": _mean(warned["lead_primera"])}


def evaluate_rules(rows: pd.DataFrame, observed: pd.DataFrame, *, tau: float, k: int, window: int, alpha: float,
                   access: RecipeAccess, period: str):
    positives = {"regla_actual": emission_positive("umbral", rows["p_modelo"], rows["zona_t"], tau=tau),
                 "B01": emission_positive("umbral_fuera_de_epidemia", rows["p_modelo"], rows["zona_t"], tau=tau),
                 "canal": channel_positive(rows["zona_t"])}
    events, table, onset_frames = {}, [], []
    years = rows["target_week"].map(lambda week: week.year)
    for system, positive in positives.items():
        result = evaluate_period(rows, positive, observed, k=k, window=window, access=access)
        onsets = result.onsets.copy()
        onsets["lead_primera"] = first_alert_leads(onsets, result.alerts, window)
        covered = weeks_under_alert(rows, result.alerts)
        events[system] = (onsets, result.alerts)
        groups = [("global", None, None, onsets, result.alerts, covered)]
        for code in sorted(rows["municipality_code"].unique()):
            own = rows["municipality_code"] == code
            groups.append(("municipio", code, None, onsets[onsets["municipality_code"] == code],
                           result.alerts[result.alerts["municipality_code"] == code], covered[own]))
        for year in sorted(years.unique()):
            own = years == year
            groups.append(("anio", None, int(year), onsets[onsets["year"] == year],
                           result.alerts[result.alerts["year"] == year], covered[own]))
        for code in sorted(rows["municipality_code"].unique()):
            for year in sorted(years.unique()):
                own = (rows["municipality_code"] == code) & (years == year)
                groups.append(("municipio_anio", code, int(year),
                               onsets[(onsets["municipality_code"] == code) & (onsets["year"] == year)],
                               result.alerts[(result.alerts["municipality_code"] == code)
                                             & (result.alerts["year"] == year)], covered[own]))
        for level, code, year, own_onsets, own_alerts, own_covered in groups:
            table.append({"periodo": period, "sistema": system, "nivel": level, "municipality_code": code,
                          "anio": year, **_metrics(own_onsets, own_alerts, own_covered, len(own_covered))})
        frame = onsets[["municipality_code", "kind", "week", "year", "evaluable", "censoring", "warned", "lead",
                        "lead_primera"]].copy()
        frame.insert(0, "sistema", system)
        frame.insert(0, "periodo", period)
        onset_frames.append(frame)

    comparisons = {}
    for option, reference in (("B01", "regla_actual"), ("B01", "canal"), ("regla_actual", "canal")):
        a, b = events[option][0], events[reference][0]
        a = a[(a["kind"] == ONSET) & a["evaluable"]]
        b = b[(b["kind"] == ONSET) & b["evaluable"]]
        require(list(zip(a["municipality_code"], a["week"])) == list(zip(b["municipality_code"], b["week"])),
                "Systems must share the evaluable onsets.")
        wins, d = discordant_wins(a["warned"].to_numpy(dtype=bool), b["warned"].to_numpy(dtype=bool))
        comparisons[f"{option}_vs_{reference}"] = {"discrepantes": d, "a_favor": wins, "en_contra": d - wins,
                                                   "nivel": event_level(wins, d, alpha=alpha)}

    b01 = positives["B01"]
    zones = rows["zona_t"].to_numpy()
    in_alert_zone = zones == _ALERT_ZONE
    composition = {
        "emisiones_B01": int(b01.sum()),
        "emisiones_B01_por_zona": {zone: int((b01 & (zones == zone)).sum()) for zone in sorted(set(zones))},
        "semanas_zona_alerta": int(in_alert_zone.sum()),
        "semanas_zona_alerta_descartadas_por_p": int((in_alert_zone & ~b01).sum()),
        "fraccion_descartada": float((in_alert_zone & ~b01).sum() / in_alert_zone.sum()),
        "semanas_zona_epidemia": int((zones == _EPIDEMIC_ZONE).sum()),
        "emisiones_regla_actual_en_epidemia": int((positives["regla_actual"] & (zones == _EPIDEMIC_ZONE)).sum()),
    }
    return pd.DataFrame(table), pd.concat(onset_frames, ignore_index=True), comparisons, composition


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alternative emission rule B01 on the frozen recipe's predictions.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--salida", type=Path, default=None, help="New evidence folder (omitted: print only).")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    access = RecipeAccess.from_config(config)
    recipe = load_frozen_recipe(access.recipe_path, channel_version=access.channel_version)
    tau = float(recipe["decision"]["tau"])
    k, window = config.decision.k, config.decision.W
    require((int(recipe["decision"]["k"]), int(recipe["decision"]["W"])) == (k, window),
            "k and W of the configuration differ from the frozen recipe.")
    observed = load_channel_product(resolve_path(config.datos.canal),
                                    expected_md5=config.datos.version_dvc_canal).to_pandas()

    tables, onsets, summary = [], [], {"receta_hash": recipe["hash_sha256"], "tau": tau, "k": k, "W": window,
                                       "periodos": {}}
    for period, path in PERIODS.items():
        rows = _read_series(path)
        table, period_onsets, comparisons, composition = evaluate_rules(
            rows, observed, tau=tau, k=k, window=window, alpha=config.comparacion.alfa, access=access, period=period)
        current = rows["emision_modelo"].astype(bool).to_numpy()
        require(bool((emission_positive("umbral", rows["p_modelo"], rows["zona_t"], tau=tau) == current).all()),
                f"{path}: the contract rule does not reproduce the saved emissions.")
        tables.append(table)
        onsets.append(period_onsets)
        overall = table[table["nivel"] == "global"].set_index("sistema").drop(
            columns=["periodo", "nivel", "municipality_code", "anio"])
        summary["periodos"][period] = {"fuente": path.as_posix(), "filas": len(rows),
                                       "global": json.loads(overall.to_json(orient="index")),
                                       "comparaciones": comparisons, "composicion": composition}
    result_table, result_onsets = pd.concat(tables, ignore_index=True), pd.concat(onsets, ignore_index=True)
    if args.salida is not None:
        output = resolve_path(args.salida)
        require(not output.exists(), f"{output} exists; evidence folders are never overwritten.")
        output.mkdir(parents=True)
        result_table.to_csv(output / "b01_eventos.csv", index=False, lineterminator="\n")
        result_onsets.to_csv(output / "b01_inicios_entradas.csv", index=False, lineterminator="\n")
        (output / "b01_resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
