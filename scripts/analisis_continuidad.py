"""Descriptive continuity analysis (protocol declared in docs/forecasting/analisis_complementario.md).

Question: in the weeks in which a municipality is already in the epidemic zone, is the probability of the frozen
recipe of still being above U1e at t + 8 better than that of B2? Saved predictions only (nothing is refitted or
chosen): development out of year and the single test run, development first.

Per stratum (zona_t = epidemia, zona_t != epidemia) and system (recipe, B2), over the rows of the stratum: log-loss,
Brier, BSS and log-loss improvement against B2, the Murphy decomposition with 10 equal-frequency intervals
(reports.murphy_decomposition) and the municipality-year cells with at least 5 weeks in the stratum. Additional
count: false alarms of rule B01 that start (Alert.start) within 13 weeks after the end of an episode (run of >= 3
eligible weeks with target_u1e = 1, contrato §8.8).

Run from the repository root:
    .venv/Scripts/python.exe -B scripts/analisis_continuidad.py --salida reports/forecasting/evidence/continuidad_20261120
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analisis_b01 import _read_series  # noqa: E402
from src.epidemiology.validation import require  # noqa: E402
from src.forecasting.comparison import count_wins  # noqa: E402
from src.forecasting.config import DEFAULT_CONFIG_PATH, load_config, resolve_path  # noqa: E402
from src.forecasting.dataset import load_channel_product  # noqa: E402
from src.forecasting.diagnostics import EPISODE_MIN_RUN  # noqa: E402
from src.forecasting.events import _runs, emission_positive, evaluate_period  # noqa: E402
from src.forecasting.guard import RecipeAccess, load_frozen_recipe  # noqa: E402
from src.forecasting.metrics import brier, improvement, log_loss  # noqa: E402
from src.forecasting.reports import murphy_decomposition  # noqa: E402

PERIODS = {  # development first (protocol order)
    "desarrollo": Path("reports/forecasting/evidence/diagnostico_desarrollo_sin_clima_20261112/diag_serie.csv"),
    "prueba": Path("reports/forecasting/evidence/prueba_20261112/diag_serie.csv"),
}
STRATA = {"zona_epidemia": lambda zones: zones == "epidemia", "zona_no_epidemia": lambda zones: zones != "epidemia"}
SYSTEMS = {"receta": "p_modelo", "B2": "p_b2"}
MIN_CELL_WEEKS = 5
AFTER_EPISODE_WEEKS = 13


def stratum_metrics(rows: pd.DataFrame, *, access: RecipeAccess, clip: float) -> tuple[dict, pd.DataFrame]:
    y, weeks = rows["y"].to_numpy(dtype=int), list(rows["target_week"])
    out = {"filas": len(rows), "positivas": int(y.sum()), "prevalencia": float(y.mean())}
    for system, column in SYSTEMS.items():
        p = rows[column].to_numpy(dtype=float)
        murphy = murphy_decomposition(y, p, weeks, access=access)
        out[system] = {"logloss": log_loss(y, p, weeks, access=access, clip=clip),
                       "brier": brier(y, p, weeks, access=access), "p_media": float(p.mean()),
                       "confiabilidad": murphy["confiabilidad"], "resolucion": murphy["resolucion"],
                       "incertidumbre": murphy["incertidumbre"], "intervalos": murphy["intervalos"]}
    out["bss_vs_b2"] = improvement(out["receta"]["brier"], out["B2"]["brier"])
    out["mejora_logloss_vs_b2"] = improvement(out["receta"]["logloss"], out["B2"]["logloss"])

    cells = []
    years = rows["target_week"].map(lambda week: week.year)
    for (code, year), group in rows.groupby([rows["municipality_code"], years]):
        if len(group) < MIN_CELL_WEEKS:
            continue
        gy, gw = group["y"].to_numpy(dtype=int), list(group["target_week"])
        record = {"municipality_code": code, "anio": int(year), "filas": len(group), "positivas": int(gy.sum())}
        for system, column in SYSTEMS.items():
            gp = group[column].to_numpy(dtype=float)
            record[f"logloss_{system}"] = log_loss(gy, gp, gw, access=access, clip=clip)
            record[f"brier_{system}"] = brier(gy, gp, gw, access=access)
        cells.append(record)
    cells = pd.DataFrame(cells)
    out["casillas"] = {"casillas": len(cells), "minimo_semanas": MIN_CELL_WEEKS,
                       "ganadas_brier": count_wins(cells["brier_receta"], cells["brier_B2"]) if len(cells) else 0,
                       "ganadas_logloss": count_wins(cells["logloss_receta"], cells["logloss_B2"]) if len(cells) else 0}
    return out, cells


def episode_ends(observed: pd.DataFrame, code: str) -> list:
    own = observed[(observed["municipality_code"] == code) & observed["channel_eligible"]].sort_values("week_start")
    weeks, values = list(own["week_start"]), own["target_u1e"].to_numpy(dtype=int)
    return [weeks[first + length - 1] for first, length in _runs(values) if length >= EPISODE_MIN_RUN]


def false_alarms_after_episodes(rows, observed, *, tau, k, window, access) -> dict:
    positive = emission_positive("umbral_fuera_de_epidemia", rows["p_modelo"], rows["zona_t"], tau=tau)
    events = evaluate_period(rows, positive, observed, k=k, window=window, access=access)
    false_alarms = events.alerts[events.alerts["false_alarm"]]
    ends = {code: episode_ends(observed, code) for code in rows["municipality_code"].unique()}
    after = [any(end + timedelta(weeks=1) <= alert.start <= end + timedelta(weeks=AFTER_EPISODE_WEEKS)
                 for end in ends[alert.municipality_code]) for alert in false_alarms.itertuples(index=False)]
    by_code = pd.Series(after, index=false_alarms["municipality_code"].to_numpy(), dtype=bool)
    return {"falsas_alarmas_B01": len(false_alarms), "empiezan_tras_episodio": int(sum(after)),
            "semanas_tras_episodio": AFTER_EPISODE_WEEKS,
            "por_municipio": {code: {"falsas_alarmas": int((by_code.index == code).sum()),
                                     "tras_episodio": int(by_code[by_code.index == code].sum())}
                              for code in sorted(set(by_code.index))}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Descriptive continuity analysis (saved predictions only).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--salida", type=Path, default=None, help="New evidence folder (omitted: print only).")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    access = RecipeAccess.from_config(config)
    recipe = load_frozen_recipe(access.recipe_path, channel_version=access.channel_version)
    tau, clip = float(recipe["decision"]["tau"]), config.metricas.recorte_log_loss
    observed = load_channel_product(resolve_path(config.datos.canal),
                                    expected_md5=config.datos.version_dvc_canal).to_pandas()
    summary, all_cells = {"tau": tau, "k": config.decision.k, "W": config.decision.W, "periodos": {}}, []
    for period, path in PERIODS.items():
        rows = _read_series(path)
        require(rows[list(SYSTEMS.values())].notna().all().all(), f"{path}: missing probabilities.")
        zones = rows["zona_t"].to_numpy()
        strata = {}
        for name, select in STRATA.items():
            metrics, cells = stratum_metrics(rows[select(zones)], access=access, clip=clip)
            strata[name] = metrics
            all_cells.append(cells.assign(periodo=period, estrato=name))
        summary["periodos"][period] = {
            "fuente": path.as_posix(), "filas": len(rows), "estratos": strata,
            "falsas_alarmas_B01": false_alarms_after_episodes(rows, observed, tau=tau, k=config.decision.k,
                                                              window=config.decision.W, access=access)}
    if args.salida is not None:
        output = resolve_path(args.salida)
        require(not output.exists(), f"{output} exists; evidence folders are never overwritten.")
        output.mkdir(parents=True)
        cells = pd.concat(all_cells, ignore_index=True)
        cells = cells[["periodo", "estrato"] + [c for c in cells.columns if c not in ("periodo", "estrato")]]
        cells.to_csv(output / "continuidad_casillas.csv", index=False, lineterminator="\n")
        (output / "continuidad_resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                         encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
