"""Reference M0: minimal position model (logistic L2 on posicion_4s alone), development and test report.

M0 measures how much of the recipe's performance is already contained in the continuous position against the
channel. Same grid of C as the frozen recipe, same preprocessing (training-only standardisation), same training cut
D7.1 and the same out-of-year common configuration. Nothing is selected on the test period: the test figures use the
configuration fixed in development (the common C with the four development years) and are reported as they are.

Development: out-of-year cells and summary, wins against B2 and against R0 (the frozen recipe rebuilt with
complementary.logistic_table, verified in reproduccion_r0_20261120). Test: M0 refitted once per (municipality, test
year) as in evaluate.run_test; pooled log-loss, Brier and AP over the test rows, and cells against B2 and against
the recipe taken from the saved predictions of the single test run (predicciones_prueba.parquet, not rerun). The
test-period barrier applies (approved frozen recipe).

Run from the repository root:
    .venv/Scripts/python.exe -B scripts/analisis_m0.py --salida reports/forecasting/evidence/m0_posicion_20261120
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.epidemiology.validation import require  # noqa: E402
from src.forecasting.complementary import (  # noqa: E402
    candidate_c,
    common_by_year,
    common_choice,
    development_trial_data,
    logistic_table,
    out_of_year_cells,
    paired_wins,
    position_table,
    position_test_predictions,
    summarize,
)
from src.forecasting.comparison import count_wins  # noqa: E402
from src.forecasting.config import DEFAULT_CONFIG_PATH, load_config, resolve_path  # noqa: E402
from src.forecasting.dataset import FEATURE_SETS, build_examples, load_channel_product  # noqa: E402
from src.forecasting.guard import RecipeAccess  # noqa: E402
from src.forecasting.metrics import average_precision, brier, improvement, log_loss  # noqa: E402

TEST_PREDICTIONS = Path("data/processed/forecasting/prueba/predicciones_prueba.parquet")
_KEYS = ["municipality_code", "t"]


def _test_report(m0: pd.DataFrame, saved: pd.DataFrame, access: RecipeAccess, clip: float):
    systems = {}
    for system in ("modelo", "B2"):
        own = saved[saved["sistema"] == system][_KEYS + ["target_week", "y", "p"]]
        merged = m0[_KEYS + ["target_week", "y"]].merge(own, on=_KEYS, how="left", suffixes=("", "_s"),
                                                        validate="one_to_one")
        require(bool(merged["p"].notna().all()) and len(merged) == len(m0), f"{system} does not cover the M0 rows.")
        require(bool((merged["target_week"] == merged["target_week_s"]).all()
                     and (merged["y"].astype(int) == merged["y_s"].astype(int)).all()), f"{system}: rows differ.")
        systems[system] = merged["p"].to_numpy(dtype=float)
    systems["M0"] = m0["p"].to_numpy(dtype=float)
    y, weeks = m0["y"].to_numpy(dtype=int), list(m0["target_week"])
    pooled = {s: {"logloss": log_loss(y, p, weeks, access=access, clip=clip), "brier": brier(y, p, weeks, access=access),
                  "ap": average_precision(y, p, weeks, access=access)} for s, p in systems.items()}
    cells = []
    for (code, year), idx in m0.groupby(["municipality_code", "anio"]).groups.items():
        positions = m0.index.get_indexer(idx)
        cell_weeks = [weeks[i] for i in positions]
        record = {"municipality_code": code, "anio": int(year), "filas": len(positions),
                  "positivas": int(y[positions].sum())}
        for s, p in systems.items():
            record[f"logloss_{s}"] = log_loss(y[positions], p[positions], cell_weeks, access=access, clip=clip)
            record[f"brier_{s}"] = brier(y[positions], p[positions], cell_weeks, access=access)
        cells.append(record)
    cells = pd.DataFrame(cells).sort_values(["municipality_code", "anio"]).reset_index(drop=True)
    wins = {f"{metric}_vs_{ref}": {"ganadas": count_wins(cells[f"{metric}_M0"], cells[f"{metric}_{ref}"]),
                                   "perdidas": count_wins(cells[f"{metric}_{ref}"], cells[f"{metric}_M0"]),
                                   "casillas": len(cells)}
            for metric in ("brier", "logloss") for ref in ("B2", "modelo")}
    summary = {"filas": len(m0), "prevalencia": float(y.mean()), "global": pooled,
               "bss_vs_b2": improvement(pooled["M0"]["brier"], pooled["B2"]["brier"]),
               "mejora_logloss_vs_b2": improvement(pooled["M0"]["logloss"], pooled["B2"]["logloss"]),
               "casillas": wins}
    return cells, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reference M0 (position only), development and test report.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--salida", type=Path, default=None, help="New evidence folder (omitted: print only).")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    access = RecipeAccess.from_config(config)
    clip = config.metricas.recorte_log_loss
    grid = config.busqueda.logistica
    product = load_channel_product(resolve_path(config.datos.canal), expected_md5=config.datos.version_dvc_canal)
    examples = build_examples(product, h=config.horizonte.h, canal=config.canal, features=config.features)
    trial_data = development_trial_data(examples, config, access)

    r0_table, _ = logistic_table(trial_data, grid.C, max_iter=grid.max_iter, features=FEATURE_SETS["sin_clima"],
                                 access=access, clip=clip)
    m0_table = position_table(trial_data, grid.C, max_iter=grid.max_iter, access=access, clip=clip)
    r0_cells, m0_cells = out_of_year_cells(r0_table), out_of_year_cells(m0_table)
    c_of = candidate_c(m0_table)
    fixed = {"C": c_of[common_choice(m0_table)], "max_iter": grid.max_iter}
    development = {
        "C_comun_cuatro_anios": fixed["C"],
        "C_fuera_de_anio_por_anio": {year: c_of[c] for year, c in common_by_year(m0_table).items()},
        "M0_fuera_de_anio": summarize(m0_cells),
        "R0_fuera_de_anio": summarize(r0_cells),
        "M0_frente_a_R0": paired_wins(m0_cells, r0_cells),
    }

    m0_test = position_test_predictions(examples, config, fixed, access)
    saved = pd.read_parquet(resolve_path(TEST_PREDICTIONS))
    test_cells, test = _test_report(m0_test, saved, access, clip)
    result = {"configuracion_fija": fixed, "desarrollo": development, "prueba": test}
    if args.salida is not None:
        output = resolve_path(args.salida)
        require(not output.exists(), f"{output} exists; evidence folders are never overwritten.")
        output.mkdir(parents=True)
        cells = pd.concat([m0_cells.assign(sistema="M0"), r0_cells.assign(sistema="R0")], ignore_index=True)
        cells.to_csv(output / "casillas_desarrollo.csv", index=False, lineterminator="\n")
        test_cells.to_csv(output / "casillas_prueba.csv", index=False, lineterminator="\n")
        (output / "resumen_m0.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
