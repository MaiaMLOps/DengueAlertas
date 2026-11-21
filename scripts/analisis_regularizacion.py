"""Regularisation of the frozen logistic recipe: reproduction of R0 and the extended grid of C (development only).

Two parts, both with the selection procedure of the frozen recipe (single table, common configuration by the mean
log-loss improvement over B2, out-of-year scoring, training cut D7.1, sin_clima features); only the grid of C changes:

  reproduccion  rebuilds the logistic single table of the pre-registered grid with the current code and compares it,
                with tolerance 1e-9, with the stored sin_clima search table (DVC) and with the versioned out-of-year
                cells of the recipe (reports/forecasting/evidence/seleccion_conjuntos_20261112/).
  ampliada      scores the 19-value grid of configs/contraste_familias.yaml (logistica_ampliada) out of year, C = 2
                fixed in every year, and compares both with R0 cell by cell (protocol declared in
                docs/forecasting/analisis_complementario.md before the calculation).
  c2c3          compares C = 2 and C = 3 fixed in every development year, cell by cell (log-loss wins) and by year
                (mean difference of the improvement over B2); adds two new files to the extended-grid folder.

No test-period target week is scored. Evidence goes to a new folder; existing files are never overwritten.

Run from the repository root:
    .venv/Scripts/python.exe -B scripts/analisis_regularizacion.py reproduccion --salida reports/forecasting/evidence/reproduccion_r0_20261120
    .venv/Scripts/python.exe -B scripts/analisis_regularizacion.py ampliada --salida reports/forecasting/evidence/regularizacion_c_ampliada_20261120
    .venv/Scripts/python.exe -B scripts/analisis_regularizacion.py c2c3 --salida reports/forecasting/evidence/regularizacion_c_ampliada_20261120
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
    fixed_cells,
    logistic_table,
    out_of_year_cells,
    paired_wins,
    summarize,
)
from src.forecasting.config import (  # noqa: E402
    CONTRAST_CONFIG_PATH,
    DEFAULT_CONFIG_PATH,
    load_config,
    load_contrast_config,
    resolve_path,
)
from src.forecasting.dataset import FEATURE_SETS, build_examples, load_channel_product  # noqa: E402
from src.forecasting.guard import RecipeAccess  # noqa: E402

TOLERANCE = 1e-9
STORED_TABLE = Path("data/processed/forecasting/sin_clima/tabla_unica_desarrollo.parquet")
VERSIONED_CELLS = Path("reports/forecasting/evidence/seleccion_conjuntos_20261112/casillas_fuera_de_anio.csv")
FIXED_C = 2.0   # value reported by the family contrast (table 01, in-sample)


def _prepare(config_path: Path):
    config = load_config(config_path)
    access = RecipeAccess.from_config(config)
    product = load_channel_product(resolve_path(config.datos.canal), expected_md5=config.datos.version_dvc_canal)
    examples = build_examples(product, h=config.horizonte.h, canal=config.canal, features=config.features)
    return config, access, development_trial_data(examples, config, access)


def _max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def _write(output: Path, files: dict) -> None:
    require(not output.exists(), f"{output} exists; evidence folders are never overwritten.")
    output.mkdir(parents=True)
    for name, content in files.items():
        path = output / name
        if isinstance(content, pd.DataFrame):
            content.to_csv(path, index=False, lineterminator="\n")
        else:
            path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def reproduction(config_path: Path, output: Path | None) -> dict:
    config, access, trial_data = _prepare(config_path)
    grid = config.busqueda.logistica.C
    table, _ = logistic_table(trial_data, grid, max_iter=config.busqueda.logistica.max_iter,
                              features=FEATURE_SETS["sin_clima"], access=access, clip=config.metricas.recorte_log_loss)
    keys = ["municipality_code", "candidato", "anio"]
    stored = pd.read_parquet(resolve_path(STORED_TABLE))
    stored = stored[stored["familia"] == "logistica"].sort_values(keys).reset_index(drop=True)
    rebuilt = table.sort_values(keys).reset_index(drop=True)
    require(rebuilt[keys].equals(stored[keys]), "The rebuilt table has other cells than the stored one.")
    table_diff = {column: _max_abs(rebuilt[column], stored[column])
                  for column in ("logloss", "brier", "logloss_b2", "brier_b2", "mejora_b2", "logloss_b1", "brier_b1")}

    cells = out_of_year_cells(table)
    versioned = pd.read_csv(resolve_path(VERSIONED_CELLS), dtype={"municipality_code": str})
    merged = cells.merge(versioned, on=["municipality_code", "anio"], suffixes=("", "_versionada"),
                         validate="one_to_one")
    require(len(merged) == len(cells) == 24, "The out-of-year cells do not match the versioned ones.")
    same_choice = bool((merged["candidato"] == merged["candidato_logistica"]).all())
    cells_diff = {"logloss": _max_abs(merged["logloss"], merged["logloss_logistica"]),
                  "mejora_b2": _max_abs(merged["mejora_b2"], merged["mejora_b2_logistica"]),
                  "logloss_b2": _max_abs(merged["logloss_b2"], merged["logloss_b2_versionada"])}
    summary = summarize(cells)
    result = {
        "rejilla_C": list(grid),
        "C_comun_cuatro_anios": candidate_c(table)[common_choice(table)],
        "C_fuera_de_anio_por_anio": {year: candidate_c(table)[c] for year, c in common_by_year(table).items()},
        "resumen_fuera_de_anio": summary,
        "max_diferencia_tabla_almacenada": table_diff,
        "max_diferencia_casillas_versionadas": cells_diff,
        "misma_eleccion_por_casilla": same_choice,
        "tolerancia": TOLERANCE,
        "coincide": bool(same_choice and max(table_diff.values()) <= TOLERANCE
                         and max(cells_diff.values()) <= TOLERANCE),
    }
    if output is not None:
        _write(output, {"casillas_fuera_de_anio_r0.csv": cells, "resumen_reproduccion.json": result})
    return result


def extended(config_path: Path, contrast_path: Path, output: Path | None) -> dict:
    config, access, trial_data = _prepare(config_path)
    clip, features = config.metricas.recorte_log_loss, FEATURE_SETS["sin_clima"]
    pre = config.busqueda.logistica
    contrast = load_contrast_config(contrast_path)
    ext = contrast.corridas["logistica_ampliada"].busqueda.logistica
    require(FIXED_C in ext.C, f"C = {FIXED_C} is not in the extended grid.")

    r0_table, _ = logistic_table(trial_data, pre.C, max_iter=pre.max_iter, features=features, access=access,
                                 clip=clip)
    ext_table, _ = logistic_table(trial_data, ext.C, max_iter=ext.max_iter, features=features, access=access,
                                  clip=clip)
    r0_cells, ext_cells = out_of_year_cells(r0_table), out_of_year_cells(ext_table)
    fixed_name = next(name for name, c in candidate_c(ext_table).items() if c == FIXED_C)
    c2_cells = fixed_cells(ext_table, fixed_name)
    in_sample = ext_table.groupby("candidato")["mejora_b2"].mean()

    ext_c = candidate_c(ext_table)
    common_all = ext_c[common_choice(ext_table)]
    ext_summary, r0_summary = summarize(ext_cells), summarize(r0_cells)
    wins_vs_r0 = paired_wins(ext_cells, r0_cells)
    rule = {
        "C_comun_distinto_de_3": common_all != 3.0,
        "mejora_supera_r0": ext_summary["mejora_b2_media"] > r0_summary["mejora_b2_media"],
        "gana_al_menos_17_de_24_a_r0": wins_vs_r0["ganadas"] >= 17,
    }
    result = {
        "rejilla_C": list(ext.C),
        "C_comun_cuatro_anios": common_all,
        "C_fuera_de_anio_por_anio": {year: ext_c[c] for year, c in common_by_year(ext_table).items()},
        "ampliada_fuera_de_anio": ext_summary,
        "r0_fuera_de_anio": r0_summary,
        "ampliada_frente_a_r0": wins_vs_r0,
        "C2_fijo": summarize(c2_cells),
        "C2_fijo_frente_a_r0": paired_wins(c2_cells, r0_cells),
        "C2_mejora_media_dentro_de_muestra": float(in_sample[fixed_name]),
        "regla": rule,
        "reemplaza_receta": all(rule.values()),
    }
    if output is not None:
        cells = pd.concat([r0_cells.assign(sistema="R0"), ext_cells.assign(sistema="ampliada"),
                           c2_cells.assign(sistema="C2_fijo")], ignore_index=True)
        per_c = (ext_table.groupby("candidato")
                 .agg(mejora_b2_media=("mejora_b2", "mean"), logloss_medio=("logloss", "mean"),
                      brier_medio=("brier", "mean"))
                 .reset_index())
        per_c.insert(1, "C", per_c["candidato"].map(ext_c))
        _write(output, {"casillas_fuera_de_anio.csv": cells, "rejilla_dentro_de_muestra.csv": per_c,
                        "resumen_regularizacion.json": result})
    return result


def c2_against_c3(config_path: Path, contrast_path: Path, output: Path | None) -> dict:
    """C = 2 and C = 3 fixed in every development year, on the extended-grid table (same cells and B2)."""
    config, access, trial_data = _prepare(config_path)
    ext = load_contrast_config(contrast_path).corridas["logistica_ampliada"].busqueda.logistica
    table, _ = logistic_table(trial_data, ext.C, max_iter=ext.max_iter, features=FEATURE_SETS["sin_clima"],
                              access=access, clip=config.metricas.recorte_log_loss)
    name_of = {c: name for name, c in candidate_c(table).items()}
    require(FIXED_C in name_of and 3.0 in name_of, "C = 2 and C = 3 must be in the extended grid.")
    c2, c3 = fixed_cells(table, name_of[FIXED_C]), fixed_cells(table, name_of[3.0])
    cells = c2.merge(c3, on=["municipality_code", "anio"], suffixes=("_c2", "_c3"), validate="one_to_one")
    cells["diferencia_mejora_b2"] = cells["mejora_b2_c2"] - cells["mejora_b2_c3"]
    cells["diferencia_logloss"] = cells["logloss_c2"] - cells["logloss_c3"]
    cells["gana_c2"] = cells["logloss_c2"] < cells["logloss_c3"]
    by_year = cells.groupby("anio").agg(diferencia_mejora_b2_media=("diferencia_mejora_b2", "mean"),
                                        diferencia_logloss_media=("diferencia_logloss", "mean"),
                                        casillas_ganadas_c2=("gana_c2", "sum"))
    result = {"C2_frente_a_C3": paired_wins(c2, c3),
              "diferencia_mejora_b2_media": float(cells["diferencia_mejora_b2"].mean()),
              "por_anio": {int(year): {"diferencia_mejora_b2_media": float(row.diferencia_mejora_b2_media),
                                       "diferencia_logloss_media": float(row.diferencia_logloss_media),
                                       "casillas_ganadas_c2": int(row.casillas_ganadas_c2)}
                           for year, row in by_year.iterrows()}}
    if output is not None:
        files = {"c2_frente_a_c3_casillas.csv": cells, "c2_frente_a_c3.json": result}
        existing = [name for name in files if (output / name).exists()]
        require(output.is_dir() and not existing, f"{output} must exist without {existing}; files are never "
                                                  f"overwritten.")
        for name, content in files.items():
            if isinstance(content, pd.DataFrame):
                content.to_csv(output / name, index=False, lineterminator="\n")
            else:
                (output / name).write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regularisation analyses of the frozen logistic recipe "
                                                 "(development only).")
    parser.add_argument("parte", choices=["reproduccion", "ampliada", "c2c3"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--contraste", type=Path, default=CONTRAST_CONFIG_PATH)
    parser.add_argument("--salida", type=Path, default=None, help="New evidence folder (omitted: print only).")
    args = parser.parse_args(argv)
    output = resolve_path(args.salida) if args.salida is not None else None
    if args.parte == "reproduccion":
        result = reproduction(args.config, output)
    elif args.parte == "c2c3":
        result = c2_against_c3(args.config, args.contraste, output)
    else:
        result = extended(args.config, args.contraste, output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
