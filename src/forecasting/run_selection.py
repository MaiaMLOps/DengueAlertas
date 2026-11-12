"""Command line of the F6 selection: evidence and recipe draft from the saved development search
(contrato §6.2, §7.5 and §8.3; especificación §6.F6, §7 and §8).

Flow (nothing is retrained):
    load_config -> check the F5b inputs against their DVC descriptors (md5) -> search_provenance
    -> load_channel_product (md5) -> build_examples -> development rows of splits.yearly_trials
    -> selection.select twice (the result must be identical) -> evidence tables
    -> write the evidence folder and the recipe draft (selection.write_recipe_draft), never overwriting.

Evidence files (folder given by --salida, which must be empty or new):
    resumen_seleccion.json       steps 1-4 and tau in the pre-registered order, inputs and margins < 1e-12
    casillas_fuera_de_anio.csv   per (municipality, year): common out-of-year configuration of each family
    mejora_por_municipio.csv     mean out-of-year improvement per municipality and minimum B2 log-loss
    comparaciones_excepciones.csv  family and hyperparameter comparisons with their annual differences
    curva_tau.csv                event metrics per tau, with the channel as a reference row
    margenes_de_eleccion.csv     margin of every choice over the second candidate
    receta_borrador.yaml         recipe draft of especificación §8, without approval or hash

The test period is never read: selection uses development rows only, and the recipe is a draft.

Usage:
    python -m src.forecasting.run_selection --salida reports/forecasting/evidence/seleccion_YYYYMMDD
        [--config P] [--busqueda data/processed/forecasting] [--permitir-cambios]
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
import yaml

from src.epidemiology.validation import require

from .comparison import out_of_year_scores
from .config import DEFAULT_CONFIG_PATH, ROOT, ForecastConfig, load_config
from .dataset import build_examples, load_channel_product
from .guard import RecipeAccess
from .models import FAMILIES
from .search import DEFAULT_OUTPUT_DIR, MANIFEST_FILENAME, PREDICTIONS_FILENAME, TABLE_FILENAME
from .selection import CURVE_COLUMNS, Selection, build_recipe, select, write_recipe_draft
from .splits import yearly_trials

CONTRACT_PATH = "docs/forecasting/contrato_predictivo.md"
PROVENANCE_INPUTS = (TABLE_FILENAME, PREDICTIONS_FILENAME, MANIFEST_FILENAME)
DVC_DIR = DEFAULT_OUTPUT_DIR            # the F5b descriptors are versioned next to the data
DRAFT_FILENAME = "receta_borrador.yaml"
EVIDENCE_FILES = ("resumen_seleccion.json", "casillas_fuera_de_anio.csv", "mejora_por_municipio.csv",
                  "comparaciones_excepciones.csv", "curva_tau.csv", "margenes_de_eleccion.csv", DRAFT_FILENAME)
NEAR_TIE = 1e-12                        # a choice decided by less than this is reported (Jesús, U6.3)


# --------------------------------------------------------------------------- provenance and inputs
def _git(*args: str) -> str:
    result = subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, capture_output=True, text=True)
    require(result.returncode == 0 and result.stdout.strip() != "", f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout.strip()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def search_provenance(search_dir: str | Path, *, config: ForecastConfig) -> dict:
    """Commit of the code (HEAD), of the contract, channel version, search id and md5 of the search inputs."""
    search_dir = Path(search_dir)
    for name in PROVENANCE_INPUTS:
        require((search_dir / name).is_file(), f"Search input {search_dir / name} is missing.")
    manifest = json.loads((search_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    return {
        "commit_codigo": _git("rev-parse", "HEAD"),
        "commit_contrato": _git("log", "-1", "--format=%H", "--", CONTRACT_PATH),
        "version_dvc_canal": config.datos.version_dvc_canal,
        "busqueda_id": manifest["busqueda_id"],
        "insumos_md5": {name: _md5(search_dir / name) for name in PROVENANCE_INPUTS},
    }


def check_inputs_against_dvc(provenance: dict, dvc_dir: Path = DVC_DIR) -> None:
    """Every input must have the md5 of its versioned DVC descriptor (the search of F5b, not another one)."""
    for name, digest in provenance["insumos_md5"].items():
        descriptor = dvc_dir / f"{name}.dvc"
        require(descriptor.is_file(), f"DVC descriptor {descriptor} is missing.")
        expected = yaml.safe_load(descriptor.read_text(encoding="utf-8"))["outs"][0]["md5"]
        require(digest == expected, f"{name} has md5 {digest}, but its DVC descriptor says {expected}.")


def development_rows(examples: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    first, last = config.validacion.desarrollo
    trials = yearly_trials(examples, range(first, last + 1),
                           first_train_year=config.validacion.primer_anio_entrenamiento)
    labels = trials[0].evaluate.append([trial.evaluate for trial in trials[1:]])
    return examples.loc[labels]


def _same(first: Selection, second: Selection) -> bool:
    if first.familias != second.familias or first.tau != second.tau:
        return False
    for field in ("elecciones", "comparaciones", "predicciones", "curva"):
        if not getattr(first, field).equals(getattr(second, field)):
            return False
    return json.dumps(first.canal, sort_keys=True, default=str) == json.dumps(second.canal, sort_keys=True, default=str)


# --------------------------------------------------------------------------- evidence
def _ranking(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean mejora_b2 per candidate (years within municipality, then municipalities), best first."""
    per = rows.groupby(["municipality_code", "candidato", "orden"])["mejora_b2"].mean()
    means = per.groupby(["candidato", "orden"]).mean().reset_index()
    return means.sort_values(["mejora_b2", "orden"], ascending=[False, True]).reset_index(drop=True)


def _margin(ranked: pd.DataFrame) -> float:
    return float(ranked.loc[0, "mejora_b2"] - ranked.loc[1, "mejora_b2"]) if len(ranked) > 1 else float("inf")


def evidence(selection: Selection, table: pd.DataFrame, config: ForecastConfig) -> dict:
    """Evidence tables and the summary of a selection, in the pre-registered order."""
    canonical = table.sort_values(["municipality_code", "familia", "orden", "anio"], kind="mergesort")
    flags = []

    common = {}
    for family in FAMILIES:
        ranked = _ranking(canonical[canonical["familia"] == family])
        name = ranked.loc[0, "candidato"]
        params = canonical.loc[(canonical["familia"] == family) & (canonical["candidato"] == name), "parametros"].iloc[0]
        margin = _margin(ranked)
        common[family] = {"candidato": name, "orden": int(ranked.loc[0, "orden"]), "parametros": json.loads(params),
                          "mejora_media_4_anios": float(ranked.loc[0, "mejora_b2"]),
                          "segundo": ranked.loc[1, "candidato"], "mejora_media_segundo": float(ranked.loc[1, "mejora_b2"]),
                          "ventaja_sobre_segundo": margin}
        if margin < NEAR_TIE:
            flags.append(f"paso 1 {family}: ventaja {margin:.3e} sobre {ranked.loc[1, 'candidato']}")

    margins = []
    for family in FAMILIES:
        own = canonical[canonical["familia"] == family]
        for year in sorted(set(own["anio"])):
            others = own[own["anio"] != year]
            ranked = _ranking(others)
            margins.append({"familia": family, "eleccion": "comun_sin_y", "municipality_code": "", "anio": year,
                            "candidato": ranked.loc[0, "candidato"], "ventaja": _margin(ranked)})
            for code in sorted(set(own["municipality_code"])):
                ranked = _ranking(others[others["municipality_code"] == code])
                margins.append({"familia": family, "eleccion": "propia_sin_y", "municipality_code": code, "anio": year,
                                "candidato": ranked.loc[0, "candidato"], "ventaja": _margin(ranked)})
        for code in sorted(set(own["municipality_code"])):
            ranked = _ranking(own[own["municipality_code"] == code])
            margins.append({"familia": family, "eleccion": "propia_4_anios", "municipality_code": code, "anio": 0,
                            "candidato": ranked.loc[0, "candidato"], "ventaja": _margin(ranked)})
    margins = pd.DataFrame(margins)
    for _, row in margins[margins["ventaja"] < NEAR_TIE].iterrows():
        flags.append(f"elección {row['eleccion']} {row['familia']} {row['municipality_code']} {row['anio']}: "
                     f"ventaja {row['ventaja']:.3e}")

    simple, complex_ = FAMILIES
    scores = {family: out_of_year_scores(table, familia=family) for family in FAMILIES}
    cells = scores[simple][["municipality_code", "anio"]].copy()
    b2 = table.drop_duplicates(["municipality_code", "anio"]).set_index(["municipality_code", "anio"])["logloss_b2"]
    cells["logloss_b2"] = [float(b2.loc[(code, year)]) for code, year in zip(cells["municipality_code"], cells["anio"])]
    for family in FAMILIES:
        cells[f"candidato_{family}"] = scores[family]["candidato_comun"].to_numpy()
        cells[f"logloss_{family}"] = scores[family]["logloss_comun"].to_numpy()
        cells[f"mejora_b2_{family}"] = scores[family]["mejora_comun"].to_numpy()
    cells["diferencia_logloss"] = cells[f"logloss_{complex_}"] - cells[f"logloss_{simple}"]
    cells["diferencia_mejora_b2"] = cells[f"mejora_b2_{complex_}"] - cells[f"mejora_b2_{simple}"]
    cells["gana_lightgbm"] = cells["diferencia_logloss"] < 0
    for _, row in cells[cells["diferencia_logloss"].abs() < NEAR_TIE].iterrows():
        flags.append(f"casilla {row['municipality_code']} {row['anio']}: |diferencia logloss| < 1e-12")
    by_municipality = cells.groupby("municipality_code").agg(
        logloss_b2_min=("logloss_b2", "min"), **{f"mejora_{f}": (f"mejora_b2_{f}", "mean") for f in FAMILIES},
        victorias_lightgbm=("gana_lightgbm", "sum")).reset_index()
    by_municipality["diferencia_mejora"] = by_municipality[f"mejora_{complex_}"] - by_municipality[f"mejora_{simple}"]
    gap = selection.familias.mejora[complex_] - selection.familias.mejora[simple]
    if abs(gap) < NEAR_TIE:
        flags.append(f"paso 2: diferencia de mejora media {gap:.3e}")

    comparisons = selection.comparaciones
    for _, row in comparisons[comparisons["diferencia"].abs() < NEAR_TIE].iterrows():
        if (row["candidato_opcion"], row["familia_opcion"]) != (row["candidato_referencia"], row["familia_referencia"]):
            flags.append(f"{row['regla']} {row['municipality_code']} {row['anio']}: |diferencia| < 1e-12 entre "
                         f"candidatos distintos")

    curve = selection.curva
    channel = {key: selection.canal[key] for key in CURVE_COLUMNS}
    curve_out = pd.concat([curve.assign(sistema="modelo"),
                           pd.DataFrame([{"tau": np.nan, **channel, "sistema": "canal"}])], ignore_index=True)
    probabilities = selection.predicciones["p"].to_numpy()
    for tau in config.decision.tau_grid:
        near = int((np.abs(probabilities - tau) < NEAR_TIE).sum())
        if near:
            flags.append(f"τ {tau}: {near} predicciones a menos de 1e-12 del umbral")

    point = curve[curve["tau"] == selection.tau.tau].iloc[0]
    summary = {
        "paso_1_configuracion_comun": common,
        "paso_2_familias": {"mejora_media_fuera_de_anio": selection.familias.mejora,
                            "victorias_lightgbm": selection.familias.victorias, "casillas": selection.familias.casillas,
                            "umbral": selection.familias.umbral, "reemplaza": selection.familias.reemplaza,
                            "familia_comun": selection.familias.familia_comun},
        "pasos_3_4_elecciones": selection.elecciones.to_dict(orient="records"),
        "tau": {"tau": selection.tau.tau, "restriccion_cumplida": selection.tau.restriccion_cumplida,
                "k": config.decision.k, "W": config.decision.W,
                "modelo": {"tau": float(point["tau"]), **{key: (float(point[key]) if isinstance(channel[key], float)
                                                               else int(point[key])) for key in CURVE_COLUMNS}},
                "canal": channel},
        "decisiones_por_menos_de_1e-12": flags,
        "reproducible": True,
    }
    return {"summary": summary, "cells": cells, "by_municipality": by_municipality, "comparisons": comparisons,
            "curve": curve_out, "margins": margins}


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"{type(value).__name__} is not JSON-serializable")


def write_evidence(output_dir: Path, tables: dict, recipe: dict, provenance: dict) -> None:
    """Write every evidence file; the folder must not hold any of them already."""
    output_dir.mkdir(parents=True, exist_ok=True)
    present = [name for name in EVIDENCE_FILES if (output_dir / name).exists()]
    require(not present, f"{output_dir} already has {present}; evidence is never overwritten.")
    summary = {"fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "commit_codigo": provenance["commit_codigo"], "busqueda_id": provenance["busqueda_id"],
               "insumos_md5": provenance["insumos_md5"], "version_dvc_canal": provenance["version_dvc_canal"],
               **tables["summary"]}
    (output_dir / "resumen_seleccion.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    tables["cells"].to_csv(output_dir / "casillas_fuera_de_anio.csv", index=False)
    tables["by_municipality"].to_csv(output_dir / "mejora_por_municipio.csv", index=False)
    tables["comparisons"].to_csv(output_dir / "comparaciones_excepciones.csv", index=False)
    tables["curve"].to_csv(output_dir / "curva_tau.csv", index=False)
    tables["margins"].to_csv(output_dir / "margenes_de_eleccion.csv", index=False)
    write_recipe_draft(recipe, output_dir / DRAFT_FILENAME)


# --------------------------------------------------------------------------- command line
def _dirty_paths() -> list[str]:
    result = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain=v1", "--untracked-files=all", "--",
                             "src", "configs"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.splitlines()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F6 selection: evidence and recipe draft (no retraining).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--busqueda", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--salida", type=Path, required=True)
    parser.add_argument("--permitir-cambios", action="store_true",
                        help="Run with uncommitted changes in src/ or configs/ (never for the recipe to approve).")
    args = parser.parse_args(argv)
    try:
        dirty = _dirty_paths()
        require(not dirty or args.permitir_cambios,
                f"src/ or configs/ have uncommitted changes; the draft must match a commit. Changes: {dirty[:10]}")
        config = load_config(args.config)
        provenance = search_provenance(args.busqueda, config=config)
        check_inputs_against_dvc(provenance)
        channel = Path(config.datos.canal)
        product = load_channel_product(channel if channel.is_absolute() else ROOT / channel,
                                       expected_md5=config.datos.version_dvc_canal)
        examples = build_examples(product, h=config.horizonte.h, canal=config.canal, features=config.features)
        rows, observed = development_rows(examples, config), product.to_pandas()
        table = pd.read_parquet(args.busqueda / TABLE_FILENAME)
        predictions = pd.read_parquet(args.busqueda / PREDICTIONS_FILENAME)
        with tempfile.TemporaryDirectory() as folder:
            access = RecipeAccess.from_config(config, recipe_path=Path(folder) / "sin_receta.yaml")
            result = select(table, predictions, rows, observed, config=config, access=access)
            again = select(table, predictions, rows, observed, config=config, access=access)
        require(_same(result, again), "Two runs of the selection differ; nothing is written.")
        recipe = build_recipe(result, config=config, provenance=provenance)
        write_evidence(args.salida, evidence(result, table, config), recipe, provenance)
        print(f"Selection written to {args.salida}: family {result.familias.familia_comun}, tau {result.tau.tau} "
              f"(constraint met: {result.tau.restriccion_cumplida}).")
    except Exception as exc:   # reported with its traceback
        traceback.print_exc()
        print(f"Selection stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
