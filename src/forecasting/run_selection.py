"""Command line of the F6 selection: evidence and recipe draft from the saved development search
(contrato §6.2, §7.5 and §8.3; especificación §6.F6, §7 and §8).

Flow (nothing is retrained):
    uncommitted-changes check -> load_config -> search_provenance -> check the inputs against their DVC
    descriptors (md5)
    -> load_channel_product (md5) -> build_examples -> development rows of splits.yearly_trials
    -> selection.select twice (the result must be identical) -> evidence tables
    -> write the evidence folder and the recipe draft (selection.write_recipe_draft), never overwriting.

Evidence files (folder given by --salida; none of these files may exist in it):
    resumen_seleccion.json       steps 1-4 and tau in the pre-registered order, inputs and margins < 1e-12
    casillas_fuera_de_anio.csv   per (municipality, year): common out-of-year configuration of each family
    mejora_por_municipio.csv     mean out-of-year improvement per municipality and minimum B2 log-loss
    comparaciones_excepciones.csv  family and hyperparameter comparisons with their annual differences
    curva_tau.csv                event metrics per tau, with the channel as a reference row
    margenes_de_eleccion.csv     margin of every choice over the second candidate
    receta_borrador.yaml         recipe draft of especificación §8, without approval or hash

D7.2 mode (contrato §6.6), with --busqueda-sin-clima: the two searches (full set in --busqueda, F5b; sin_clima in
--busqueda-sin-clima) must share examples and configuration; selection.select_with_feature_sets runs twice (must be
identical) and the evidence adds:
    comparacion_conjuntos.csv     step 0: per family and (municipality, year), common out-of-year configuration of
                                  each set, log-loss, improvement and win of the full set (24 cells per family)
    ablacion.csv                  ablation per municipality: final family, the other set and its configuration
    comparaciones_ablacion.csv    hyperparameter comparisons of the ablation, with their annual differences
and resumen_seleccion.json adds paso_0_conjuntos, conjuntos_municipio and ablacion. The draft is
selection.build_recipe_with_feature_sets. --conjunto-forzado completo reproduces F6 (check only).
The sin_clima inputs are checked against their DVC descriptors when these exist; otherwise their md5 is recorded
and the summary says that the descriptors are pending.

The test period is never read: selection uses development rows only, and the recipe is a draft.

Usage:
    python -m src.forecasting.run_selection --salida reports/forecasting/evidence/seleccion_YYYYMMDD
        [--config P] [--busqueda data/processed/forecasting] [--permitir-cambios]
        [--busqueda-sin-clima data/processed/forecasting/sin_clima [--conjunto-forzado completo|sin_clima]]
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
from .dataset import FEATURE_SETS, build_examples, load_channel_product
from .guard import RecipeAccess
from .models import FAMILIES
from .search import DEFAULT_OUTPUT_DIR, MANIFEST_FILENAME, PREDICTIONS_FILENAME, TABLE_FILENAME
from .selection import (CURVE_COLUMNS, Selection, SetSelection, build_recipe, build_recipe_with_feature_sets, select,
                        select_with_feature_sets, write_recipe_draft)
from .splits import yearly_trials

CONTRACT_PATH = "docs/forecasting/contrato_predictivo.md"
PROVENANCE_INPUTS = (TABLE_FILENAME, PREDICTIONS_FILENAME, MANIFEST_FILENAME)
DVC_DIR = DEFAULT_OUTPUT_DIR            # the F5b descriptors are versioned next to the data
DRAFT_FILENAME = "receta_borrador.yaml"
EVIDENCE_FILES = ("resumen_seleccion.json", "casillas_fuera_de_anio.csv", "mejora_por_municipio.csv",
                  "comparaciones_excepciones.csv", "curva_tau.csv", "margenes_de_eleccion.csv", DRAFT_FILENAME)
SET_EVIDENCE_FILES = ("comparacion_conjuntos.csv", "ablacion.csv", "comparaciones_ablacion.csv")
NEAR_TIE = 1e-12                        # a choice decided by less than this is reported (F6)


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
    names = EVIDENCE_FILES + (SET_EVIDENCE_FILES if "set_cells" in tables else ())
    present = [name for name in names if (output_dir / name).exists()]
    require(not present, f"{output_dir} already has {present}; evidence is never overwritten.")
    if "busquedas" in provenance:
        searches = {"busquedas": provenance["busquedas"]}
    else:
        searches = {"busqueda_id": provenance["busqueda_id"], "insumos_md5": provenance["insumos_md5"]}
    summary = {"fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "commit_codigo": provenance["commit_codigo"], **searches,
               "version_dvc_canal": provenance["version_dvc_canal"], **tables["summary"]}
    (output_dir / "resumen_seleccion.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    tables["cells"].to_csv(output_dir / "casillas_fuera_de_anio.csv", index=False)
    tables["by_municipality"].to_csv(output_dir / "mejora_por_municipio.csv", index=False)
    tables["comparisons"].to_csv(output_dir / "comparaciones_excepciones.csv", index=False)
    tables["curve"].to_csv(output_dir / "curva_tau.csv", index=False)
    tables["margins"].to_csv(output_dir / "margenes_de_eleccion.csv", index=False)
    if "set_cells" in tables:
        tables["set_cells"].to_csv(output_dir / "comparacion_conjuntos.csv", index=False)
        tables["ablation"].to_csv(output_dir / "ablacion.csv", index=False)
        tables["ablation_comparisons"].to_csv(output_dir / "comparaciones_ablacion.csv", index=False)
    write_recipe_draft(recipe, output_dir / DRAFT_FILENAME)


# --------------------------------------------------------------------------- D7.2 mode
def _manifest(search_dir: Path) -> dict:
    return json.loads((Path(search_dir) / MANIFEST_FILENAME).read_text(encoding="utf-8"))


def check_paired_searches(full_dir: Path, simple_dir: Path) -> None:
    """The two searches must differ only in the feature set: same examples, configuration, years, municipalities
    and candidates; the full-set search has no variant and the other is the sin_clima variant."""
    full, simple = _manifest(full_dir)["contenido"], _manifest(simple_dir)["contenido"]
    require("variante" not in full, f"{full_dir} must be the full-set search (it has a variant).")
    variant = simple.get("variante", {})
    require(variant.get("nombre") == "sin_clima" and tuple(variant.get("features", ())) == FEATURE_SETS["sin_clima"],
            f"{simple_dir} must be the sin_clima search with the features {list(FEATURE_SETS['sin_clima'])}.")
    for key in ("anios", "prueba", "primer_anio_entrenamiento", "municipios", "configuracion", "candidatos",
                "ejemplos_sha256"):
        require(full.get(key) == simple.get(key), f"The two searches differ in {key}; the comparison would be unpaired.")


def _dvc_status(provenance: dict, folder: Path) -> str:
    descriptors = [folder / f"{name}.dvc" for name in provenance["insumos_md5"]]
    if not any(path.is_file() for path in descriptors):
        return "pendiente (sin descriptores .dvc)"
    check_inputs_against_dvc(provenance, folder)
    return "verificado contra los descriptores .dvc"


def _same_sets(first: SetSelection, second: SetSelection) -> bool:
    return (_same(first.seleccion, second.seleccion) and first.conjuntos == second.conjuntos
            and first.conjuntos_municipio == second.conjuntos_municipio
            and first.celdas_conjuntos.equals(second.celdas_conjuntos) and first.ablacion.equals(second.ablacion)
            and first.comparaciones_ablacion.equals(second.comparaciones_ablacion))


def set_evidence(result: SetSelection, tables: dict, config: ForecastConfig, forced_set: str | None) -> dict:
    """F6 evidence on the chosen sets plus the step-0 comparison and the ablation."""
    chosen = {family: forced_set or result.conjuntos[family].conjunto for family in FAMILIES}
    table = pd.concat([tables[chosen[family]][tables[chosen[family]]["familia"] == family] for family in FAMILIES],
                      ignore_index=True)
    out = evidence(result.seleccion, table, config)
    summary = out["summary"]
    step0 = {}
    for family, choice in result.conjuntos.items():
        step0[family] = {"conjunto": choice.conjunto, "mejora_media_fuera_de_anio": choice.mejora,
                         "victorias_completo": choice.victorias, "casillas": choice.casillas, "umbral": choice.umbral,
                         "reemplaza": choice.reemplaza, "usado": chosen[family]}
        gap = choice.mejora["completo"] - choice.mejora["sin_clima"]
        if abs(gap) < NEAR_TIE:
            summary["decisiones_por_menos_de_1e-12"].append(f"paso 0 {family}: diferencia de mejora media {gap:.3e}")
    cells = result.celdas_conjuntos
    for row in cells[(cells["logloss_completo"] - cells["logloss_sin_clima"]).abs() < NEAR_TIE].itertuples():
        summary["decisiones_por_menos_de_1e-12"].append(
            f"paso 0 {row.familia} {row.municipality_code} {row.anio}: |diferencia logloss| < 1e-12")
    out["summary"] = {"conjunto_forzado": forced_set, "paso_0_conjuntos": step0,
                      "conjuntos_municipio": result.conjuntos_municipio, **summary,
                      "ablacion": result.ablacion.to_dict(orient="records")}
    out.update(set_cells=cells, ablation=result.ablacion, ablation_comparisons=result.comparaciones_ablacion)
    return out


def _run_feature_sets(args, config: ForecastConfig) -> None:
    full_provenance = search_provenance(args.busqueda, config=config)
    check_inputs_against_dvc(full_provenance)
    simple_provenance = search_provenance(args.busqueda_sin_clima, config=config)
    check_paired_searches(args.busqueda, args.busqueda_sin_clima)
    status = _dvc_status(simple_provenance, args.busqueda_sin_clima)
    provenance = {key: full_provenance[key] for key in ("commit_codigo", "commit_contrato", "version_dvc_canal")}
    provenance["busquedas"] = {
        "completo": {"busqueda_id": full_provenance["busqueda_id"], "insumos_md5": full_provenance["insumos_md5"]},
        "sin_clima": {"busqueda_id": simple_provenance["busqueda_id"], "insumos_md5": simple_provenance["insumos_md5"]}}
    channel = Path(config.datos.canal)
    product = load_channel_product(channel if channel.is_absolute() else ROOT / channel,
                                   expected_md5=config.datos.version_dvc_canal)
    examples = build_examples(product, h=config.horizonte.h, canal=config.canal, features=config.features)
    rows, observed = development_rows(examples, config), product.to_pandas()
    folders = {"completo": args.busqueda, "sin_clima": args.busqueda_sin_clima}
    tables = {name: pd.read_parquet(folder / TABLE_FILENAME) for name, folder in folders.items()}
    predictions = {name: pd.read_parquet(folder / PREDICTIONS_FILENAME) for name, folder in folders.items()}
    with tempfile.TemporaryDirectory() as folder:
        access = RecipeAccess.from_config(config, recipe_path=Path(folder) / "sin_receta.yaml")
        result = select_with_feature_sets(tables, predictions, rows, observed, config=config, access=access,
                                          forced_set=args.conjunto_forzado)
        again = select_with_feature_sets(tables, predictions, rows, observed, config=config, access=access,
                                         forced_set=args.conjunto_forzado)
    require(_same_sets(result, again), "Two runs of the selection differ; nothing is written.")
    recipe = build_recipe_with_feature_sets(result, config=config, provenance=provenance)
    tables_out = set_evidence(result, tables, config, args.conjunto_forzado)
    tables_out["summary"]["descriptores_dvc_sin_clima"] = status
    write_evidence(args.salida, tables_out, recipe, provenance)
    print(f"Selection written to {args.salida}: sets {dict((f, c.conjunto) for f, c in result.conjuntos.items())}, "
          f"family {result.seleccion.familias.familia_comun}, tau {result.seleccion.tau.tau} "
          f"(constraint met: {result.seleccion.tau.restriccion_cumplida}).")


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
    parser.add_argument("--busqueda-sin-clima", type=Path, default=None,
                        help="Folder of the sin_clima search: runs the selection of D7.2 (contrato §6.6).")
    parser.add_argument("--conjunto-forzado", choices=sorted(FEATURE_SETS), default=None,
                        help="Skip step 0 and use this set for both families (check only; requires "
                             "--busqueda-sin-clima).")
    args = parser.parse_args(argv)
    try:
        dirty = _dirty_paths()
        require(not dirty or args.permitir_cambios,
                f"src/ or configs/ have uncommitted changes; the draft must match a commit. Changes: {dirty[:10]}")
        config = load_config(args.config)
        require(args.conjunto_forzado is None or args.busqueda_sin_clima is not None,
                "--conjunto-forzado requires --busqueda-sin-clima.")
        if args.busqueda_sin_clima is not None:
            _run_feature_sets(args, config)
            return 0
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
