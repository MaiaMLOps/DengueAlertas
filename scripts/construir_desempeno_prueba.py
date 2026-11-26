"""Static data of the frontend view «Desempeño en prueba» (docs/forecasting/desempeno_prueba.md).

Builds frontend/public/desempeno_prueba.json from the versioned evidence of the single test run and of the
complementary analysis; the frontend presents it without computing anything. No model is loaded, refitted or
scored: every figure is copied from the evidence. The only derivation is the weeks under alert of the product rule
(B01), recomputed per municipality with events.emission_positive("umbral_fuera_de_epidemia") and events.alerts with
the tau and k of the frozen recipe, from the confirmation week to the last week of each alert (both included); its
totals must equal the evidence.

Inputs (SOURCES) are listed in the output with their SHA-256. The digest is taken over the bytes stored in git:
the raw bytes for files marked -text in .gitattributes (the evidence) and the bytes with CRLF normalised to LF for
text files, so the output does not depend on the line endings of the checkout.

Output: UTF-8 JSON without ASCII escaping, indentation or whitespace, with the key order of the schema, a final
newline and no timestamps; probabilities and metrics rounded to 4 decimals. An existing output is never
overwritten, and writing is refused with uncommitted changes in src/ or configs/.

Run from the repository root:
    .venv/Scripts/python.exe -B scripts/construir_desempeno_prueba.py              # writes the default output
    .venv/Scripts/python.exe -B scripts/construir_desempeno_prueba.py --salida X   # writes X
    .venv/Scripts/python.exe -B scripts/construir_desempeno_prueba.py --verificar  # compares with the versioned file
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.epidemiology.validation import require  # noqa: E402
from src.forecasting.config import load_config  # noqa: E402
from src.forecasting.events import alerts, emission_positive  # noqa: E402
from src.forecasting.figures_reports import NAMES  # noqa: E402
from src.forecasting.guard import RecipeAccess, load_frozen_recipe  # noqa: E402

SCHEMA_VERSION = "1.0"
OUTPUT = Path("frontend/public/desempeno_prueba.json")
PERIOD = "prueba"
CARD = Path("reports/forecasting/evidence/prueba_20261112/ficha_desempeno.json")
SERIES = Path("reports/forecasting/evidence/prueba_20261112/diag_serie.csv")
TEST_EVENTS = Path("reports/forecasting/evidence/prueba_20261112/diag_eventos.csv")
B01_EVENTS = Path("reports/forecasting/evidence/complementario_20261120/b01_eventos.csv")
B01_ONSETS = Path("reports/forecasting/evidence/complementario_20261120/b01_inicios_entradas.csv")
B01_SUMMARY = Path("reports/forecasting/evidence/complementario_20261120/b01_resumen.json")
RECIPE = Path("configs/receta_congelada.yaml")
CONFIG = Path("configs/forecasting.yaml")
SOURCES = (CARD, SERIES, TEST_EVENTS, B01_EVENTS, B01_ONSETS, B01_SUMMARY, RECIPE)

NOTES = (
    "Resultados retrospectivos de la prueba 2023–2025, evaluada una sola vez con la receta congelada.",
    "Las probabilidades son las de los modelos de origen móvil de la prueba, no las del modelo desplegado.",
    "La regla del producto se definió después de la evaluación: sus cifras son descriptivas, no confirmatorias.",
)
# Rule key of the schema -> system of the complementary evidence; only the product rule was defined afterwards.
RULES = {"umbral": "regla_actual", "producto": "B01", "canal": "canal"}
POSTERIOR = {"umbral": False, "producto": True, "canal": False}
COUNT_FIELDS = ("inicios_avisados", "inicios_evaluables", "entradas_avisadas", "entradas_evaluables",
                "falsas_alarmas", "semanas_bajo_alerta", "semanas")
LEVELS = ("favorable", "mixto", "desfavorable")
PRODUCT_RULE = "umbral_fuera_de_epidemia"
DECIMALS = 4


# --------------------------------------------------------------------------- formatting helpers
def _num(value) -> float:
    """A metric or probability rounded to DECIMALS (never -0.0); NaN is not accepted."""
    number = float(value)
    require(math.isfinite(number), f"Non-finite value {value!r} where a number is required.")
    rounded = round(number, DECIMALS)
    return 0.0 if rounded == 0 else rounded


def _optional(value):
    """A lead in weeks, or None when the evidence leaves it empty (no warned event)."""
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else _num(value)


def _int(value) -> int:
    number = float(value)
    require(number.is_integer(), f"Expected an integer count; got {value!r}.")
    return int(number)


def _iso(value) -> str:
    return date.fromisoformat(str(value)).isoformat()


def _bool(value) -> bool:
    require(isinstance(value, (bool,)) or str(value) in ("True", "False"), f"Expected a boolean; got {value!r}.")
    return value if isinstance(value, bool) else str(value) == "True"


def serialize(document: dict) -> bytes:
    """Deterministic UTF-8 bytes: insertion key order, no ASCII escaping, no whitespace, final newline."""
    text = json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return (text + "\n").encode("utf-8")


# --------------------------------------------------------------------------- sources
def _git_text_attribute(path: Path, root: Path) -> str:
    result = subprocess.run(["git", "--no-optional-locks", "check-attr", "text", "--", path.as_posix()], cwd=root,
                            capture_output=True, text=True, check=True)
    return result.stdout.rsplit(":", 1)[-1].strip()


def source_digest(path: Path, root: Path = ROOT) -> str:
    """SHA-256 of the bytes stored in git: raw for -text files, CRLF normalised to LF for text files."""
    full = root / path
    require(full.is_file(), f"Missing source {path}.")
    content = full.read_bytes()
    if _git_text_attribute(path, root) != "unset":
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def _read_csv(path: Path, root: Path) -> pd.DataFrame:
    full = root / path
    require(full.is_file(), f"Missing source {path}.")
    return pd.read_csv(full, dtype={"municipality_code": str})


def _read_json(path: Path, root: Path) -> dict:
    full = root / path
    require(full.is_file(), f"Missing source {path}.")
    return json.loads(full.read_text(encoding="utf-8"))


def _load_recipe(root: Path) -> dict:
    """The frozen recipe, validated by the guard (hash, provenance and approval)."""
    config = load_config(root / CONFIG)
    access = RecipeAccess.from_config(config, recipe_path=root / RECIPE)
    return load_frozen_recipe(access.recipe_path, channel_version=access.channel_version)


# --------------------------------------------------------------------------- blocks
def recipe_block(card: dict, recipe: dict, summary: dict) -> dict:
    """hash_sha256 and familia_comun of the card; tau, k and W of the recipe; card and B01 summary must agree."""
    decision = recipe["decision"]
    block = {"hash_sha256": card["receta"]["hash_sha256"], "familia_comun": card["receta"]["familia_comun"],
             "tau": decision["tau"], "k": decision["k"], "W": decision["W"]}
    require(block["hash_sha256"] == recipe["hash_sha256"], "The card and the frozen recipe have different hashes.")
    require(block["familia_comun"] == recipe["familia_comun"], "The card and the frozen recipe differ in familia_comun.")
    for name in ("tau", "k", "W"):
        require(card["receta"][name] == block[name], f"The card and the frozen recipe differ in {name}.")
        require(summary[name] == block[name], f"The B01 summary and the frozen recipe differ in {name}.")
    require(summary["receta_hash"] == block["hash_sha256"], "The B01 summary was not built with the frozen recipe.")
    return block


def probability_block(estimate: dict) -> dict:
    """global.probabilidad from ficha_desempeno.json -> global.estimacion."""
    vs_b2, vs_b1 = estimate["vs_b2"]["conclusion"], estimate["vs_b1"]["conclusion"]
    return {"brier": {"modelo": _num(estimate["brier"]["modelo"]), "b2": _num(estimate["brier"]["B2"]),
                      "b1": _num(estimate["brier"]["B1"])},
            "bss_vs_b2": _num(estimate["bss_vs_b2"]), "bss_vs_b1": _num(estimate["bss_vs_b1"]),
            "ap": {"modelo": _num(estimate["ap"]["modelo"]), "b2": _num(estimate["ap"]["B2"])},
            "prevalencia": _num(estimate["prevalencia"]),
            "pares_ganados_vs_b2": _int(vs_b2["casillas_ganadas"]), "pares_total": _int(vs_b2["casillas_total"]),
            "nivel_vs_b2": vs_b2["nivel_conclusion"], "nivel_vs_b1": vs_b1["nivel_conclusion"]}


def card_block(estimate: dict) -> dict:
    """municipios.<code>.ficha from ficha_desempeno.json -> municipios.<code>.estimacion."""
    vs_b2 = estimate["vs_b2"]["conclusion"]
    require(vs_b2["nivel_descriptivo"] in LEVELS, f"Unexpected descriptive level {vs_b2['nivel_descriptivo']!r}.")
    return {"nivel": vs_b2["nivel_descriptivo"], "bss_vs_b2": _num(estimate["bss_vs_b2"]),
            "pares_ganados": _int(vs_b2["casillas_ganadas"]), "pares_total": _int(vs_b2["casillas_total"]),
            "ap": {"modelo": _num(estimate["ap"]["modelo"]), "b2": _num(estimate["ap"]["B2"])},
            "prevalencia": _num(estimate["prevalencia"])}


def _event_row(table: pd.DataFrame, system: str, level: str, code: str | None) -> pd.Series:
    rows = table[(table["sistema"] == system) & (table["nivel"] == level)]
    if code is not None:
        rows = rows[rows["municipality_code"] == code]
    require(len(rows) == 1, f"b01_eventos.csv: expected one row for {system}/{level}/{code}; got {len(rows)}.")
    return rows.iloc[0]


def events_block(table: pd.DataFrame, level: str, code: str | None = None) -> dict:
    """eventos of the three rules, from b01_eventos.csv (periodo = prueba)."""
    block = {}
    for rule, system in RULES.items():
        row = _event_row(table, system, level, code)
        entry = {field: _int(row[field]) for field in COUNT_FIELDS}
        entry["anticipacion_mediana"] = _optional(row["anticipacion_mediana_contrato"])
        entry["anticipacion_mediana_primera"] = _optional(row["anticipacion_mediana_primera"])
        entry["posterior"] = POSTERIOR[rule]
        block[rule] = entry
    return block


def onsets_block(onsets: pd.DataFrame, code: str) -> list[dict]:
    """Evaluable onsets of the test with whether each rule warned them, in chronological order."""
    own = onsets[onsets["municipality_code"] == code]
    result = []
    for week in sorted(own["week"].unique()):
        rows = own[own["week"] == week]
        warned = {}
        for rule, system in RULES.items():
            match = rows[rows["sistema"] == system]
            require(len(match) == 1, f"b01_inicios_entradas.csv: onset {code} {week} has {len(match)} rows of {system}.")
            warned[rule] = _bool(match.iloc[0]["warned"])
        result.append({"semana": _iso(week), "avisado": warned})
    return result


def product_under_alert(rows: pd.DataFrame, *, tau: float, k: int) -> list[bool]:
    """Weeks under alert of the product rule for one municipality's emissions (rows in chronological order)."""
    weeks = [date.fromisoformat(value) for value in rows["t"]]
    zones = list(rows["zona_t"])
    positive = emission_positive(PRODUCT_RULE, rows["p_modelo"].to_numpy(), zones, tau=tau)
    found = alerts(weeks, positive, zones, k=k)
    return [any(alert.confirmation <= week <= alert.end for alert in found) for week in weeks]


def series_block(rows: pd.DataFrame, product: list[bool]) -> list[dict]:
    """municipios.<code>.serie from diag_serie.csv, in chronological order."""
    result = []
    for row, under_product in zip(rows.itertuples(index=False), product):
        require(row.y in (0, 1), f"diag_serie.csv: y must be 0 or 1; got {row.y!r}.")
        result.append({"semana_emision": _iso(row.t), "semana_objetivo": _iso(row.target_week),
                       "probabilidad": _num(row.p_modelo), "b2_estado_canal": _num(row.p_b2),
                       "b1_climatologia_estacional": _num(row.p_b1), "zona_t": row.zona_t,
                       "zona_epidemica_observada": bool(row.y == 1), "casos_objetivo": _int(row.casos_objetivo),
                       "central_objetivo": _num(row.central_objetivo), "u1e_objetivo": _num(row.u1e_objetivo),
                       "bajo_alerta": {"umbral": _bool(row.alerta_modelo), "producto": bool(under_product),
                                       "canal": _bool(row.alerta_canal)}})
    return result


# --------------------------------------------------------------------------- document
def build_document(root: Path = ROOT) -> dict:
    """The whole document, validated against the evidence it copies."""
    card = _read_json(CARD, root)
    summary = _read_json(B01_SUMMARY, root)
    recipe = _load_recipe(root)
    receta = recipe_block(card, recipe, summary)

    series = _read_csv(SERIES, root)
    require(set(series["rotulo"]) == {PERIOD}, "diag_serie.csv must hold only test rows.")
    require(not series.duplicated(["municipality_code", "t"]).any(), "diag_serie.csv: duplicated emissions.")
    events = _read_csv(B01_EVENTS, root)
    events = events[events["periodo"] == PERIOD]
    onsets = _read_csv(B01_ONSETS, root)
    onsets = onsets[(onsets["periodo"] == PERIOD) & (onsets["kind"] == "inicio")
                    & (onsets["evaluable"].map(_bool))]

    codes = sorted(card["municipios"])
    require(codes == sorted(NAMES), "The card and figures_reports.NAMES list different municipalities.")
    require(codes == sorted(series["municipality_code"].unique()), "diag_serie.csv and the card differ in municipalities.")

    global_events = events_block(events, "global")
    for rule, system in RULES.items():
        expected = summary["periodos"][PERIOD]["global"][system]
        require(all(global_events[rule][field] == expected[field] for field in COUNT_FIELDS),
                f"b01_eventos.csv and b01_resumen.json differ in the global events of {system}.")

    municipalities = {}
    for code in codes:
        rows = series[series["municipality_code"] == code].sort_values("t").reset_index(drop=True)
        product = product_under_alert(rows, tau=receta["tau"], k=receta["k"])
        own_events = events_block(events, "municipio", code)
        observed = {"umbral": int(rows["alerta_modelo"].map(_bool).sum()), "producto": int(sum(product)),
                    "canal": int(rows["alerta_canal"].map(_bool).sum())}
        for rule, weeks in observed.items():
            require(weeks == own_events[rule]["semanas_bajo_alerta"],
                    f"{code}: {weeks} weeks under alert of {rule} in the series; the evidence has "
                    f"{own_events[rule]['semanas_bajo_alerta']}.")
            require(len(rows) == own_events[rule]["semanas"], f"{code}: series length differs from the evidence.")
        municipalities[code] = {"nombre": NAMES[code], "ficha": card_block(card["municipios"][code]["estimacion"]),
                                "eventos": own_events, "inicios": onsets_block(onsets, code),
                                "serie": series_block(rows, product)}

    return {"version_esquema": SCHEMA_VERSION,
            "fuentes": [{"ruta": path.as_posix(), "sha256": source_digest(path, root)} for path in SOURCES],
            "receta": receta, "notas": list(NOTES),
            "global": {"probabilidad": probability_block(card["global"]["estimacion"]), "eventos": global_events},
            "municipios": municipalities}


def _dirty_paths(root: Path) -> list[str]:
    result = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain=v1", "--untracked-files=all", "--",
                             "src", "configs"], cwd=root, capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Data of the frontend view «Desempeño en prueba».")
    parser.add_argument("--salida", type=Path, default=OUTPUT,
                        help=f"Output path, relative to the repository root (default: {OUTPUT.as_posix()}).")
    parser.add_argument("--verificar", action="store_true",
                        help="Rebuild in memory and compare byte by byte with the output; exit 1 if they differ.")
    args = parser.parse_args(argv)
    target = args.salida if args.salida.is_absolute() else ROOT / args.salida
    content = serialize(build_document(ROOT))
    if args.verificar:
        if not target.is_file():
            print(f"{args.salida.as_posix()} does not exist.", file=sys.stderr)
            return 1
        if target.read_bytes() != content:
            print(f"{args.salida.as_posix()} differs from the data rebuilt from the evidence.", file=sys.stderr)
            return 1
        print(f"{args.salida.as_posix()} matches the evidence ({len(content)} bytes).")
        return 0
    dirty = _dirty_paths(ROOT)
    require(not dirty, f"src/ or configs/ have uncommitted changes; the output must match a commit: {dirty[:10]}")
    require(not target.exists(), f"{args.salida.as_posix()} already exists and is never overwritten; check it with "
                                 "--verificar or remove it before an approved regeneration.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    print(f"Wrote {args.salida.as_posix()} ({len(content)} bytes, sha256 {hashlib.sha256(content).hexdigest()}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
