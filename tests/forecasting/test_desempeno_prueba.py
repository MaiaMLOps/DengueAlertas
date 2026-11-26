"""frontend/public/desempeno_prueba.json against its schema (docs/forecasting/desempeno_prueba.md) and the evidence.

The file is rebuilt from the versioned evidence by scripts/construir_desempeno_prueba.py; the tests are skipped
when that evidence is not available locally. Expected figures are the published ones (report tables), compared at
their published rounding.
"""

from datetime import date
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("construir_desempeno_prueba",
                                               ROOT / "scripts" / "construir_desempeno_prueba.py")
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)

OUTPUT = ROOT / builder.OUTPUT
CODES = ["08001", "08758", "41001", "50001", "54001", "76001"]
RULES = ["umbral", "producto", "canal"]
LAST_EMISSION = date(2025, 10, 27)

TOP_KEYS = ["version_esquema", "fuentes", "receta", "notas", "global", "municipios"]
SOURCE_KEYS = ["ruta", "sha256"]
RECIPE_KEYS = ["hash_sha256", "familia_comun", "tau", "k", "W"]
GLOBAL_KEYS = ["probabilidad", "eventos"]
PROBABILITY_KEYS = ["brier", "bss_vs_b2", "bss_vs_b1", "ap", "prevalencia", "pares_ganados_vs_b2", "pares_total",
                    "nivel_vs_b2", "nivel_vs_b1"]
EVENT_KEYS = ["inicios_avisados", "inicios_evaluables", "entradas_avisadas", "entradas_evaluables", "falsas_alarmas",
              "semanas_bajo_alerta", "semanas", "anticipacion_mediana", "anticipacion_mediana_primera", "posterior"]
MUNICIPALITY_KEYS = ["nombre", "ficha", "eventos", "inicios", "serie"]
CARD_KEYS = ["nivel", "bss_vs_b2", "pares_ganados", "pares_total", "ap", "prevalencia"]
ONSET_KEYS = ["semana", "avisado"]
SERIES_KEYS = ["semana_emision", "semana_objetivo", "probabilidad", "b2_estado_canal", "b1_climatologia_estacional",
               "zona_t", "zona_epidemica_observada", "casos_objetivo", "central_objetivo", "u1e_objetivo",
               "bajo_alerta"]
COUNTS = ["inicios_avisados", "inicios_evaluables", "entradas_avisadas", "entradas_evaluables", "falsas_alarmas",
          "semanas_bajo_alerta", "semanas"]

# Global events (Tabla 5): rule -> (onsets warned of 10, entries warned of 41, false alarms, weeks of 936).
GLOBAL_EVENTS = {"umbral": (4, 13, 18, 565), "producto": (9, 24, 26, 192), "canal": (8, 19, 30, 393)}
# Tabla 6: level, BSS vs B2 (3 decimals), pairs won of 3, AP model / B2 (2 decimals), onsets warned per rule and
# evaluable onsets (None: no onsets), false alarms per rule (umbral, producto, canal).
CARDS = {
    "41001": ("favorable", 0.197, 3, (0.78, 0.62), (2, 3, 3), 3, (5, 6, 6)),
    "50001": ("favorable", 0.106, 2, (0.82, 0.67), (0, 0, 0), 1, (5, 5, 4)),
    "76001": ("favorable", 0.239, 3, (0.95, 0.81), (0, 0, 0), None, (1, 1, 3)),
    "08758": ("mixto", 0.029, 2, (0.69, 0.48), (1, 3, 2), 3, (3, 5, 6)),
    "54001": ("mixto", 0.006, 1, (0.66, 0.59), (1, 2, 2), 2, (2, 4, 6)),
    "08001": ("desfavorable", -0.053, 1, (0.59, 0.60), (0, 1, 1), 1, (2, 5, 5)),
}


@pytest.fixture(scope="module")
def document():
    missing = [path.as_posix() for path in builder.SOURCES if not (ROOT / path).is_file()]
    if missing:
        pytest.skip(f"Evidence of the test run not available locally; missing {missing}.")
    return json.loads(OUTPUT.read_text(encoding="utf-8"))


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_iso(value) -> bool:
    return isinstance(value, str) and date.fromisoformat(value).isoformat() == value


# --------------------------------------------------------------------------- 1. reproducibility
def test_the_versioned_file_is_rebuilt_byte_by_byte(document):
    assert builder.main(["--verificar"]) == 0


def test_verify_fails_when_the_file_differs(document, tmp_path):
    altered = tmp_path / "desempeno_prueba.json"
    altered.write_bytes(OUTPUT.read_bytes().replace(b'"version_esquema":"1.0"', b'"version_esquema":"0.9"'))
    assert builder.main(["--verificar", "--salida", str(altered)]) == 1


def test_serialization_is_compact_utf8_with_a_final_newline(document):
    content = OUTPUT.read_bytes()
    assert content.endswith(b"}\n") and content.count(b"\n") == 1
    assert b"\\u" not in content and "Cúcuta".encode("utf-8") in content
    compact = json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    assert content == compact.encode("utf-8")


# --------------------------------------------------------------------------- 2. schema
def _check_events(block):
    assert list(block) == RULES
    for rule, entry in block.items():
        assert list(entry) == EVENT_KEYS
        assert all(_is_count(entry[key]) for key in COUNTS)
        for key in ("anticipacion_mediana", "anticipacion_mediana_primera"):
            assert entry[key] is None or _is_number(entry[key])
        assert entry["posterior"] is (rule == "producto")


def test_header_follows_the_schema(document):
    assert list(document) == TOP_KEYS
    assert document["version_esquema"] == "1.0"
    assert [entry["ruta"] for entry in document["fuentes"]] == [path.as_posix() for path in builder.SOURCES]
    assert all(list(entry) == SOURCE_KEYS and len(entry["sha256"]) == 64 for entry in document["fuentes"])
    assert list(document["receta"]) == RECIPE_KEYS
    assert document["receta"] == {"hash_sha256": "de74d4abd33537433ffc9d1572c6cdd8620a2608fcc2b67ee4a1752aa865fda5",
                                  "familia_comun": "logistica", "tau": 0.05, "k": 2, "W": 12}
    assert document["notas"] == [
        "Resultados retrospectivos de la prueba 2023–2025, evaluada una sola vez con la receta congelada.",
        "Las probabilidades son las de los modelos de origen móvil de la prueba, no las del modelo desplegado.",
        "La regla del producto se definió después de la evaluación: sus cifras son descriptivas, no confirmatorias.",
    ]


def test_global_block_follows_the_schema(document):
    block = document["global"]
    assert list(block) == GLOBAL_KEYS
    probability = block["probabilidad"]
    assert list(probability) == PROBABILITY_KEYS
    assert list(probability["brier"]) == ["modelo", "b2", "b1"] and list(probability["ap"]) == ["modelo", "b2"]
    assert all(_is_number(value) for value in [*probability["brier"].values(), *probability["ap"].values(),
                                               probability["bss_vs_b2"], probability["bss_vs_b1"],
                                               probability["prevalencia"]])
    assert _is_count(probability["pares_ganados_vs_b2"]) and _is_count(probability["pares_total"])
    _check_events(block["eventos"])


def test_municipalities_follow_the_schema(document):
    municipalities = document["municipios"]
    assert list(municipalities) == CODES
    for code, entry in municipalities.items():
        assert list(entry) == MUNICIPALITY_KEYS
        assert entry["nombre"] == builder.NAMES[code]
        card = entry["ficha"]
        assert list(card) == CARD_KEYS and list(card["ap"]) == ["modelo", "b2"]
        assert card["nivel"] in ("favorable", "mixto", "desfavorable")
        assert _is_number(card["bss_vs_b2"]) and _is_number(card["prevalencia"])
        assert _is_count(card["pares_ganados"]) and _is_count(card["pares_total"])
        _check_events(entry["eventos"])
        for onset in entry["inicios"]:
            assert list(onset) == ONSET_KEYS and _is_iso(onset["semana"])
            assert list(onset["avisado"]) == RULES and all(isinstance(v, bool) for v in onset["avisado"].values())
        assert [onset["semana"] for onset in entry["inicios"]] == sorted(onset["semana"] for onset in entry["inicios"])
        series = entry["serie"]
        assert len(series) == 156
        assert [row["semana_emision"] for row in series] == sorted(row["semana_emision"] for row in series)
        for row in series:
            assert list(row) == SERIES_KEYS
            assert _is_iso(row["semana_emision"]) and _is_iso(row["semana_objetivo"])
            assert all(_is_number(row[key]) for key in ("probabilidad", "b2_estado_canal",
                                                        "b1_climatologia_estacional", "central_objetivo",
                                                        "u1e_objetivo"))
            assert _is_count(row["casos_objetivo"]) and isinstance(row["zona_epidemica_observada"], bool)
            assert row["zona_t"] in ("seguridad", "alerta", "epidemia")
            assert list(row["bajo_alerta"]) == RULES
            assert all(isinstance(value, bool) for value in row["bajo_alerta"].values())


def test_numbers_have_at_most_four_decimals(document):
    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, float):
            assert round(value, 4) == value

    walk(document)


# --------------------------------------------------------------------------- 3. global figures
def test_global_probability_figures_match_the_report(document):
    probability = document["global"]["probabilidad"]
    assert round(probability["bss_vs_b2"], 3) == 0.081
    assert round(probability["bss_vs_b1"], 3) == 0.468
    assert round(probability["brier"]["modelo"], 3) == 0.141
    assert round(probability["brier"]["b2"], 3) == 0.154
    assert round(probability["ap"]["modelo"], 2) == 0.77
    assert round(probability["ap"]["b2"], 2) == 0.61
    assert (probability["pares_ganados_vs_b2"], probability["pares_total"]) == (12, 18)
    assert probability["nivel_vs_b2"] == "tendencia_favorable"
    assert probability["nivel_vs_b1"] == "mejora_consistente"


# --------------------------------------------------------------------------- 4. global events
def test_global_events_match_the_report(document):
    events = document["global"]["eventos"]
    for rule, (onsets, entries, false_alarms, weeks) in GLOBAL_EVENTS.items():
        entry = events[rule]
        assert (entry["inicios_avisados"], entry["inicios_evaluables"]) == (onsets, 10)
        assert (entry["entradas_avisadas"], entry["entradas_evaluables"]) == (entries, 41)
        assert entry["falsas_alarmas"] == false_alarms
        assert (entry["semanas_bajo_alerta"], entry["semanas"]) == (weeks, 936)


# --------------------------------------------------------------------------- 5. municipal cards
@pytest.mark.parametrize("code", list(CARDS))
def test_municipal_card_matches_the_report(document, code):
    level, bss, pairs, (ap_model, ap_b2), warned, evaluable, false_alarms = CARDS[code]
    entry = document["municipios"][code]
    card = entry["ficha"]
    assert card["nivel"] == level
    assert round(card["bss_vs_b2"], 3) == bss
    assert (card["pares_ganados"], card["pares_total"]) == (pairs, 3)
    assert (round(card["ap"]["modelo"], 2), round(card["ap"]["b2"], 2)) == (ap_model, ap_b2)
    events = entry["eventos"]
    assert tuple(events[rule]["inicios_avisados"] for rule in RULES) == warned
    assert all(events[rule]["inicios_evaluables"] == (evaluable or 0) for rule in RULES)
    assert len(entry["inicios"]) == (evaluable or 0)
    assert tuple(events[rule]["falsas_alarmas"] for rule in RULES) == false_alarms


# --------------------------------------------------------------------------- 6. internal coherence
def test_municipalities_add_up_to_the_global_events(document):
    for rule in RULES:
        for key in COUNTS:
            total = sum(entry["eventos"][rule][key] for entry in document["municipios"].values())
            assert total == document["global"]["eventos"][rule][key], (rule, key)


def test_series_weeks_under_alert_match_the_events(document):
    for code, entry in document["municipios"].items():
        for rule in RULES:
            weeks = sum(row["bajo_alerta"][rule] for row in entry["serie"])
            assert weeks == entry["eventos"][rule]["semanas_bajo_alerta"], (code, rule)


def test_onsets_and_warnings_match_the_test_events(document):
    events = pd.read_csv(ROOT / builder.TEST_EVENTS, dtype={"municipality_code": str})
    events = events[(events["tipo"] == "inicio") & events["evaluable"]]
    for code, entry in document["municipios"].items():
        own = events[events["municipality_code"] == code].sort_values("semana")
        expected = [{"semana": row.semana, "umbral": bool(row.avisado_modelo), "canal": bool(row.avisado_canal)}
                    for row in own.itertuples(index=False)]
        actual = [{"semana": onset["semana"], "umbral": onset["avisado"]["umbral"],
                   "canal": onset["avisado"]["canal"]} for onset in entry["inicios"]]
        assert actual == expected, code


def test_onset_warnings_add_up_to_the_events(document):
    for code, entry in document["municipios"].items():
        for rule in RULES:
            warned = sum(onset["avisado"][rule] for onset in entry["inicios"])
            assert warned == entry["eventos"][rule]["inicios_avisados"], (code, rule)


# --------------------------------------------------------------------------- 7. no final model
def test_no_emission_after_the_last_test_emission(document):
    emissions = [date.fromisoformat(row["semana_emision"]) for entry in document["municipios"].values()
                 for row in entry["serie"]]
    assert max(emissions) == LAST_EMISSION
    assert min(emissions) == date(2022, 11, 7)
