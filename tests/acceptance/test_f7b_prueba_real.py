"""F7b acceptance with the outputs of the single test run (--prueba-dir, --channel-parquet and --sin-clima-dir), with
the recipe of D7.2 (contrato §6.1, §6.6, §8.4, §9.4 and §10.18; api_salida_d10.md §6 and §7; especificación §6.F7;
DV-02). Replaces the five tests of test_f7_prueba_real.py superseded by DV-02 (log, predictions, card, history and
final models); the other tests of that file stay in force. Approved by Jesús on 2026-11-12 (U7.3): same claims as
F7, with the values of the new recipe and the ablation read from it.

Every test is named `test_prueba_real_*`: until Jesús approves and runs the test they are deselected with
-k "not prueba_real" (reported as omitted, pending). They check structure, provenance and internal consistency,
never a predetermined result. Recipe of D7.2 (configs/receta_congelada.yaml, 6fc68f6): hash de74d4ab…fda5; model
logistica_05 (C = 3) with the sin_clima set in the six municipalities; ablation logistica_03 (C = 0.3) with the full
set; tau = 0.05. History (decision D2): development shows the saved predictions of the recipe configuration
(logistica_05 of the sin_clima search, «usado para elegir»); test shows the predictions of the single run.

Files expected in --prueba-dir: predicciones_prueba.parquet, ensayos_prueba.parquet, historial.parquet,
ficha_desempeno.json and modelos_finales/manifiesto.json (+ <code>.joblib). Log: reports/forecasting/bitacora_prueba.jsonl.
"""

from datetime import date
import functools
import json
from pathlib import Path
import re

import pandas as pd
import pytest

pytestmark = pytest.mark.acceptance

ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "reports" / "forecasting" / "bitacora_prueba.jsonl"
RECIPE_HASH = "de74d4abd33537433ffc9d1572c6cdd8620a2608fcc2b67ee4a1752aa865fda5"
CHANNEL_MD5 = "99fd12311a261264019645d906631cfc"
CODES = ("08001", "08758", "41001", "50001", "54001", "76001")
SYSTEMS = ("modelo", "ablacion", "B2", "B1")
ROWS_PER_YEAR = {2023: 52, 2024: 53, 2025: 51}   # per municipality; 2025-12-29 is partial and not eligible
LAST_ORIGIN, LAST_TARGET = date(2025, 10, 27), date(2025, 12, 22)
MODEL = ("logistica_05", "sin_clima")
ABLATION = ("logistica_03", "completo")
TAU = 0.05


@functools.lru_cache(maxsize=None)
def _examples(channel_file: str) -> pd.DataFrame:
    from src.forecasting.config import load_config
    from src.forecasting.dataset import build_examples, load_channel_product

    config = load_config()
    product = load_channel_product(channel_file, expected_md5=config.datos.version_dvc_canal)
    return build_examples(product, h=config.horizonte.h, canal=config.canal, features=config.features)


def _predictions(prueba_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(prueba_dir / "predicciones_prueba.parquet")


def test_prueba_real_log_has_one_execution(prueba_dir):
    entries = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["ejecucion"] == 1 and entry["razon"] is None
    assert entry["receta_hash"] == RECIPE_HASH and entry["version_dvc_canal"] == CHANNEL_MD5
    assert re.fullmatch(r"[0-9a-f]{40}", entry["commit"])


def test_prueba_real_predictions_cover_the_test_rows(prueba_dir, channel_path):
    predictions = _predictions(prueba_dir)
    examples = _examples(str(channel_path))
    test_rows = examples[examples["target_week"].map(lambda week: 2023 <= week.year <= 2025)]
    assert len(test_rows) == 936 and int(test_rows["y"].sum()) == 322
    expected = set(zip(test_rows["municipality_code"], test_rows["t"]))
    labels = dict(zip(zip(test_rows["municipality_code"], test_rows["t"]), test_rows["y"]))
    assert sorted(set(predictions["sistema"])) == sorted(SYSTEMS)
    for system in SYSTEMS:
        own = predictions[predictions["sistema"] == system]
        assert len(own) == 936 and set(zip(own["municipality_code"], own["t"])) == expected
        assert all(labels[(c, t)] == y for c, t, y in zip(own["municipality_code"], own["t"], own["y"]))
        counts = own.groupby(["municipality_code", "anio"]).size()
        assert all(counts[(code, year)] == n for code in CODES for year, n in ROWS_PER_YEAR.items())
        assert own["p"].between(0, 1).all()
    model = predictions[predictions["sistema"] == "modelo"]
    ablation = predictions[predictions["sistema"] == "ablacion"]
    assert set(zip(model["candidato"], model["conjunto_variables"])) == {MODEL}
    assert set(zip(ablation["candidato"], ablation["conjunto_variables"])) == {ABLATION}


def test_prueba_real_card_fields_and_sizes(prueba_dir):
    card = json.loads((prueba_dir / "ficha_desempeno.json").read_text(encoding="utf-8"))
    assert set(card["municipios"]) == set(CODES)
    total = card["global"]
    assert total["tau"] == TAU and total["periodo"] == "prueba"
    assert len(total["casillas"]) == 18 and len(total["curva_amoc"]) == 19
    assert {"bss_vs_b2", "bss_vs_b1", "mejora_logloss_vs_b2", "mejora_logloss_vs_b1"} <= set(total["estimacion"])
    for comparison in ("vs_b2", "vs_b1", "ablacion_clima"):
        conclusion = total["estimacion"][comparison]["conclusion"]
        assert conclusion["nivel_conclusion"] in {"mejora_consistente", "tendencia_favorable", "sin_mejora"}
        assert conclusion["casillas_total"] == 18
    events = total["eventos"]
    assert events["inicios_evaluables"] == 10 and events["municipio_anios"] == 18
    assert any("76001" in note and "No hubo inicios de brote" in note for note in total["notas"])
    assert card["municipios"]["76001"]["eventos"]["inicios_evaluables"] == 0
    assert sum(item["n"] for item in total["calibracion"]) == 936


def test_prueba_real_history_has_only_rolling_origin_predictions(prueba_dir, sin_clima_dir):
    history = pd.read_parquet(prueba_dir / "historial.parquet")
    assert len(history) == 1248 + 936
    assert history.groupby("municipality_code").size().to_dict() == {code: 364 for code in CODES}
    assert set(history["origen"]) == {"desarrollo_origen_movil", "prueba_origen_movil"}
    assert (history["semana_emision"] <= LAST_ORIGIN).all()
    assert (history.loc[history["origen"] == "prueba_origen_movil", "periodo"] == "prueba").all()

    key = ["municipality_code", "semana_emision"]
    test = _predictions(prueba_dir)
    for system, column in (("modelo", "probabilidad"), ("B2", "b2_estado_canal"), ("B1", "b1_climatologia_estacional")):
        saved = test[test["sistema"] == system].rename(columns={"t": "semana_emision"})
        merged = history[history["origen"] == "prueba_origen_movil"].merge(saved, on=key, validate="one_to_one")
        assert len(merged) == 936 and (merged[column] == merged["p"]).all()

    development = pd.read_parquet(sin_clima_dir / "predicciones_desarrollo.parquet")
    chosen = development[(development["familia"] == "logistica") & (development["candidato"] == MODEL[0])]
    merged = history[history["origen"] == "desarrollo_origen_movil"].merge(
        chosen.rename(columns={"t": "semana_emision"}), on=key, validate="one_to_one")
    assert len(merged) == 1248 and (merged["probabilidad"] == merged["p"]).all()


def test_prueba_real_final_models(prueba_dir):
    manifest = json.loads((prueba_dir / "modelos_finales" / "manifiesto.json").read_text(encoding="utf-8"))
    assert manifest["receta_hash"] == RECIPE_HASH
    models = manifest["modelos"]
    assert sorted(models) == list(CODES)
    for code, item in models.items():
        assert (item["filas"], item["ultima_emision"], item["ultima_objetivo"]) == (711, LAST_ORIGIN.isoformat(),
                                                                                    LAST_TARGET.isoformat())
        assert (item["familia"], item["candidato"], item["conjunto_variables"]) == ("logistica", *MODEL)
        assert (prueba_dir / "modelos_finales" / f"{code}.joblib").is_file()
