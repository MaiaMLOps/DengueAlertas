"""F7 acceptance with the outputs of the single test run (--prueba-dir, --channel-parquet and --search-dir)
(contrato §6.1, §8.4, §8.8, §9.4 and §10.18 and §10.22; api_salida_d10.md §6 and §7; especificación §6.F7).
Approved by Jesús on 2026-11-12 (U7.1).

Every test is named `test_prueba_real_*`. Until Jesús approves and the test is run in U7.3 they are deselected in
the step reports with -k "not prueba_real" (reported as omitted, pending U7.3), as in F6. They check structure,
provenance and internal consistency, never a predetermined result: no figure of the test period is literal here
except the sizes of contrato §6.1 and §7.4 (936 rows, 322 positives, 10 evaluable onsets, 18 municipality-years),
which come from the observed series.

Files expected in --prueba-dir (written by the F7 command line):
- predicciones_prueba.parquet and ensayos_prueba.parquet (evaluate.run_test);
- historial.parquet and historial_eventos.parquet (artifacts.history_table, development and test);
- ficha_desempeno.json (artifacts.performance_card);
- modelos_finales/manifiesto.json and modelos_finales/<code>.joblib (artifacts.train_final_models);
- diagnostico_prueba/diag_<input>.csv for the six inputs of §8.8.
The log of executions is reports/forecasting/bitacora_prueba.jsonl in the repository.
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
RECIPE_HASH = "d2135a39a9de0b9fd55e63caf5cc22859e2c82a4b4437afe2a96bfbe2c39da98"
CHANNEL_MD5 = "99fd12311a261264019645d906631cfc"
CODES = ("08001", "08758", "41001", "50001", "54001", "76001")
SYSTEMS = ("modelo", "sin_clima", "B2", "B1")
ROWS_PER_YEAR = {2023: 52, 2024: 53, 2025: 51}   # per municipality; 2025-12-29 is partial and not eligible
LAST_ORIGIN, LAST_TARGET = date(2025, 10, 27), date(2025, 12, 22)
E, P, T = "epidemica", "post_epidemica", "tranquila"
TEST_REGIMES = {  # 2023, 2024, 2025 (observed series only; same literals as test_f7_diagnostico_desarrollo.py)
    "08001": (E, E, P), "08758": (E, E, E), "41001": (E, E, P), "50001": (E, P, E), "54001": (T, E, E),
    "76001": (E, E, P)}
DIAGNOSTIC_FILES = ("casillas", "resumenes", "transicion", "eventos", "alertas", "serie")


def _config():
    from src.forecasting.config import load_config

    return load_config()


@functools.lru_cache(maxsize=None)
def _examples(channel_file: str) -> pd.DataFrame:
    from src.forecasting.dataset import build_examples, load_channel_product

    config = _config()
    product = load_channel_product(channel_file, expected_md5=config.datos.version_dvc_canal)
    return build_examples(product, h=config.horizonte.h, canal=config.canal, features=config.features)


def _predictions(prueba_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(prueba_dir / "predicciones_prueba.parquet")


def _card(prueba_dir: Path) -> dict:
    return json.loads((prueba_dir / "ficha_desempeno.json").read_text(encoding="utf-8"))


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
    assert set(predictions.loc[predictions["sistema"] == "modelo", "candidato"]) == {"logistica_03"}


def test_prueba_real_trials_follow_d71(prueba_dir):
    trials = pd.read_parquet(prueba_dir / "ensayos_prueba.parquet")
    assert len(trials) == 18
    assert sorted(set(zip(trials["municipality_code"], trials["anio"]))) == [
        (code, year) for code in CODES for year in (2023, 2024, 2025)]
    assert (trials["ultima_objetivo_entrenamiento"] <= trials["t0"]).all()
    assert trials["filas_evaluadas"].tolist() == [ROWS_PER_YEAR[year] for year in trials["anio"]]


def test_prueba_real_card_fields_and_sizes(prueba_dir):
    card = _card(prueba_dir)
    assert set(card["municipios"]) == set(CODES)
    total = card["global"]
    assert total["tau"] == 0.1 and total["periodo"] == "prueba"
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


def test_prueba_real_card_cells_recomputed_from_saved_predictions(prueba_dir):
    import numpy as np

    predictions = _predictions(prueba_dir)
    clip = _config().metricas.recorte_log_loss
    for cell in _card(prueba_dir)["global"]["casillas"]:
        for system, key in (("modelo", "modelo"), ("B2", "b2"), ("B1", "b1")):
            own = predictions[(predictions["sistema"] == system) & (predictions["municipality_code"] == cell["municipio"])
                              & (predictions["anio"] == cell["anio"])]
            y, p = own["y"].to_numpy(dtype=float), own["p"].to_numpy(dtype=float)
            assert cell[f"brier_{key}"] == pytest.approx(float(np.mean((p - y) ** 2)), rel=1e-12)
            if key != "b1":
                q = np.clip(p, clip, 1 - clip)
                expected = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
                assert cell[f"logloss_{key}"] == pytest.approx(expected, rel=1e-12)
        assert cell["gana_vs_b2"] == (cell["brier_modelo"] < cell["brier_b2"])


def test_prueba_real_history_has_only_rolling_origin_predictions(prueba_dir, search_dir):
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

    development = pd.read_parquet(search_dir / "predicciones_desarrollo.parquet")
    chosen = development[(development["familia"] == "logistica") & (development["candidato"] == "logistica_03")]
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
        assert (item["familia"], item["candidato"]) == ("logistica", "logistica_03")
        assert (prueba_dir / "modelos_finales" / f"{code}.joblib").is_file()


def test_prueba_real_diagnostic_inputs(prueba_dir):
    folder = prueba_dir / "diagnostico_prueba"
    tables = {name: pd.read_csv(folder / f"diag_{name}.csv", dtype={"municipality_code": str})
              for name in DIAGNOSTIC_FILES}
    cells = tables["casillas"]
    assert len(cells) == 18 and set(cells["rotulo"]) == {"prueba"}
    for cell in cells.itertuples():
        assert cell.regimen == TEST_REGIMES[cell.municipality_code][cell.anio - 2023]
    assert len(tables["serie"]) == 936
    onsets = tables["eventos"][tables["eventos"]["tipo"] == "inicio"]
    assert int(onsets["evaluable"].sum()) == 10
