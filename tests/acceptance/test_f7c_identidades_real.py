"""F7c acceptance with real data (--channel-parquet, --search-dir, --sin-clima-dir and --prueba-dir): the derivable
reports of unit U7.5 reproduce the evidence already frozen (contrato §7.2–§7.4, §8.4 and §8.8; especificación §6.F7).
Plan approved by Jesús on 2026-11-12. Nothing is retrained and the test is not run again: only saved predictions are
read. These are identities with committed evidence, not new figures; the new figures (Murphy, classification at tau,
alert ratio, durations) are frozen in a separate file after Jesús's independent check.

Inputs of each period (decision D5 of the plan):
- development (1,248 rows, 24 cells, rotulo "usado_para_elegir"): the recipe is the common out-of-year configuration
  of logistica in the sin_clima search (ae6225180d37d025224b); the ablation is the common out-of-year configuration of
  logistica in the full-set search (F5b, 7f34b69832eea69407c6), the cells compared by step 0 of D7.2; B2 and B1 are the
  saved baselines of the sin_clima search (identical row by row to F5b);
- test (936 rows, 18 cells, rotulo "prueba"): the four systems of predicciones_prueba.parquet (execution 1 of the
  log, recipe de74d4ab…fda5).
The development AMOC curve of the tau choice uses logistica_05 in the four years (curva_tau.csv of D7.2).

API frozen here (`src.forecasting.reports`), besides the one of test_f7c_formulas.py:
- PeriodSystems(periodo, rotulo, rows, p, candidatos): rows sorted by (municipality_code, t) with municipality_code,
  t, target_week, anio, zona_t, y; p maps "modelo", "ablacion", "B2" and "B1" to arrays in the order of rows;
  candidatos has municipality_code, anio, sistema and candidato for "modelo" and "ablacion";
- development_systems(rows, *, search_dir, sin_clima_dir, familia) -> PeriodSystems;
- development_recipe_probabilities(rows, *, sin_clima_dir, familia, candidato) -> array in the order of the sorted
  rows;
- test_systems(prueba_dir) -> PeriodSystems;
- period_reports(systems, observed, *, config, tau, access) -> dict of DataFrames with the keys casillas,
  habilidad, clasificacion_tau, murphy, confiabilidad, calibracion_anual, eventos_razon, duracion_alertas and
  curva_amoc (columns periodo, rotulo and, where applicable, sistema, referencia, nivel and grupo).
Tolerances: exact for counts and labels; rel 1e-12 for scores recomputed from the same predictions.
"""

import functools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytestmark = pytest.mark.acceptance

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "reports" / "forecasting" / "evidence"
RECIPE_HASH = "de74d4abd33537433ffc9d1572c6cdd8620a2608fcc2b67ee4a1752aa865fda5"
TAU = 0.05
RECIPE_COMMON = {2019: "logistica_04", 2020: "logistica_05", 2021: "logistica_04", 2022: "logistica_04"}
ABLATION_COMMON = {2019: "logistica_03", 2020: "logistica_04", 2021: "logistica_03", 2022: "logistica_03"}
SYSTEMS = ("modelo", "ablacion", "B2", "B1")


def _config():
    from src.forecasting.config import load_config

    return load_config()


@functools.lru_cache(maxsize=None)
def _product(channel_file: str):
    from src.forecasting.dataset import load_channel_product

    return load_channel_product(channel_file, expected_md5=_config().datos.version_dvc_canal)


@functools.lru_cache(maxsize=None)
def _development_rows(channel_file: str) -> pd.DataFrame:
    from src.forecasting.dataset import build_examples
    from src.forecasting.run_selection import development_rows

    config = _config()
    examples = build_examples(_product(channel_file), h=config.horizonte.h, canal=config.canal,
                              features=config.features)
    return development_rows(examples, config)


_CACHE = {}


@pytest.fixture
def development(channel_path, search_dir, sin_clima_dir, tmp_path):
    from src.forecasting.guard import RecipeAccess
    from src.forecasting import reports

    key = ("desarrollo", str(channel_path), str(search_dir), str(sin_clima_dir))
    if key not in _CACHE:
        config = _config()
        rows = _development_rows(str(channel_path))
        systems = reports.development_systems(rows, search_dir=search_dir, sin_clima_dir=sin_clima_dir,
                                              familia="logistica")
        access = RecipeAccess.from_config(config, recipe_path=tmp_path / "receta_inexistente.yaml")
        observed = _product(str(channel_path)).to_pandas()
        tables = reports.period_reports(systems, observed, config=config, tau=TAU, access=access)
        recipe_p = reports.development_recipe_probabilities(rows, sin_clima_dir=sin_clima_dir, familia="logistica",
                                                            candidato="logistica_05")
        curve = reports.amoc_curve(systems.rows, recipe_p, observed, tau_grid=config.decision.tau_grid,
                                   k=config.decision.k, window=config.decision.W, access=access)
        _CACHE[key] = (systems, tables, curve)
    return _CACHE[key]


@pytest.fixture
def test_period(channel_path, prueba_dir):
    from src.forecasting.guard import RecipeAccess
    from src.forecasting import reports

    key = ("prueba", str(channel_path), str(prueba_dir))
    if key not in _CACHE:
        config = _config()
        systems = reports.test_systems(prueba_dir)
        access = RecipeAccess.from_config(config, recipe_path=ROOT / "configs" / "receta_congelada.yaml")
        tables = reports.period_reports(systems, _product(str(channel_path)).to_pandas(), config=config, tau=TAU,
                                        access=access)
        _CACHE[key] = (systems, tables)
    return _CACHE[key]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _same(value, expected) -> bool:
    if expected is None or (isinstance(expected, float) and math.isnan(expected)):
        return value is None or (isinstance(value, float) and math.isnan(value))
    return value == pytest.approx(expected, rel=1e-12)


def _one(frame: pd.DataFrame, **where) -> pd.Series:
    mask = np.ones(len(frame), dtype=bool)
    for column, value in where.items():
        mask &= (frame[column] == value).to_numpy()
    assert int(mask.sum()) == 1, where
    return frame[mask].iloc[0]


# --------------------------------------------------------------------------- recipe and inputs
def test_recipe_is_the_one_of_d72():
    recipe = yaml.safe_load((ROOT / "configs" / "receta_congelada.yaml").read_text(encoding="utf-8"))
    assert recipe["hash_sha256"] == RECIPE_HASH and recipe["decision"]["tau"] == TAU
    assert {entry["candidato"] for entry in recipe["ablacion"].values()} == {"logistica_03"}


def test_development_inputs(development):
    systems, _, _ = development
    assert (systems.periodo, systems.rotulo) == ("desarrollo", "usado_para_elegir")
    assert len(systems.rows) == 1248 and set(systems.p) == set(SYSTEMS)
    assert all(len(systems.p[name]) == 1248 for name in SYSTEMS)
    for system, expected in (("modelo", RECIPE_COMMON), ("ablacion", ABLATION_COMMON)):
        own = systems.candidatos[systems.candidatos["sistema"] == system]
        assert {year: set(own.loc[own["anio"] == year, "candidato"]) for year in expected} == {
            year: {name} for year, name in expected.items()}


def test_test_inputs(test_period):
    systems, _ = test_period
    assert (systems.periodo, systems.rotulo) == ("prueba", "prueba")
    assert len(systems.rows) == 936 and set(systems.p) == set(SYSTEMS)
    own = systems.candidatos
    assert set(own.loc[own["sistema"] == "modelo", "candidato"]) == {"logistica_05"}
    assert set(own.loc[own["sistema"] == "ablacion", "candidato"]) == {"logistica_03"}


# --------------------------------------------------------------------------- development identities
def test_development_cells_reproduce_step_0_and_the_diagnostic(development):
    _, tables, _ = development
    cells = tables["casillas"].set_index(["municipality_code", "anio"])
    assert len(cells) == 24
    step0 = pd.read_csv(EVIDENCE / "seleccion_conjuntos_20261112" / "comparacion_conjuntos.csv",
                        dtype={"municipality_code": str})
    for row in step0[step0["familia"] == "logistica"].itertuples():
        cell = cells.loc[(row.municipality_code, row.anio)]
        assert cell["logloss_modelo"] == pytest.approx(row.logloss_sin_clima, rel=1e-12)
        assert cell["logloss_ablacion"] == pytest.approx(row.logloss_completo, rel=1e-12)
    diag = pd.read_csv(EVIDENCE / "diagnostico_desarrollo_sin_clima_20261112" / "diag_casillas.csv",
                       dtype={"municipality_code": str})
    for row in diag.itertuples():
        cell = cells.loc[(row.municipality_code, row.anio)]
        for mine, theirs in (("brier_modelo", "brier_modelo"), ("logloss_B2", "logloss_b2"), ("brier_B2", "brier_b2"),
                             ("logloss_B1", "logloss_b1"), ("brier_B1", "brier_b1")):
            assert cell[mine] == pytest.approx(getattr(row, theirs), rel=1e-12)


def test_development_skill_reproduces_the_evidence(development):
    _, tables, _ = development
    skill = tables["habilidad"]
    diag = pd.read_csv(EVIDENCE / "diagnostico_desarrollo_sin_clima_20261112" / "diag_casillas.csv",
                       dtype={"municipality_code": str})
    summary = pd.read_csv(EVIDENCE / "diagnostico_desarrollo_sin_clima_20261112" / "diag_resumenes.csv",
                          dtype={"grupo": str})
    pooled = _one(summary, nivel="global")
    for reference, suffix in (("B2", "b2"), ("B1", "b1")):
        row = _one(skill, sistema="modelo", referencia=reference)
        assert row["casillas"] == 24
        assert row["ganadas_logloss"] == int(diag[f"gana_logloss_{suffix}"].sum())
        assert row["ganadas_brier"] == int(diag[f"gana_brier_{suffix}"].sum())
        assert row["mejora_logloss_casillas"] == pytest.approx(diag[f"mejora_logloss_{suffix}"].mean(), rel=1e-12)
        assert row["bss_casillas"] == pytest.approx(diag[f"bss_{suffix}"].mean(), rel=1e-12)
        assert row["mejora_logloss_filas"] == pytest.approx(pooled[f"mejora_logloss_{suffix}_filas"], rel=1e-12)
        assert row["bss_filas"] == pytest.approx(pooled[f"bss_{suffix}_filas"], rel=1e-12)
    step0 = pd.read_csv(EVIDENCE / "seleccion_conjuntos_20261112" / "comparacion_conjuntos.csv")
    step0 = step0[step0["familia"] == "logistica"]
    ablation = _one(skill, sistema="modelo", referencia="ablacion")
    assert ablation["ganadas_logloss"] == int((step0["logloss_sin_clima"] < step0["logloss_completo"]).sum())
    assert int(step0["gana_completo"].sum()) == 9


def test_development_events_reproduce_the_diagnostic(development):
    _, tables, _ = development
    folder = EVIDENCE / "diagnostico_desarrollo_sin_clima_20261112"
    alerts = pd.read_csv(folder / "diag_alertas.csv", dtype={"municipality_code": str})
    onsets = pd.read_csv(folder / "diag_eventos.csv", dtype={"municipality_code": str})
    onsets = onsets[(onsets["tipo"] == "inicio") & onsets["evaluable"]]
    events = tables["eventos_razon"]
    for system in ("modelo", "canal"):
        row = _one(events, sistema=system, nivel="global")
        own = alerts[(alerts["sistema"] == system) & ~alerts["continuidad"] & alerts["juzgable"]]
        assert row["inicios_evaluables"] == len(onsets) == 10
        assert row["inicios_avisados"] == int(onsets[f"avisado_{system}"].sum())
        assert row["alertas_juzgables"] == len(own)
        assert row["falsas_alarmas"] == int(own["falsa_alarma"].sum())
        assert row["falsas_seguidas_epidemia"] == int((own["falsa_alarma"] & own["seguida_de_epidemia"]).sum())
        assert row["municipio_anios"] == 24


def test_development_amoc_of_the_tau_choice_reproduces_curva_tau(development):
    _, _, curve = development
    stored = pd.read_csv(EVIDENCE / "seleccion_conjuntos_20261112" / "curva_tau.csv")
    pairs = (("inicios_evaluables", "evaluable_onsets"), ("inicios_avisados", "warned_onsets"),
             ("deteccion", "detection"), ("falsas_alarmas", "false_alarms"), ("municipio_anios", "municipality_years"),
             ("falsas_por_municipio_anio", "false_alarms_per_municipality_year"),
             ("alertas_juzgables", "judgeable_alerts"), ("alertas_continuidad", "continuity_alerts"))
    model, model_stored = curve[curve["sistema"] == "modelo"], stored[stored["sistema"] == "modelo"]
    assert model["tau"].tolist() == pytest.approx(model_stored["tau"].tolist())
    for mine, theirs in pairs:
        assert model[mine].tolist() == pytest.approx(model_stored[theirs].tolist(), rel=1e-12)
        assert _one(curve, sistema="canal")[mine] == pytest.approx(_one(stored, sistema="canal")[theirs], rel=1e-12)


# --------------------------------------------------------------------------- test identities
def test_test_cells_reproduce_casillas_prueba(test_period):
    _, tables = test_period
    cells = tables["casillas"].set_index(["municipality_code", "anio"])
    stored = pd.read_csv(EVIDENCE / "prueba_20261112" / "casillas_prueba.csv", dtype={"municipality_code": str})
    assert len(cells) == len(stored) == 18
    for row in stored.itertuples():
        cell = cells.loc[(row.municipality_code, row.anio)]
        assert (cell["filas"], cell["positivas"]) == (row.filas, row.positivas)
        for system in SYSTEMS:
            for metric in ("brier", "logloss"):
                assert cell[f"{metric}_{system}"] == pytest.approx(getattr(row, f"{metric}_{system}"), rel=1e-12)


def test_test_skill_reproduces_resumen_prueba(test_period):
    _, tables = test_period
    summary = _read_json(EVIDENCE / "prueba_20261112" / "resumen_prueba.json")
    estimation, conclusions = summary["estimacion"], summary["conclusiones"]
    skill = tables["habilidad"]
    for reference, name, suffix in (("B2", "vs_b2", "vs_b2"), ("B1", "vs_b1", "vs_b1"),
                                    ("ablacion", "ablacion_clima", None)):
        row = _one(skill, sistema="modelo", referencia=reference)
        assert row["casillas"] == conclusions[name]["casillas"] == 18
        assert row["ganadas_brier"] == conclusions[name]["casillas_ganadas_brier"]
        assert row["ganadas_logloss"] == conclusions[name]["casillas_ganadas_logloss"]
        if suffix is not None:
            assert row["bss_filas"] == pytest.approx(estimation[f"bss_{suffix}"], rel=1e-12)
            assert row["mejora_logloss_filas"] == pytest.approx(estimation[f"mejora_logloss_{suffix}"], rel=1e-12)
    murphy = tables["murphy"]
    for system in ("modelo", "ablacion"):
        row = _one(murphy, sistema=system, nivel="global")
        assert row["brier"] == pytest.approx(estimation["brier"][system], rel=1e-12)
        assert row["brier"] == pytest.approx(
            row["confiabilidad"] - row["resolucion"] + row["incertidumbre"] + row["residuo"], abs=1e-12)
    total = _one(tables["calibracion_anual"], anio="todos")
    assert total["prevalencia"] == pytest.approx(estimation["prevalencia"], rel=1e-12)


def test_test_reliability_intervals_are_those_of_the_card(test_period):
    _, tables = test_period
    card = _read_json(EVIDENCE / "prueba_20261112" / "ficha_desempeno.json")["global"]["calibracion"]
    bins = tables["confiabilidad"]
    bins = bins[(bins["sistema"] == "modelo") & (bins["nivel"] == "global")]
    assert bins["n"].tolist() == [entry["n"] for entry in card] == [94, 94, 93, 94, 93, 94, 93, 94, 93, 94]
    assert bins["p_media"].tolist() == pytest.approx([entry["probabilidad_media"] for entry in card], rel=1e-12)
    assert bins["frecuencia_observada"].tolist() == pytest.approx(
        [entry["frecuencia_observada"] for entry in card], rel=1e-12)


def test_test_amoc_is_the_one_of_the_card(test_period):
    _, tables = test_period
    card = _read_json(EVIDENCE / "prueba_20261112" / "ficha_desempeno.json")["global"]
    curve = tables["curva_amoc"]
    model = curve[curve["sistema"] == "modelo"]
    assert model["tau"].tolist() == pytest.approx([point["tau"] for point in card["curva_amoc"]])
    for (_, row), point in zip(model.iterrows(), card["curva_amoc"]):
        assert _same(row["deteccion"], point["deteccion"])
        assert _same(row["falsas_por_municipio_anio"], point["falsas_alarmas_por_municipio_anio"])
        assert _same(row["anticipacion_mediana"], point["anticipacion_mediana_semanas"])
    channel = _one(curve, sistema="canal")
    point = card["punto_canal"]
    assert _same(channel["deteccion"], point["deteccion"])
    assert _same(channel["falsas_por_municipio_anio"], point["falsas_alarmas_por_municipio_anio"])
    assert _same(channel["anticipacion_mediana"], point["anticipacion_mediana_semanas"])


def test_test_events_reproduce_resumen_prueba(test_period):
    _, tables = test_period
    events = _read_json(EVIDENCE / "prueba_20261112" / "resumen_prueba.json")["eventos"]
    for system, warned in (("modelo", "avisados_modelo"), ("canal", "avisados_canal")):
        row = _one(tables["eventos_razon"], sistema=system, nivel="global")
        assert row["inicios_evaluables"] == events["inicios_evaluables"]
        assert row["inicios_avisados"] == events[warned]
        assert row["falsas_alarmas"] == events["falsas_alarmas"][system]
        assert row["municipio_anios"] == events["municipio_anios"]
        assert row["falsas_por_municipio_anio"] == pytest.approx(
            events["falsas_alarmas_por_municipio_anio"][system], rel=1e-12)
    assert _one(tables["eventos_razon"], sistema="modelo", nivel="global")["alertas_continuidad"] == events[
        "alertas_continuidad_modelo"]
