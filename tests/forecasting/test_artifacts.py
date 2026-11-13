"""Unit tests of src/forecasting/artifacts.py beyond the F7 and F7b acceptance (synthetic data only)."""

from datetime import date, timedelta
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.epidemiology.config import EpidemiologyValidationError
from src.forecasting import artifacts
from src.forecasting.config import load_config
from src.forecasting.guard import RecipeAccess

WEEK = timedelta(weeks=1)
_SPEC = importlib.util.spec_from_file_location("evaluate_world", Path(__file__).with_name("test_evaluate.py"))
_WORLD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_WORLD)


def _search(folder: Path, codes=("00001",)):
    """A fake search folder: predictions of two candidates and the B2/B1 baselines for 2019."""
    t = [date(2018, 11, 12) + i * WEEK for i in range(52)]
    model, baselines = [], []
    for code in codes:
        for name in ("logistica_03", "logistica_05"):
            model.append(pd.DataFrame({"municipality_code": code, "t": t, "target_week": [w + 8 * WEEK for w in t],
                                       "familia": "logistica", "candidato": name, "anio": 2019, "y": 0,
                                       "p": 0.1 if name == "logistica_05" else 0.2}))
        for system in ("B2", "B1"):
            baselines.append(pd.DataFrame({"municipality_code": code, "t": t, "target_week": [w + 8 * WEEK for w in t],
                                           "familia": system, "candidato": system, "anio": 2019, "y": 0, "p": 0.3}))
    folder.mkdir(parents=True)
    pd.concat(model, ignore_index=True).to_parquet(folder / "predicciones_desarrollo.parquet", index=False)
    pd.concat(baselines, ignore_index=True).to_parquet(folder / "predicciones_lineas_base_desarrollo.parquet",
                                                        index=False)
    rows = pd.DataFrame({"municipality_code": codes[0], "t": t, "target_week": [w + 8 * WEEK for w in t]})
    return rows, hashlib.md5((folder / "predicciones_desarrollo.parquet").read_bytes()).hexdigest()


def _recipe(md5):
    return {"municipios": {"00001": {"familia": "logistica", "candidato": "logistica_05",
                                     "conjunto_variables": "sin_clima"}},
            "procedencia": {"busquedas": {"sin_clima": {"insumos_md5": {"predicciones_desarrollo.parquet": md5}}}}}


def test_development_history_uses_the_recipe_candidate_of_its_search(tmp_path):
    rows, md5 = _search(tmp_path / "sin_clima")
    long = artifacts.development_history_inputs(_recipe(md5), rows, {"sin_clima": tmp_path / "sin_clima"})
    assert long.groupby("sistema").size().to_dict() == {"B1": 52, "B2": 52, "modelo": 52}
    assert set(long.loc[long["sistema"] == "modelo", "candidato"]) == {"logistica_05"}
    assert (long.loc[long["sistema"] == "modelo", "p"] == 0.1).all()


def test_development_history_checks_the_md5_of_the_search(tmp_path):
    rows, _ = _search(tmp_path / "sin_clima")
    with pytest.raises(EpidemiologyValidationError, match="provenance"):
        artifacts.development_history_inputs(_recipe("0" * 32), rows, {"sin_clima": tmp_path / "sin_clima"})


def test_history_of_two_periods_with_the_synthetic_world(tmp_path):
    """Development and test histories of the evaluate world; the alert state restarts at each period (decision U7.1-D4)."""
    config = load_config()
    _WORLD.write_recipe(tmp_path / "receta.yaml", config)
    access = RecipeAccess.from_config(config, recipe_path=tmp_path / "receta.yaml")
    examples, observed = _WORLD.world()
    years = examples["target_week"].map(lambda week: week.year)
    rng = np.random.default_rng(1)
    parts = []
    for origin, mask in (("desarrollo_origen_movil", (years >= 2019) & (years <= 2022)),
                         ("prueba_origen_movil", years >= 2023)):
        rows = examples[mask].reset_index(drop=True)
        predictions = pd.concat([rows[["municipality_code", "t", "target_week", "y"]].assign(
            sistema=system, p=rng.uniform(0, 1, len(rows))) for system in ("modelo", "B2", "B1")], ignore_index=True)
        parts.append(artifacts.history_table(rows, predictions, observed, origen=origin, tau=0.5, k=2, window=12,
                                             access=access))
    history = pd.concat([part.filas for part in parts], ignore_index=True)
    assert list(history.columns) == artifacts.HISTORY_COLUMNS
    assert set(history["periodo"]) == {"desarrollo", "prueba"}
    for part in parts:
        first = part.filas.sort_values("semana_emision").groupby("municipality_code").head(1)
        assert set(first["estado_alerta"]) <= {"sin_alerta", "senal_en_observacion"}
        assert set(part.eventos["tipo_evento"]) <= {"inicio_avisado", "inicio_no_avisado", "falsa_alarma",
                                                    "alerta_continuidad", "inicio_censurado", "alerta_censurada"}


def test_cli_refuses_uncommitted_changes_before_reading_anything(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("nothing must be read")

    monkeypatch.setattr(artifacts, "_dirty_paths", lambda: [" M src/forecasting/artifacts.py"])
    monkeypatch.setattr(artifacts, "load_channel_product", forbidden)
    assert artifacts.main(["--prueba", str(tmp_path)]) == 1
    assert "uncommitted changes" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
