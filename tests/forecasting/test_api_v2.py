"""Pruebas de la vista v0.2 (api_salida_d10.md §4) con un panel sintetico.

El panel real se gestiona con DVC y no siempre esta materializado; estas pruebas validan la forma de
la respuesta y las validaciones de entrada sin depender de el. La receta y el contrato no cambian.
"""

from datetime import date, timedelta
import math
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.epidemiology.validation import SOURCE_SCHEMA
from src.forecasting import api_v2

CODES = ("08001", "08758", "41001", "50001", "54001", "76001")
ALLOWED_STATES = {"sin_alerta", "senal_en_observacion", "alerta_inicio_confirmada",
                  "continuidad_epidemica"}


class DummyModel:
    """Modelo determinista con la interfaz que usa la API (predict_proba de dos columnas)."""

    def predict_proba(self, frame):
        p = np.linspace(0.1, 0.4, len(frame))
        return np.column_stack([1 - p, p])

    def predict(self, frame):
        return np.asarray([int(p >= 0.05) for p in self.predict_proba(frame)[:, 1]])


def _panel(path):
    rows = []
    week = date(2007, 1, 1)
    end = date(2025, 12, 22)
    while week <= end:
        ordinal = week.timetuple().tm_yday // 7 + 1
        for index, code in enumerate(CODES):
            seasonal = 6 * math.sin(2 * math.pi * ordinal / 52.18 + index)
            rows.append({
                "municipality_code": code,
                "municipality": code,
                "week_start": week,
                "population": 100000,
                "area_km2": 100.0,
                "population_density_km2": 1000.0,
                "temperature_c": 26.0 + seasonal / 10,
                "precipitation_mm": 100.0 + seasonal,
                "relative_humidity_pct": 70.0,
                "confirmed_records_screening": max(0, int(12 + seasonal + (week.year - 2007) * 0.4 + index * 3)),
                "no_eligible_records_observed": False,
                "week_end_nominal": week + timedelta(days=6),
                "week_end_effective": week + timedelta(days=6),
                "n_days": 7,
                "partial_week": False,
            })
        week += timedelta(weeks=1)
    # Semana parcial de cierre: existe en el panel pero no es emisible (contrato §3.7).
    for index, code in enumerate(CODES):
        week = date(2025, 12, 29)
        rows.append({
            "municipality_code": code,
            "municipality": code,
            "week_start": week,
            "population": 100000,
            "area_km2": 100.0,
            "population_density_km2": 1000.0,
            "temperature_c": 26.0,
            "precipitation_mm": 100.0,
            "relative_humidity_pct": 70.0,
            "confirmed_records_screening": 3 + index,
            "no_eligible_records_observed": False,
            "week_end_nominal": week + timedelta(days=6),
            "week_end_effective": week + timedelta(days=3),
            "n_days": 3,
            "partial_week": True,
        })
    table = pa.Table.from_pandas(pd.DataFrame(rows), schema=SOURCE_SCHEMA, preserve_index=False)
    pq.write_table(table, path)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    path = tmp_path_factory.mktemp("panel") / "panel.parquet"
    _panel(path)
    previous = os.environ.get("FORECAST_PANEL_PATH")
    os.environ["FORECAST_PANEL_PATH"] = str(path)
    api_v2.clear_context()
    try:
        yield api_v2.get_context()
    finally:
        api_v2.clear_context()
        if previous is None:
            os.environ.pop("FORECAST_PANEL_PATH", None)
        else:
            os.environ["FORECAST_PANEL_PATH"] = previous


def test_ventana_y_respuesta_0_2(ctx):
    assert (ctx.first_week, ctx.last_week) == (date(2025, 10, 27), date(2025, 12, 22))
    response = api_v2.build_prediction(
        codigo="41001", nombre="Neiva", semana_emision=ctx.last_week,
        model=DummyModel(), model_name="DengueAlertas_Neiva", model_version="v1.0")
    assert response["version_esquema"] == "0.2"
    assert response["municipio"] == {"codigo": "41001", "nombre": "Neiva"}
    assert response["semana_emision"] == "2025-12-22"
    assert response["semana_objetivo"] == "2026-02-16"
    assert 0 <= response["probabilidad"] <= 1
    assert response["decision"]["estado_alerta"] in ALLOWED_STATES
    assert response["decision"]["tau"] == 0.05 and response["decision"]["k"] == 2
    assert len(response["serie_canal"]) == 60
    assert len(response["trayectoria"]) == 12
    assert len(response["advertencias"]) == 4
    assert set(response["canal_actual"]) == {"casos", "central", "u1e", "zona"}
    assert set(response["referencias"]) == {"b2_estado_canal", "b1_climatologia_estacional"}


def test_semana_fuera_de_ventana(ctx):
    with pytest.raises(api_v2.V2Error) as error:
        api_v2.build_prediction(
            codigo="41001", nombre="Neiva", semana_emision=date(2025, 10, 20),
            model=DummyModel(), model_name="DengueAlertas_Neiva", model_version="v1.0")
    assert error.value.codigo == "fuera_de_ventana_vigente"


def test_semana_parcial_no_es_vigente(ctx):
    with pytest.raises(api_v2.V2Error) as error:
        api_v2.build_prediction(
            codigo="41001", nombre="Neiva", semana_emision=date(2025, 12, 29),
            model=DummyModel(), model_name="DengueAlertas_Neiva", model_version="v1.0")
    assert error.value.codigo == "fuera_de_ventana_vigente"


def test_municipio_no_piloto(ctx):
    with pytest.raises(api_v2.V2Error) as error:
        api_v2.build_prediction(
            codigo="99999", nombre="X", semana_emision=ctx.last_week,
            model=DummyModel(), model_name="X", model_version="v1.0")
    assert error.value.codigo == "municipio_no_piloto"
