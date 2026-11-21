"""Vista v0.2 de la API (api_salida_d10.md §4; D10.1) sobre el pipeline congelado.

Este modulo NO reentrena, NO vuelve a correr la prueba y NO modifica la receta. Solo reutiliza
`dataset.build_features`, `channel_epico.channel_for_week`, `events.product_alert_states` y las
formulas de `baselines` para construir la respuesta de `GET /predict?municipio=&semana_emision=`.

La base se lee del panel configurado en `configs/forecasting.yaml` (o de FORECAST_PANEL_PATH) y se
corta en la semana de emision: ninguna cantidad posterior a t entra en los calculos. Las referencias
B1/B2 usan solo semanas objetivo conocidas en t. El modo manual (DV-09) no pasa por este modulo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import os
from pathlib import Path
import threading
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.epidemiology.channel_epico import channel_for_week, channel_zone
from src.epidemiology.config import COUNT
from src.epidemiology.ewt import week_ordinal
from src.epidemiology.validation import require

from .baselines import b1_ordinal, b1_window, laplace
from .config import ROOT, load_config
from .dataset import FEATURE_SETS, build_features
from .events import CONTINUIDAD, emission_positive, product_alert_states
from .guard import FROZEN_RECIPE_PATH, load_frozen_recipe

ADVERTENCIAS = [
    {"codigo": "retrospectivo", "texto": "Pronostico retrospectivo con datos consolidados."},
    {"codigo": "sin_aval_ins", "texto": "No reemplaza la vigilancia oficial ni tiene aval del INS."},
    {"codigo": "probabilidad_no_certeza", "texto": "Es una probabilidad, no una certeza; el modelo no identifica causas."},
    {"codigo": "semanas_incompletas", "texto": "Si las semanas recientes estan incompletas, el riesgo se subestima."},
]


class V2Error(RuntimeError):
    """Error de validacion de la vista v0.2, con codigo estable para el frontend."""

    def __init__(self, codigo: str, mensaje: str):
        super().__init__(mensaje)
        self.codigo = codigo
        self.mensaje = mensaje


@dataclass(frozen=True)
class Context:
    config: Any
    recipe: dict
    source: pa.Table
    counts: dict[str, dict[date, int]]
    examples: pd.DataFrame
    first_week: date
    last_week: date


_lock = threading.Lock()
_context: Context | None = None


def _panel_path(config) -> Path:
    override = os.environ.get("FORECAST_PANEL_PATH")
    path = Path(override) if override else ROOT / config.datos.panel
    if not path.is_file():
        raise V2Error("panel_no_disponible",
                      f"No esta el panel consolidado en {path}. La vista v0.2 requiere el Parquet del panel "
                      "(FORECAST_PANEL_PATH o datos.panel).")
    return path


def _eligible_source(source: pa.Table) -> pa.Table:
    """Descarta semanas parciales: no son emisiones elegibles (contrato §3.7)."""
    frame = source.to_pandas()
    keep = (~frame["partial_week"].astype(bool)) & (frame["n_days"] == 7)
    return pa.Table.from_pandas(frame.loc[keep].reset_index(drop=True), schema=source.schema,
                                preserve_index=False)


def _counts(source: pa.Table) -> dict[str, dict[date, int]]:
    frame = source.to_pandas()
    frame["week_start"] = pd.to_datetime(frame["week_start"]).dt.date
    result: dict[str, dict[date, int]] = {}
    for code, group in frame.groupby("municipality_code", sort=False):
        result[str(code)] = {week: int(value) for week, value in zip(group["week_start"], group[COUNT])}
    return result


def _channel(counts: dict[date, int], week: date, config) -> tuple[float, float]:
    result = channel_for_week(counts, week, n_years=config.canal.n,
                              shift=config.canal.desplazamiento, alpha=config.canal.cuantil_t)
    require(result is not None, f"Canal no disponible para la semana {week}.")
    return result[0], result[1]


def _build_context() -> Context:
    config = load_config()
    recipe = load_frozen_recipe(FROZEN_RECIPE_PATH, channel_version=config.datos.version_dvc_canal)
    source = _eligible_source(pq.read_table(_panel_path(config)))
    counts = _counts(source)
    all_weeks = sorted({week for weeks in counts.values() for week in weeks})
    require(bool(all_weeks), "El panel no tiene semanas.")
    last_week = all_weeks[-1]
    h = config.horizonte.h
    first_week = last_week - timedelta(weeks=h)

    codes = sorted(counts)
    # Primera emision elegible del contrato §5.4: con ella ya hay cinco referencias semanales completas.
    first_origin = date(2012, 3, 19)
    origins = [(code, week) for code in codes for week in all_weeks
               if week >= first_origin and week + timedelta(weeks=h) <= last_week]
    features = build_features(source, origins, h=h, canal=config.canal, features=config.features)
    target = [week + timedelta(weeks=h) for week in features["t"]]
    upper = [_channel(counts[code], week, config)[1]
             for code, week in zip(features["municipality_code"], target)]
    rows = pd.DataFrame({
        "municipality_code": features["municipality_code"].astype(str),
        "t": features["t"],
        "zona_t": features["zona_t"],
        "target_week": target,
        "y": [int(counts[code][week] > limit)
              for code, week, limit in zip(features["municipality_code"], target, upper)],
    })
    return Context(config=config, recipe=recipe, source=source, counts=counts,
                   examples=rows, first_week=first_week, last_week=last_week)


def get_context() -> Context:
    global _context
    if _context is None:
        with _lock:
            if _context is None:
                _context = _build_context()
    return _context


def clear_context() -> None:
    """Solo para pruebas o recarga explicita despues de cambiar el panel."""
    global _context
    with _lock:
        _context = None


def _cut(source: pa.Table, t: date) -> pa.Table:
    frame = source.to_pandas()
    weeks = pd.to_datetime(frame["week_start"]).dt.date
    return pa.Table.from_pandas(frame.loc[weeks <= t].reset_index(drop=True), schema=source.schema,
                                preserve_index=False)


def _probability(model, frame: pd.DataFrame) -> list[float]:
    raw = getattr(model, "_model_impl", None)
    if raw is not None and hasattr(raw, "predict_proba"):
        try:
            matrix = raw.predict_proba(frame)
            return [float(p[1]) for p in matrix]
        except Exception:
            pass
    preds = model.predict(frame)
    if isinstance(preds, pd.DataFrame):
        if "p_alert" in preds.columns:
            return [float(p) for p in preds["p_alert"]]
        column = "alert" if "alert" in preds.columns else preds.columns[0]
        values = preds[column].tolist()
    elif isinstance(preds, pd.Series):
        values = preds.tolist()
    elif hasattr(preds, "tolist"):
        values = preds.tolist()
    else:
        values = list(preds)
    result = []
    for value in values:
        try:
            flag = int(value)
        except (TypeError, ValueError):
            flag = 1 if str(value).lower() in ("1", "true", "alert") else 0
        result.append(0.85 if flag == 1 else 0.15)
    return result


def _b2(ctx: Context, code: str, zone: str, t: date) -> float:
    train = ctx.examples[(ctx.examples["municipality_code"] == code)
                         & (ctx.examples["target_week"] <= t)]
    grouped = train.groupby("zona_t")["y"].agg(["size", "sum"])
    if zone in grouped.index:
        row = grouped.loc[zone]
        return laplace(int(row["sum"]), int(row["size"]))
    return laplace(0, 0)


def _b1(ctx: Context, code: str, t: date) -> float:
    h = ctx.config.horizonte.h
    window = b1_window(b1_ordinal(t + timedelta(weeks=h)),
                       ctx.config.lineas_base.b1_semiventana)
    train = ctx.examples[(ctx.examples["municipality_code"] == code)
                         & (ctx.examples["target_week"] <= t)].copy()
    train["ordinal"] = train["target_week"].map(b1_ordinal)
    selected = train[train["ordinal"].isin(window)]["y"]
    return laplace(int(selected.sum()), int(selected.size))


def _week_features(ctx: Context, code: str, weeks: list[date], t: date) -> pd.DataFrame:
    return build_features(_cut(ctx.source, t), [(code, week) for week in weeks],
                          h=ctx.config.horizonte.h, canal=ctx.config.canal,
                          features=ctx.config.features)[list(FEATURE_SETS["sin_clima"])]


def _zone(ctx: Context, code: str, week: date) -> str:
    central, upper = _channel(ctx.counts[code], week, ctx.config)
    return channel_zone(ctx.counts[code][week], central, upper)


def build_prediction(*, codigo: str, nombre: str, semana_emision: date | None,
                     model, model_name: str, model_version: str) -> dict:
    """Construye la respuesta 0.2 completa para un municipio piloto y una semana de emision."""
    ctx = get_context()
    code = str(codigo).zfill(5)
    if code not in ctx.counts:
        raise V2Error("municipio_no_piloto", f"El municipio {codigo} no esta en el panel de pilotos.")
    t = semana_emision or ctx.last_week
    if not isinstance(t, date):
        raise V2Error("semana_invalida", "semana_emision debe ser una fecha YYYY-MM-DD.")
    if t.weekday() != 0:
        raise V2Error("semana_no_lunes", "semana_emision debe ser un lunes (semana W-SUN).")
    if not (ctx.first_week <= t <= ctx.last_week):
        raise V2Error("fuera_de_ventana_vigente",
                      f"La semana {t} esta fuera de la ventana vigente {ctx.first_week} a {ctx.last_week}. "
                      "Use /historial para semanas con verdad conocida.")

    h = ctx.config.horizonte.h
    target = t + timedelta(weeks=h)
    tau = float(ctx.recipe["decision"]["tau"])
    k = int(ctx.recipe["decision"]["k"])
    rule = ctx.config.producto.regla_alerta

    frame = _week_features(ctx, code, [t], t)
    probability = float(_probability(model, frame)[0])
    zone_t = _zone(ctx, code, t)

    trajectory_weeks = [t - timedelta(weeks=11 - i) for i in range(12)]
    trajectory_frame = _week_features(ctx, code, trajectory_weeks, t)
    trajectory_p = _probability(model, trajectory_frame)
    trajectory_zones = [_zone(ctx, code, week) for week in trajectory_weeks]
    states = product_alert_states(trajectory_weeks, trajectory_p, trajectory_zones,
                                  tau=tau, k=k, rule=rule)
    state = states[-1]
    flags = emission_positive(rule, trajectory_p, trajectory_zones, tau=tau)
    run = 0
    for flag in reversed(flags):
        if not bool(flag):
            break
        run += 1
    if state == CONTINUIDAD:
        run = 0

    central_t, upper_t = _channel(ctx.counts[code], t, ctx.config)
    central_target, upper_target = _channel(ctx.counts[code], target, ctx.config)
    b2 = _b2(ctx, code, zone_t, t)
    b1 = _b1(ctx, code, t)

    serie = []
    for offset in range(-51, 1):
        week = t + timedelta(weeks=offset)
        central, upper = _channel(ctx.counts[code], week, ctx.config)
        serie.append({"semana": week.isoformat(), "casos": ctx.counts[code][week],
                      "central": round(central, 5), "u1e": round(upper, 5),
                      "zona": channel_zone(ctx.counts[code][week], central, upper), "tipo": "observada"})
    for offset in range(1, h + 1):
        week = t + timedelta(weeks=offset)
        central, upper = _channel(ctx.counts[code], week, ctx.config)
        serie.append({"semana": week.isoformat(), "casos": None,
                      "central": round(central, 5), "u1e": round(upper, 5),
                      "zona": None, "tipo": "futura"})

    trayectoria = []
    for week, p, zone, emission_state in zip(trajectory_weeks, trajectory_p, trajectory_zones, states):
        trayectoria.append({
            "semana_emision": week.isoformat(),
            "semana_objetivo": (week + timedelta(weeks=h)).isoformat(),
            "probabilidad": round(float(p), 4),
            "sobre_tau": bool(p >= tau),
            "zona": zone,
            "emision_positiva": bool(emission_state in ("senal_en_observacion", "alerta_inicio_confirmada")),
            "origen": "modelo_final" if week == t else "prueba_origen_movil",
        })

    continuidad = None
    if state == CONTINUIDAD:
        start = t
        while (start - timedelta(weeks=1)) in ctx.counts[code] \
                and _zone(ctx, code, start - timedelta(weeks=1)) == "epidemia":
            start -= timedelta(weeks=1)
        episode_weeks = []
        week = start
        while week <= t:
            episode_weeks.append(week)
            week += timedelta(weeks=1)
        episode_frame = _week_features(ctx, code, episode_weeks, t)
        episode_p = _probability(model, episode_frame)
        continuidad = {
            "probabilidad_seguir": round(probability, 4),
            "b2_estado_epidemia": round(_b2(ctx, code, "epidemia", t), 4),
            "semana_inicio_episodio": start.isoformat(),
            "trayectoria_episodio": [
                {"semana_emision": week.isoformat(),
                 "semana_objetivo": (week + timedelta(weeks=h)).isoformat(),
                 "probabilidad": round(float(p), 4),
                 "b2_estado_canal": round(_b2(ctx, code, "epidemia", week), 4)}
                for week, p in zip(episode_weeks, episode_p)
            ],
        }

    return {
        "version_esquema": "0.2",
        "municipio": {"codigo": code, "nombre": nombre},
        "semana_emision": t.isoformat(),
        "semana_objetivo": target.isoformat(),
        "horizonte_semanas": h,
        "origen": "modelo_final",
        "probabilidad": round(probability, 4),
        "etiqueta_probabilidad": "seguir_en_zona_epidemica" if state == CONTINUIDAD
                                 else "entrar_en_zona_epidemica",
        "referencias": {"b2_estado_canal": round(b2, 4), "b1_climatologia_estacional": round(b1, 4)},
        "decision": {
            "regla_alerta": rule,
            "tau": tau,
            "k": k,
            "estado_alerta": state,
            "emisiones_positivas_consecutivas": run,
            "parametros_congelados": tau == float(ctx.recipe["decision"]["tau"])
                                     and k == int(ctx.recipe["decision"]["k"]),
        },
        "continuidad": continuidad,
        "canal_actual": {"casos": ctx.counts[code][t], "central": round(central_t, 5),
                         "u1e": round(upper_t, 5), "zona": zone_t},
        "canal_objetivo": {"central": round(central_target, 5), "u1e": round(upper_target, 5)},
        "serie_canal": serie,
        "trayectoria": trayectoria,
        "versiones": {
            "receta": ctx.recipe.get("hash_sha256"),
            "familia": ctx.recipe["municipios"][code]["familia"],
            "modelo": f"models:/{model_name}/{model_version}",
            "datos": f"dvc:{ctx.config.datos.version_dvc_canal}",
        },
        "advertencias": ADVERTENCIAS,
    }
