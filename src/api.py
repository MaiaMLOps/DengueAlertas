import os
import threading
import time
import datetime
import json
import logging
import traceback
import numpy as np
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

load_dotenv()

# ---------------------------------------------------------------------------
# Autenticacion transparente con Cloudflare Zero Trust (Service Tokens)
# ---------------------------------------------------------------------------
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider
from mlflow.tracking.request_header.registry import _request_header_provider_registry

CF_CLIENT_ID = os.environ.get("CF_ACCESS_CLIENT_ID")
CF_CLIENT_SECRET = os.environ.get("CF_ACCESS_CLIENT_SECRET")

class CloudflareAccessHeaderProvider(RequestHeaderProvider):
    def in_context(self):
        return True

    def request_headers(self):
        cid = os.environ.get("CF_ACCESS_CLIENT_ID") or CF_CLIENT_ID
        csec = os.environ.get("CF_ACCESS_CLIENT_SECRET") or CF_CLIENT_SECRET
        headers = {}
        if cid and csec:
            headers["CF-Access-Client-Id"] = cid
            headers["CF-Access-Client-Secret"] = csec
        return headers

try:
    _request_header_provider_registry.register(CloudflareAccessHeaderProvider)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Variables de entorno del Stage y MLflow
# ---------------------------------------------------------------------------
STAGE = os.getenv("STAGE", "Production").capitalize() # "Staging" o "Production"
MODEL_NAME = os.getenv("MODEL_NAME", "ModeloPrueba")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://platform.arkoiot.com")
DEFAULT_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.205"))
current_version = "None"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---------------------------------------------------------------------------
# Telemetria y Logging Estructurado (JSON rotativo)
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

telemetry_logger = logging.getLogger("dengue_telemetry")
telemetry_logger.setLevel(logging.INFO)
if not telemetry_logger.handlers:
    log_file = os.path.join(LOG_DIR, f"api_{STAGE.lower()}.jsonl")
    handler = RotatingFileHandler(log_file, maxBytes=50 * 1024 * 1024, backupCount=10, encoding="utf-8")
    handler.setFormatter(logging.Formatter('%(message)s'))
    telemetry_logger.addHandler(handler)

app = FastAPI(
    title=f"Dengue Risk API [{STAGE}]",
    description=f"API de inferencia de Dengue",
    version=f"{current_version}"
)

# Middleware de telemetria HTTP inteligente (filtra ruido de /schema y /health)
class TelemetryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or request.headers.get("CF-Ray") or f"req_{int(time.time()*1000)}"
        user_email = (
            request.headers.get("Cf-Access-Authenticated-User-Email")
            or request.headers.get("cf-access-authenticated-user-email")
            or request.headers.get("x-user-email")
            or ""
        )
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = round(time.time() - start_time, 4)
            response.headers["X-Request-ID"] = req_id

            # Ignorar el polling rutinario (/schema, /health) salvo que de error
            # Tambien ignoramos /telemetry/event en el middleware para evitar log duplicado
            if (request.url.path not in ("/schema", "/health", "/telemetry/event")) or response.status_code >= 400:
                telemetry_logger.info(json.dumps({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "type": "http_access",
                    "stage": STAGE,
                    "request_id": req_id,
                    "user_email": user_email,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_sec": duration,
                    "client_ip": request.headers.get("CF-Connecting-IP") or (request.client.host if request.client else None),
                    "origin": request.headers.get("origin")
                }))
            return response

        except Exception as exc:
            duration = round(time.time() - start_time, 4)
            telemetry_logger.error(json.dumps({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "type": "http_error",
                "stage": STAGE,
                "request_id": req_id,
                "user_email": user_email,
                "method": request.method,
                "path": request.url.path,
                "duration_sec": duration,
                "error_type": type(exc).__name__,
                "error_detail": str(exc),
                "traceback": traceback.format_exc()
            }))
            raise exc

app.add_middleware(TelemetryMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"]
)

# ---------------------------------------------------------------------------
# Estado global de Modelos (Base y 5 Modelos Municipales)
# ---------------------------------------------------------------------------
loaded_model = None
current_version = "None"
current_mlflow_version = "None"
current_schema = []
model_lock = threading.Lock()

# Diccionarios para enrutador multi-modelo por municipio
loaded_municipal_models: Dict[str, Any] = {}
municipal_models_metadata: Dict[str, Dict[str, Any]] = {}

# Mapeo maestro de municipios conocidos a metadatos DANE
MUNICIPALITY_GEO_REGISTRY = {
    "cali": {
        "codigo": "76001",
        "deptoId": "COVAC",
        "nombre": "Cali",
        "departamento": "Valle del Cauca",
        "aliases": ["76001", "cali", "valle del cauca", "covac"],
        "densidad": 3910.0
    },
    "bucaramanga": {
        "codigo": "68001",
        "deptoId": "COSAN",
        "nombre": "Bucaramanga",
        "departamento": "Santander",
        "aliases": ["68001", "bucaramanga", "santander", "cosan"],
        "densidad": 3655.0
    },
    "cartagena": {
        "codigo": "13001",
        "deptoId": "COBOL",
        "nombre": "Cartagena",
        "departamento": "Bolívar",
        "aliases": ["13001", "cartagena", "bolivar", "cobol"],
        "densidad": 1581.0
    },
    "ibague": {
        "codigo": "73001",
        "deptoId": "COTOL",
        "nombre": "Ibagué",
        "departamento": "Tolima",
        "aliases": ["73001", "ibague", "tolima", "cotol"],
        "densidad": 377.0
    },
    "villavicencio": {
        "codigo": "50001",
        "deptoId": "COMET",
        "nombre": "Villavicencio",
        "departamento": "Meta",
        "aliases": ["50001", "villavicencio", "meta", "comet"],
        "densidad": 395.0
    }
}

MUNICIPALITY_MODELS_MAP: Dict[str, str] = {
    "76001": "DengueAlertas_Cali", "cali": "DengueAlertas_Cali", "covac": "DengueAlertas_Cali",
    "68001": "DengueAlertas_Bucaramanga", "bucaramanga": "DengueAlertas_Bucaramanga", "cosan": "DengueAlertas_Bucaramanga",
    "13001": "DengueAlertas_Cartagena", "cartagena": "DengueAlertas_Cartagena", "cobol": "DengueAlertas_Cartagena",
    "73001": "DengueAlertas_Ibague", "ibague": "DengueAlertas_Ibague", "cotol": "DengueAlertas_Ibague",
    "50001": "DengueAlertas_Villavicencio", "villavicencio": "DengueAlertas_Villavicencio", "comet": "DengueAlertas_Villavicencio"
}

def load_single_model(client: MlflowClient, model_name: str):
    """
    Intenta cargar un modelo por alias (MLflow 3.x ej @staging) o por Stage clasico (Staging/Production).
    """
    model_obj = None
    target_version = "None"
    tag_version = "None"

    # 1. Probar por Alias (MLflow 3.x: @staging o @production)
    try:
        alias_name = STAGE.lower()
        mv = client.get_model_version_by_alias(model_name, alias_name)
        if mv:
            target_version = mv.version
            tags = mv.tags or {}
            tag_version = tags.get("current_version") or tags.get("version") or f"v{target_version}"
            model_uri = f"models:/{model_name}@{alias_name}"
            model_obj = mlflow.pyfunc.load_model(model_uri)
    except Exception:
        pass

    # 2. Probar por Stage clasico (MLflow 2.x)
    if model_obj is None:
        try:
            latest_versions = client.get_latest_versions(model_name, stages=[STAGE])
            if latest_versions:
                mv = latest_versions[0]
                target_version = mv.version
                tags = mv.tags or {}
                tag_version = tags.get("current_version") or tags.get("version") or f"v{target_version}"
                model_uri = f"models:/{model_name}/{STAGE}"
                model_obj = mlflow.pyfunc.load_model(model_uri)
        except Exception:
            pass

    # 3. Ultimo recurso: version mas reciente sin stage
    if model_obj is None:
        try:
            latest_versions = client.get_latest_versions(model_name)
            if latest_versions:
                mv = latest_versions[0]
                target_version = mv.version
                tags = mv.tags or {}
                tag_version = tags.get("current_version") or tags.get("version") or f"v{target_version}"
                model_uri = f"models:/{model_name}/{target_version}"
                model_obj = mlflow.pyfunc.load_model(model_uri)
        except Exception:
            pass

    if model_obj is None:
        return None, "None", "None", []

    schema_info = []
    try:
        input_schema = model_obj.metadata.get_input_schema()
        if input_schema:
            for col in input_schema.inputs:
                schema_info.append({
                    "name": col.name,
                    "type": str(col.type),
                    "required": getattr(col, "required", True)
                })
    except Exception:
        pass

    return model_obj, target_version, tag_version, schema_info

def fetch_and_load_model():
    """
    Carga el modelo general por defecto (ModeloPrueba) para retrocompatibilidad total,
    y precarga en paralelo los 5 modelos especializados por ciudad desde MLflow.
    """
    global loaded_model, current_version, current_mlflow_version, current_schema
    global loaded_municipal_models, municipal_models_metadata
    try:
        client = MlflowClient()

        # 1. Cargar modelo base por defecto (ModeloPrueba)
        m_base, m_ver, t_ver, s_info = load_single_model(client, MODEL_NAME)
        if m_base is not None:
            with model_lock:
                loaded_model = m_base
                current_mlflow_version = m_ver
                current_version = t_ver
                current_schema = s_info
                app.version = t_ver
            print(f"[OK] [MLflow] Modelo base '{MODEL_NAME}' ({t_ver}) cargado con exito.")
        else:
            print(f"[WARN] [MLflow] No se encontro version para el modelo base '{MODEL_NAME}'.")

        # 2. Descubrimiento dinamico de modelos DengueAlertas_* con tag 'enabled == true'
        discovered_models = []
        try:
            registered = client.search_registered_models()
            for rm in registered:
                name = rm.name
                tags = rm.tags or {}
                if name.startswith("DengueAlertas_") and name != MODEL_NAME:
                    enabled_val = str(tags.get("enabled", "")).strip().lower()
                    if enabled_val in ("true", "1", "yes"):
                        discovered_models.append((name, tags))
                    else:
                        print(f"[Multi-Model] Modelo '{name}' omitido (tag 'enabled' es '{tags.get('enabled')}')")
        except Exception as e:
            print(f"[WARN] Error buscando modelos registrados en MLflow: {e}")
            discovered_models = [
                ("DengueAlertas_Cali", {"enabled": "true"}),
                ("DengueAlertas_Bucaramanga", {"enabled": "true"}),
                ("DengueAlertas_Cartagena", {"enabled": "true"}),
                ("DengueAlertas_Ibague", {"enabled": "true"}),
                ("DengueAlertas_Villavicencio", {"enabled": "true"})
            ]

        new_loaded_muns = {}
        new_metadata_muns = {}
        new_routing_map = {}
        loaded_count = 0

        for mun_model_name, mun_tags in discovered_models:
            m_mun, m_mun_ver, t_mun_ver, s_mun_info = load_single_model(client, mun_model_name)
            if m_mun is not None:
                new_loaded_muns[mun_model_name] = m_mun

                # Extraer clave de la ciudad (ej: DengueAlertas_Cali -> cali)
                raw_city = mun_model_name.replace("DengueAlertas_", "").strip().lower()
                geo = MUNICIPALITY_GEO_REGISTRY.get(raw_city, {
                    "codigo": mun_tags.get("municipality_code", raw_city),
                    "nombre": mun_model_name.replace("DengueAlertas_", "").strip(),
                    "departamento": mun_tags.get("department", "Colombia"),
                    "deptoId": mun_tags.get("depto_id", ""),
                    "aliases": [raw_city]
                })

                new_metadata_muns[mun_model_name] = {
                    "mlflow_version": m_mun_ver,
                    "tag_version": t_mun_ver,
                    "schema": s_mun_info,
                    "tags": mun_tags,
                    "city_key": raw_city,
                    "codigo": geo.get("codigo", raw_city),
                    "nombre": geo.get("nombre", raw_city.capitalize()),
                    "departamento": geo.get("departamento", ""),
                    "deptoId": geo.get("deptoId", "")
                }

                aliases = geo.get("aliases", [raw_city, geo.get("codigo", raw_city)])
                for alias in aliases:
                    new_routing_map[str(alias).strip().lower()] = mun_model_name

                loaded_count += 1
                print(f"[OK] [Multi-Model] Modelo municipal '{mun_model_name}' ({t_mun_ver}) [ENABLED: TRUE] listo en memoria.")
            else:
                print(f"[WARN] [Multi-Model] Modelo municipal '{mun_model_name}' no pudo ser cargado.")

        with model_lock:
            loaded_municipal_models.clear()
            loaded_municipal_models.update(new_loaded_muns)
            municipal_models_metadata.clear()
            municipal_models_metadata.update(new_metadata_muns)
            MUNICIPALITY_MODELS_MAP.clear()
            MUNICIPALITY_MODELS_MAP.update(new_routing_map)

        print(f"[Hot-Reload] Estado: 1 Modelo Base + {loaded_count}/{len(discovered_models)} Modelos Municipales habilitados.")
        return True

    except Exception as e:
        print(f"[Error] Fallo general al sincronizar modelos desde MLflow: {e}")
        return False

# ---------------------------------------------------------------------------
# Ciclo de Vida: Inicializacion
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup():
    print(f"[START] Iniciando API para entorno [{STAGE}]...")
    fetch_and_load_model()


# ---------------------------------------------------------------------------
# Endpoints de la API
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "stage": STAGE,
        "model_name": MODEL_NAME,
        "model_loaded": loaded_model is not None,
        "version": current_version,
        "mlflow_version": current_mlflow_version,
        "schema_fields_count": len(current_schema),
        "municipal_models_active": list(loaded_municipal_models.keys()),
        "active_codes": [m.get("codigo") for m in municipal_models_metadata.values() if m.get("codigo")],
        "total_models_ready": (1 if loaded_model else 0) + len(loaded_municipal_models)
    }

@app.get("/municipios")
def get_active_municipios():
    """
    Devuelve los municipios que tienen modelos DengueAlertas_* con tag 'enabled == true' en MLflow.
    Permite que la UI autoconfigure el mapa dinamicamente en tiempo real.
    """
    items = []
    for model_name, meta in municipal_models_metadata.items():
        items.append({
            "model_name": model_name,
            "city_key": meta.get("city_key"),
            "nombre": meta.get("nombre"),
            "codigo": meta.get("codigo"),
            "departamento": meta.get("departamento"),
            "deptoId": meta.get("deptoId"),
            "version": meta.get("tag_version", "v1.0"),
            "mlflow_version": meta.get("mlflow_version"),
            "enabled": True
        })
    return {
        "stage": STAGE,
        "total_activos": len(items),
        "active_codes": [m["codigo"] for m in items if m.get("codigo")],
        "active_models": [m["model_name"] for m in items],
        "municipios": items
    }

@app.get("/schema")
def get_schema(response: Response, municipio: Optional[str] = Query(default=None)):
    """
    Devuelve dinamicamente las variables que espera el modelo.
    Soporta consultar el esquema especifico de un municipio o el general por defecto.
    """
    response.headers["Cache-Control"] = "public, max-age=60"

    # Si se consulta esquema de un municipio especifico
    if municipio:
        mun_key = str(municipio).strip().lower()
        mapped_name = MUNICIPALITY_MODELS_MAP.get(mun_key)
        if mapped_name and mapped_name in municipal_models_metadata:
            meta = municipal_models_metadata[mapped_name]
            return {
                "stage": STAGE,
                "model_name": mapped_name,
                "municipio": municipio,
                "version": meta.get("tag_version", "v1.0"),
                "input_schema": meta.get("schema", current_schema)
            }

    # Esquema por defecto (retrocompatible)
    if loaded_model is None and not loaded_municipal_models:
        raise HTTPException(status_code=503, detail="Modelos no disponibles en este momento.")

    return {
        "stage": STAGE,
        "model_name": MODEL_NAME,
        "version": current_version,
        "input_schema": current_schema
    }

@app.post("/sync")
def sync_model():
    """
    Permite forzar la recarga en caliente inmediata desde un webhook o curl.
    """
    success = fetch_and_load_model()
    if not success:
        raise HTTPException(status_code=500, detail="No se pudo actualizar el modelo desde MLflow.")
    return {
        "message": "Modelos sincronizados en caliente",
        "stage": STAGE,
        "default_model": MODEL_NAME,
        "municipal_models_loaded": list(loaded_municipal_models.keys())
    }

@app.post("/telemetry/event")
def record_ui_event(
    request: Request,
    event_data: Dict[str, Any]
):
    """
    Captura eventos de interaccion y clics en la interfaz de usuario (UI Clickstream)
    """
    user_email = (
        request.headers.get("Cf-Access-Authenticated-User-Email")
        or request.headers.get("cf-access-authenticated-user-email")
        or request.headers.get("x-user-email")
        or ""
    )
    req_id = request.headers.get("X-Request-ID") or f"evt_{int(time.time()*1000)}"

    telemetry_logger.info(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "type": "ui_click_event",
        "stage": STAGE,
        "request_id": req_id,
        "user_email": user_email,
        "client_ip": request.headers.get("CF-Connecting-IP") or (request.client.host if request.client else None),
        "origin": request.headers.get("origin"),
        "action": event_data.get("action", "click"),
        "target": event_data.get("target"),
        "section": event_data.get("section"),
        "metadata": event_data.get("metadata", {})
    }))
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# Capa de Retrocompatibilidad y Adaptacion Dinamica de Features
# ---------------------------------------------------------------------------
DANE_DENSITY_MAP = {
    "76001": 3910.0, "cali": 3910.0,
    "68001": 3655.0, "bucaramanga": 3655.0,
    "13001": 1581.0, "cartagena": 1581.0,
    "73001": 377.0,  "ibague": 377.0,
    "50001": 395.0,  "villavicencio": 395.0
}

def adapt_features_for_model(df: pd.DataFrame, expected_schema: list) -> pd.DataFrame:
    """
    Capa de adaptacion y retrocompatibilidad inteligente:
    Permite procesar indistintamente payloads con nombres en español (5 vars clasicas),
    payloads MLOps modernos (15 features con lags), o payloads minimalistas con 'municipio' y 'week'.
    """
    df = df.copy()
    expected_cols = [f["name"] for f in expected_schema] if expected_schema else []

    # 1. Homologacion de alias (Español <-> Ingles)
    alias_pairs = [
        ("temperatura", "temperature_c"),
        ("precipitacion", "precipitation_mm"),
        ("humedad", "relative_humidity_pct"),
        ("densidad_poblacional", "population_density_km2"),
    ]
    for es_col, en_col in alias_pairs:
        if es_col in df.columns and en_col not in df.columns:
            df[en_col] = df[es_col]
        elif en_col in df.columns and es_col not in df.columns:
            df[es_col] = df[en_col]

    # 2. Densidad poblacional por municipio si no viene explicita
    mun_col = next((c for c in ["municipio", "municipality_code", "codigo_dane", "codigo", "deptoId"] if c in df.columns), None)
    if mun_col:
        for idx, row in df.iterrows():
            key = str(row[mun_col]).strip().lower()
            if key in DANE_DENSITY_MAP:
                val = DANE_DENSITY_MAP[key]
                if "population_density_km2" not in df.columns or pd.isna(df.at[idx, "population_density_km2"]):
                    df.at[idx, "population_density_km2"] = val
                if "densidad_poblacional" not in df.columns or pd.isna(df.at[idx, "densidad_poblacional"]):
                    df.at[idx, "densidad_poblacional"] = val

    # 3. Semana del año y componentes ciclicos (sin_week, cos_week)
    week_col = next((c for c in ["week", "semana", "semana_epidemiologica"] if c in df.columns), None)
    current_week = datetime.date.today().isocalendar()[1]
    
    if "sin_week" not in df.columns or "cos_week" not in df.columns:
        weeks = df[week_col].astype(float) if week_col else float(current_week)
        df["sin_week"] = np.sin(2 * np.pi * weeks / 52.0).astype(float)
        df["cos_week"] = np.cos(2 * np.pi * weeks / 52.0).astype(float)

    # 4. Inercia epidemiologica (casos_previos <-> 5 lags)
    if "casos_previos" in df.columns:
        c_prev = df["casos_previos"].astype(float)
        if "casos_lag_1" not in df.columns: df["casos_lag_1"] = c_prev
        if "casos_lag_2" not in df.columns: df["casos_lag_2"] = c_prev * 0.90
        if "casos_lag_3" not in df.columns: df["casos_lag_3"] = c_prev * 0.85
        if "casos_lag_4" not in df.columns: df["casos_lag_4"] = c_prev * 0.80
        if "casos_rolling_mean_4w" not in df.columns: df["casos_rolling_mean_4w"] = c_prev
    elif "casos_lag_1" in df.columns and "casos_previos" not in df.columns:
        df["casos_previos"] = df["casos_lag_1"]

    # 5. Lags climaticos (si faltan en el payload, usar valores contemporaneos)
    for base_col, lag_col in [("temperature_c", "temperatura_lag_2"),
                             ("precipitation_mm", "precipitacion_lag_2"),
                             ("precipitation_mm", "precipitacion_lag_4"),
                             ("relative_humidity_pct", "humedad_lag_2")]:
        if lag_col not in df.columns and base_col in df.columns:
            df[lag_col] = df[base_col]

    # 6. Proyectar y alinear estrictamente las columnas que el modelo activo espera
    if expected_cols:
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0.0
        df = df[expected_cols]

    return df

@app.post("/predict")
def predict(
    request: Request,
    payload: Union[Dict[str, Any], List[Dict[str, Any]]],
    threshold: Optional[float] = Query(default=None, description="Umbral de corte opcional")
):
    """
    Recibe cualquier payload JSON (uno o varios registros).
    Convierte a DataFrame y MLflow valida el esquema automaticamente.
    Registra entradas (features) y resultados para analiticas de Data Drift.
    """
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="El modelo aun no ha sido cargado en memoria.")
    
    cutoff = threshold if threshold is not None else DEFAULT_THRESHOLD
    user_email = (
        request.headers.get("Cf-Access-Authenticated-User-Email")
        or request.headers.get("cf-access-authenticated-user-email")
        or request.headers.get("x-user-email")
        or ""
    )
    req_id = request.headers.get("X-Request-ID") or ""

    # Normalizar entrada a lista de registros
    records = [payload] if isinstance(payload, dict) else payload
    if not records:
        raise HTTPException(status_code=400, detail="El payload esta vacio.")

    df = pd.DataFrame(records)

    # 1. Seleccion del modelo objetivo (Enrutador municipal con fallback al modelo base)
    target_model = loaded_model
    target_model_name = MODEL_NAME
    target_version = current_version
    target_schema = current_schema
    evaluated_mun = None

    if records and isinstance(records[0], dict):
        mun_val = (
            records[0].get("municipio")
            or records[0].get("municipality_code")
            or records[0].get("codigo_dane")
            or records[0].get("codigo")
            or records[0].get("deptoId")
        )
        if mun_val:
            mun_key = str(mun_val).strip().lower()
            mapped_name = MUNICIPALITY_MODELS_MAP.get(mun_key)
            if mapped_name and mapped_name in loaded_municipal_models:
                target_model = loaded_municipal_models[mapped_name]
                target_model_name = mapped_name
                meta = municipal_models_metadata.get(mapped_name, {})
                target_version = meta.get("tag_version", "v1.0")
                target_schema = meta.get("schema") or current_schema
                evaluated_mun = str(mun_val)

    if target_model is None:
        raise HTTPException(status_code=503, detail="Ningun modelo disponible en memoria para inferencia.")

    # 2. Adaptador de retrocompatibilidad proyectado al esquema exacto del modelo seleccionado
    df = adapt_features_for_model(df, target_schema)

    # 3. Adaptar tipos de datos automaticamente al esquema inferido por MLflow para target_model
    if target_schema:
        for field in target_schema:
            col_name = field["name"]
            col_type = field["type"].lower()
            if col_name in df.columns:
                try:
                    if "integer" in col_type or "int32" in col_type:
                        df[col_name] = df[col_name].astype("int32")
                    elif "long" in col_type or "int64" in col_type:
                        df[col_name] = df[col_name].astype("int64")
                    elif "double" in col_type or "float" in col_type:
                        df[col_name] = df[col_name].astype("float64")
                except Exception:
                    pass

    try:
        infer_start = time.time()
        with model_lock:
            probs = None
            try:
                raw_model = getattr(target_model, "_model_impl", None)
                if raw_model and hasattr(raw_model, "predict_proba"):
                    probs_matrix = raw_model.predict_proba(df)
                    probs = [float(p[1]) for p in probs_matrix]
            except Exception:
                probs = None

            raw_preds = target_model.predict(df)
            
            # Soporte polimorfico: LightGBM v5 devuelve DataFrame ('alert', 'p_alert', 'pred_count'),
            # mientras que modelos sklearn devuelven ndarray o Series
            if isinstance(raw_preds, pd.DataFrame):
                alert_col = "alert" if "alert" in raw_preds.columns else raw_preds.columns[0]
                preds_list = raw_preds[alert_col].tolist()
                if "p_alert" in raw_preds.columns:
                    probs = raw_preds["p_alert"].astype(float).tolist()
            elif isinstance(raw_preds, pd.Series):
                preds_list = raw_preds.tolist()
            elif hasattr(raw_preds, "tolist"):
                preds_list = raw_preds.tolist()
            else:
                preds_list = list(raw_preds)

        infer_duration = round(time.time() - infer_start, 4)

        results = []
        for i, pred in enumerate(preds_list):
            try:
                pred_int = int(pred)
            except (ValueError, TypeError):
                pred_int = 1 if str(pred).lower() in ("1", "true", "alert") else 0

            prob_val = probs[i] if probs is not None and i < len(probs) else (0.85 if pred_int == 1 else 0.15)
            is_alert = bool(prob_val >= cutoff)
            res_item = {
                "index": i,
                "prediction": pred_int,
                "probability": round(prob_val, 4),
                "alerta_activa": is_alert
            }
            if isinstance(raw_preds, pd.DataFrame) and "pred_count" in raw_preds.columns:
                try:
                    res_item["pred_count"] = round(float(raw_preds["pred_count"].iloc[i]), 2)
                except Exception:
                    pass
            results.append(res_item)

        alertas_count = sum(1 for r in results if r["alerta_activa"])
        avg_prob = round(sum(r["probability"] for r in results) / len(results), 4) if results else 0.0

        # Telemetria rica de inferencia: incluye entradas completas para analisis de Data Drift
        telemetry_logger.info(json.dumps({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": "inference_metrics",
            "stage": STAGE,
            "request_id": req_id,
            "user_email": user_email,
            "model_name": target_model_name,
            "model_version": target_version,
            "municipio_evaluado": evaluated_mun or "Base / No especificado",
            "threshold_used": cutoff,
            "records_count": len(results),
            "alertas_activadas": alertas_count,
            "avg_probability": avg_prob,
            "inputs": records if len(records) > 1 else records[0],
            "results": results if len(results) > 1 else results[0],
            "inference_time_sec": infer_duration
        }))

        return {
            "stage": STAGE,
            "model_name": target_model_name,
            "model_version": target_version,
            "municipio_evaluado": evaluated_mun or "Base / No especificado",
            "threshold_used": cutoff,
            "total_records": len(results),
            "results": results if len(results) > 1 else results[0]
        }

    except Exception as e:
        telemetry_logger.error(json.dumps({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": "inference_error",
            "stage": STAGE,
            "request_id": req_id,
            "user_email": user_email,
            "model_name": MODEL_NAME,
            "model_version": current_version,
            "inputs": records if len(records) > 1 else records[0],
            "error_detail": str(e),
            "traceback": traceback.format_exc()
        }))
        raise HTTPException(
            status_code=422,
            detail=f"Error de validacion del esquema para el modelo {STAGE} v{current_version}: {str(e)}"
        )
