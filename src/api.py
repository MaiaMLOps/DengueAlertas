import os
import threading
import time
import json
import logging
import traceback
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
# Estado global del Modelo (Hot-Reloading en memoria)
# ---------------------------------------------------------------------------
loaded_model = None
current_version = "None"
current_mlflow_version = "None"
current_schema = []
model_lock = threading.Lock()

def fetch_and_load_model():
    """
    Consulta MLflow para el STAGE configurado (Staging o Production),
    obtiene la version activa mas reciente, extrae la version desde el TAG 'current_version',
    y carga el modelo en caliente en memoria actualizando app.version.
    """
    global loaded_model, current_version, current_mlflow_version, current_schema
    try:
        client = MlflowClient()
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=[STAGE])
        
        if not latest_versions:
            print(f"[WARN] [MLflow] No se encontro ninguna version en Stage '{STAGE}' para el modelo '{MODEL_NAME}'.")
            return False

        model_version_obj = latest_versions[0]
        target_mlflow_version = model_version_obj.version
        
        tags = model_version_obj.tags or {}
        tag_version = tags.get("current_version") or tags.get("version") or f"v{target_mlflow_version}"

        if loaded_model is not None and current_mlflow_version == target_mlflow_version and current_version == tag_version:
            return True

        print(f"[Hot-Reload] Cargando '{MODEL_NAME}' Version MLflow {target_mlflow_version} (Tag: {tag_version}, Stage: {STAGE})...")
        model_uri = f"models:/{MODEL_NAME}/{STAGE}"
        new_model = mlflow.pyfunc.load_model(model_uri)
        
        schema_info = []
        try:
            input_schema = new_model.metadata.get_input_schema()
            if input_schema:
                for col in input_schema.inputs:
                    schema_info.append({
                        "name": col.name,
                        "type": str(col.type),
                        "required": getattr(col, "required", True)
                    })
        except Exception as se:
            print(f"Aviso al extraer schema: {se}")

        with model_lock:
            loaded_model = new_model
            current_version = tag_version
            current_mlflow_version = target_mlflow_version
            current_schema = schema_info
            app.version = tag_version

        print(f"[OK] [Hot-Reload] Modelo {tag_version} (MLflow v{target_mlflow_version}) cargado con exito. Variables en schema: {len(schema_info)}")
        
        telemetry_logger.info(json.dumps({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": "model_reloaded",
            "stage": STAGE,
            "model_name": MODEL_NAME,
            "model_version": tag_version,
            "mlflow_version": target_mlflow_version,
            "variables_count": len(schema_info)
        }))
        return True

    except Exception as e:
        print(f"[Error] Fallo al cargar modelo desde MLflow: {e}")
        telemetry_logger.error(json.dumps({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": "model_load_error",
            "stage": STAGE,
            "model_name": MODEL_NAME,
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        return False

# ---------------------------------------------------------------------------
# Ciclo de Vida: Inicializacion (Event-Driven: Sincronizacion por Webhooks)
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
        "schema_fields_count": len(current_schema)
    }

@app.get("/schema")
def get_schema(response: Response):
    """
    Devuelve dinamicamente las variables que espera el modelo activo actualmente.
    Incluye Cache-Control para que el navegador almacene en cache 60s y reduzca llamadas.
    """
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible en este momento.")
    
    response.headers["Cache-Control"] = "public, max-age=60"
    
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
        "message": "Modelo sincronizado en caliente",
        "stage": STAGE,
        "version": current_version,
        "schema": current_schema
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

    # Adaptar tipos de datos automaticamente al esquema inferido por MLflow
    if current_schema:
        for field in current_schema:
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
                raw_model = getattr(loaded_model, "_model_impl", None)
                if raw_model and hasattr(raw_model, "predict_proba"):
                    probs_matrix = raw_model.predict_proba(df)
                    probs = [float(p[1]) for p in probs_matrix]
            except Exception:
                probs = None

            raw_preds = loaded_model.predict(df)
            
            if hasattr(raw_preds, "tolist"):
                preds_list = raw_preds.tolist()
            else:
                preds_list = list(raw_preds)

        infer_duration = round(time.time() - infer_start, 4)

        results = []
        for i, pred in enumerate(preds_list):
            prob_val = probs[i] if probs is not None else (0.85 if pred == 1 else 0.15)
            is_alert = bool(prob_val >= cutoff)
            results.append({
                "index": i,
                "prediction": int(pred),
                "probability": round(prob_val, 4),
                "alerta_activa": is_alert
            })

        alertas_count = sum(1 for r in results if r["alerta_activa"])
        avg_prob = round(sum(r["probability"] for r in results) / len(results), 4) if results else 0.0

        # Telemetria rica de inferencia: incluye entradas completas para analisis de Data Drift
        telemetry_logger.info(json.dumps({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": "inference_metrics",
            "stage": STAGE,
            "request_id": req_id,
            "user_email": user_email,
            "model_name": MODEL_NAME,
            "model_version": current_version,
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
            "model_version": current_version,
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
