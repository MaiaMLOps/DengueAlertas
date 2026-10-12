import os
import threading
import time
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# ---------------------------------------------------------------------------
# Autenticación transparente con Cloudflare Zero Trust (Service Tokens)
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

app = FastAPI(
    title=f"Dengue Risk API [{STAGE}]",
    description=f"API de inferencia de Dengue",
    version=f"{current_version}"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    obtiene la versión activa más reciente, extrae la versión desde el TAG 'current_version',
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
        
        # Extraer la versión semántica desde el tag (ej. 'current_version' o 'version')
        tags = model_version_obj.tags or {}
        tag_version = tags.get("current_version") or tags.get("version") or f"v{target_mlflow_version}"

        # Si ya tenemos esa versión de MLflow cargada, no hacemos nada
        if loaded_model is not None and current_mlflow_version == target_mlflow_version and current_version == tag_version:
            return True

        print(f"[Hot-Reload] Cargando '{MODEL_NAME}' Version MLflow {target_mlflow_version} (Tag: {tag_version}, Stage: {STAGE})...")
        model_uri = f"models:/{MODEL_NAME}/{STAGE}"
        new_model = mlflow.pyfunc.load_model(model_uri)
        
        # Extraer el esquema dinámico de entrada
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
            app.version = tag_version  # Actualiza la versión de Swagger / OpenAPI dinámicamente

        print(f"[OK] [Hot-Reload] Modelo {tag_version} (MLflow v{target_mlflow_version}) cargado con exito. Variables en schema: {len(schema_info)}")
        return True

    except Exception as e:
        print(f"[Error] Fallo al cargar modelo desde MLflow: {e}")
        return False

# Hilo en segundo plano para verificar cambios de versión cada 45 segundos
def auto_sync_worker():
    while True:
        time.sleep(45)
        try:
            fetch_and_load_model()
        except Exception:
            pass

@app.on_event("startup")
def startup():
    print(f"[START] Iniciando API para entorno [{STAGE}]...")
    fetch_and_load_model()
    threading.Thread(target=auto_sync_worker, daemon=True).start()

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
def get_schema():
    """
    Devuelve dinámicamente las variables que espera el modelo activo actualmente.
    El frontend puede consumir esto para saber qué campos mostrar.
    """
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible en este momento.")
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

@app.post("/predict")
def predict(
    payload: Union[Dict[str, Any], List[Dict[str, Any]]],
    threshold: Optional[float] = Query(default=None, description="Umbral de corte opcional")
):
    """
    Recibe cualquier payload JSON (uno o varios registros).
    Convierte a DataFrame y MLflow valida el esquema automáticamente.
    """
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="El modelo aún no ha sido cargado en memoria.")
    
    cutoff = threshold if threshold is not None else DEFAULT_THRESHOLD

    # Normalizar entrada a lista de registros
    records = [payload] if isinstance(payload, dict) else payload
    if not records:
        raise HTTPException(status_code=400, detail="El payload está vacío.")

    df = pd.DataFrame(records)

    # Adaptar tipos de datos automáticamente al esquema inferido por MLflow
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
        # MLflow valida tipos y columnas requeridas contra su firma interna
        with model_lock:
            # Intentar obtener probabilidades si el estimador base lo soporta
            probs = None
            try:
                raw_model = getattr(loaded_model, "_model_impl", None)
                if raw_model and hasattr(raw_model, "predict_proba"):
                    probs_matrix = raw_model.predict_proba(df)
                    probs = [float(p[1]) for p in probs_matrix]
            except Exception:
                probs = None

            raw_preds = loaded_model.predict(df)
            
            # Normalizar predicciones
            if hasattr(raw_preds, "tolist"):
                preds_list = raw_preds.tolist()
            else:
                preds_list = list(raw_preds)

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

        return {
            "stage": STAGE,
            "model_version": current_version,
            "threshold_used": cutoff,
            "total_records": len(results),
            "results": results if len(results) > 1 else results[0]
        }

    except Exception as e:
        # MLflow genera errores descriptivos si falta alguna columna o el tipo de dato es erróneo
        raise HTTPException(
            status_code=422,
            detail=f"Error de validación del esquema para el modelo {STAGE} v{current_version}: {str(e)}"
        )
