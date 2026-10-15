import os
import time
import json
import hmac
import hashlib
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, Header
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
LOG_DIR = os.getenv("LOG_DIR", "/opt/dengue-api/logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Structured JSON Logger
# ---------------------------------------------------------------------------
webhook_logger = logging.getLogger("dengue_webhook")
webhook_logger.setLevel(logging.INFO)
if not webhook_logger.handlers:
    log_file = os.path.join(LOG_DIR, "webhook_events.jsonl")
    handler = RotatingFileHandler(log_file, maxBytes=50 * 1024 * 1024, backupCount=10, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    webhook_logger.addHandler(handler)

app = FastAPI(
    title="MLflow Webhook Receiver",
    description="Microservicio receptor y telemetría de Webhooks de MLflow",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import base64

def verify_mlflow_hmac(raw_body: bytes, signature_header: str, delivery_id: str, timestamp_hdr: str, secret: str) -> bool:
    if not secret:
        return True
    if not signature_header:
        return False
    
    clean_sig = signature_header.strip()
    
    # 1. Standard MLflow format: v1,<base64>
    if clean_sig.startswith("v1,"):
        received_b64 = clean_sig.split("v1,")[1]
        raw_str = raw_body.decode("utf-8", errors="replace")
        signed_content = f"{delivery_id}.{timestamp_hdr}.{raw_str}"
        expected_digest = hmac.new(secret.encode("utf-8"), signed_content.encode("utf-8"), hashlib.sha256).digest()
        expected_b64 = base64.b64encode(expected_digest).decode("utf-8")
        if hmac.compare_digest(expected_b64, received_b64):
            return True

    # 2. Fallback direct sha256 hex
    if clean_sig.startswith("sha256="):
        clean_sig = clean_sig.split("sha256=")[1]
    expected_hex = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected_hex, clean_sig)

@app.get("/webhook/health")
async def health():
    return {
        "status": "healthy",
        "service": "mlflow-webhook",
        "secret_configured": bool(WEBHOOK_SECRET),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

@app.post("/webhook/mlflow")
@app.post("/webhook")
async def receive_mlflow_webhook(request: Request):
    start_time = time.time()
    req_id = request.headers.get("X-Request-ID") or request.headers.get("CF-Ray") or f"wh_{int(start_time * 1000)}"
    
    # Read raw body
    body_bytes = await request.body()
    
    # Try parsing JSON
    payload = {}
    try:
        if body_bytes:
            payload = json.loads(body_bytes.decode("utf-8", errors="replace"))
    except Exception as e:
        payload = {"_raw_body": body_bytes.decode("utf-8", errors="replace"), "_error": str(e)}

    # Extract signature headers (MLflow sends X-Mlflow-Signature, X-Mlflow-Delivery-Id, X-Mlflow-Timestamp)
    sig_header = (
        request.headers.get("X-Mlflow-Signature")
        or request.headers.get("X-Hub-Signature-256")
        or request.headers.get("X-Signature-256")
        or request.headers.get("X-Signature")
        or ""
    )
    delivery_id = request.headers.get("X-Mlflow-Delivery-Id", "")
    timestamp_hdr = request.headers.get("X-Mlflow-Timestamp", "")

    sig_valid = None
    if WEBHOOK_SECRET:
        sig_valid = verify_mlflow_hmac(body_bytes, sig_header, delivery_id, timestamp_hdr, WEBHOOK_SECRET)

    # Extract event metadata
    entity = payload.get("entity", "")
    action = payload.get("action", "")
    if entity and action:
        event_type = f"{entity}.{action}"
    else:
        event_type = payload.get("event") or payload.get("event_type") or "unknown_event"

    data = payload.get("data", {}) if isinstance(payload.get("data"), dict) else {}
    model_name = (
        data.get("name")
        or data.get("model_name")
        or payload.get("model_name")
        or payload.get("registered_model", {}).get("name")
        or ""
    )
    version = (
        data.get("version")
        or payload.get("version")
        or payload.get("model_version", {}).get("version")
        or ""
    )
    alias = (
        data.get("alias")
        or payload.get("alias")
        or payload.get("model_version_alias", {}).get("alias")
        or ""
    )
    tag = (
        data.get("key")
        or payload.get("tag")
        or payload.get("model_version_tag", {}).get("key")
        or ""
    )

    client_ip = (
        request.headers.get("CF-Connecting-IP")
        or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or request.client.host
        if request.client else "unknown"
    )

    # -----------------------------------------------------------------------
    # Sincronizacion automatica de Stage (Retrocompatibilidad sin duplicados)
    # -----------------------------------------------------------------------
    stage_sync_result = None
    api_notify_result = None
    if sig_valid and model_name and version:
        action_clean = action.lower() if action else ""
        alias_clean = alias.lower() if alias else ""

        target_stage = None
        if "created" in action_clean or "created" in event_type.lower():
            if alias_clean in ("production", "champion"):
                target_stage = "Production"
            elif alias_clean in ("staging", "challenger"):
                target_stage = "Staging"
        elif "deleted" in action_clean or "deleted" in event_type.lower():
            if alias_clean in ("production", "champion", "staging", "challenger"):
                target_stage = "Archived"

        if target_stage:
            try:
                import mlflow
                from mlflow.tracking import MlflowClient
                mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:////opt/mlflow/mlflow.db")
                client = MlflowClient(tracking_uri=mlflow_uri)
                client.transition_model_version_stage(
                    name=model_name,
                    version=str(version),
                    stage=target_stage,
                    archive_existing_versions=True
                )
                stage_sync_result = {
                    "synced": True,
                    "target_stage": target_stage,
                    "archive_existing_versions": True
                }
            except Exception as e:
                stage_sync_result = {
                    "synced": False,
                    "target_stage": target_stage,
                    "error": str(e)
                }

            # -------------------------------------------------------------------
            # Notificar en caliente a la API correspondiente (/sync)
            # -------------------------------------------------------------------
            if stage_sync_result and stage_sync_result.get("synced"):
                sync_url = "http://127.0.0.1:8000/sync" if target_stage == "Production" else "http://127.0.0.1:8001/sync"
                try:
                    import requests
                    sync_resp = requests.post(sync_url, timeout=10)
                    api_notify_result = {
                        "notified": True,
                        "target_stage": target_stage,
                        "url": sync_url,
                        "status_code": sync_resp.status_code,
                        "response": sync_resp.json() if sync_resp.headers.get("content-type", "").startswith("application/json") else sync_resp.text[:200]
                    }
                except Exception as sync_err:
                    api_notify_result = {
                        "notified": False,
                        "target_stage": target_stage,
                        "url": sync_url,
                        "error": str(sync_err)
                    }


    duration = round(time.time() - start_time, 4)

    # Log structured event
    log_record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "type": "mlflow_webhook_event",
        "request_id": req_id,
        "event_type": event_type,
        "model_name": model_name,
        "version": str(version),
        "alias": str(alias),
        "tag": str(tag),
        "signature_valid": sig_valid,
        "has_signature_header": bool(sig_header),
        "client_ip": client_ip,
        "stage_sync": stage_sync_result,
        "api_sync_notification": api_notify_result,
        "processing_time_sec": duration,
        "payload": payload
    }
    
    webhook_logger.info(json.dumps(log_record))

    return {
        "status": "received",
        "request_id": req_id,
        "signature_valid": sig_valid,
        "event": event_type,
        "model_name": model_name,
        "version": str(version),
        "alias": str(alias),
        "stage_sync": stage_sync_result,
        "api_sync_notification": api_notify_result
    }

