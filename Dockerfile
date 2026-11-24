# Imagen de referencia para un eventual despliegue en un servicio PaaS.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements/serve.txt .
RUN pip install --no-cache-dir -r serve.txt
COPY configs/ ./configs/
COPY src/ ./src/

# MODEL_NAME es el modelo base retrocompatible que informa /health, /schema y
# el /predict manual sin municipio. Debe ser un modelo DengueAlertas_* real;
# los seis municipales se descubren solos por el prefijo y el tag enabled=true.
ENV STAGE="Production" \
    MODEL_NAME="DengueAlertas_Barranquilla" \
    MLFLOW_TRACKING_URI="https://platform.arkoiot.com" \
    MODEL_THRESHOLD="0.205" \
    FORECAST_PANEL_PATH="/app/data/processed/pilot_municipalities_weekly.parquet"

RUN mkdir -p /tmp/dengue-logs && ln -s /tmp/dengue-logs /app/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
