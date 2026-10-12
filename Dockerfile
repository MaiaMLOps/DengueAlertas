FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias de inferencia
COPY requirements/serve.txt .
RUN pip install --no-cache-dir -r serve.txt

# Variables de entorno por defecto (se sobreescriben en docker run)
ENV STAGE="Production"
ENV MODEL_NAME="ModeloPrueba"
ENV MLFLOW_TRACKING_URI="https://platform.arkoiot.com"
ENV MODEL_THRESHOLD="0.205"

COPY src/ /app/src/
EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]