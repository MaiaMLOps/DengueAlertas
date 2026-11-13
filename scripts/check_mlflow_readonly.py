"""Read-only check of the team MLflow server (F0, U0.7).

Uses src/forecasting/tracking.py in server mode only inside this script; configs/forecasting.yaml
is not changed. Lists experiments and registered models (tags, aliases, versions) and the server
version when /version is exposed. It creates, updates or deletes nothing and never prints
credentials or request headers.

Run from the repository root:
    .venv/Scripts/python.exe -B scripts/check_mlflow_readonly.py
"""

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

import mlflow  # noqa: E402
import requests  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.entities import ViewType  # noqa: E402
from mlflow.tracking.request_header.registry import resolve_request_headers  # noqa: E402

from src.forecasting.config import load_config  # noqa: E402
from src.forecasting.tracking import configure_tracking  # noqa: E402


def _short(value, limit=80):
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _server_version(uri: str) -> str:
    response = requests.get(uri.rstrip("/") + "/version", headers=resolve_request_headers(), timeout=30)
    if response.status_code != 200:
        return f"not exposed (HTTP {response.status_code})"
    return _short(response.text.strip(), 40)


def main() -> None:
    config = load_config()
    server_config = config.model_copy(
        update={"seguimiento": config.seguimiento.model_copy(update={"modo": "servidor"})}
    )
    uri = configure_tracking(server_config)
    parsed = urlparse(uri)
    print(f"Tracking server: {parsed.scheme}://{parsed.hostname}")
    print(f"Client: mlflow-skinny {mlflow.__version__}")
    print(f"Server version: {_server_version(uri)}")

    client = MlflowClient()

    experiments = client.search_experiments(view_type=ViewType.ALL)
    print(f"\nExperiments: {len(experiments)}")
    for experiment in experiments:
        tags = {key: _short(value, 40) for key, value in experiment.tags.items() if not key.startswith("mlflow.")}
        print(f"- [{experiment.experiment_id}] {experiment.name} ({experiment.lifecycle_stage}) tags={tags}")

    models = client.search_registered_models()
    print(f"\nRegistered models: {len(models)}")
    for model in models:
        print(f"- {model.name}")
        print(f"  tags={ {key: _short(value, 40) for key, value in model.tags.items()} }")
        print(f"  aliases={dict(model.aliases)}")
        versions = client.search_model_versions(f"name='{model.name}'")
        for version in sorted(versions, key=lambda item: int(item.version)):
            version_tags = {key: _short(value, 40) for key, value in version.tags.items()}
            print(f"  v{version.version}: stage={version.current_stage} status={version.status} tags={version_tags}")


if __name__ == "__main__":
    main()
