"""MLflow connection with Cloudflare Access headers (especificación §9).

Server mode needs CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET and fails before any network
call when either is missing. Local mode uses the file store under mlruns/. There is no automatic
fallback from server to local: the mode is chosen explicitly in configs/forecasting.yaml.
Credential values are never logged, printed or included in error messages.
"""

import os
from collections.abc import Mapping
from pathlib import Path

import mlflow
from dotenv import dotenv_values
from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider
from mlflow.tracking.request_header.registry import _request_header_provider_registry

from src.forecasting.config import ROOT, ForecastConfig

CLIENT_ID_VAR = "CF_ACCESS_CLIENT_ID"
CLIENT_SECRET_VAR = "CF_ACCESS_CLIENT_SECRET"
TRACKING_URI_VAR = "MLFLOW_TRACKING_URI"
DEFAULT_DOTENV_PATH = ROOT / ".env"

# Headers served by the registered provider; empty until server mode is configured.
_active_headers: dict[str, str] = {}


class MissingCredentialsError(RuntimeError):
    """Cloudflare Access credentials are absent; the message names variables, never values."""


class CloudflareAccessHeaderProvider(RequestHeaderProvider):
    """Adds the Cloudflare Access headers to every MLflow request once server mode is configured."""

    def in_context(self):
        return bool(_active_headers)

    def request_headers(self):
        return dict(_active_headers)


def cloudflare_headers(environ: Mapping[str, str]) -> dict[str, str]:
    """Build the two Cloudflare Access headers from an explicit environment mapping."""
    missing = [name for name in (CLIENT_ID_VAR, CLIENT_SECRET_VAR) if not environ.get(name)]
    if missing:
        raise MissingCredentialsError(
            "Missing Cloudflare Access credentials for MLflow: "
            + ", ".join(missing)
            + ". Define them in .env or in the environment."
        )
    return {
        "CF-Access-Client-Id": environ[CLIENT_ID_VAR],
        "CF-Access-Client-Secret": environ[CLIENT_SECRET_VAR],
    }


def load_environment(dotenv_path: str | Path = DEFAULT_DOTENV_PATH) -> dict[str, str]:
    """Merge .env (if present) with the process environment; the process environment wins.

    os.environ is not modified.
    """
    path = Path(dotenv_path)
    from_file = {key: value for key, value in dotenv_values(path).items() if value is not None} if path.is_file() else {}
    return {**from_file, **os.environ}


def _register_provider_once() -> None:
    if not any(isinstance(provider, CloudflareAccessHeaderProvider) for provider in _request_header_provider_registry):
        _request_header_provider_registry.register(CloudflareAccessHeaderProvider)


def configure_tracking(config: ForecastConfig, environ: Mapping[str, str] | None = None) -> str:
    """Point MLflow at the server or the local file store, as configured, and return the URI."""
    tracking = config.seguimiento
    if tracking.modo == "local":
        _active_headers.clear()
        directory = Path(tracking.directorio_local)
        if not directory.is_absolute():
            directory = ROOT / directory
        uri = directory.resolve().as_uri()
    else:
        env = load_environment() if environ is None else environ
        headers = cloudflare_headers(env)
        uri = env.get(TRACKING_URI_VAR) or tracking.uri_servidor
        _active_headers.clear()
        _active_headers.update(headers)
        _register_provider_once()
    mlflow.set_tracking_uri(uri)
    return uri
