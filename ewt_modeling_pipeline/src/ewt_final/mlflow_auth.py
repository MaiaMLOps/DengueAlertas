"""Cabeceras Cloudflare Access para clientes MLflow."""
from __future__ import annotations

import os


def configure_cloudflare_access() -> None:
    """Registra el proveedor de cabeceras sin imprimir secretos."""
    try:
        from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider
        from mlflow.tracking.request_header.registry import _request_header_provider_registry
    except ImportError:
        return

    class CloudflareAccessHeaderProvider(RequestHeaderProvider):
        def in_context(self):
            return bool(os.getenv("CF_ACCESS_CLIENT_ID") and os.getenv("CF_ACCESS_CLIENT_SECRET"))

        def request_headers(self):
            return {
                "CF-Access-Client-Id": os.environ["CF_ACCESS_CLIENT_ID"],
                "CF-Access-Client-Secret": os.environ["CF_ACCESS_CLIENT_SECRET"],
            }

    try:
        _request_header_provider_registry.register(CloudflareAccessHeaderProvider)
    except Exception:
        # Puede estar registrado por la API en el mismo proceso.
        pass
