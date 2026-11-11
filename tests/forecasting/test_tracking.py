"""Unit checks of tracking.py with fake credentials, no network and isolated MLflow state."""

import os
import socket

import pytest
from mlflow.tracking.request_header import registry as header_registry

from src.forecasting import tracking
from src.forecasting.config import ROOT, load_config

FAKE_ENV = {"CF_ACCESS_CLIENT_ID": "fake-id.access", "CF_ACCESS_CLIENT_SECRET": "fake-secret"}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Unit tests must not open network connections.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.delenv("FORECAST_HORIZON", raising=False)
    monkeypatch.setattr(header_registry._request_header_provider_registry, "_registry", [])
    monkeypatch.setattr(tracking, "_active_headers", {})
    uris = []
    monkeypatch.setattr(tracking.mlflow, "set_tracking_uri", uris.append)
    return uris


def _with_mode(mode):
    config = load_config()
    return config.model_copy(update={"seguimiento": config.seguimiento.model_copy(update={"modo": mode})})


def test_headers_use_cloudflare_names():
    assert tracking.cloudflare_headers(FAKE_ENV) == {
        "CF-Access-Client-Id": "fake-id.access",
        "CF-Access-Client-Secret": "fake-secret",
    }


def test_empty_value_counts_as_missing():
    with pytest.raises(tracking.MissingCredentialsError, match="CF_ACCESS_CLIENT_SECRET"):
        tracking.cloudflare_headers({**FAKE_ENV, "CF_ACCESS_CLIENT_SECRET": ""})


def test_load_environment_reads_given_file_and_process_env_wins(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("CF_ACCESS_CLIENT_ID=from-file\nONLY_IN_FILE=yes\n", encoding="utf-8")
    monkeypatch.setenv("CF_ACCESS_CLIENT_ID", "from-process")

    env = tracking.load_environment(dotenv)

    assert env["CF_ACCESS_CLIENT_ID"] == "from-process"
    assert env["ONLY_IN_FILE"] == "yes"
    assert "ONLY_IN_FILE" not in os.environ


def test_load_environment_without_file_returns_process_env(tmp_path):
    env = tracking.load_environment(tmp_path / "absent.env")
    assert "ONLY_IN_FILE" not in env


def test_local_mode_needs_no_credentials(_isolate):
    uri = tracking.configure_tracking(_with_mode("local"), environ={})

    assert uri == (ROOT / "mlruns").resolve().as_uri()
    assert uri.startswith("file:")
    assert _isolate == [uri]
    assert list(header_registry._request_header_provider_registry) == []


def test_server_mode_uses_tracking_uri_from_environment(_isolate):
    uri = tracking.configure_tracking(_with_mode("servidor"), environ={**FAKE_ENV, "MLFLOW_TRACKING_URI": "https://example.test"})

    assert uri == "https://example.test"
    assert _isolate == ["https://example.test"]


def test_server_mode_defaults_to_configured_uri():
    uri = tracking.configure_tracking(_with_mode("servidor"), environ=FAKE_ENV)
    assert uri == "https://platform.arkoiot.com"


def test_server_mode_without_credentials_fails_and_does_not_fall_back(_isolate):
    with pytest.raises(tracking.MissingCredentialsError):
        tracking.configure_tracking(_with_mode("servidor"), environ={})
    assert _isolate == []


def test_configuring_twice_registers_a_single_header_provider():
    config = _with_mode("servidor")
    tracking.configure_tracking(config, environ=FAKE_ENV)
    tracking.configure_tracking(config, environ=FAKE_ENV)

    providers = [
        provider
        for provider in header_registry._request_header_provider_registry
        if isinstance(provider, tracking.CloudflareAccessHeaderProvider)
    ]
    assert len(providers) == 1
    assert header_registry.resolve_request_headers() == {
        "CF-Access-Client-Id": "fake-id.access",
        "CF-Access-Client-Secret": "fake-secret",
    }


def test_switching_to_local_deactivates_headers():
    tracking.configure_tracking(_with_mode("servidor"), environ=FAKE_ENV)
    tracking.configure_tracking(_with_mode("local"), environ={})
    assert header_registry.resolve_request_headers() == {}
