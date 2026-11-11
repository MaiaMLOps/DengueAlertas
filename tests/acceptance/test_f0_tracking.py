"""F0 acceptance: Cloudflare Access credentials for MLflow (especificación §6.F0 and §9).

tracking.cloudflare_headers receives an explicit environment mapping, so these tests never
read .env and never open a network connection.
"""

import logging
import socket

import pytest

pytestmark = pytest.mark.acceptance

ID_VAR = "CF_ACCESS_CLIENT_ID"
SECRET_VAR = "CF_ACCESS_CLIENT_SECRET"
DECOY = "señuelo-no-imprimir-7f3a"


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("No network connection is allowed when credentials are missing.")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.mark.parametrize(
    ("environ", "missing"),
    [
        ({}, [ID_VAR, SECRET_VAR]),
        ({SECRET_VAR: DECOY}, [ID_VAR]),
        ({ID_VAR: DECOY}, [SECRET_VAR]),
    ],
    ids=["both_missing", "id_missing", "secret_missing"],
)
def test_missing_credentials_fail_fast(environ, missing):
    from src.forecasting.tracking import MissingCredentialsError, cloudflare_headers

    with pytest.raises(MissingCredentialsError) as error:
        cloudflare_headers(environ)

    for name in missing:
        assert name in str(error.value)


@pytest.mark.parametrize(
    "environ",
    [{ID_VAR: DECOY}, {SECRET_VAR: DECOY}],
    ids=["id_present", "secret_present"],
)
def test_present_credential_value_is_never_printed(environ, capsys, caplog):
    from src.forecasting.tracking import MissingCredentialsError, cloudflare_headers

    caplog.set_level(logging.DEBUG)

    with pytest.raises(MissingCredentialsError) as error:
        cloudflare_headers(environ)

    captured = capsys.readouterr()
    assert DECOY not in str(error.value)
    assert DECOY not in repr(error.value.args)
    assert DECOY not in captured.out
    assert DECOY not in captured.err
    assert DECOY not in caplog.text
