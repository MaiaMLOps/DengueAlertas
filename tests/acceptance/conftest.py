"""Acceptance data paths require an explicit pytest option (convention of tests/epidemiology).

Without the option the test is skipped; with the option and a missing file it fails. Option
names are distinct from the legacy --pilot-parquet and --labels-parquet so both conftests can
load in one session.
"""

from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--panel-parquet", default=None, help="Six-pilot weekly panel for acceptance tests.")
    parser.addoption("--channel-parquet", default=None, help="Channel product (F1) for acceptance tests.")
    parser.addoption("--legacy-labels-parquet", default=None, help="Legacy labels Parquet, read only by tests.")
    parser.addoption("--search-dir", default=None, help="Folder with the development search outputs (F5b).")
    parser.addoption("--prueba-dir", default=None, help="Folder with the outputs of the single test run (F7).")
    parser.addoption("--sin-clima-dir", default=None, help="Folder with the sin_clima search outputs (F6b, D7.2).")


def _explicit_path(request, option, purpose):
    value = request.config.getoption(option)
    if value is None:
        pytest.skip(f"{purpose} not requested ({option}).")
    path = Path(value).resolve()
    if not path.is_file():
        pytest.fail(f"Explicit {option} file is unavailable.")
    return path


@pytest.fixture
def panel_path(request):
    return _explicit_path(request, "--panel-parquet", "Six-pilot panel acceptance")


@pytest.fixture
def channel_path(request):
    return _explicit_path(request, "--channel-parquet", "Channel product acceptance")


@pytest.fixture
def legacy_labels_path(request):
    return _explicit_path(request, "--legacy-labels-parquet", "Legacy labels comparison")


@pytest.fixture
def search_dir(request):
    value = request.config.getoption("--search-dir")
    if value is None:
        pytest.skip("Development search acceptance not requested (--search-dir).")
    path = Path(value).resolve()
    required = ("tabla_unica_desarrollo.parquet", "predicciones_desarrollo.parquet")
    if not path.is_dir() or not all((path / name).is_file() for name in required):
        pytest.fail("Explicit --search-dir folder or its search outputs are unavailable.")
    return path


@pytest.fixture
def prueba_dir(request):
    value = request.config.getoption("--prueba-dir")
    if value is None:
        pytest.skip("Single test run acceptance not requested (--prueba-dir).")
    path = Path(value).resolve()
    if not path.is_dir() or not (path / "predicciones_prueba.parquet").is_file():
        pytest.fail("Explicit --prueba-dir folder or its test predictions are unavailable.")
    return path


@pytest.fixture
def sin_clima_dir(request):
    value = request.config.getoption("--sin-clima-dir")
    if value is None:
        pytest.skip("sin_clima search acceptance not requested (--sin-clima-dir).")
    path = Path(value).resolve()
    required = ("tabla_unica_desarrollo.parquet", "predicciones_desarrollo.parquet", "busqueda_manifest.json")
    if not path.is_dir() or not all((path / name).is_file() for name in required):
        pytest.fail("Explicit --sin-clima-dir folder or its search outputs are unavailable.")
    return path
