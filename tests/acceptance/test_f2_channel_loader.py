"""F2 acceptance: the channel loader checks the product against datos.version_dvc_canal
(especificación §4, «Verificación del producto (F2)»).

The loader computes the md5 of the Parquet on disk; the md5 declared in the .dvc file is not enough.
A copy with one altered byte must be rejected with a message that names both md5 values.
"""

import hashlib

import pytest

pytestmark = pytest.mark.acceptance

PRODUCT_ROWS = 5952


def _md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _expected_md5() -> str:
    from src.forecasting.config import load_config

    return load_config().datos.version_dvc_canal


def test_channel_file_matches_configured_version(channel_path):
    """Data check without the implementation: the file passed is the versioned product."""
    assert _md5(channel_path) == _expected_md5()


def test_loader_accepts_the_versioned_product(channel_path):
    from src.forecasting.dataset import load_channel_product

    product = load_channel_product(channel_path, expected_md5=_expected_md5())

    assert product.num_rows == PRODUCT_ROWS


def test_loader_rejects_an_altered_copy(channel_path, tmp_path):
    from src.forecasting.dataset import load_channel_product

    content = bytearray(channel_path.read_bytes())
    middle = len(content) // 2
    content[middle] ^= 0xFF
    altered = tmp_path / "altered_channel.parquet"
    altered.write_bytes(bytes(content))
    expected, observed = _expected_md5(), _md5(altered)
    assert observed != expected

    with pytest.raises(ValueError) as error:
        load_channel_product(altered, expected_md5=expected)
    assert expected in str(error.value) and observed in str(error.value)
