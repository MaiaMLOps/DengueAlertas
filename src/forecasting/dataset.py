"""Supervised examples (t, t + h) and features (contrato §3, §5, §10.5 and §10.7; especificación §4).

U2.2 adds only the channel loader. Before any feature is built, the channel product must be the
version fixed in configs/forecasting.yaml (`datos.version_dvc_canal`, the md5 of its .dvc file):
the loader hashes the bytes on disk, never trusting the md5 declared by the .dvc, and parses the
Parquet from those same bytes.
"""

import hashlib
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq

from src.epidemiology.channel_epico import CHANNEL_SCHEMA
from src.epidemiology.validation import SOURCE_SCHEMA

CHANNEL_PRODUCT_COLUMNS = SOURCE_SCHEMA.names + CHANNEL_SCHEMA.names
_BLOCK_BYTES = 1024 * 1024
_MD5_PATTERN = re.compile(r"[0-9a-f]{32}")


class ChannelVersionError(ValueError):
    """The channel product on disk is not the version fixed in the configuration."""


def load_channel_product(path: str | Path, *, expected_md5: str) -> pa.Table:
    """Read the channel product after checking its md5 against `expected_md5`.

    The file is read in 1 MiB blocks; each block updates the md5 and is kept in memory, and the
    Parquet is parsed from that buffer, so the verified bytes are exactly the bytes used. Fails with
    ChannelVersionError on a different md5 and with ValueError on unexpected columns.
    """
    if not isinstance(expected_md5, str) or not _MD5_PATTERN.fullmatch(expected_md5):
        raise ValueError(f"datos.version_dvc_canal is not set or is not a lowercase md5 hex digest: {expected_md5!r}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Channel product not found: {path}")

    digest, buffer = hashlib.md5(), bytearray()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_BLOCK_BYTES), b""):
            digest.update(block)
            buffer.extend(block)
    observed = digest.hexdigest()
    if observed != expected_md5:
        raise ChannelVersionError(
            f"Channel product {path} has md5 {observed}, but datos.version_dvc_canal is {expected_md5}. "
            "The file on disk is not the configured version: restore it with DVC or review the configuration."
        )

    table = pq.read_table(pa.BufferReader(bytes(buffer)))
    if table.column_names != CHANNEL_PRODUCT_COLUMNS:
        raise ValueError(f"Channel product {path} has columns {table.column_names}; "
                         f"expected {CHANNEL_PRODUCT_COLUMNS}.")
    return table
