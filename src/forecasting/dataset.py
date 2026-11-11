"""Supervised examples (t, t + h) and features (contrato §3, §5, §10.5 and §10.7; especificación §4).

Flow:
    load_channel_product (md5 against datos.version_dvc_canal)
      -> build_examples: rows (municipality, t) and target y from the product
           -> build_features: features from the fifteen SOURCE columns only

Rows (contrato §3.7 and §5.4): t is an origin when the `features.ventana_semanas` weeks t-11..t
and the target week t + h are channel-eligible in the product. The rule depends on the configured
window, not on which features are implemented, so rows are the same for every family.

Features never read the channel product: `build_features` accepts only the fifteen source columns
(validate_source rejects anything else) and recomputes every channel quantity with
`channel_for_week` from the counts of complete weeks in the base it receives. With a base cut at t
there is no row after t, so U1e_{t+h} can only come from weeks <= t (prueba de truncamiento).
The target y = target_u1e of the product at t + h is not a feature and may read that row.
"""

from collections.abc import Sequence
from datetime import date, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.epidemiology.channel_epico import CHANNEL_SCHEMA, channel_for_week, channel_zone
from src.epidemiology.config import COUNT
from src.epidemiology.ewt import week_ordinal
from src.epidemiology.validation import SOURCE_SCHEMA, require, validate_source

from .config import ChannelSection, FeatureSection

CHANNEL_PRODUCT_COLUMNS = SOURCE_SCHEMA.names + CHANNEL_SCHEMA.names
_BLOCK_BYTES = 1024 * 1024
_MD5_PATTERN = re.compile(r"[0-9a-f]{32}")

FEATURES = (  # contrato §5.2, in table order
    "zona_t", "posicion_4s", "crecimiento_4s", "cambio_umbral", "sem_sin", "sem_cos",
    "anom_temp_m1", "anom_temp_m2", "anom_temp_m3",
    "anom_prec_m1", "anom_prec_m2", "anom_prec_m3",
    "anom_hum_m1", "anom_hum_m2", "anom_hum_m3",
)
# Period of sem_sin and sem_cos, part of their definition in contrato §5.2 (2π·(w_{t+h} − 1)/52.18);
# a constant of the formula, not a tunable parameter. Used literally: 52.18, not 365.25 / 7.
ANNUAL_PERIOD_WEEKS = 52.18
KEY_COLUMNS = ("municipality_code", "t")
EXAMPLE_KEY_COLUMNS = ("municipality_code", "t", "target_week", "y")
_CHANNEL_METADATA = b"denguealertas.channel"


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


# --------------------------------------------------------------------------- features
def _check_horizon(h: int) -> None:
    require(isinstance(h, int) and not isinstance(h, bool) and h >= 1, f"h must be an integer >= 1; got {h!r}.")


def _check_origins(origins) -> list[tuple[str, date]]:
    checked = []
    for origin in origins:
        require(isinstance(origin, tuple) and len(origin) == 2, f"Origins must be (municipality_code, t) pairs; got {origin!r}.")
        code, t = origin
        require(isinstance(code, str), f"municipality_code must be text; got {code!r}.")
        require(isinstance(t, date) and not isinstance(t, datetime) and t.weekday() == 0,
                f"Origin t must be a Monday datetime.date; got {t!r} for {code}.")
        checked.append((code, t))
    require(len(set(checked)) == len(checked), "Duplicate origins.")
    return checked


def _complete_counts(source: pa.Table) -> dict[str, dict[date, int]]:
    """{municipality: {Monday: count}} over complete weeks; partial weeks are left out (D1)."""
    counts: dict[str, dict[date, int]] = {}
    for row in source.select(["municipality_code", "week_start", "partial_week", COUNT]).to_pylist():
        if not row["partial_week"]:
            counts.setdefault(row["municipality_code"], {})[row["week_start"]] = row[COUNT]
    return counts


def _log1(value: float) -> float:
    return math.log1p(value)


def build_features(source: pa.Table, origins: Sequence[tuple[str, date]], *, h: int,
                   canal: ChannelSection, features: FeatureSection) -> pd.DataFrame:
    """Features of each origin (municipality, t) computed only from the base `source`.

    `source` must be the fifteen panel columns (validate_source). For each municipality the channel
    is computed once per needed week: t-(w-1)..t and t+h for every origin, without repetitions.
    Any missing quantity raises an error naming municipality, week and quantity; no NaN is produced.
    Returns municipality_code, t and the implemented features in FEATURES order.
    """
    _check_horizon(h)
    validate_source(source)
    origins = _check_origins(origins)
    w = features.w
    counts = _complete_counts(source)
    channel_kwargs = {"n_years": canal.n, "shift": canal.desplazamiento, "alpha": canal.cuantil_t}

    channel: dict[tuple[str, date], tuple[float, float]] = {}
    for code in sorted({code for code, _ in origins}):
        require(code in counts, f"Municipality {code} is not in the base.")
        needed = set()
        for origin_code, t in origins:
            if origin_code == code:
                needed.update(t - timedelta(weeks=k) for k in range(w))
                needed.add(t + timedelta(weeks=h))
        for week in sorted(needed):
            result = channel_for_week(counts[code], week, **channel_kwargs)
            require(result is not None,
                    f"Channel (central, U1e) unavailable for {code} week {week}: fewer than {canal.n} "
                    "prior-year references of the same ordinal in the base.")
            channel[(code, week)] = result

    def count(code: str, week: date, t: date) -> int:
        value = counts[code].get(week)
        require(value is not None, f"Count C unavailable for {code} week {week} (origin t = {t}); "
                                   "it is not a complete week of the base.")
        return value

    rows = []
    for code, t in origins:
        target = t + timedelta(weeks=h)
        central_t, upper_t = channel[(code, t)]
        recent = [t - timedelta(weeks=k) for k in range(w)]
        angle = 2 * math.pi * (week_ordinal(target) - 1) / ANNUAL_PERIOD_WEEKS
        rows.append({
            "municipality_code": code,
            "t": t,
            "zona_t": channel_zone(count(code, t, t), central_t, upper_t),
            "posicion_4s": sum(_log1(count(code, s, t)) - _log1(channel[(code, s)][1]) for s in recent) / w,
            "crecimiento_4s": _log1(count(code, t, t)) - _log1(count(code, t - timedelta(weeks=w), t)),
            "cambio_umbral": _log1(channel[(code, target)][1]) - _log1(upper_t),
            "sem_sin": math.sin(angle),
            "sem_cos": math.cos(angle),
        })
    implemented = [name for name in FEATURES if name in rows[0]] if rows else []
    return pd.DataFrame(rows, columns=[*KEY_COLUMNS, *implemented])


def _check_channel_parameters(product: pa.Table, canal: ChannelSection) -> None:
    """The product's channel must be the one the configuration recomputes for the features."""
    metadata = product.schema.metadata or {}
    require(_CHANNEL_METADATA in metadata,
            "Channel product lacks its channel parameters (denguealertas.channel metadata).")
    stored = json.loads(metadata[_CHANNEL_METADATA])
    configured = {"n_years": canal.n, "shift": canal.desplazamiento, "alpha": canal.cuantil_t}
    differences = {name: (stored.get(name), value) for name, value in configured.items() if stored.get(name) != value}
    require(not differences,
            f"Channel product parameters differ from the canal section of the configuration "
            f"(product, configuration): {differences}. Rebuild the product or review the configuration.")


def build_examples(product: pa.Table, *, h: int, canal: ChannelSection, features: FeatureSection) -> pd.DataFrame:
    """Rows (municipality, t) with target week t + h, target y and features.

    t is an origin when the `features.ventana_semanas` weeks ending at t and the week t + h are
    channel-eligible in the product (contrato §3.7, §5.4). y = target_u1e of the product at t + h.
    """
    _check_horizon(h)
    require(product.column_names == CHANNEL_PRODUCT_COLUMNS, "Unexpected channel product columns.")
    _check_channel_parameters(product, canal)
    window = features.ventana_semanas

    status = {(row["municipality_code"], row["week_start"]): (row["channel_eligible"], row["target_u1e"])
              for row in product.select(["municipality_code", "week_start", "channel_eligible", "target_u1e"]).to_pylist()}

    def eligible(code: str, week: date) -> bool:
        return status.get((code, week), (False, None))[0]

    origins, targets = [], {}
    for code, t in sorted(status):
        target = t + timedelta(weeks=h)
        if eligible(code, target) and all(eligible(code, t - timedelta(weeks=k)) for k in range(window)):
            origins.append((code, t))
            targets[(code, t)] = (target, status[(code, target)][1])

    frame = build_features(product.select(SOURCE_SCHEMA.names), origins, h=h, canal=canal, features=features)
    feature_columns = [name for name in frame.columns if name not in KEY_COLUMNS]
    frame.insert(2, "target_week", [targets[(code, t)][0] for code, t in zip(frame["municipality_code"], frame["t"])])
    frame.insert(3, "y", pd.array([targets[(code, t)][1] for code, t in zip(frame["municipality_code"], frame["t"])],
                                  dtype="int8"))
    return frame[[*EXAMPLE_KEY_COLUMNS, *feature_columns]].reset_index(drop=True)
