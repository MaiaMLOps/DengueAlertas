"""Supervised examples (t, t + h), tabular features and GRU input (contrato §3, §5, §10.5 and §10.7;
especificación §4).

Flow:
    load_channel_product (md5 against datos.version_dvc_canal)
      -> build_examples / build_gru_examples: rows (municipality, t) and target y from the product
           -> build_features / build_gru_input: inputs from the fifteen SOURCE columns only
                -> _weekly_quantities: per-week counts, channel and anomalies, computed once

Rows (contrato §3.7 and §5.4): t is an origin when the `features.ventana_semanas` weeks t-11..t
and the target week t + h are channel-eligible in the product. The rule depends on the configured
window, not on which inputs are built, so rows are the same for every family.

Inputs never read the channel product: the builders accept only the fifteen source columns
(validate_source rejects anything else) and recompute every channel quantity with
`channel_for_week` from the counts of complete weeks in the base they receive. With a base cut at t
there is no row after t, so U1e_{t+h} can only come from weeks <= t (prueba de truncamiento).
The target y = target_u1e of the product at t + h is not an input and may read that row.

Aggregation convention: every sum or mean over weeks is a left-to-right Python sum from the oldest
week to the most recent one (then divided by the number of weeks for a mean). Tabular features and
the GRU sequence share the same weekly values, so aggregating the sequence this way reproduces the
tabular features bit for bit.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.epidemiology.channel_epico import CHANNEL_SCHEMA, ZONES, channel_for_week, channel_zone, reference_weeks
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
# Climate block of contrato §5.2: anomaly a_s(X) = X_s - clim_s(X), aggregated per monthly block by
# sum (precipitation) or mean (temperature, humidity). Definition of the features, not a parameter.
CLIMATE_VARIABLES = {
    "temp": ("temperature_c", "mean"),
    "prec": ("precipitation_mm", "sum"),
    "hum": ("relative_humidity_pct", "mean"),
}
CLIMATE_COLUMNS = tuple(column for column, _ in CLIMATE_VARIABLES.values())
# GRU input (contrato §5.3): per week, log(C_s + 1), pos_s, zona_s one-hot and the three anomalies.
GRU_SEQUENCE_CHANNELS = ("log_c", "pos", "zona_seguridad", "zona_alerta", "zona_epidemia",
                         "anom_temp", "anom_prec", "anom_hum")
GRU_STATIC = ("cambio_umbral", "sem_sin", "sem_cos")
KEY_COLUMNS = ("municipality_code", "t")
EXAMPLE_KEY_COLUMNS = ("municipality_code", "t", "target_week", "y")
_CHANNEL_METADATA = b"denguealertas.channel"


class ChannelVersionError(ValueError):
    """The channel product on disk is not the version fixed in the configuration."""


@dataclass(frozen=True)
class GruInput:
    """GRU input: `keys` (municipality_code, t, and target_week, y for examples), `sequence` of shape
    (n, ventana_semanas, 8) in chronological order (index 0 = t-11, last = t) and `static` (n, 3)."""
    keys: pd.DataFrame
    sequence: np.ndarray
    static: np.ndarray


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


# --------------------------------------------------------------------------- validation helpers
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


# --------------------------------------------------------------------------- weekly quantities
def _complete_weeks(source: pa.Table) -> tuple[dict[str, dict[date, int]], dict[str, dict[date, dict[str, float]]]]:
    """Counts and climate of complete weeks per municipality; partial weeks are left out (D1), so
    they can never be a reference of the channel or of clim_s."""
    counts: dict[str, dict[date, int]] = {}
    climate: dict[str, dict[date, dict[str, float]]] = {}
    for row in source.select(["municipality_code", "week_start", "partial_week", COUNT,
                              *CLIMATE_COLUMNS]).to_pylist():
        if not row["partial_week"]:
            code, week = row["municipality_code"], row["week_start"]
            counts.setdefault(code, {})[week] = row[COUNT]
            climate.setdefault(code, {})[week] = {column: row[column] for column in CLIMATE_COLUMNS}
    return counts, climate


def _climate_blocks(features: FeatureSection) -> list[tuple[str, str, str, list[int]]]:
    """(feature name, column, aggregation, week offsets relative to t, chronological) for variables
    7-15. Blocks come from `features.bloques` (both ends included). The generated names must be
    exactly the climate block of FEATURES, and no block may start before the row window."""
    earliest = -(features.ventana_semanas - 1)
    blocks = []
    for short, (column, how) in CLIMATE_VARIABLES.items():
        for block, (start, end) in features.bloques.items():
            require(start >= earliest, f"Block {block} = [{start}, {end}] starts before the "
                                       f"{features.ventana_semanas}-week row window.")
            blocks.append((f"anom_{short}_{block}", column, how, list(range(start, end + 1))))
    names = {name for name, *_ in blocks}
    require(names == set(FEATURES[6:]) and len(blocks) == len(names),
            f"features.bloques generates {sorted(names)}; expected the climate block {list(FEATURES[6:])}.")
    order = {name: i for i, name in enumerate(FEATURES)}
    return sorted(blocks, key=lambda block: order[block[0]])


@dataclass(frozen=True)
class _Weekly:
    counts: dict[str, dict[date, int]]
    channel: dict[tuple[str, date], tuple[float, float]]
    anomaly: dict[tuple[str, date], dict[str, float]]

    def count(self, code: str, week: date, t: date) -> int:
        value = self.counts[code].get(week)
        require(value is not None, f"Count C unavailable for {code} week {week} (origin t = {t}); "
                                   "it is not a complete week of the base.")
        return value


def _weekly_quantities(source: pa.Table, origins: list[tuple[str, date]], *, h: int, canal: ChannelSection,
                       channel_offsets: Sequence[int], climate_offsets: Sequence[int]) -> _Weekly:
    """Per-week quantities, each computed once per municipality and week without repetitions:
    channel (central, U1e) for t + channel_offsets and t + h, and anomalies a_s for t + climate_offsets.
    Any missing quantity raises an error naming municipality, week and quantity."""
    counts, climate = _complete_weeks(source)
    channel_kwargs = {"n_years": canal.n, "shift": canal.desplazamiento, "alpha": canal.cuantil_t}
    channel: dict[tuple[str, date], tuple[float, float]] = {}
    anomaly: dict[tuple[str, date], dict[str, float]] = {}
    for code in sorted({code for code, _ in origins}):
        require(code in counts, f"Municipality {code} is not in the base.")
        channel_weeks, climate_weeks = set(), set()
        for origin_code, t in origins:
            if origin_code == code:
                channel_weeks.update(t + timedelta(weeks=offset) for offset in channel_offsets)
                channel_weeks.add(t + timedelta(weeks=h))
                climate_weeks.update(t + timedelta(weeks=offset) for offset in climate_offsets)
        for week in sorted(channel_weeks):
            result = channel_for_week(counts[code], week, **channel_kwargs)
            require(result is not None,
                    f"Channel (central, U1e) unavailable for {code} week {week}: fewer than {canal.n} "
                    "prior-year references of the same ordinal in the base.")
            channel[(code, week)] = result
        for week in sorted(climate_weeks):
            require(week in climate[code], f"Climate unavailable for {code} week {week}; "
                                           "it is not a complete week of the base.")
            references = reference_weeks(climate[code], week, n_years=canal.n)
            require(references is not None,
                    f"Climate references (clim_s) unavailable for {code} week {week}: fewer than {canal.n} "
                    "prior-year complete weeks of the same ordinal in the base.")
            anomaly[(code, week)] = {
                column: climate[code][week][column]
                - sum(climate[code][reference][column] for reference in references) / len(references)
                for column in CLIMATE_COLUMNS}
    return _Weekly(counts, channel, anomaly)


def _position(count: int, upper: float) -> float:
    """pos_s = log(C_s + 1) - log(U1e_s + 1) (contrato §5.1)."""
    return math.log1p(count) - math.log1p(upper)


def _static(weekly: _Weekly, code: str, t: date, h: int) -> tuple[float, float, float]:
    """(cambio_umbral, sem_sin, sem_cos): the target-week inputs, shared by both families."""
    target = t + timedelta(weeks=h)
    angle = 2 * math.pi * (week_ordinal(target) - 1) / ANNUAL_PERIOD_WEEKS
    cambio = math.log1p(weekly.channel[(code, target)][1]) - math.log1p(weekly.channel[(code, t)][1])
    return cambio, math.sin(angle), math.cos(angle)


def _chronological(t: date, offsets: Sequence[int]) -> list[date]:
    return [t + timedelta(weeks=offset) for offset in sorted(offsets)]


# --------------------------------------------------------------------------- tabular features
def build_features(source: pa.Table, origins: Sequence[tuple[str, date]], *, h: int,
                   canal: ChannelSection, features: FeatureSection) -> pd.DataFrame:
    """The fifteen features of each origin (municipality, t) computed only from the base `source`.

    `source` must be the fifteen panel columns (validate_source). Channel and clim_s use the same
    reference weeks (`reference_weeks`). Weekly quantities are computed once and aggregated per
    origin in chronological order (module convention). Any missing quantity raises an error; no NaN
    is produced. Returns municipality_code, t and FEATURES.
    """
    _check_horizon(h)
    validate_source(source)
    origins = _check_origins(origins)
    w = features.w
    blocks = _climate_blocks(features)
    recent_offsets = list(range(-(w - 1), 1))
    weekly = _weekly_quantities(source, origins, h=h, canal=canal, channel_offsets=recent_offsets,
                                climate_offsets=sorted({o for *_, offsets in blocks for o in offsets}))

    rows = []
    for code, t in origins:
        central_t, upper_t = weekly.channel[(code, t)]
        count_t = weekly.count(code, t, t)
        cambio, sem_sin, sem_cos = _static(weekly, code, t, h)
        positions = [_position(weekly.count(code, s, t), weekly.channel[(code, s)][1])
                     for s in _chronological(t, recent_offsets)]
        row = {
            "municipality_code": code,
            "t": t,
            "zona_t": channel_zone(count_t, central_t, upper_t),
            "posicion_4s": sum(positions) / w,
            "crecimiento_4s": math.log1p(count_t) - math.log1p(weekly.count(code, t - timedelta(weeks=w), t)),
            "cambio_umbral": cambio,
            "sem_sin": sem_sin,
            "sem_cos": sem_cos,
        }
        for name, column, how, offsets in blocks:
            values = [weekly.anomaly[(code, s)][column] for s in _chronological(t, offsets)]
            row[name] = sum(values) if how == "sum" else sum(values) / len(values)
        rows.append(row)
    return pd.DataFrame(rows, columns=[*KEY_COLUMNS, *FEATURES])


# --------------------------------------------------------------------------- GRU input
def build_gru_input(source: pa.Table, origins: Sequence[tuple[str, date]], *, h: int,
                    canal: ChannelSection, features: FeatureSection) -> GruInput:
    """GRU input of each origin (contrato §5.3) computed only from the base `source`.

    Sequence of the `features.ventana_semanas` weeks t-11..t in chronological order with the channels
    GRU_SEQUENCE_CHANNELS; static inputs GRU_STATIC. Same weekly values and helpers as
    `build_features`. No scaling here: it is fitted per fold in the modelling pipeline (contrato §5.5).
    """
    _check_horizon(h)
    validate_source(source)
    origins = _check_origins(origins)
    window_offsets = list(range(-(features.ventana_semanas - 1), 1))
    weekly = _weekly_quantities(source, origins, h=h, canal=canal, channel_offsets=window_offsets,
                                climate_offsets=window_offsets)

    sequence = np.empty((len(origins), len(window_offsets), len(GRU_SEQUENCE_CHANNELS)), dtype=np.float64)
    static = np.empty((len(origins), len(GRU_STATIC)), dtype=np.float64)
    for i, (code, t) in enumerate(origins):
        for j, s in enumerate(_chronological(t, window_offsets)):
            count = weekly.count(code, s, t)
            central, upper = weekly.channel[(code, s)]
            zone = channel_zone(count, central, upper)
            sequence[i, j] = [math.log1p(count), _position(count, upper),
                              *(1.0 if zone == name else 0.0 for name in ZONES),
                              *(weekly.anomaly[(code, s)][column] for column in CLIMATE_COLUMNS)]
        static[i] = _static(weekly, code, t, h)
    keys = pd.DataFrame(origins, columns=list(KEY_COLUMNS))
    return GruInput(keys=keys, sequence=sequence, static=static)


# --------------------------------------------------------------------------- rows and targets
def _check_channel_parameters(product: pa.Table, canal: ChannelSection) -> None:
    """The product's channel must be the one the configuration recomputes for the inputs."""
    metadata = product.schema.metadata or {}
    require(_CHANNEL_METADATA in metadata,
            "Channel product lacks its channel parameters (denguealertas.channel metadata).")
    stored = json.loads(metadata[_CHANNEL_METADATA])
    configured = {"n_years": canal.n, "shift": canal.desplazamiento, "alpha": canal.cuantil_t}
    differences = {name: (stored.get(name), value) for name, value in configured.items() if stored.get(name) != value}
    require(not differences,
            f"Channel product parameters differ from the canal section of the configuration "
            f"(product, configuration): {differences}. Rebuild the product or review the configuration.")


def _origins_and_targets(product: pa.Table, *, h: int, canal: ChannelSection, features: FeatureSection):
    """Origins (sorted by municipality and t) and {origin: (target_week, y)} from the product.

    t is an origin when the `features.ventana_semanas` weeks ending at t and the week t + h are
    channel-eligible (contrato §3.7, §5.4); y = target_u1e of the product at t + h.
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
    return origins, targets


def _target_columns(frame: pd.DataFrame, targets: dict) -> pd.DataFrame:
    pairs = list(zip(frame["municipality_code"], frame["t"]))
    frame.insert(2, "target_week", [targets[pair][0] for pair in pairs])
    frame.insert(3, "y", pd.array([targets[pair][1] for pair in pairs], dtype="int8"))
    return frame


def build_examples(product: pa.Table, *, h: int, canal: ChannelSection, features: FeatureSection) -> pd.DataFrame:
    """Rows (municipality, t) with target week t + h, target y and the fifteen features."""
    origins, targets = _origins_and_targets(product, h=h, canal=canal, features=features)
    frame = build_features(product.select(SOURCE_SCHEMA.names), origins, h=h, canal=canal, features=features)
    feature_columns = [name for name in frame.columns if name not in KEY_COLUMNS]
    frame = _target_columns(frame, targets)
    return frame[[*EXAMPLE_KEY_COLUMNS, *feature_columns]].reset_index(drop=True)


def build_gru_examples(product: pa.Table, *, h: int, canal: ChannelSection, features: FeatureSection) -> GruInput:
    """Same rows as `build_examples`, with the GRU input and keys municipality_code, t, target_week, y."""
    origins, targets = _origins_and_targets(product, h=h, canal=canal, features=features)
    gru = build_gru_input(product.select(SOURCE_SCHEMA.names), origins, h=h, canal=canal, features=features)
    keys = _target_columns(gru.keys.copy(), targets)
    return GruInput(keys=keys[list(EXAMPLE_KEY_COLUMNS)].reset_index(drop=True), sequence=gru.sequence, static=gru.static)
