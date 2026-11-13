"""F2 acceptance: GRU input (contrato §5.3 and §10.7; especificación §6.F2), with h = 8.

Sequence of the weeks s = t-11 .. t, chronological (index 0 = t-11, index 11 = t), with eight
channels: log(C_s + 1), pos_s, zona_s one-hot (seguridad, alerta, epidemia) and the anomalies
a_s(temperature), a_s(precipitation), a_s(humidity). Static inputs: cambio_umbral, sem_sin, sem_cos.

Coherence with the tabular features is EXACT: both come from the same weekly values, and every
aggregation is a left-to-right Python sum from the oldest week to the most recent one (convention
documented in src/forecasting/dataset.py), so the same floating-point operations give the same bits.

The truncation sample and the origins are those of the frozen tests/acceptance/test_f2_dataset.py,
imported by path.
"""

from datetime import timedelta
import functools
import importlib.util
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

pytestmark = pytest.mark.acceptance

F2_FILE = Path(__file__).resolve().parent / "test_f2_dataset.py"
H = 8
ROWS = 4266
STEPS = 12
SEQUENCE_CHANNELS = ("log_c", "pos", "zona_seguridad", "zona_alerta", "zona_epidemia",
                     "anom_temp", "anom_prec", "anom_hum")
STATIC = ("cambio_umbral", "sem_sin", "sem_cos")
ZONES = ("seguridad", "alerta", "epidemia")
BLOCK_STEPS = {"m1": range(8, 12), "m2": range(4, 8), "m3": range(0, 4)}
CLIMATE = {"temp": ("anom_temp", "mean"), "prec": ("anom_prec", "sum"), "hum": ("anom_hum", "mean")}
KEY_COLUMNS = ["municipality_code", "t", "target_week", "y"]


def _load_f2():
    spec = importlib.util.spec_from_file_location("frozen_f2_dataset_for_gru", F2_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


f2 = _load_f2()


def _config():
    from src.forecasting.config import load_config

    return load_config()


@functools.lru_cache(maxsize=None)
def _examples_and_gru(channel_file: str):
    from src.forecasting.dataset import build_examples, build_gru_examples, load_channel_product

    config = _config()
    product = load_channel_product(channel_file, expected_md5=config.datos.version_dvc_canal)
    kwargs = {"h": H, "canal": config.canal, "features": config.features}
    return build_examples(product, **kwargs), build_gru_examples(product, **kwargs)


def _left_to_right(values) -> float:
    total = 0.0
    for value in values:
        total = total + float(value)
    return total


# --------------------------------------------------------------------------- shape and rows
def test_channel_and_static_names():
    from src.forecasting.dataset import GRU_SEQUENCE_CHANNELS, GRU_STATIC

    assert tuple(GRU_SEQUENCE_CHANNELS) == SEQUENCE_CHANNELS
    assert tuple(GRU_STATIC) == STATIC


def test_shapes_types_and_finiteness(channel_path):
    _, gru = _examples_and_gru(str(channel_path))

    assert gru.sequence.shape == (ROWS, STEPS, len(SEQUENCE_CHANNELS))
    assert gru.static.shape == (ROWS, len(STATIC))
    assert gru.sequence.dtype == np.float64 and gru.static.dtype == np.float64
    assert np.isfinite(gru.sequence).all() and np.isfinite(gru.static).all()


def test_zone_one_hot(channel_path):
    _, gru = _examples_and_gru(str(channel_path))
    one_hot = gru.sequence[:, :, 2:5]

    assert set(np.unique(one_hot)) <= {0.0, 1.0}
    assert (one_hot.sum(axis=2) == 1.0).all()


def test_rows_identical_to_tabular_examples(channel_path):
    examples, gru = _examples_and_gru(str(channel_path))

    assert list(gru.keys.columns) == KEY_COLUMNS
    assert len(gru.keys) == ROWS
    for name in KEY_COLUMNS:
        left = [f2._as_date(v) if name in ("t", "target_week") else v for v in gru.keys[name]]
        right = [f2._as_date(v) if name in ("t", "target_week") else v for v in examples[name]]
        assert left == right, name


# --------------------------------------------------------------------------- truncation
def test_identical_with_full_and_truncated_base(panel_path):
    from src.forecasting.dataset import build_gru_input

    config = _config()
    kwargs = {"h": H, "canal": config.canal, "features": config.features}
    panel = pq.read_table(panel_path)
    origins = [(code, t) for code in f2.CODES for t in f2.sample_origins()]
    full = build_gru_input(panel, origins, **kwargs)
    full_index = {(code, f2._as_date(t)): i for i, (code, t) in
                  enumerate(zip(full.keys["municipality_code"], full.keys["t"]))}

    mismatches = []
    for code, t in origins:
        cut = f2._single_municipality_cut(panel, code, t)
        truncated = build_gru_input(cut, [(code, t)], **kwargs)
        i = full_index[(code, t)]
        if not (np.array_equal(full.sequence[i], truncated.sequence[0])
                and np.array_equal(full.static[i], truncated.static[0])):
            mismatches.append((code, t))
    assert len(origins) == 6 * len(f2.sample_origins())
    assert mismatches == []


# --------------------------------------------------------------------------- coherence (exact)
def test_coherent_with_tabular_features(channel_path):
    examples, gru = _examples_and_gru(str(channel_path))
    channel = {name: i for i, name in enumerate(SEQUENCE_CHANNELS)}

    for i, row in enumerate(examples.to_dict("records")):
        sequence, static = gru.sequence[i], gru.static[i]
        label = (row["municipality_code"], row["t"])
        assert _left_to_right(sequence[8:12, channel["pos"]]) / 4 == row["posicion_4s"], label
        assert sequence[11, channel["log_c"]] - sequence[7, channel["log_c"]] == row["crecimiento_4s"], label
        assert ZONES[int(np.argmax(sequence[11, 2:5]))] == row["zona_t"], label
        assert tuple(static) == (row["cambio_umbral"], row["sem_sin"], row["sem_cos"]), label
        for short, (name, how) in CLIMATE.items():
            for block, steps in BLOCK_STEPS.items():
                total = _left_to_right(sequence[list(steps), channel[name]])
                value = total if how == "sum" else total / 4
                assert value == row[f"anom_{short}_{block}"], (label, short, block)


def test_target_week_and_horizon(channel_path):
    _, gru = _examples_and_gru(str(channel_path))

    for t, target in zip(gru.keys["t"], gru.keys["target_week"]):
        assert f2._as_date(target) == f2._as_date(t) + timedelta(weeks=H)
