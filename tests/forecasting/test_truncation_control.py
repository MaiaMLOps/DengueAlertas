"""Negative control of the frozen truncation comparison (contrato §10.7).

Imports `truncation_mismatches` from tests/acceptance/test_f2_dataset.py by path, so what is checked is
the comparison that is frozen, not a copy. On a synthetic series, a correct builder must pass, and two
builders with deliberate leaks must be detected:
- clim_s computed with all years of the base (future years included);
- C_{t+1} used in place of C_t (pandas shift(-1)).
"""

from datetime import date, timedelta
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

ACCEPTANCE_FILE = Path(__file__).resolve().parents[1] / "acceptance" / "test_f2_dataset.py"
COUNT = "confirmed_records_screening"
CODE = "99999"
WEEKS = 8 * 52
ORIGIN_INDICES = (300, 330, 360)  # at least five prior years and future weeks after each origin


def _load_frozen_comparison():
    spec = importlib.util.spec_from_file_location("frozen_f2_dataset_acceptance", ACCEPTANCE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.truncation_mismatches


truncation_mismatches = _load_frozen_comparison()


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(20261111)
    weeks = [date(2007, 1, 1) + timedelta(weeks=i) for i in range(WEEKS)]
    table = pa.table({
        "municipality_code": pa.array([CODE] * WEEKS, pa.string()),
        "week_start": pa.array(weeks, pa.date32()),
        COUNT: pa.array(rng.poisson(20, WEEKS), pa.int64()),
        "temperature_c": pa.array(27 + rng.normal(0, 1, WEEKS), pa.float64()),
    })
    return table, [(CODE, weeks[i]) for i in ORIGIN_INDICES]


def _builder(*, leak_clim=False, leak_next=False, columns=("crecimiento_4s", "anom_temp_m1")):
    def build(base: pa.Table, origins):
        frame = base.to_pandas()
        frame["week_start"] = pd.to_datetime(frame["week_start"])
        frame = frame.sort_values("week_start").set_index("week_start")
        count = frame[COUNT].astype(float)
        if leak_next:
            count = count.shift(-1)
        log_count = np.log1p(count)
        temperature = frame["temperature_c"]
        if leak_clim:
            clim = temperature.groupby(np.arange(len(frame)) % 52).transform("mean")
        else:
            clim = sum(temperature.shift(52 * k) for k in range(1, 6)) / 5
        values = {"crecimiento_4s": log_count - log_count.shift(4),
                  "anom_temp_m1": (temperature - clim).rolling(4).mean(),
                  "secret_feature": temperature}
        rows = []
        for code, t in origins:
            stamp = pd.Timestamp(t)
            rows.append({"municipality_code": code, "t": t, **{name: values[name].loc[stamp] for name in columns}})
        return pd.DataFrame(rows, columns=["municipality_code", "t", *columns])
    return build


def test_correct_builder_passes(synthetic):
    table, origins = synthetic

    assert truncation_mismatches(_builder(), table, origins) == []


def test_climatology_with_all_years_is_detected(synthetic):
    table, origins = synthetic

    mismatches = truncation_mismatches(_builder(leak_clim=True), table, origins)

    assert len(mismatches) == len(origins)
    assert all("anom_temp_m1" in line for line in mismatches)


def test_next_week_count_is_detected(synthetic):
    table, origins = synthetic

    mismatches = truncation_mismatches(_builder(leak_next=True), table, origins)

    assert len(mismatches) == len(origins)
    assert all("crecimiento_4s" in line for line in mismatches)


def test_builder_without_feature_columns_is_rejected(synthetic):
    table, origins = synthetic

    with pytest.raises(AssertionError, match="no feature columns"):
        truncation_mismatches(_builder(columns=()), table, origins)


def test_columns_outside_the_contract_are_rejected(synthetic):
    table, origins = synthetic

    with pytest.raises(AssertionError, match="outside the contract"):
        truncation_mismatches(_builder(columns=("crecimiento_4s", "secret_feature")), table, origins)
