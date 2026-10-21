"""Synthetic defaults; actual local Parquet regressions require an explicit option."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pilot_selection.config import COUNT, PILOTS, ROOT


def pytest_addoption(parser):
    parser.addoption("--master-parquet", default=None, help="Opt in to regressions against the current local master.")


@pytest.fixture(scope="session")
def master_path(request):
    value = request.config.getoption("--master-parquet")
    if value is None:
        pytest.skip("Local Parquet regression not requested.")
    path = (ROOT / Path(value)).resolve()
    if not path.is_file():
        pytest.fail("Explicitly requested local master is unavailable.")
    return path


def make_panel(codes, dates, counts):
    dates = pd.DatetimeIndex(dates)
    nominal = dates + pd.Timedelta(days=6)
    effective = nominal.where(nominal <= pd.Timestamp("2025-12-31"), pd.Timestamp("2025-12-31"))
    days = (effective - dates).days + 1
    frames = []
    for index, code in enumerate(codes):
        values = np.asarray(counts[index], dtype="int64")
        frames.append(pd.DataFrame({
            "municipality_code": code, "municipality": "Synthetic " + code,
            "week_start": dates, "week_end_nominal": nominal, "week_end_effective": effective,
            "n_days": days.astype("uint8"), "partial_week": days < 7,
            COUNT: values, "no_eligible_records_observed": values == 0,
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def short_panel():
    return make_panel(["08001"], pd.date_range("2022-12-12", periods=6, freq="W-MON"),
                      [[2, 0, 0, 0, 3, 0]])


@pytest.fixture(scope="session")
def synthetic_master():
    dates = pd.date_range("2007-01-01", "2025-12-31", freq="W-MON")
    codes = [pilot.code for pilot in PILOTS] + [f"01{i:03d}" for i in range(1, 13)]
    counts = [np.where(np.arange(len(dates)) % (i + 2) == 0, 0, i + 1) for i in range(len(codes))]
    return make_panel(codes, dates, counts)
