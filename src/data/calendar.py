"""Monday-Sunday calendar with explicit effective coverage at window limits."""

from collections.abc import Iterable

import pandas as pd

from .config import END, START


def build_calendar(codes: Iterable[str], start: str = START, end: str = END) -> pd.DataFrame:
    lower, upper = pd.Timestamp(start), pd.Timestamp(end)
    if lower > upper:
        raise ValueError("Calendar start must not exceed end.")
    first = lower - pd.Timedelta(days=lower.weekday())
    weeks = pd.date_range(first, upper, freq="7D")
    frame = pd.MultiIndex.from_product(
        [sorted(set(codes)), weeks], names=["municipality_code", "week_start"]
    ).to_frame(index=False)
    frame["municipality_code"] = frame["municipality_code"].astype("string")
    frame["week_end_nominal"] = frame["week_start"] + pd.Timedelta(days=6)
    frame["week_end_effective"] = frame["week_end_nominal"].clip(upper=upper)
    effective_start = frame["week_start"].clip(lower=lower)
    frame["n_days"] = (frame["week_end_effective"] - effective_start).dt.days.add(1).astype("uint8")
    frame["partial_week"] = frame["n_days"].lt(7)
    return frame
