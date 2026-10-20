"""Explicit territorial/date conversions without inferring epidemiological rules."""

import pandas as pd

from .config import DataValidationError


def normalize_component(values: pd.Series, width: int) -> pd.Series:
    text = values.astype("string").str.strip()
    valid = text.str.fullmatch(rf"[0-9]{{1,{width}}}", na=False)
    if not valid.all():
        # Never put source values or row identifiers in exceptions.
        raise DataValidationError(f"Invalid territorial component: {int((~valid).sum())} rows (width {width}).")
    return text.str.zfill(width)


def municipality_codes(department: pd.Series, municipality: pd.Series) -> pd.Series:
    return normalize_component(department, 2) + normalize_component(municipality, 3)


def parse_source_dates(values: pd.Series) -> tuple[pd.Series, dict]:
    text = values.astype("string").fillna("").str.strip()
    normalized = (
        text.str.replace("\u00a0", " ", regex=False)
        .str.replace("\u202f", " ", regex=False)
        .str.replace(r"(?i)a\.\s*m\.", "AM", regex=True)
        .str.replace(r"(?i)p\.\s*m\.", "PM", regex=True)
    )
    parsed = pd.to_datetime(normalized, format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
    return parsed, {
        "blank": int(text.eq("").sum()),
        "parse_fail_nonblank": int((text.ne("") & parsed.isna()).sum()),
    }


def week_start(dates: pd.Series) -> pd.Series:
    days = dates.dt.normalize()
    return days - pd.to_timedelta(days.dt.weekday, unit="D")
