"""Aggregate-only quality summaries and strict merge/acceptance controls."""

import numpy as np
import pandas as pd

from .calendar import build_calendar
from .config import COUNT, COVARIATE_COLUMNS, EXPECTED, KEYS, DataValidationError


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DataValidationError(message)


def frame_profile(frame: pd.DataFrame, keys: list[str] | None = None) -> dict:
    result = {
        "rows": len(frame), "columns": len(frame.columns),
        "types": {c: str(t) for c, t in frame.dtypes.items()},
        "nulls": {c: int(n) for c, n in frame.isna().sum().items()},
        "blank_strings": {}, "numeric_ranges": {}, "date_ranges": {},
    }
    for c in frame:
        values = frame[c]
        if pd.api.types.is_string_dtype(values.dtype):
            result["blank_strings"][c] = int(values.astype("string").str.strip().eq("").sum())
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            numeric = values.to_numpy(dtype=float, na_value=np.nan)
            finite = np.isfinite(numeric)
            result["numeric_ranges"][c] = {
                "min": float(numeric[finite].min()) if finite.any() else None,
                "max": float(numeric[finite].max()) if finite.any() else None,
                "nonfinite": int((~finite).sum()),
            }
        if pd.api.types.is_datetime64_any_dtype(values):
            result["date_ranges"][c] = {
                "min": values.min().isoformat() if values.notna().any() else None,
                "max": values.max().isoformat() if values.notna().any() else None,
            }
    if keys:
        result["unique_keys"] = len(frame[keys].drop_duplicates())
        result["duplicate_rows_beyond_first"] = int(frame.duplicated(keys).sum())
    return result


def validate_keys(frame: pd.DataFrame, label: str) -> None:
    require(all(c in frame for c in KEYS), f"{label}: missing key columns.")
    require(not frame[KEYS].isna().any().any(), f"{label}: null keys.")
    require(not frame.duplicated(KEYS).any(), f"{label}: duplicated municipality-week keys.")


def validate_covariates(frame: pd.DataFrame) -> pd.DataFrame:
    require(list(frame.columns) == COVARIATE_COLUMNS, "Unexpected covariate columns or order.")
    require(not frame.isna().any().any(), "Covariates contain missing values; no imputation is authorized.")
    result = frame.copy()
    codes = result["municipality_code"].astype("string")
    require(codes.str.fullmatch(r"[0-9]{5}", na=False).all(), "Covariate codes must contain five digits.")
    result["municipality_code"] = codes
    require(result["municipality"].astype("string").str.strip().ne("").all(), "Empty municipal names.")
    require(pd.api.types.is_datetime64_any_dtype(result["week_start"]), "Covariate dates must be typed dates.")
    dates = result["week_start"]
    require(dates.dt.tz is None, "Covariate dates must be timezone-naive calendar dates.")
    require((dates.eq(dates.dt.normalize()) & dates.dt.weekday.eq(0)).all(), "Expected Mondays at midnight.")
    validate_keys(result, "Covariates")
    for c in COVARIATE_COLUMNS[3:]:
        require(pd.api.types.is_numeric_dtype(result[c]), f"{c}: expected numeric data.")
        require(np.isfinite(result[c].to_numpy(dtype=float)).all(), f"{c}: nonfinite values.")
    require((result["population"].gt(0) & result["population"].mod(1).eq(0)).all(), "Population must be positive integers.")
    result["population"] = result["population"].astype("int64")
    require(result["area_km2"].gt(0).all(), "Municipal area must be positive.")
    require(result["precipitation_mm"].ge(0).all(), "Negative precipitation detected.")
    require(result["relative_humidity_pct"].between(0, 100).all(), "Humidity outside 0-100 percent.")
    require(np.allclose(result["population"] / result["area_km2"], result["population_density_km2"], rtol=1e-10, atol=1e-8), "Density does not agree with population/area.")
    require(result.groupby("municipality_code")["municipality"].nunique().eq(1).all(), "Conflicting municipal names.")
    require(result.groupby("municipality_code")["area_km2"].nunique().eq(1).all(), "Area changes within municipality.")
    annual = result.assign(year=dates.dt.year).groupby(["municipality_code", "year"])["population"].nunique()
    require(annual.eq(1).all(), "Population changes inside a municipality-year.")
    expected = build_calendar(codes.unique())[KEYS]
    comparison = expected.merge(result[KEYS], on=KEYS, how="outer", validate="one_to_one", indicator=True)
    require(comparison["_merge"].eq("both").all(), "Covariates differ from the complete authorized weekly calendar.")
    return result.sort_values(KEYS).reset_index(drop=True)


def integrate(covariates: pd.DataFrame, counts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    validate_keys(covariates, "Covariates")
    validate_keys(counts, "Weekly counts")
    require(COUNT in counts, "Weekly counts: missing count column.")
    values = counts[COUNT]
    require(values.notna().all(), "Weekly counts contain missing values; only unmatched keys may become zero.")
    require(pd.api.types.is_integer_dtype(values) and values.ge(0).all(), "Counts must be nonnegative integers.")
    coverage = counts[KEYS].merge(covariates[KEYS], on=KEYS, how="left", validate="one_to_one", indicator=True)
    unmatched = int(coverage["_merge"].ne("both").sum())
    require(unmatched == 0, f"Weekly counts have {unmatched} keys outside covariates.")
    result = covariates.merge(counts, on=KEYS, how="left", validate="one_to_one", indicator=True)
    unmatched_keys = result["_merge"].eq("left_only")
    zero_positions = int(unmatched_keys.sum())
    result.loc[unmatched_keys, COUNT] = 0
    result[COUNT] = result[COUNT].astype("int64")
    result["no_eligible_records_observed"] = result[COUNT].eq(0)
    result = result.drop(columns="_merge")
    calendar = build_calendar(result["municipality_code"].unique())
    result = result.merge(calendar, on=KEYS, how="left", validate="one_to_one")
    require(len(result) == len(covariates), "Merge changed the row count.")
    require(int(result[COUNT].sum()) == int(counts[COUNT].sum()), "Merge changed the record total.")
    require(not result.isna().any().any(), "Integrated panel contains unexpected missing values.")
    validate_keys(result, "Integrated panel")
    return result.sort_values(KEYS).reset_index(drop=True), {
        "covariate_rows": len(covariates), "count_keys": len(counts), "output_rows": len(result),
        "count_keys_unmatched": unmatched, "positions_filled_with_zero": zero_positions,
        "rows_multiplied": len(result) - len(covariates), "cardinality": "one_to_one",
        "record_total_before": int(counts[COUNT].sum()), "record_total_after": int(result[COUNT].sum()),
    }


def acceptance_checks(panel: pd.DataFrame) -> dict:
    sizes = panel.groupby("municipality_code").size()
    observed = {
        "municipalities": len(sizes), "weeks_per_municipality_min": int(sizes.min()),
        "weeks_per_municipality_max": int(sizes.max()), "rows": len(panel),
        "eligible_records": int(panel[COUNT].sum()),
        "positions_with_records": int(panel[COUNT].gt(0).sum()),
        "positions_without_records": int(panel[COUNT].eq(0).sum()),
        "partial_positions": int(panel["partial_week"].sum()),
    }
    checks = {k: {"expected": v, "observed": observed[k], "passed": observed[k] == v} for k, v in EXPECTED.items()}
    partial = panel.loc[panel["partial_week"]]
    checks["partial_week_coverage"] = {
        "expected": "2025-12-29 / nominal 2026-01-04 / effective 2025-12-31 / 3 days",
        "passed": bool(len(partial) and partial["week_start"].eq(pd.Timestamp("2025-12-29")).all()
                       and partial["week_end_nominal"].eq(pd.Timestamp("2026-01-04")).all()
                       and partial["week_end_effective"].eq(pd.Timestamp("2025-12-31")).all()
                       and partial["n_days"].eq(3).all()
                       and panel.loc[~panel["partial_week"], "n_days"].eq(7).all()),
    }
    checks["unique_keys"] = {"passed": bool(not panel.duplicated(KEYS).any())}
    return checks
