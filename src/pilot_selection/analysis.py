"""Pure descriptive calculations; no source reconstruction or pilot selection."""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

from .config import AnalysisValidationError, COUNT, INDICATORS, KEYS, PERIODS, PILOTS


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisValidationError(message)


def prepare_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the observed panel without imputing, deduplicating or repairing."""
    required = KEYS + ["municipality", COUNT, "partial_week", "n_days",
                       "week_end_nominal", "week_end_effective", "no_eligible_records_observed"]
    require(set(required).issubset(frame.columns), "Missing required panel columns.")
    require(len(frame) > 0, "Panel is empty.")
    require(not frame.isna().any().any(), "Panel contains nulls.")
    out = frame.copy()
    require(out.municipality_code.map(
        lambda x: isinstance(x, str) and len(x) == 5 and x.isascii() and x.isdigit()
    ).all(), "DIVIPOLA must be five ASCII digits stored as text.")
    require(out.municipality.map(lambda x: isinstance(x, str) and bool(x.strip())).all(),
            "Municipality names must be nonempty text.")
    require(out.groupby("municipality_code").municipality.nunique().eq(1).all(),
            "One descriptive name per municipality is required.")
    for col in (COUNT, "n_days"):
        require(is_integer_dtype(out[col]) and not is_bool_dtype(out[col]),
                f"{col} must contain integers.")
    require(out[COUNT].ge(0).all(), "Negative counts are invalid.")
    require(out.n_days.between(1, 7).all(), "Calendar duration must be between one and seven days.")
    for col in ("partial_week", "no_eligible_records_observed"):
        require(is_bool_dtype(out[col]), f"{col} must be boolean.")
    for col in ("week_start", "week_end_nominal", "week_end_effective"):
        try:
            out[col] = pd.to_datetime(out[col], errors="raise", format="ISO8601")
            valid = out[col].dt.tz is None and out[col].eq(out[col].dt.normalize()).all()
        except (ValueError, TypeError, AttributeError):
            raise AnalysisValidationError("Calendar fields must be timezone-free dates.") from None
        require(valid, "Calendar fields must be timezone-free dates.")
    require(not out.duplicated(KEYS).any(), "Duplicate municipality-week keys.")
    require(out.week_start.dt.dayofweek.eq(0).all(), "week_start must be Monday.")
    require(out.week_end_nominal.eq(out.week_start + pd.Timedelta(days=6)).all(),
            "Nominal week end disagrees with Monday-Sunday calendar.")
    require(out.n_days.eq((out.week_end_effective - out.week_start).dt.days + 1).all(),
            "Effective week end disagrees with n_days.")
    require(out.partial_week.eq(out.n_days.lt(7)).all(), "partial_week disagrees with n_days.")
    require(out.no_eligible_records_observed.eq(out[COUNT].eq(0)).all(),
            "Zero-record flag disagrees with counts.")
    out = out.sort_values(KEYS).reset_index(drop=True)
    grid = pd.date_range(out.week_start.min(), out.week_start.max(), freq="W-MON")
    for _, group in out.groupby("municipality_code", sort=True):
        require(group.week_start.tolist() == grid.tolist(), "Missing or unequal weekly calendars.")
    return out


def complete_weeks(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[~frame.partial_week].copy().sort_values(KEYS).reset_index(drop=True)
    require(len(out) > 0, "No complete weeks remain.")
    require(out.n_days.eq(7).all(), "Complete rows must contain seven calendar days.")
    require(out.groupby("municipality_code").week_start.diff().dropna().eq(
        pd.Timedelta(days=7)).all(), "Complete series contains a calendar gap.")
    return out


def period_frame(frame: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    """Filter before computing any period-dependent statistic."""
    start, end = years
    require(start <= end, "Invalid period.")
    out = frame.loc[frame.week_start.dt.year.between(start, end)].copy()
    require(len(out) > 0, "Requested period is empty.")
    return out.sort_values(KEYS).reset_index(drop=True)


def annual_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.assign(year=frame.week_start.dt.year, active=frame[COUNT].gt(0))
    annual = working.groupby(["municipality_code", "municipality", "year"], sort=True).agg(
        total_records=(COUNT, "sum"), n_weeks=(COUNT, "size"), active_weeks=("active", "sum")
    ).reset_index()
    annual["records_per_week"] = annual.total_records / annual.n_weeks
    annual["active_proportion"] = annual.active_weeks / annual.n_weeks
    require(int(annual.total_records.sum()) == int(frame[COUNT].sum()), "Annual totals disagree.")
    return annual


def longest_zero_run(counts) -> int:
    current = longest = 0
    for value in counts:
        current = current + 1 if value == 0 else 0
        longest = max(longest, current)
    return longest


def municipal_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    annual = annual_profiles(frame).set_index("municipality_code")
    rows = []
    for code, group in frame.sort_values(KEYS).groupby("municipality_code", sort=True):
        values = group[COUNT]
        yearly = annual.loc[[code]].sort_values("year")
        total = int(values.sum())
        peak = int(yearly.total_records.max())
        peak_years = yearly.loc[yearly.total_records.eq(peak), "year"].tolist() if total else []
        min_years = yearly.loc[yearly.total_records.eq(yearly.total_records.min()), "year"].tolist()
        rows.append({
            "municipality_code": code, "municipality": group.municipality.iloc[0],
            "n_weeks": len(group), "total_records": total,
            "active_weeks": int(values.gt(0).sum()), "active_pct": 100 * values.gt(0).mean(),
            "weekly_median": float(values.median()),
            "annual_active_median": float(yearly.active_weeks.median()),
            "annual_active_min": int(yearly.active_weeks.min()),
            "max_zero_run": longest_zero_run(values),
            "peak_year": min(peak_years) if peak_years else None,
            "peak_years": "|".join(map(str, peak_years)),
            "min_year": min(min_years), "min_years": "|".join(map(str, min_years)),
            "peak_concentration_pct": 100 * peak / total if total else np.nan,
        })
    result = pd.DataFrame(rows)
    result["peak_year"] = result.peak_year.astype("Int64")
    require(int(result.total_records.sum()) == int(frame[COUNT].sum()), "Municipal totals disagree.")
    return result


def distribution_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = metrics[list(INDICATORS)].quantile([0, .25, .5, .75, 1], interpolation="linear").T
    summary.columns = ["min", "p25", "median", "p75", "max"]
    summary.index.name = "indicator"
    summary["valid_municipalities"] = metrics[list(INDICATORS)].count()
    return summary.reset_index()


def comparison_15(metrics: pd.DataFrame) -> pd.DataFrame:
    require(len(metrics) >= 15 and metrics.n_weeks.nunique() == 1,
            "Comparison requires at least 15 municipalities with a common denominator.")
    high = metrics.sort_values(["active_weeks", "municipality_code"], ascending=[False, True]).head(5)
    low = metrics.sort_values(["active_weeks", "municipality_code"]).head(5)
    mid = metrics.assign(distance=(metrics.active_weeks - metrics.active_weeks.median()).abs()).sort_values(
        ["distance", "municipality_code"]).head(5).drop(columns="distance")
    result = pd.concat([high.assign(comparison_group="highest"), mid.assign(comparison_group="median"),
                        low.assign(comparison_group="lowest")], ignore_index=True)
    require(result.municipality_code.nunique() == 15,
            "Comparison groups overlap under ties; cannot report 15 distinct municipalities using this rule.")
    return result


def peak_year_distribution(metrics: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    counts = metrics.peak_year.value_counts().reindex(range(years[0], years[1] + 1), fill_value=0)
    result = counts.rename_axis("year").rename("municipalities").reset_index()
    result["municipalities"] = result.municipalities.astype(int)
    result["municipality_pct"] = 100 * result.municipalities / len(metrics)
    return result


def sensitivity_comparison(full: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    result = full.merge(historical, on=["municipality_code", "municipality"], suffixes=("_full", "_historical"),
                        validate="one_to_one")
    require(len(result) == len(full) == len(historical), "Period municipality sets differ.")
    for column in INDICATORS:
        result[column + "_historical_minus_full"] = result[column + "_historical"] - result[column + "_full"]
    result["recent_records"] = result.total_records_full - result.total_records_historical
    result["recent_share_pct"] = 100 * result.recent_records / result.total_records_full.replace(0, np.nan)
    require(result.recent_records.ge(0).all(), "Historical total exceeds full total.")
    return result


def pilot_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = metrics.set_index("municipality_code")
    codes = [pilot.code for pilot in PILOTS]
    require(set(codes).issubset(indexed.index), "A team-selected municipality is missing.")
    result = indexed.loc[codes].reset_index()
    result.insert(2, "pilot_name", [pilot.name for pilot in PILOTS])
    return result


def build_tables(complete: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Each period gets its own filtered input and its own annual statistics.
    frames = {label: period_frame(complete, years) for label, years in PERIODS.items()}
    metrics = {label: municipal_indicators(frame) for label, frame in frames.items()}
    return {
        "municipal_indicators": pd.concat([m.assign(period=p) for p, m in metrics.items()], ignore_index=True),
        "annual_profiles": annual_profiles(frames["full"]),
        "distribution_summary": pd.concat([distribution_summary(m).assign(period=p) for p, m in metrics.items()], ignore_index=True),
        "comparison_15": comparison_15(metrics["full"]),
        "sensitivity_top10": metrics["historical"].sort_values(
            ["active_pct", "municipality_code"], ascending=[False, True]).head(10).reset_index(drop=True),
        "sensitivity_comparison": sensitivity_comparison(metrics["full"], metrics["historical"]),
        "peak_year_distribution": pd.concat([peak_year_distribution(m, PERIODS[p]).assign(period=p)
                                             for p, m in metrics.items()], ignore_index=True),
        "selected_pilots": pd.concat([pilot_rows(m).assign(period=p) for p, m in metrics.items()], ignore_index=True),
    }
