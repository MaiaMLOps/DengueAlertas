"""Pure historical alignment and shifted geometric mean."""

from dataclasses import dataclass
from datetime import date, timedelta
import math
from collections.abc import Iterable

import pyarrow as pa

from .config import COUNT, EPSILON, EPSILON_DENOMINATOR, FIRST_LABEL_YEAR, HISTORY_YEARS
from .validation import require, validate_source


def week_ordinal(start: date) -> int:
    """Enumerate Mondays within the year of the Monday, independently of ISO."""
    require(start.weekday() == 0, "Ordinal alignment requires a Monday.")
    first_day = date(start.year, 1, 1)
    first_monday = first_day + timedelta(days=(-first_day.weekday()) % 7)
    return (start - first_monday).days // 7 + 1


def shifted_geometric_mean(counts: Iterable[int]) -> float:
    values = tuple(counts)
    require(len(values) == HISTORY_YEARS, "Exactly five historical counts are required.")
    require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values),
            "Historical counts must be nonnegative integers.")
    # Exact for constant integer histories, including five zeros and equality cases.
    if len(set(values)) == 1:
        return float(values[0])
    # Algebraically equivalent to exp(mean(log(C + eps))) - eps. Avoid subtraction
    # cancellation near zero; no clipping, epidemiological floor or comparison tolerance.
    result = EPSILON * math.expm1(math.fsum(math.log1p(value / EPSILON) for value in values) / HISTORY_YEARS)
    require(math.isfinite(result) and result >= 0, "Invalid numeric EWT; calculation stopped.")
    # With eps=1/1000, an integer k is the EXACT mean iff this integer identity
    # holds. Test both enclosing integers; never round a noninteger threshold.
    # This also fixes nonconstant histories whose true mean is an integer but
    # whose log/exp evaluation lies one floating-point residue below it.
    scaled_product = math.prod(EPSILON_DENOMINATOR * value + 1 for value in values)
    for candidate in {math.floor(result), math.ceil(result)}:
        if (EPSILON_DENOMINATOR * candidate + 1) ** HISTORY_YEARS == scaled_product:
            return float(candidate)
    return result


@dataclass(frozen=True)
class HistoricalWeek:
    week_start: date
    ordinal: int
    count: int
    partial: bool


def history_index(table: pa.Table) -> dict[tuple[str, int, int], HistoricalWeek]:
    index = {}
    for row in table.select(["municipality_code", "week_start", COUNT, "partial_week"]).to_pylist():
        start = row["week_start"]
        ordinal = week_ordinal(start)
        key = (row["municipality_code"], start.year, ordinal)
        require(key not in index, "Duplicate historical alignment key.")
        index[key] = HistoricalWeek(start, ordinal, row[COUNT], row["partial_week"])
    return index


def historical_references(index: dict, code: str, year: int, ordinal: int) -> tuple[HistoricalWeek, ...]:
    """Return five prior-year weeks; only ordinal 53 may fall back to 52."""
    require(1 <= ordinal <= 53, "Weekly ordinal must be between 1 and 53.")
    references = []
    for historical_year in range(year - HISTORY_YEARS, year):
        reference = index.get((code, historical_year, ordinal))
        if reference is None and ordinal == 53:
            reference = index.get((code, historical_year, 52))
        if reference is None or reference.partial:
            return ()
        require(reference.week_start.year == historical_year and historical_year < year,
                "Historical reference violates the prior-year restriction.")
        references.append(reference)
    return tuple(references)


def calculate_ewt(table: pa.Table) -> tuple[list[float | None], dict]:
    validate_source(table)
    index = history_index(table)
    thresholds = []
    audit = {"initialization": 0, "partial_week": 0, "missing_history": 0,
             "eligible": 0, "historical_references": 0, "week_53_fallback_references": 0}
    for row in table.select(["municipality_code", "week_start", "partial_week"]).to_pylist():
        start = row["week_start"]
        if start.year < FIRST_LABEL_YEAR:
            audit["initialization"] += 1
            thresholds.append(None)
            continue
        if row["partial_week"]:
            audit["partial_week"] += 1
            thresholds.append(None)
            continue
        ordinal = week_ordinal(start)
        references = historical_references(index, row["municipality_code"], start.year, ordinal)
        if not references:
            audit["missing_history"] += 1
            thresholds.append(None)
            continue
        thresholds.append(shifted_geometric_mean(reference.count for reference in references))
        audit["eligible"] += 1
        audit["historical_references"] += len(references)
        audit["week_53_fallback_references"] += sum(ordinal == 53 and r.ordinal == 52 for r in references)
    return thresholds, audit
