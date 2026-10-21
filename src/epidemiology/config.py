"""Fixed methodological contract approved by the team on 2026-10-21."""

from pathlib import Path

from src.data.config import COUNT, KEYS

ROOT = Path(__file__).resolve().parents[2]
HISTORY_YEARS = 5
EPSILON_DENOMINATOR = 1000
EPSILON = 0.001
FIRST_LABEL_YEAR = 2012
ADDED_COLUMNS = ["ewt", "excess_label", "label_eligible"]
DEFAULT_INPUT = Path("data/processed/pilot_municipalities_weekly.parquet")


class EpidemiologyValidationError(ValueError):
    """An integrity failure described without individual health records."""


def method_contract() -> dict:
    """Serializable parameters; these are decisions, not optimization options."""
    return {
        "version": 1,
        "approved_date": "2026-10-21",
        "count": COUNT,
        "history_years": HISTORY_YEARS,
        "epsilon": EPSILON,
        "formula": "exp(sum(log(C + 0.001)) / 5) - 0.001",
        "numerics": "0.001 * expm1(mean(log1p(C / 0.001))); exact constant and integer-mean identities; no tolerance",
        "calendar": "Monday-Sunday W-SUN; no INS equivalence asserted",
        "year": "year of week_start; weeks crossing years are not split",
        "ordinal": "1-based position of Monday within its calendar year, not ISO week",
        "reference": "same municipality and ordinal in years y-5 through y-1 only",
        "week_53": "use ordinal 53 when present; otherwise ordinal 52 of that historical year",
        "epidemic_years": "included without exclusion or substitution",
        "first_label_year": FIRST_LABEL_YEAR,
        "partial_week": "retain 2025-12-29 with null ewt and label; never extrapolate",
        "ineligible": "null ewt and excess_label; label_eligible=false",
        "comparison": "observed count > unrounded ewt; no comparison tolerance or epidemiological floor",
        "scope": "retrospective observed weekly state; no horizon, features, model or sustained episode",
    }
