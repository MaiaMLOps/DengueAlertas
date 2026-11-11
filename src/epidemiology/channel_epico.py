"""Endemic channel of epiCo for each municipality-week (contrato §1 and §10.1; D1.1).

Transcription of epiCo `endemic_channel` (epiverse-trace/epiCo, commit 39a45ef,
method = "geometric", shift = 1) over the n prior years of the same weekly ordinal:

    central  = round(exp(mean(log(x + shift))) - shift, 5)
    gsd      = round(sd(log(x + shift)), 5)                     (ddof = 1)
    interval = |qt(alpha; n - 1)| * gsd / sqrt(n)
    upper    = central * exp(interval)                          (0 if central = 0)

Zones: "seguridad" if C <= central; "alerta" if central < C <= upper; "epidemia" if C > upper.

Calendar conventions (D1) are reused from the legacy module `ewt.py` without modifying it:
the weekly ordinal is the 1-based Monday within its year, history comes from years y-5..y-1
only, and ordinal 53 falls back to 52 in years without it. Eligibility follows the same rule
as the legacy `label_eligible`: year >= first_year, not the partial week, and five complete
references.

Parameters are explicit keyword-only arguments without defaults; callers take them from the
`canal` section of configs/forecasting.yaml and `first_year` from `config.FIRST_LABEL_YEAR`.
This module never reads configuration nor files.

`channel_for_week` computes central and upper for any week, also after 2025 and without that week
being a panel row, from a municipality's counts of complete weeks. It follows the same chain as
`build_channel_table` (historical_references -> epico_channel), so on panel weeks it reproduces the
product bit for bit.

Limitation: what is verified is this transcription of the epiCo formula, not epiCo executed
in R. Python `round` and R `round` could differ only on an exact tie at the sixth decimal;
the acceptance figures were computed with Python as well, so they would not detect it.
"""

from collections.abc import Collection, Mapping, Sequence
from datetime import date, datetime, timedelta
import math

import numpy as np
import pyarrow as pa
from scipy import stats

from .config import COUNT, HISTORY_YEARS
from .ewt import HistoricalWeek, history_index, historical_references, week_ordinal
from .validation import require, validate_source

ROUND_DECIMALS = 5  # part of the transcribed epiCo formula, not a tunable parameter
ZONES = ("seguridad", "alerta", "epidemia")
CHANNEL_SCHEMA = pa.schema([
    ("epico_central", pa.float64()),
    ("epico_upper", pa.float64()),
    ("target_u1e", pa.int8()),
    ("channel_zone", pa.string()),
    pa.field("channel_eligible", pa.bool_(), nullable=False),
])


def _check_parameters(n_years: int, alpha: float, shift: float | None = None) -> None:
    require(isinstance(n_years, int) and not isinstance(n_years, bool) and n_years >= 2,
            f"n_years must be an integer >= 2; got {n_years!r}.")
    require(isinstance(alpha, (int, float)) and not isinstance(alpha, bool) and 0 < alpha < 0.5,
            f"alpha must lie strictly between 0 and 0.5; got {alpha!r}.")
    if shift is not None:
        require(isinstance(shift, (int, float)) and not isinstance(shift, bool)
                and math.isfinite(shift) and shift > 0,
                f"shift must be a finite number > 0; got {shift!r}.")


def t_critical(*, n_years: int, alpha: float) -> float:
    """Return |qt(alpha; n_years - 1)|, the Student t quantile used by the channel interval.

    `alpha` is the probability of the LOWER tail of the t quantile (0.025 in the contract),
    not a two-sided significance level: qt(0.025; 4) = -2.776445, whose absolute value is used.
    """
    _check_parameters(n_years, alpha)
    return abs(float(stats.t.ppf(alpha, n_years - 1)))


def epico_channel(history: Sequence[int], *, n_years: int, shift: float,
                  alpha: float) -> tuple[float, float, float]:
    """Return (central, gsd, upper) for the counts of the same ordinal in the n prior years.

    `alpha` is the lower-tail probability of the t quantile (see `t_critical`). Central and
    gsd are rounded to five decimals as in epiCo; upper is not rounded.
    """
    _check_parameters(n_years, alpha, shift)
    values = tuple(history)
    require(len(values) == n_years, f"Expected {n_years} historical counts; got {len(values)}.")
    require(all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) and v >= 0
                for v in values), "Historical counts must be nonnegative integers.")
    logs = np.log(np.asarray(values, dtype=float) + shift)
    central = round(float(np.exp(logs.mean())) - shift, ROUND_DECIMALS)
    gsd = round(float(np.std(logs, ddof=1)), ROUND_DECIMALS)
    if central == 0:
        return central, gsd, 0.0
    interval = t_critical(n_years=n_years, alpha=alpha) * gsd / math.sqrt(n_years)
    upper = central * math.exp(interval)
    require(math.isfinite(central) and math.isfinite(upper) and central >= 0 and upper >= central,
            "Invalid numeric channel; calculation stopped.")
    return central, gsd, upper


def channel_zone(count: int, central: float, upper: float) -> str:
    """Zone of the observed count against the channel, with strict upper comparisons."""
    require(central <= upper, "Channel upper limit below its central line.")
    if count <= central:
        return ZONES[0]
    if count <= upper:
        return ZONES[1]
    return ZONES[2]


def _is_monday_date(value) -> bool:
    # datetime and pandas.Timestamp are date subclasses; they would never match a date key.
    return isinstance(value, date) and not isinstance(value, datetime) and value.weekday() == 0


_SERIES = "series"  # single-municipality key for the legacy history index


def _candidate_index(weeks: Collection[date], year: int, ordinal: int) -> dict | None:
    """Minimal history index: the weeks of ordinal `ordinal` (and 52 when it is 53) in the
    n prior years that are present in `weeks`. Only dates matter here; callers read the values.

    A candidate date that falls outside its year (ordinal 53 in a 52-Monday year) does not exist,
    so historical_references falls back to 52, as in the panel. If a year does have ordinal 53 but
    `weeks` lacks it, the reference is missing and None is returned: the fallback to 52 applies
    only to years without that week, never to a missing one."""
    index = {}
    wanted = (ordinal, 52) if ordinal == 53 else (ordinal,)
    for historical_year in range(year - HISTORY_YEARS, year):
        first_day = date(historical_year, 1, 1)
        first_monday = first_day + timedelta(days=(-first_day.weekday()) % 7)
        for candidate_ordinal in wanted:
            start = first_monday + timedelta(weeks=candidate_ordinal - 1)
            if start.year != historical_year:
                continue
            if start not in weeks:
                if candidate_ordinal == 53:
                    return None
                continue
            require(week_ordinal(start) == candidate_ordinal, "Inconsistent ordinal alignment.")
            index[(_SERIES, historical_year, candidate_ordinal)] = HistoricalWeek(
                start, candidate_ordinal, None, False)  # count unused: only the date is selected
    return index


def reference_weeks(weeks: Collection[date], week_start: date, *, n_years: int) -> tuple[date, ...] | None:
    """Return the n reference Mondays of `week_start`, in year order y-5..y-1, or None.

    Single source of the D1 alignment for the channel and for the climate anomalies (contrato §1
    and §5.1): same ordinal in the n prior years; ordinal 53 falls back to 52 only when the year has
    no ordinal 53 in its calendar; a reference week absent from `weeks` gives None. `weeks` are the
    Mondays of the COMPLETE weeks available (precondition: partial weeks excluded by the caller).
    Delegates to `ewt.historical_references` without modifying it.
    """
    require(n_years == HISTORY_YEARS,
            f"The legacy calendar utilities use {HISTORY_YEARS} history years; got n_years={n_years!r}.")
    require(_is_monday_date(week_start), f"week_start must be a Monday datetime.date; got {week_start!r}.")
    require(isinstance(weeks, Collection) and not isinstance(weeks, (str, bytes)),
            "weeks must be a collection of Monday dates.")
    bad = next((key for key in weeks if not _is_monday_date(key)), None)
    require(bad is None, f"Week keys must be Monday datetime.date values; got {bad!r}.")
    ordinal = week_ordinal(week_start)
    index = _candidate_index(weeks, week_start.year, ordinal)
    if index is None:
        return None
    references = historical_references(index, _SERIES, week_start.year, ordinal)
    return tuple(reference.week_start for reference in references) or None


def channel_for_week(counts: Mapping[date, int], week_start: date, *, n_years: int, shift: float,
                     alpha: float) -> tuple[float, float] | None:
    """Return (central, upper) of any week from one municipality's counts, or None without history.

    `counts` maps the Monday of each COMPLETE week to its count (precondition: the caller excludes
    partial weeks, such as 2025-12-29; this function cannot detect them). `week_start` need not be a
    key, so weeks after the panel (e.g. 2026-02-16) are allowed. References come from
    `reference_weeks`; then `epico_channel` gets the same vector as in `build_channel_table`.
    D1 eligibility (year >= first_year, complete week) is the caller's decision.
    """
    _check_parameters(n_years, alpha, shift)
    require(isinstance(counts, Mapping), "counts must map Monday dates to weekly counts.")
    references = reference_weeks(counts, week_start, n_years=n_years)
    if references is None:
        return None
    central, _, upper = epico_channel([counts[week] for week in references],
                                      n_years=n_years, shift=shift, alpha=alpha)
    return central, upper


def build_channel_table(table: pa.Table, *, n_years: int, shift: float, alpha: float,
                        first_year: int) -> pa.Table:
    """Panel (fifteen source columns) -> same rows and columns plus the five channel columns.

    Pure transformation without I/O: row order and source Arrow arrays are preserved; source
    schema metadata is dropped (the caller documents the product). Ineligible weeks have null
    channel columns and channel_eligible = False.
    """
    require(n_years == HISTORY_YEARS,
            f"The legacy calendar utilities use {HISTORY_YEARS} history years; got n_years={n_years!r}.")
    _check_parameters(n_years, alpha, shift)
    validate_source(table)
    index = history_index(table)
    columns = {name: [] for name in CHANNEL_SCHEMA.names}
    for row in table.select(["municipality_code", "week_start", "partial_week", COUNT]).to_pylist():
        start = row["week_start"]
        references = ()
        if start.year >= first_year and not row["partial_week"]:
            references = historical_references(index, row["municipality_code"], start.year,
                                               week_ordinal(start))
        if not references:
            for name in ("epico_central", "epico_upper", "target_u1e", "channel_zone"):
                columns[name].append(None)
            columns["channel_eligible"].append(False)
            continue
        central, _, upper = epico_channel([reference.count for reference in references],
                                          n_years=n_years, shift=shift, alpha=alpha)
        count = row[COUNT]
        columns["epico_central"].append(central)
        columns["epico_upper"].append(upper)
        columns["target_u1e"].append(int(count > upper))
        columns["channel_zone"].append(channel_zone(count, central, upper))
        columns["channel_eligible"].append(True)
    result = table.replace_schema_metadata(None)
    for field in CHANNEL_SCHEMA:
        result = result.append_column(field, pa.array(columns[field.name], type=field.type))
    return result
