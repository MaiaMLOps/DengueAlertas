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

Limitation: what is verified is this transcription of the epiCo formula, not epiCo executed
in R. Python `round` and R `round` could differ only on an exact tie at the sixth decimal;
the acceptance figures were computed with Python as well, so they would not detect it.
"""

from collections.abc import Sequence
import math

import numpy as np
import pyarrow as pa
from scipy import stats

from .config import COUNT, HISTORY_YEARS
from .ewt import history_index, historical_references, week_ordinal
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
