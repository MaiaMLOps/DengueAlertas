"""Event-level evaluation: alerts, outbreak onsets and zone entries (contrato §2.4, §7.4 and §10.13;
especificación §4 and §6.F4).

Flow, per municipality:
    emissions (weeks t, positive or not, observed zone at t)
      -> model_positive(p, tau) for a model (p >= tau) or channel_positive(zones) for the channel
         (zone "alerta"): both feed the same `alerts`, so model and channel follow the same rule
      -> alerts(weeks, positive, zones, k): one Alert per maximal run of >= k positive emissions, dated at
         its confirmation week t_c (the k-th week of the run, when it is known); a continuity alert if the
         observed zone at t_c is "epidemia"
    observed series (eligible weeks and target_u1e = 1[C > U1e])
      -> outbreak_onsets: first week of a run of >= 3 weeks above U1e preceded by >= 2 weeks below
         (Umaña et al., 2026, with U1e)
      -> zone_entries: weeks above U1e whose previous week is below (secondary event)

Part 2 (U4.3) adds the period evaluation: censoring at the period borders, matching with the window W,
lead time, false alarms, metrics per municipality-year and the test-period barrier.

Everything here is a pure function on one municipality's series: no I/O, no configuration and no scores,
so nothing here calls the barrier. k and tau have no defaults; they come from the `decision` section.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math

import numpy as np

from src.epidemiology.channel_epico import ZONES
from src.epidemiology.validation import require

# Onset definition of contrato §2.4 and §7.4 (Umaña et al. with U1e): part of the event, not a tunable
# parameter (F4 decision 5). Confirmation of an onset in s happens in s + ONSET_MIN_RUN - 1.
ONSET_MIN_RUN = 3
ONSET_PRIOR_BELOW = 2
WEEK = timedelta(weeks=1)
_EPIDEMIC_ZONE = ZONES[2]
_ALERT_ZONE = ZONES[1]


@dataclass(frozen=True)
class Alert:
    """A maximal run of >= k positive emissions; `confirmation` is t_c, the k-th week of the run."""

    start: date
    confirmation: date
    end: date
    continuity: bool


# --------------------------------------------------------------------------- validation helpers
def _check_weeks(weeks: Iterable) -> list[date]:
    checked = list(weeks)
    for week in checked:
        require(isinstance(week, date) and not isinstance(week, datetime),
                f"Weeks must be datetime.date values; got {week!r}.")
        require(week.weekday() == 0, f"Weeks must be Mondays (W-SUN weeks); got {week}.")
    for previous, current in zip(checked, checked[1:]):
        require(current - previous == WEEK, f"Weeks must be consecutive: {previous} is followed by {current}.")
    return checked


def _binary(values, name: str, length: int) -> np.ndarray:
    array = np.asarray(values)
    require(array.ndim == 1 and len(array) == length,
            f"{name} must be one-dimensional with one value per week ({length}); got shape {array.shape}.")
    if array.dtype == bool:
        return array.astype(np.int8)
    require(array.dtype.kind in "iuf", f"{name} must hold booleans or 0/1; got dtype {array.dtype}.")
    require(bool(np.isin(array, (0, 1)).all()), f"{name} must hold only 0/1 (no NaN or other values).")
    return array.astype(np.int8)


def _check_k(k) -> None:
    require(isinstance(k, (int, np.integer)) and not isinstance(k, bool) and k >= 1,
            f"k must be an integer >= 1; got {k!r}.")


def _runs(values: Sequence[int]) -> list[tuple[int, int]]:
    """Maximal runs of ones as (first index, length), in order."""
    runs, i = [], 0
    while i < len(values):
        if values[i]:
            j = i
            while j < len(values) and values[j]:
                j += 1
            runs.append((i, j - i))
            i = j
        else:
            i += 1
    return runs


# --------------------------------------------------------------------------- emissions and alerts
def model_positive(p, *, tau: float) -> np.ndarray:
    """Positive model emissions: p >= tau (contrato §7.4)."""
    require(isinstance(tau, (int, float, np.floating)) and not isinstance(tau, bool)
            and math.isfinite(tau) and 0 < tau < 1, f"tau must be in (0, 1); got {tau!r}.")
    array = np.asarray(p, dtype=float)
    require(array.ndim == 1, "p must be one-dimensional.")
    require(bool(np.isfinite(array).all()) and bool(((array >= 0) & (array <= 1)).all()),
            "p must hold finite probabilities in [0, 1].")
    return array >= tau


def _check_zones(zones, length: int | None = None) -> list[str]:
    checked = list(zones)
    if length is not None:
        require(len(checked) == length, f"zones must have one value per week ({length}); got {len(checked)}.")
    unknown = sorted({str(zone) for zone in checked if zone not in ZONES})
    require(not unknown, f"Unknown zones {unknown}; expected {list(ZONES)}.")
    return checked


def channel_positive(zones) -> np.ndarray:
    """Positive channel emissions: observed zone at t is "alerta" (central < C_t <= U1e_t)."""
    return np.asarray([zone == _ALERT_ZONE for zone in _check_zones(zones)], dtype=bool)


def alerts(weeks, positive, zones, *, k: int) -> list[Alert]:
    """Alerts of one municipality's consecutive emissions, in chronological order (contrato §7.4)."""
    _check_k(k)
    checked_weeks = _check_weeks(weeks)
    flags = _binary(positive, "positive", len(checked_weeks))
    checked_zones = _check_zones(zones, len(checked_weeks))
    found = []
    for first, length in _runs(flags):
        if length < k:
            continue
        confirmation = first + k - 1
        found.append(Alert(start=checked_weeks[first], confirmation=checked_weeks[confirmation],
                           end=checked_weeks[first + length - 1],
                           continuity=checked_zones[confirmation] == _EPIDEMIC_ZONE))
    return found


# --------------------------------------------------------------------------- observed events
def outbreak_onsets(weeks, exceed) -> list[date]:
    """Onsets: first week of a run of >= 3 weeks with C > U1e preceded, inside the series, by >= 2 weeks
    with C <= U1e (contrato §2.4)."""
    checked_weeks = _check_weeks(weeks)
    values = _binary(exceed, "exceed", len(checked_weeks))
    return [checked_weeks[first] for first, length in _runs(values)
            if length >= ONSET_MIN_RUN and first >= ONSET_PRIOR_BELOW
            and not values[first - ONSET_PRIOR_BELOW:first].any()]


def zone_entries(weeks, exceed) -> list[date]:
    """Entries 0 -> 1: weeks with C > U1e whose previous week of the series has C <= U1e. A run at the
    first week of the series is not an entry (its previous week is unknown)."""
    checked_weeks = _check_weeks(weeks)
    values = _binary(exceed, "exceed", len(checked_weeks))
    return [checked_weeks[first] for first, _ in _runs(values) if first >= 1]
