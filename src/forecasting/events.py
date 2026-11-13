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

Period evaluation (part 2), for the evaluated rows of one period (development or test; the emissions of a
period are the evaluated rows of `splits.yearly_trials`). Per municipality, F = first emission (min t) and
L = last target week; no object may use weeks outside [F, L] (contrato §7.4, «Censura en los bordes»):
    period_onsets: onsets and entries with s in [first target week, L], identified on the whole eligible
      series and then censored: "ventana" if s - (W + 1) < F (the alert confirmed in s - W needs the
      emission of s - W - 1); for onsets, "confirmacion" if s + 2 > L (an onset is known in s + 2).
      No barrier: test-period onsets are counted without a recipe.
    evaluate_period: test-period barrier first, then alerts -> judgeable if t_c + W + 2 <= L; a false
      alarm if judgeable, not continuity and no onset in [t_c + 1, t_c + W]; an onset or entry in s is
      warned by any non-continuity alert with t_c in [s - W, s - 1], judgeable or not, with the lead
      s - t_c measured from the most recent one. The alert ratio counts in its numerator only onsets
      warned by judgeable alerts, so numerator and denominator share one population.

Note for F7: a judgeable alert followed by a censored onset is not a false alarm, so it enters the
denominator of the alert ratio without adding to its numerator (the censored onset is not evaluable).

Part 1 functions work on one municipality's series and call no barrier. k, tau and window have no
defaults; they come from the `decision` section of the configuration.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
import statistics

import numpy as np
import pandas as pd

from src.epidemiology.channel_epico import ZONES
from src.epidemiology.validation import require

from .guard import RecipeAccess, check_test_access

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


# =========================================================================== part 2: period evaluation
ONSET, ENTRY = "inicio", "entrada"
WINDOW_CENSORING, CONFIRMATION_CENSORING = "ventana", "confirmacion"
_ROW_COLUMNS = ("municipality_code", "t", "target_week", "zona_t")
_OBSERVED_COLUMNS = ("municipality_code", "week_start", "target_u1e", "channel_eligible")
_EVENT_COLUMNS = ["municipality_code", "kind", "week", "year", "evaluable", "censoring"]
_WARNING_COLUMNS = ["warned", "warned_by_judgeable", "lead", "warning_confirmation"]
_ALERT_COLUMNS = ["municipality_code", "start", "confirmation", "end", "continuity", "judgeable",
                  "false_alarm", "followed_by_epidemic", "year"]


@dataclass(frozen=True)
class PeriodEvents:
    """Event evaluation of one system over one period: tables of onsets/entries and alerts, and summaries."""

    onsets: pd.DataFrame
    alerts: pd.DataFrame
    summary: dict
    by_municipality: pd.DataFrame
    by_year: pd.DataFrame


@dataclass(frozen=True)
class _Municipality:
    """Rows of one municipality in the period, its borders and its whole eligible observed series."""

    code: str
    rows: pd.DataFrame          # sorted by t, original index labels kept
    first_emission: date        # F
    first_target: date
    last_target: date           # L
    weeks: list[date]
    exceed: np.ndarray


def _check_window(window) -> None:
    require(isinstance(window, (int, np.integer)) and not isinstance(window, bool) and window >= 1,
            f"window must be an integer >= 1; got {window!r}.")


def _check_frame(frame, columns, name: str) -> None:
    require(isinstance(frame, pd.DataFrame), f"{name} must be a pandas DataFrame.")
    missing = [column for column in columns if column not in frame.columns]
    require(not missing, f"{name} lacks columns {missing}.")


def _municipalities(rows: pd.DataFrame, observed: pd.DataFrame) -> list[_Municipality]:
    """Validate the period rows and the observed series; one context per municipality, in code order."""
    _check_frame(rows, _ROW_COLUMNS, "rows")
    _check_frame(observed, _OBSERVED_COLUMNS, "observed")
    require(len(rows) > 0, "rows must not be empty.")
    require(rows.index.is_unique, "rows must have a unique index.")
    require(not rows.duplicated(["municipality_code", "t"]).any(), "Duplicate (municipality_code, t) rows.")
    require(observed["channel_eligible"].dtype == bool, "observed.channel_eligible must be boolean.")

    contexts = []
    for code in sorted(rows["municipality_code"].unique()):
        group = rows[rows["municipality_code"] == code].sort_values("t")
        emissions = _check_weeks(group["t"])
        targets = _check_weeks(group["target_week"])
        offsets = {target - emission for emission, target in zip(emissions, targets)}
        require(len(offsets) == 1, f"Municipality {code}: target_week - t is not constant ({sorted(offsets)}).")
        _check_zones(group["zona_t"], len(group))

        own = observed[(observed["municipality_code"] == code) & observed["channel_eligible"]]
        own = own.sort_values("week_start")
        weeks = _check_weeks(own["week_start"])
        exceed = _binary(own["target_u1e"].to_numpy(), "observed.target_u1e", len(weeks))
        first, last = emissions[0], targets[-1]
        require(bool(weeks) and weeks[0] <= first and weeks[-1] >= last,
                f"Municipality {code}: the eligible observed series must cover the period weeks {first}..{last}.")
        contexts.append(_Municipality(code=code, rows=group, first_emission=first, first_target=targets[0],
                                      last_target=last, weeks=weeks, exceed=exceed))
    return contexts


def _period_events(context: _Municipality, window: int) -> list[dict]:
    """Onsets and entries of the period (s in [first target, L]) with their censoring."""
    records = []
    found = [(week, ONSET) for week in outbreak_onsets(context.weeks, context.exceed)]
    found += [(week, ENTRY) for week in zone_entries(context.weeks, context.exceed)]
    for week, kind in sorted(found, key=lambda item: (item[0], item[1] == ENTRY)):
        if not context.first_target <= week <= context.last_target:
            continue
        censoring = None
        if week - (window + 1) * WEEK < context.first_emission:
            censoring = WINDOW_CENSORING
        elif kind == ONSET and week + (ONSET_MIN_RUN - 1) * WEEK > context.last_target:
            censoring = CONFIRMATION_CENSORING
        records.append({"municipality_code": context.code, "kind": kind, "week": week, "year": week.year,
                        "evaluable": censoring is None, "censoring": censoring})
    return records


def _event_table(records: list[dict], columns: list[str]) -> pd.DataFrame:
    table = pd.DataFrame(records, columns=columns)
    return table.astype({"evaluable": bool})


def period_onsets(rows: pd.DataFrame, observed: pd.DataFrame, *, window: int) -> pd.DataFrame:
    """Onsets and entries of a period with their censoring (contrato §7.4). Scores nothing, so it does not
    call the test-period barrier: test-period onsets are counted with it."""
    _check_window(window)
    records = []
    for context in _municipalities(rows, observed):
        records += _period_events(context, window)
    return _event_table(records, _EVENT_COLUMNS)


def municipality_years(rows: pd.DataFrame) -> int:
    """Number of distinct (municipality, calendar year of the target week) pairs among the period rows."""
    _check_frame(rows, ("municipality_code", "target_week"), "rows")
    return len({(code, week.year) for code, week in zip(rows["municipality_code"], rows["target_week"])})


def _positive_flags(positive, rows: pd.DataFrame) -> pd.Series:
    """Positive emissions aligned with rows: a Series must carry exactly the index of rows; an array is
    positional (one value per row, in the order of rows)."""
    if isinstance(positive, pd.Series):
        require(positive.index.equals(rows.index),
                "positive is a pandas Series whose index differs from the index of rows; align it or pass an array.")
        values = positive.to_numpy()
    else:
        values = np.asarray(positive)
    return pd.Series(_binary(values, "positive", len(rows)).astype(bool), index=rows.index)


def _alert_records(context: _Municipality, flags: pd.Series, k: int, window: int, onsets: set) -> list[dict]:
    rows = context.rows
    target_year = {t: target.year for t, target in zip(rows["t"], rows["target_week"])}
    epidemic = {week for week, value in zip(context.weeks, context.exceed) if value == 1}
    records = []
    for alert in alerts(list(rows["t"]), flags.loc[rows.index].to_numpy(), list(rows["zona_t"]), k=k):
        confirmation = alert.confirmation
        following = [confirmation + d * WEEK for d in range(1, window + 1)]
        judgeable = confirmation + (window + ONSET_MIN_RUN - 1) * WEEK <= context.last_target
        false_alarm = judgeable and not alert.continuity and not any(week in onsets for week in following)
        records.append({"municipality_code": context.code, "start": alert.start, "confirmation": confirmation,
                        "end": alert.end, "continuity": alert.continuity, "judgeable": judgeable,
                        "false_alarm": false_alarm,
                        "followed_by_epidemic": any(week in epidemic for week in following),
                        "year": target_year[confirmation]})
    return records


def _warning(week: date, alert_records: list[dict], window: int) -> dict:
    candidates = [record for record in alert_records if not record["continuity"]
                  and week - window * WEEK <= record["confirmation"] <= week - WEEK]
    if not candidates:
        return {"warned": False, "warned_by_judgeable": False, "lead": math.nan, "warning_confirmation": None}
    latest = max(record["confirmation"] for record in candidates)
    return {"warned": True, "warned_by_judgeable": any(record["judgeable"] for record in candidates),
            "lead": float((week - latest).days // 7), "warning_confirmation": latest}


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def _summarize(onsets: pd.DataFrame, alert_table: pd.DataFrame, n_municipality_years: int) -> dict:
    """Event metrics of contrato §7.4 for one subset (all, one municipality or one year)."""
    require(n_municipality_years > 0, "A summary needs at least one municipality-year.")
    evaluable = onsets[onsets["evaluable"]]
    onset_rows = evaluable[evaluable["kind"] == ONSET]
    entry_rows = evaluable[evaluable["kind"] == ENTRY]
    non_continuity = alert_table[~alert_table["continuity"]]
    judgeable = non_continuity[non_continuity["judgeable"]]
    false_alarms = int(judgeable["false_alarm"].sum())
    leads = [float(lead) for lead in onset_rows.loc[onset_rows["warned"], "lead"]]
    warned = int(onset_rows["warned"].sum())
    by_judgeable = int(onset_rows["warned_by_judgeable"].sum())
    warned_entries = int(entry_rows["warned"].sum())
    continuity = int(alert_table["continuity"].sum())
    return {
        "evaluable_onsets": len(onset_rows),
        "warned_onsets": warned,
        "detection": _ratio(warned, len(onset_rows)),
        "onsets_warned_by_judgeable": by_judgeable,
        "judgeable_alerts": len(judgeable),
        "false_alarms": false_alarms,
        "false_alarms_followed_by_epidemic": int((judgeable["false_alarm"] & judgeable["followed_by_epidemic"]).sum()),
        "municipality_years": n_municipality_years,
        "false_alarms_per_municipality_year": false_alarms / n_municipality_years,
        "alert_ratio": _ratio(by_judgeable, len(judgeable)),
        "lead_median": statistics.median(leads) if leads else math.nan,
        "lead_min": min(leads) if leads else math.nan,
        "lead_max": max(leads) if leads else math.nan,
        "continuity_alerts": continuity,
        "continuity_per_municipality_year": continuity / n_municipality_years,
        "evaluable_entries": len(entry_rows),
        "warned_entries": warned_entries,
        "entry_detection": _ratio(warned_entries, len(entry_rows)),
        "censored_onsets": int(((onsets["kind"] == ONSET) & ~onsets["evaluable"]).sum()),
        "censored_entries": int(((onsets["kind"] == ENTRY) & ~onsets["evaluable"]).sum()),
        "censored_alerts": int((~non_continuity["judgeable"]).sum()),
    }


def evaluate_period(rows: pd.DataFrame, positive, observed: pd.DataFrame, *, k: int, window: int,
                    access: RecipeAccess) -> PeriodEvents:
    """Events of one system (model or channel) over one period (contrato §7.4). The test-period barrier is
    checked first with the target weeks of the rows: detection and false alarms on 2023-2025 need the
    frozen recipe."""
    check_test_access(rows["target_week"], access)
    _check_k(k)
    _check_window(window)
    contexts = _municipalities(rows, observed)
    flags = _positive_flags(positive, rows)

    event_records, alert_records = [], []
    for context in contexts:
        onsets = set(outbreak_onsets(context.weeks, context.exceed))
        own_alerts = _alert_records(context, flags, k, window, onsets)
        alert_records += own_alerts
        for record in _period_events(context, window):
            event_records.append({**record, **_warning(record["week"], own_alerts, window)})

    event_table = _event_table(event_records, _EVENT_COLUMNS + _WARNING_COLUMNS)
    event_table = event_table.astype({"warned": bool, "warned_by_judgeable": bool, "lead": float})
    alert_table = pd.DataFrame(alert_records, columns=_ALERT_COLUMNS).astype(
        {"continuity": bool, "judgeable": bool, "false_alarm": bool, "followed_by_epidemic": bool})

    target_years = rows["target_week"].map(lambda week: week.year)
    by_municipality = {
        context.code: _summarize(event_table[event_table["municipality_code"] == context.code],
                                 alert_table[alert_table["municipality_code"] == context.code],
                                 municipality_years(context.rows))
        for context in contexts}
    by_year = {
        int(year): _summarize(event_table[event_table["year"] == year], alert_table[alert_table["year"] == year],
                              int(rows.loc[target_years == year, "municipality_code"].nunique()))
        for year in sorted(target_years.unique())}
    return PeriodEvents(
        onsets=event_table, alerts=alert_table,
        summary=_summarize(event_table, alert_table, municipality_years(rows)),
        by_municipality=pd.DataFrame.from_dict(by_municipality, orient="index").rename_axis("municipality_code"),
        by_year=pd.DataFrame.from_dict(by_year, orient="index").rename_axis("year"))
