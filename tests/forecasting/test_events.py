"""Unit tests of events.py, part 1 (U4.2): alerts, emission helpers, onsets and entries (synthetic data)."""

from datetime import date, datetime, timedelta

import numpy as np
import pytest

from src.epidemiology.config import EpidemiologyValidationError
from src.forecasting.events import (Alert, alerts, channel_positive, model_positive, outbreak_onsets,
                                    zone_entries)

MONDAY = date(2020, 1, 6)


def _weeks(n, first=MONDAY):
    return [first + timedelta(weeks=i) for i in range(n)]


def _alerts(positive, k=2, zones=None):
    zones = zones or ["seguridad"] * len(positive)
    return alerts(_weeks(len(positive)), np.asarray(positive, dtype=bool), zones, k=k)


# --------------------------------------------------------------------------- alerts
def test_k_three_dates_at_third_week_and_ignores_runs_of_two():
    found = _alerts([1, 1, 0, 1, 1, 1, 1, 0], k=3)
    assert found == [Alert(start=_weeks(8)[3], confirmation=_weeks(8)[5], end=_weeks(8)[6], continuity=False)]


def test_run_reaching_the_last_emission_ends_there():
    found = _alerts([0, 0, 1, 1, 1])
    assert [(a.start, a.confirmation, a.end) for a in found] == [(_weeks(5)[2], _weeks(5)[3], _weeks(5)[4])]


def test_all_positive_series_is_one_alert():
    assert len(_alerts([1] * 10)) == 1


def test_empty_series_has_no_alerts():
    assert alerts([], np.array([], dtype=bool), [], k=2) == []
    assert outbreak_onsets([], np.array([], dtype=int)) == []
    assert zone_entries([], np.array([], dtype=int)) == []


def test_integer_flags_are_accepted_like_booleans():
    weeks = _weeks(4)
    assert alerts(weeks, np.array([0, 1, 1, 0]), ["alerta"] * 4, k=2) == alerts(
        weeks, np.array([False, True, True, False]), ["alerta"] * 4, k=2)


# --------------------------------------------------------------------------- onsets and entries
def test_long_run_onset_is_dated_at_its_first_week():
    weeks = _weeks(10)
    assert outbreak_onsets(weeks, np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0])) == [weeks[2]]


def test_two_onsets_separated_by_exactly_two_zeros():
    values = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])
    weeks = _weeks(len(values))
    assert outbreak_onsets(weeks, values) == [weeks[2], weeks[7]]


def test_entries_after_single_zeros():
    weeks = _weeks(4)
    assert zone_entries(weeks, np.array([0, 1, 0, 1])) == [weeks[1], weeks[3]]


def _reference_onsets(a):
    """Literal replica of onsets() in docs/forecasting/verificacion/eventos_d8.py (with runs of descriptivos_d3)."""
    runs, i = [], 0
    while i < len(a):
        if a[i] == 1:
            j = i
            while j < len(a) and a[j] == 1:
                j += 1
            runs.append((i, j - i))
            i = j
        else:
            i += 1
    return [st for st, ln in runs if ln >= 3 and st >= 2 and a[st - 1] == 0 and a[st - 2] == 0]


def test_onsets_match_the_verification_script_on_random_series():
    rng = np.random.default_rng(20261111)
    for _ in range(200):
        n = int(rng.integers(0, 80))
        values = (rng.random(n) < rng.uniform(0.1, 0.7)).astype(int)
        weeks = _weeks(n)
        assert outbreak_onsets(weeks, values) == [weeks[i] for i in _reference_onsets(values)]


# --------------------------------------------------------------------------- emission helpers
def test_model_positive_includes_tau():
    assert model_positive([0.2, 0.25, 0.3], tau=0.25).tolist() == [False, True, True]


@pytest.mark.parametrize("tau", [0, 1, 0.0, 1.0, float("nan"), -0.1, True, "0.3"])
def test_model_positive_rejects_invalid_tau(tau):
    with pytest.raises(EpidemiologyValidationError):
        model_positive([0.5], tau=tau)


@pytest.mark.parametrize("p", [[1.2], [-0.1], [float("nan")], [[0.5, 0.5]]])
def test_model_positive_rejects_invalid_probabilities(p):
    with pytest.raises(EpidemiologyValidationError):
        model_positive(p, tau=0.5)


def test_channel_positive_rejects_unknown_zone():
    with pytest.raises(EpidemiologyValidationError):
        channel_positive(["alerta", "brote"])


# --------------------------------------------------------------------------- validation
@pytest.mark.parametrize("weeks", [
    [MONDAY, MONDAY + timedelta(weeks=2)],                       # gap
    [MONDAY, MONDAY],                                            # duplicate
    [MONDAY + timedelta(weeks=1), MONDAY],                       # backwards
    [MONDAY, MONDAY + timedelta(days=8)],                        # not a Monday
    [datetime(2020, 1, 6), datetime(2020, 1, 13)],              # datetime, not date
    ["2020-01-06", "2020-01-13"],                                # strings
])
def test_weeks_must_be_consecutive_monday_dates(weeks):
    with pytest.raises(EpidemiologyValidationError):
        alerts(weeks, np.array([True, True]), ["alerta", "alerta"], k=2)
    with pytest.raises(EpidemiologyValidationError):
        outbreak_onsets(weeks, np.array([0, 1]))


@pytest.mark.parametrize("positive", [
    np.array([True, True, True]),                                # wrong length
    np.array([0, 2]),                                            # not 0/1
    np.array([0.0, float("nan")]),                               # NaN
    np.array(["1", "0"]),                                        # strings
])
def test_alerts_reject_invalid_positive(positive):
    with pytest.raises(EpidemiologyValidationError):
        alerts(_weeks(2), positive, ["alerta", "alerta"], k=2)


@pytest.mark.parametrize("exceed", [np.array([0, 1, 1]), np.array([0, 2]), np.array([0.0, float("nan")])])
def test_onsets_and_entries_reject_invalid_exceed(exceed):
    with pytest.raises(EpidemiologyValidationError):
        outbreak_onsets(_weeks(2), exceed)
    with pytest.raises(EpidemiologyValidationError):
        zone_entries(_weeks(2), exceed)


@pytest.mark.parametrize("zones", [["alerta"], ["alerta", "brote"]])
def test_alerts_reject_invalid_zones(zones):
    with pytest.raises(EpidemiologyValidationError):
        alerts(_weeks(2), np.array([True, True]), zones, k=2)


@pytest.mark.parametrize("k", [0, -1, 1.5, True, "2"])
def test_alerts_reject_invalid_k(k):
    with pytest.raises(EpidemiologyValidationError):
        alerts(_weeks(2), np.array([True, True]), ["alerta", "alerta"], k=k)
