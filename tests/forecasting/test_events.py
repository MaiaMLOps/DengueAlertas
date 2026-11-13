"""Unit tests of events.py (synthetic data): part 1 (U4.2) alerts, emission helpers, onsets and entries;
part 2 (U4.3) period evaluation, censoring, alignment of positive emissions and summaries."""

from datetime import date, datetime, timedelta
import math

import numpy as np
import pandas as pd
import pytest

from src.epidemiology.config import EpidemiologyValidationError
from src.forecasting.events import (Alert, alerts, channel_positive, evaluate_period, model_positive,
                                    municipality_years, outbreak_onsets, period_onsets, zone_entries)
from src.forecasting.guard import FrozenRecipeError, RecipeAccess

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


# =========================================================================== part 2: period evaluation (U4.3)
H = 8
N = 52
LAST = N + H - 1
DEV_FIRST = date(2019, 11, 11)
TEST_FIRST = date(2022, 11, 7)


def _access():
    return RecipeAccess(test_years=(2023, 2025), channel_version="0" * 32, recipe_path="receta_inexistente_unit.yaml")


def _rows(code="08001", first=DEV_FIRST, zones=None, n=N):
    zones = zones or {}
    data = [(code, first + timedelta(weeks=i), first + timedelta(weeks=i + H), zones.get(i, "seguridad"))
            for i in range(n)]
    return pd.DataFrame(data, columns=["municipality_code", "t", "target_week", "zona_t"])


def _observed(code="08001", first=DEV_FIRST, ones=(), before=20, after=5):
    marked = set(ones)
    data = [(code, first + timedelta(weeks=i), int(i in marked), True) for i in range(-before, LAST + after + 1)]
    return pd.DataFrame(data, columns=["municipality_code", "week_start", "target_u1e", "channel_eligible"])


def _flags(*confirmations, n=N):
    flags = np.zeros(n, dtype=bool)
    for c in confirmations:
        flags[[c - 1, c]] = True
    return flags


def _onset(s):
    return {s, s + 1, s + 2}


def _evaluate(rows, positive, observed):
    return evaluate_period(rows, positive, observed, k=2, window=12, access=_access())


def test_two_municipalities_use_their_own_borders():
    other_first = DEV_FIRST + timedelta(weeks=4)
    rows = pd.concat([_rows("08001"), _rows("41001", other_first)], ignore_index=True)
    observed = pd.concat([_observed("08001", ones=_onset(13)), _observed("41001", other_first, ones=_onset(12))],
                         ignore_index=True)
    result = _evaluate(rows, np.zeros(len(rows), dtype=bool), observed)
    onsets = result.onsets[result.onsets["kind"] == "inicio"].set_index("municipality_code")
    assert bool(onsets.loc["08001", "evaluable"])                    # s - 13 = F of 08001
    assert onsets.loc["41001", "censoring"] == "ventana"             # s - 13 = F - 1 of 41001
    assert result.summary["municipality_years"] == 3                 # 41001's targets reach into 2021
    assert result.by_municipality.loc["41001", "censored_onsets"] == 1


def test_shuffled_rows_give_the_same_result():
    rows, observed = _rows(zones={25: "epidemia"}), _observed(ones=_onset(30) | {44})
    positive = _flags(10, 18, 26, 40)
    base = _evaluate(rows, positive, observed)
    order = np.random.default_rng(7).permutation(len(rows))
    shuffled = _evaluate(rows.iloc[order], positive[order], observed)
    pd.testing.assert_frame_equal(base.onsets, shuffled.onsets)
    pd.testing.assert_frame_equal(base.alerts, shuffled.alerts)
    assert base.summary == shuffled.summary


def test_positive_series_must_share_the_index_of_rows():
    rows, observed = _rows(), _observed(ones=_onset(30))
    aligned = pd.Series(_flags(18), index=rows.index)
    assert _evaluate(rows, aligned, observed).summary["warned_onsets"] == 1
    with pytest.raises(EpidemiologyValidationError, match="index"):
        _evaluate(rows, pd.Series(_flags(18), index=rows.index + 1000), observed)
    with pytest.raises(EpidemiologyValidationError, match="index"):
        _evaluate(rows.iloc[::-1], aligned, observed)              # same labels, different order


def test_positive_array_is_positional():
    rows, observed = _rows(), _observed(ones=_onset(30))
    result = _evaluate(rows.iloc[::-1], _flags(18)[::-1].copy(), observed)
    assert result.summary["warned_onsets"] == 1


def test_barrier_comes_before_validation():
    rows = _rows(first=TEST_FIRST)
    rows.loc[3, "zona_t"] = "desconocida"
    with pytest.raises(FrozenRecipeError):
        _evaluate(rows, np.zeros(N, dtype=bool), _observed(first=TEST_FIRST))


@pytest.mark.parametrize("break_input", ["gap_in_t", "inconsistent_h", "short_positive", "unknown_zone",
                                         "observed_not_covering", "missing_column", "bad_window"])
def test_invalid_period_inputs_fail(break_input):
    rows, observed, positive, window = _rows(), _observed(), np.zeros(N, dtype=bool), 12
    if break_input == "gap_in_t":
        rows, positive = rows.drop(index=10), positive[:-1]
    elif break_input == "inconsistent_h":
        rows.loc[5, "target_week"] = rows.loc[5, "target_week"] + timedelta(weeks=1)
    elif break_input == "short_positive":
        positive = positive[:-1]
    elif break_input == "unknown_zone":
        rows.loc[2, "zona_t"] = "brote"
    elif break_input == "observed_not_covering":
        observed = observed[observed["week_start"] <= DEV_FIRST + timedelta(weeks=LAST - 1)]
    elif break_input == "missing_column":
        rows = rows.drop(columns="zona_t")
    elif break_input == "bad_window":
        window = 0
    with pytest.raises(EpidemiologyValidationError):
        evaluate_period(rows, positive, observed, k=2, window=window, access=_access())


def test_undefined_ratios_are_nan():
    result = _evaluate(_rows(), np.zeros(N, dtype=bool), _observed())
    assert math.isnan(result.summary["detection"]) and math.isnan(result.summary["alert_ratio"])
    assert math.isnan(result.summary["lead_median"])
    assert result.summary["false_alarms_per_municipality_year"] == 0.0


def test_lead_statistics_and_warning_confirmation():
    result = _evaluate(_rows(), _flags(12, 37), _observed(ones=_onset(20) | _onset(40)))
    onsets = result.onsets[result.onsets["kind"] == "inicio"].set_index("week")
    first, second = DEV_FIRST + timedelta(weeks=20), DEV_FIRST + timedelta(weeks=40)
    assert onsets.loc[first, "lead"] == 8
    assert onsets.loc[first, "warning_confirmation"] == DEV_FIRST + timedelta(weeks=12)
    assert onsets.loc[second, "lead"] == 3
    summary = result.summary
    assert (summary["lead_median"], summary["lead_min"], summary["lead_max"]) == (5.5, 3.0, 8.0)
    assert summary["detection"] == 1.0 and summary["alert_ratio"] == 1.0


def test_yearly_and_municipal_breakdowns_add_up():
    rows = pd.concat([_rows("08001"), _rows("41001")], ignore_index=True)
    observed = pd.concat([_observed("08001", ones=_onset(20) | {45}), _observed("41001", ones=_onset(30))],
                         ignore_index=True)
    positive = np.concatenate([_flags(5, 12, 33), _flags(18, 25)])
    result = _evaluate(rows, positive, observed)
    for key in ("evaluable_onsets", "warned_onsets", "judgeable_alerts", "false_alarms", "continuity_alerts",
                "evaluable_entries", "censored_alerts", "municipality_years"):
        assert result.by_municipality[key].sum() == result.summary[key], key
        assert result.by_year[key].sum() == result.summary[key], key


def test_judgeable_alert_before_a_censored_onset_enters_only_the_denominator():
    # Onset at s = 12 is censored (s - 13 < F); the alert t_c = 5 is followed by it, so it is not false.
    result = _evaluate(_rows(), _flags(5), _observed(ones=_onset(12)))
    summary = result.summary
    assert (summary["judgeable_alerts"], summary["false_alarms"], summary["evaluable_onsets"]) == (1, 0, 0)
    assert summary["alert_ratio"] == 0.0


def test_period_onsets_and_municipality_years_need_no_recipe():
    rows = _rows(first=TEST_FIRST)
    table = period_onsets(rows, _observed(first=TEST_FIRST, ones=_onset(30)), window=12)
    assert table["kind"].tolist() == ["inicio", "entrada"]
    assert municipality_years(rows) == 1
