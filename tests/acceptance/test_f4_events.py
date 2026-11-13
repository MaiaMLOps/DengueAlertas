"""F4 acceptance with synthetic series: alert rule and event evaluation (contrato §7.4 and §10.13;
especificación §6.F4). Decisions approved by Jesús for F4 (2026-11-11):
1. an alert that cannot be judged (t_c + W + 2 after the last target week) may still warn an evaluable
   onset, but the numerator of the alert ratio only counts onsets warned by judgeable alerts;
2. onsets are identified on the whole observed series and censored afterwards;
3. onsets in emission weeks before the first target week do not enter detection, but they do prevent a
   false alarm;
4. entries are known in s, so they are censored only by the window;
5. 3 and 2 of the onset definition are module constants; W + 1 and W + 2 derive from W;
6. by year, an onset counts in the year of s and an alert in the year of the target week of its t_c row;
7. the channel alert goes through the same code (`channel_positive` in events.py).

API frozen here (`src.forecasting.events`):
- `Alert(start, confirmation, end, continuity)`: `confirmation` is t_c, the k-th week of the run;
- `alerts(weeks, positive, zones, *, k) -> list[Alert]`: maximal runs of >= k positive emissions;
  continuity if the observed zone at t_c is "epidemia";
- `model_positive(p, *, tau)` (p >= tau) and `channel_positive(zones)` (zone == "alerta");
- `outbreak_onsets(weeks, exceed)` and `zone_entries(weeks, exceed)`: lists of dates;
- `period_onsets(rows, observed, *, window) -> DataFrame` with columns municipality_code, kind
  ("inicio" | "entrada"), week, evaluable, censoring (None | "ventana" | "confirmacion"); no barrier;
- `municipality_years(rows) -> int`;
- `evaluate_period(rows, positive, observed, *, k, window, access) -> PeriodEvents` with `onsets`
  (period_onsets columns plus warned, warned_by_judgeable, lead), `alerts` (municipality_code, start,
  confirmation, end, continuity, judgeable, false_alarm, followed_by_epidemic) and the `summary` mapping.

`rows` are the evaluated rows of a period (municipality_code, t, target_week, zona_t); `observed` is the
channel product as a DataFrame (municipality_code, week_start, target_u1e, channel_eligible). All weeks
are datetime.date Mondays. Synthetic period: one municipality, 52 emissions from F = 2019-11-11, target
weeks 2020-01-06 .. L = 2020-12-28 (development years, so no recipe is needed except in the barrier tests).
"""

from datetime import date, timedelta
import inspect
import math

import numpy as np
import pandas as pd
import pytest
import yaml

pytestmark = pytest.mark.acceptance

H = 8
CODE = "08001"
F = date(2019, 11, 11)          # first emission of the synthetic period
N_EMISSIONS = 52
LAST = 59                       # index of L = 2020-12-28, the last target week (51 + 8)
METRIC_KEYS = ("evaluable_onsets", "warned_onsets", "detection", "judgeable_alerts", "false_alarms",
               "municipality_years", "false_alarms_per_municipality_year", "alert_ratio",
               "continuity_alerts", "evaluable_entries", "warned_entries")


def _events():
    from src.forecasting import events

    return events


def _config():
    from src.forecasting.config import load_config

    return load_config()


def _week(i: int, first: date = F) -> date:
    return first + timedelta(weeks=i)


def _rows(first: date = F, zones: dict | None = None) -> pd.DataFrame:
    zones = zones or {}
    data = [(CODE, _week(i, first), _week(i + H, first), zones.get(i, "seguridad")) for i in range(N_EMISSIONS)]
    return pd.DataFrame(data, columns=["municipality_code", "t", "target_week", "zona_t"])


def _observed(ones=(), first: date = F, before: int = 30, after: int = 10, outside_ones=()) -> pd.DataFrame:
    """Observed series from first - before weeks to L + after weeks; indices relative to `first`."""
    marked = set(ones) | set(outside_ones)
    data = [(CODE, _week(i, first), int(i in marked), True) for i in range(-before, LAST + after + 1)]
    return pd.DataFrame(data, columns=["municipality_code", "week_start", "target_u1e", "channel_eligible"])


def _onset(s: int) -> set[int]:
    """Exceedance at s, s + 1 and s + 2 (with zeros before): an onset and an entry in s."""
    return {s, s + 1, s + 2}


def _positive(*alert_confirmations: int) -> np.ndarray:
    """Positive emissions at c - 1 and c for each confirmation index c (one k = 2 alert per c)."""
    positive = np.zeros(N_EMISSIONS, dtype=bool)
    for c in alert_confirmations:
        positive[[c - 1, c]] = True
    return positive


def _evaluate(positive, ones=(), zones=None, access=None, first: date = F, outside_ones=()):
    events, config = _events(), _config()
    if access is None:
        from src.forecasting.guard import RecipeAccess

        access = RecipeAccess(test_years=tuple(config.validacion.prueba),
                              channel_version=config.datos.version_dvc_canal,
                              recipe_path="receta_inexistente_f4.yaml")
    return events.evaluate_period(_rows(first, zones), positive, _observed(ones, first, outside_ones=outside_ones),
                                  k=config.decision.k, window=config.decision.W, access=access)


def _onset_row(result, s: int, kind: str = "inicio"):
    table = result.onsets
    match = table[(table["kind"] == kind) & (table["week"] == _week(s))]
    assert len(match) == 1, (kind, _week(s), len(match))
    return match.iloc[0]


def _alert_row(result, c: int):
    table = result.alerts
    match = table[table["confirmation"] == _week(c)]
    assert len(match) == 1, (_week(c), len(match))
    return match.iloc[0]


# --------------------------------------------------------------------------- configuration and signatures
def test_rule_parameters_come_from_configuration():
    decision = _config().decision
    assert (decision.k, decision.W) == (2, 12)


@pytest.mark.parametrize("function, names", [
    ("alerts", ("k",)),
    ("model_positive", ("tau",)),
    ("period_onsets", ("window",)),
    ("evaluate_period", ("k", "window", "access")),
])
def test_rule_parameters_are_keyword_only_without_defaults(function, names):
    parameters = inspect.signature(getattr(_events(), function)).parameters
    for name in names:
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, (function, name)
        assert parameters[name].default is inspect.Parameter.empty, (function, name)


# --------------------------------------------------------------------------- alerts (U4.2)
def _alerts(positive, zones=None):
    weeks = [_week(i) for i in range(len(positive))]
    zones = zones or ["seguridad"] * len(positive)
    return _events().alerts(weeks, np.asarray(positive, dtype=bool), zones, k=_config().decision.k)


def test_isolated_positive_emission_is_not_an_alert():
    assert _alerts([0, 1, 0, 0, 0, 0, 0]) == []
    found = _alerts([0, 1, 0, 0, 1, 1, 0])
    assert len(found) == 1
    assert (found[0].start, found[0].confirmation, found[0].end) == (_week(4), _week(5), _week(5))


def test_alert_is_dated_at_the_second_week_of_the_run():
    found = _alerts([0, 0, 0, 1, 1, 1, 0])
    assert len(found) == 1
    assert (found[0].start, found[0].confirmation, found[0].end) == (_week(3), _week(4), _week(5))


def test_a_maximal_run_is_one_alert_and_a_gap_splits_it():
    assert len(_alerts([1, 1, 1, 1, 1, 0, 0])) == 1
    split = _alerts([1, 1, 0, 1, 1, 0, 0])
    assert [a.confirmation for a in split] == [_week(1), _week(4)]


def test_continuity_depends_on_the_zone_at_confirmation():
    zones = ["seguridad", "epidemia", "alerta", "seguridad", "alerta", "epidemia", "seguridad"]
    found = _alerts([0, 1, 1, 0, 1, 1, 0], zones)
    assert [(a.confirmation, a.continuity) for a in found] == [(_week(2), False), (_week(5), True)]


def test_a_run_at_the_first_emission_starts_there():
    found = _alerts([1, 1, 0, 0, 0, 0, 0])
    assert [(a.start, a.confirmation) for a in found] == [(_week(0), _week(1))]


def test_emissions_must_be_consecutive_weeks():
    from src.epidemiology.config import EpidemiologyValidationError

    weeks = [_week(0), _week(1), _week(3)]
    with pytest.raises(EpidemiologyValidationError):
        _events().alerts(weeks, np.array([True, True, True]), ["seguridad"] * 3, k=2)


def test_emission_helpers():
    events = _events()
    assert events.model_positive(np.array([0.29, 0.3, 0.31]), tau=0.3).tolist() == [False, True, True]
    assert events.channel_positive(["seguridad", "alerta", "epidemia"]).tolist() == [False, True, False]


# --------------------------------------------------------------------------- onsets and entries (U4.2)
def _series(values):
    return [_week(i) for i in range(len(values))], np.asarray(values)


def test_onset_is_a_run_of_three_after_two_weeks_below():
    weeks, values = _series([0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0])
    assert _events().outbreak_onsets(weeks, values) == [_week(2)]
    assert _events().zone_entries(weeks, values) == [_week(2), _week(7), _week(12)]


def test_onset_needs_two_weeks_below_inside_the_series():
    weeks, values = _series([0, 1, 1, 1, 0, 0, 0])
    assert _events().outbreak_onsets(weeks, values) == []
    assert _events().zone_entries(weeks, values) == [_week(1)]
    weeks, values = _series([0, 0, 1, 0, 1, 1, 1, 0])
    assert _events().outbreak_onsets(weeks, values) == []        # only one week below before the run


def test_run_at_the_first_week_is_neither_onset_nor_entry():
    weeks, values = _series([1, 1, 1, 1, 0, 0])
    assert _events().outbreak_onsets(weeks, values) == []
    assert _events().zone_entries(weeks, values) == []


# --------------------------------------------------------------------------- matching (U4.3)
S = 30   # onset week used in the matching cases: 2020-06-08; s - 13 = 17 >= 0 and s + 2 = 32 <= 59


@pytest.mark.parametrize("confirmation, warned, lead, false_alarm", [
    (S - 12, True, 12, False),
    (S - 13, False, None, True),
    (S - 1, True, 1, False),
    (S, False, None, True),
])
def test_onset_is_warned_by_an_alert_confirmed_1_to_12_weeks_before(confirmation, warned, lead, false_alarm):
    result = _evaluate(_positive(confirmation), ones=_onset(S))
    onset = _onset_row(result, S)
    assert bool(onset["evaluable"]) and onset["censoring"] is None
    assert bool(onset["warned"]) is warned
    if lead is None:
        assert math.isnan(onset["lead"])
    else:
        assert onset["lead"] == lead
    assert bool(_alert_row(result, confirmation)["false_alarm"]) is false_alarm
    assert result.summary["detection"] == (1.0 if warned else 0.0)


def test_lead_is_measured_from_the_most_recent_alert_of_the_window():
    result = _evaluate(_positive(S - 10, S - 4), ones=_onset(S))
    assert _onset_row(result, S)["lead"] == 4
    assert result.summary["false_alarms"] == 0
    assert result.summary["alert_ratio"] == 0.5      # one warned onset / two judgeable alerts


def test_false_alarm_window_is_t_c_plus_1_to_t_c_plus_12():
    assert not bool(_alert_row(_evaluate(_positive(18), ones=_onset(30)), 18)["false_alarm"])   # onset at t_c + 12
    assert bool(_alert_row(_evaluate(_positive(17), ones=_onset(30)), 17)["false_alarm"])       # onset at t_c + 13


def test_false_alarm_breakdown_by_epidemic_weeks_that_follow():
    result = _evaluate(_positive(10, 30), ones={15})
    assert bool(_alert_row(result, 10)["false_alarm"]) and bool(_alert_row(result, 10)["followed_by_epidemic"])
    assert bool(_alert_row(result, 30)["false_alarm"]) and not bool(_alert_row(result, 30)["followed_by_epidemic"])
    assert result.summary["false_alarms"] == 2
    assert result.summary["municipality_years"] == 1
    assert result.summary["false_alarms_per_municipality_year"] == 2.0


# --------------------------------------------------------------------------- continuity (U4.3)
def test_continuity_alert_is_neither_detection_nor_false_alarm():
    result = _evaluate(_positive(S - 4, 45), ones=_onset(S), zones={S - 4: "epidemia", 45: "epidemia"})
    assert not bool(_onset_row(result, S)["warned"])
    for c in (S - 4, 45):
        row = _alert_row(result, c)
        assert bool(row["continuity"]) and not bool(row["false_alarm"])
    summary = result.summary
    assert (summary["detection"], summary["false_alarms"], summary["judgeable_alerts"],
            summary["continuity_alerts"]) == (0.0, 0, 0, 2)


def test_zone_at_the_start_of_the_run_does_not_make_continuity():
    result = _evaluate(_positive(S - 4), ones=_onset(S), zones={S - 5: "epidemia", S - 4: "alerta"})
    assert not bool(_alert_row(result, S - 4)["continuity"])
    assert bool(_onset_row(result, S)["warned"])


# --------------------------------------------------------------------------- censoring (U4.3)
@pytest.mark.parametrize("s, evaluable, censoring", [
    (13, True, None),            # s - 13 = F: the window and its first confirmation are covered
    (12, False, "ventana"),      # s - 13 = F - 1
    (57, True, None),            # s + 2 = L
    (58, False, "confirmacion"),  # s + 2 = L + 1
])
def test_onset_censoring_at_the_period_borders(s, evaluable, censoring):
    result = _evaluate(_positive(), ones=_onset(s))
    onset = _onset_row(result, s)
    assert bool(onset["evaluable"]) is evaluable
    assert onset["censoring"] == censoring
    assert result.summary["evaluable_onsets"] == int(evaluable)


def test_entries_are_censored_only_by_the_window():
    result = _evaluate(_positive(), ones={LAST})                 # an entry in L, not an onset
    entry = _onset_row(result, LAST, kind="entrada")
    assert bool(entry["evaluable"]) and entry["censoring"] is None
    assert result.summary["evaluable_entries"] == 1 and result.summary["evaluable_onsets"] == 0
    early = _onset_row(_evaluate(_positive(), ones={12}), 12, kind="entrada")
    assert not bool(early["evaluable"]) and early["censoring"] == "ventana"


@pytest.mark.parametrize("c, judgeable", [(LAST - 14, True), (LAST - 13, False)])
def test_alert_censoring_at_the_last_target_week(c, judgeable):
    result = _evaluate(_positive(c))
    row = _alert_row(result, c)
    assert bool(row["judgeable"]) is judgeable
    assert bool(row["false_alarm"]) is judgeable                 # no onset follows; censored alerts are not judged
    assert result.summary["judgeable_alerts"] == int(judgeable)
    assert result.summary["false_alarms"] == int(judgeable)


def test_unjudgeable_alert_warns_but_does_not_enter_the_alert_ratio():
    # Onset in s = 57 (s + 2 = L). Alert t_c = 50: t_c + 14 > L, not judgeable, but it warns the onset.
    # Alert t_c = 10: judgeable false alarm. Detection 1/1; alert ratio 0/1, not 1/1.
    result = _evaluate(_positive(10, 50), ones=_onset(57))
    onset = _onset_row(result, 57)
    assert bool(onset["warned"]) and not bool(onset["warned_by_judgeable"]) and onset["lead"] == 7
    assert not bool(_alert_row(result, 50)["judgeable"])
    summary = result.summary
    assert (summary["warned_onsets"], summary["detection"]) == (1, 1.0)
    assert (summary["judgeable_alerts"], summary["false_alarms"], summary["alert_ratio"]) == (1, 1, 0.0)


def test_onset_before_the_first_target_week_prevents_a_false_alarm_but_is_not_evaluated():
    # Onset in index 5 (2019-12-16): an emission week of the period, before its first target week.
    result = _evaluate(_positive(2), ones=_onset(5))
    assert not bool(_alert_row(result, 2)["false_alarm"])
    assert result.summary["evaluable_onsets"] == 0
    assert result.onsets[result.onsets["kind"] == "inicio"].empty


def test_observed_weeks_outside_the_period_do_not_change_evaluated_objects():
    positive = _positive(1, 10, 18, 40, 50)
    ones = _onset(13) | _onset(30) | _onset(57) | {44}
    outside = {-12, -11, -10, -3, -2, -1, LAST + 1, LAST + 2, LAST + 3, LAST + 7}
    base = _evaluate(positive, ones=ones)
    altered = _evaluate(positive, ones=ones, outside_ones=outside)

    def evaluated(result):
        onsets = result.onsets[result.onsets["evaluable"]].reset_index(drop=True)
        alerts = result.alerts[result.alerts["judgeable"]].reset_index(drop=True)
        return onsets, alerts

    for left, right in zip(evaluated(base), evaluated(altered)):
        pd.testing.assert_frame_equal(left, right)
    for key in METRIC_KEYS:
        a, b = base.summary[key], altered.summary[key]
        assert a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b)), key


# --------------------------------------------------------------------------- barrier (U4.3)
TEST_FIRST = date(2022, 11, 7)   # first emission whose target week is 2023-01-02


def _recipe_access(tmp_path, valid: bool):
    from src.forecasting.guard import RecipeAccess, recipe_hash

    config = _config()
    path = tmp_path / "receta.yaml"
    if valid:
        recipe = {
            "decision": {"tau": 0.3, "k": 2, "W": 12},
            "procedencia": {"commit_codigo": "a" * 40, "commit_contrato": "b" * 40,
                            "version_dvc_canal": config.datos.version_dvc_canal},
            "aprobacion": {"nombre": "Receta sintética de prueba (no es una aprobación real)",
                           "fecha": date(2026, 11, 13)},
        }
        recipe["hash_sha256"] = recipe_hash(recipe)
        path.write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return RecipeAccess(test_years=tuple(config.validacion.prueba), channel_version=config.datos.version_dvc_canal,
                        recipe_path=path)


@pytest.mark.parametrize("system", ["modelo", "canal"])
def test_test_period_detection_requires_the_frozen_recipe(tmp_path, system):
    from src.forecasting.guard import FrozenRecipeError

    zones = {i: ("alerta" if i in (9, 10) else "seguridad") for i in range(N_EMISSIONS)}
    rows = _rows(TEST_FIRST, zones)
    positive = (_positive(10) if system == "modelo" else _events().channel_positive(rows["zona_t"]))
    observed = _observed(_onset(30), TEST_FIRST)
    config = _config()
    with pytest.raises(FrozenRecipeError):
        _events().evaluate_period(rows, positive, observed, k=config.decision.k, window=config.decision.W,
                                  access=_recipe_access(tmp_path, valid=False))
    result = _events().evaluate_period(rows, positive, observed, k=config.decision.k, window=config.decision.W,
                                       access=_recipe_access(tmp_path, valid=True))
    assert result.summary["evaluable_onsets"] == 1


def test_counting_test_period_onsets_needs_no_recipe():
    table = _events().period_onsets(_rows(TEST_FIRST), _observed(_onset(30), TEST_FIRST), window=_config().decision.W)
    onsets = table[table["kind"] == "inicio"]
    assert onsets["week"].tolist() == [_week(30, TEST_FIRST)]
    assert onsets["evaluable"].tolist() == [True]
    assert _events().municipality_years(_rows(TEST_FIRST)) == 1
