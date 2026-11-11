"""Unit tests of build_features and build_examples, part 1 (variables 1-6), with synthetic data (U2.3).

Synthetic municipality: history years 2007-2011 repeat a per-ordinal count c(ordinal), so for 2012
weeks the channel has gsd = 0 and central = U1e = c(ordinal) exactly; values follow by hand.
"""

from datetime import date, timedelta
import json
import math

import pyarrow as pa
import pytest

from src.epidemiology.channel_epico import build_channel_table, channel_for_week
from src.epidemiology.config import FIRST_LABEL_YEAR, EpidemiologyValidationError
from src.epidemiology.ewt import week_ordinal
from src.epidemiology.validation import SOURCE_SCHEMA
from src.forecasting.config import ChannelSection, FeatureSection
from src.forecasting.dataset import ANNUAL_PERIOD_WEEKS, FEATURES, build_examples, build_features

CANAL = ChannelSection(n=5, desplazamiento=1, cuantil_t=0.025)
FEATURES_CFG = FeatureSection(w=4, bloques={"m1": (-3, 0), "m2": (-7, -4), "m3": (-11, -8)}, ventana_semanas=12)
IMPLEMENTED = list(FEATURES[:6])
CODE = "08001"


def history_count(start: date) -> int:
    return week_ordinal(start) % 7 + 1


def panel(last=date(2013, 12, 30), overrides=None) -> pa.Table:
    overrides = overrides or {}
    rows, current = [], date(2007, 1, 1)
    while current <= last:
        value = overrides.get(current, history_count(current) if current.year < 2012 else 3)
        nominal = current + timedelta(days=6)
        rows.append({"municipality_code": CODE, "municipality": "Synthetic", "week_start": current,
                     "population": 1000, "area_km2": 10.0, "population_density_km2": 100.0,
                     "temperature_c": 25.0, "precipitation_mm": 0.0, "relative_humidity_pct": 70.0,
                     "confirmed_records_screening": value, "no_eligible_records_observed": value == 0,
                     "week_end_nominal": nominal, "week_end_effective": nominal, "n_days": 7,
                     "partial_week": False})
        current += timedelta(weeks=1)
    return pa.Table.from_pylist(rows, schema=SOURCE_SCHEMA)


def product(table: pa.Table, alpha=0.025) -> pa.Table:
    built = build_channel_table(table, n_years=5, shift=1, alpha=0.025, first_year=FIRST_LABEL_YEAR)
    return built.replace_schema_metadata(
        {b"denguealertas.channel": json.dumps({"n_years": 5, "shift": 1, "alpha": alpha}).encode()})


def features_of(table, origins, h=8, features=FEATURES_CFG):
    return build_features(table, origins, h=h, canal=CANAL, features=features)


# --------------------------------------------------------------------------- constants
def test_feature_names_and_period():
    assert len(FEATURES) == 15 and len(set(FEATURES)) == 15
    assert ANNUAL_PERIOD_WEEKS == 52.18


# --------------------------------------------------------------------------- rows
def test_origin_range_follows_the_configured_window():
    examples = build_examples(product(panel()), h=8, canal=CANAL, features=FEATURES_CFG)

    # Eligible weeks: 2012-01-02 .. 2013-12-30 (105). First t = 2012-01-02 + 11 weeks; last t + 8 = 2013-12-30.
    assert examples["t"].iloc[0] == date(2012, 3, 19)
    assert examples["t"].iloc[-1] == date(2013, 11, 4)
    assert len(examples) == 105 - 11 - 8
    assert list(examples.columns) == ["municipality_code", "t", "target_week", "y", *IMPLEMENTED]


def test_origin_range_moves_with_the_window_and_h():
    shorter = FeatureSection(w=4, bloques={"m1": (-3, 0)}, ventana_semanas=6)
    examples = build_examples(product(panel()), h=4, canal=CANAL, features=shorter)

    assert examples["t"].iloc[0] == date(2012, 2, 6)
    assert examples["t"].iloc[-1] == date(2013, 12, 2)
    assert (examples["target_week"] == [t + timedelta(weeks=4) for t in examples["t"]]).all()


def test_target_is_the_product_label_at_t_plus_h():
    target = date(2012, 7, 2)
    # 50 is far above its U1e = c(ordinal) <= 7; 0 is never above U1e = c(ordinal) >= 1.
    table = panel(overrides={target: 50, target + timedelta(weeks=1): 0})
    examples = build_examples(product(table), h=8, canal=CANAL, features=FEATURES_CFG)
    labelled = dict(zip(examples["target_week"], examples["y"]))

    assert labelled[target] == 1
    assert labelled[target + timedelta(weeks=1)] == 0
    assert str(examples["y"].dtype) == "int8"


def test_parameter_guard_rejects_a_different_channel():
    with pytest.raises(EpidemiologyValidationError, match="differ"):
        build_examples(product(panel(), alpha=0.05), h=8, canal=CANAL, features=FEATURES_CFG)
    with pytest.raises(EpidemiologyValidationError, match="metadata"):
        build_examples(product(panel()).replace_schema_metadata(None), h=8, canal=CANAL, features=FEATURES_CFG)


# --------------------------------------------------------------------------- values by hand
def test_variables_one_to_six_by_hand():
    t = date(2012, 6, 4)
    recent = [t - timedelta(weeks=k) for k in range(4)]
    counts = {recent[0]: 9, recent[1]: 2, recent[2]: 0, recent[3]: 5, t - timedelta(weeks=4): 1}
    row = features_of(panel(overrides=counts), [(CODE, t)]).iloc[0]
    upper = {s: history_count(s) for s in [*recent, t + timedelta(weeks=8)]}  # gsd = 0: U1e = central = c(ordinal)

    assert row["zona_t"] == ("epidemia" if 9 > upper[t] else "seguridad")
    assert row["posicion_4s"] == pytest.approx(
        sum(math.log(counts[s] + 1) - math.log(upper[s] + 1) for s in recent) / 4, rel=1e-12)
    assert row["crecimiento_4s"] == pytest.approx(math.log(10) - math.log(2), rel=1e-12)
    assert row["cambio_umbral"] == pytest.approx(
        math.log(upper[t + timedelta(weeks=8)] + 1) - math.log(upper[t] + 1), rel=1e-12, abs=1e-15)
    angle = 2 * math.pi * (week_ordinal(t + timedelta(weeks=8)) - 1) / 52.18
    assert (row["sem_sin"], row["sem_cos"]) == (math.sin(angle), math.cos(angle))


def same_ordinal(year: int, ordinal: int) -> date:
    first = date(year, 1, 1)
    return first + timedelta(days=(-first.weekday()) % 7) + timedelta(weeks=ordinal - 1)


def test_zone_equal_to_a_constant_channel_is_seguridad():
    t = date(2012, 6, 4)  # constant history: central = U1e = c(ordinal)
    row = features_of(panel(overrides={t: history_count(t)}), [(CODE, t)]).iloc[0]

    assert row["zona_t"] == "seguridad"


@pytest.mark.parametrize("position, zone", [("below_central", "seguridad"), ("above_central", "alerta"),
                                            ("below_upper", "alerta"), ("above_upper", "epidemia")])
def test_zone_against_a_spread_channel(position, zone):
    t = date(2012, 6, 4)
    history = {same_ordinal(2007 + k, week_ordinal(t)): value for k, value in enumerate((2, 30, 5, 60, 9))}
    table = panel(overrides=history)
    counts = dict(zip(table.column("week_start").to_pylist(), table.column("confirmed_records_screening").to_pylist()))
    central, upper = channel_for_week(counts, t, n_years=5, shift=1, alpha=0.025)
    value = {"below_central": math.floor(central), "above_central": math.floor(central) + 1,
             "below_upper": math.floor(upper), "above_upper": math.floor(upper) + 1}[position]
    assert central < math.floor(central) + 1 <= math.floor(upper) < upper  # the four values are distinct zones

    row = features_of(panel(overrides={**history, t: value}), [(CODE, t)]).iloc[0]

    assert row["zona_t"] == zone


def test_target_week_with_ordinal_53():
    t = date(2012, 11, 5)  # t + 8 = 2012-12-31, ordinal 53
    row = features_of(panel(), [(CODE, t)]).iloc[0]

    assert row["sem_sin"] == math.sin(2 * math.pi * 52 / ANNUAL_PERIOD_WEEKS)
    assert row["sem_cos"] == math.cos(2 * math.pi * 52 / ANNUAL_PERIOD_WEEKS)


def test_channel_weeks_beyond_the_base_are_computed_from_history():
    # Base ends at t; t + 8 is not a row, its U1e comes from 2008-2012 weeks <= t.
    t = date(2012, 10, 29)
    cut = panel(last=t)
    full = panel()

    assert features_of(cut, [(CODE, t)]).equals(features_of(full, [(CODE, t)]))


def test_no_missing_values():
    examples = build_examples(product(panel()), h=8, canal=CANAL, features=FEATURES_CFG)

    assert not examples.isna().any().any()


# --------------------------------------------------------------------------- errors instead of NaN
@pytest.mark.parametrize("origin, message", [
    ((CODE, date(2011, 6, 6)), "Channel .* unavailable"),          # no five-year history
    ((CODE, date(2007, 1, 15)), "unavailable"),                    # C_{t-4} before the base
    ((CODE, date(2014, 1, 6)), "Count C unavailable|Channel"),     # origin after the base
    (("99999", date(2012, 6, 4)), "not in the base"),
    ((CODE, date(2012, 6, 5)), "Monday"),
])
def test_missing_inputs_raise_clear_errors(origin, message):
    with pytest.raises(EpidemiologyValidationError, match=message):
        features_of(panel(), [origin])


def test_duplicate_origins_and_invalid_h_fail():
    with pytest.raises(EpidemiologyValidationError, match="Duplicate"):
        features_of(panel(), [(CODE, date(2012, 6, 4))] * 2)
    with pytest.raises(EpidemiologyValidationError, match="h must be"):
        features_of(panel(), [(CODE, date(2012, 6, 4))], h=0)


def test_base_with_channel_columns_is_rejected():
    with pytest.raises(EpidemiologyValidationError, match="fifteen original columns"):
        features_of(product(panel()), [(CODE, date(2012, 6, 4))])
