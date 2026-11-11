"""Unit tests of the GRU input (contrato §5.3; U2.5) with synthetic data.

History years 2007-2011 repeat a per-ordinal count c(ordinal) (gsd = 0, so U1e = central = c for 2012
weeks) and a per-year climate whose clim_s for any 2012 week is 22, 30 and 64.
"""

from datetime import date, timedelta
import json
import math

import numpy as np
import pyarrow as pa
import pytest

from src.epidemiology.channel_epico import ZONES, build_channel_table, channel_for_week, channel_zone
from src.epidemiology.config import FIRST_LABEL_YEAR, EpidemiologyValidationError
from src.epidemiology.ewt import week_ordinal
from src.epidemiology.validation import SOURCE_SCHEMA
from src.forecasting.config import ChannelSection, FeatureSection
from src.forecasting.dataset import (
    GRU_SEQUENCE_CHANNELS, GRU_STATIC, build_examples, build_gru_examples, build_gru_input,
)

CANAL = ChannelSection(n=5, desplazamiento=1, cuantil_t=0.025)
FEATURES_CFG = FeatureSection(w=4, bloques={"m1": (-3, 0), "m2": (-7, -4), "m3": (-11, -8)}, ventana_semanas=12)
CODE = "08001"
T = date(2012, 6, 4)
CH = {name: i for i, name in enumerate(GRU_SEQUENCE_CHANNELS)}


def count_of(start: date) -> int:
    if start.year < 2012:
        return week_ordinal(start) % 7 + 1
    return start.toordinal() % 11  # varies week to week in 2012-2013


def climate_of(start: date) -> tuple[float, float, float]:
    if start.year < 2012:
        k = start.year - 2007
        return 20.0 + k, 10.0 * (k + 1), 60.0 + 2 * k
    return 25.0 + (start.toordinal() % 3), 5.0 * (start.toordinal() % 4), 70.0 - (start.toordinal() % 5)


def panel(overrides=None) -> pa.Table:
    overrides = overrides or {}
    rows, current = [], date(2007, 1, 1)
    while current <= date(2013, 12, 30):
        value = overrides.get(current, count_of(current))
        temperature, precipitation, humidity = climate_of(current)
        nominal = current + timedelta(days=6)
        rows.append({"municipality_code": CODE, "municipality": "Synthetic", "week_start": current,
                     "population": 1000, "area_km2": 10.0, "population_density_km2": 100.0,
                     "temperature_c": temperature, "precipitation_mm": precipitation,
                     "relative_humidity_pct": humidity, "confirmed_records_screening": value,
                     "no_eligible_records_observed": value == 0, "week_end_nominal": nominal,
                     "week_end_effective": nominal, "n_days": 7, "partial_week": False})
        current += timedelta(weeks=1)
    return pa.Table.from_pylist(rows, schema=SOURCE_SCHEMA)


def gru(table, origins, h=8):
    return build_gru_input(table, origins, h=h, canal=CANAL, features=FEATURES_CFG)


def test_shape_and_names():
    result = gru(panel(), [(CODE, T), (CODE, T + timedelta(weeks=1))])

    assert result.sequence.shape == (2, 12, 8) and result.static.shape == (2, 3)
    assert len(GRU_SEQUENCE_CHANNELS) == 8 and GRU_STATIC == ("cambio_umbral", "sem_sin", "sem_cos")
    assert list(result.keys.columns) == ["municipality_code", "t"]


def test_time_axis_is_chronological():
    sequence = gru(panel(), [(CODE, T)]).sequence[0]

    for j in range(12):
        week = T - timedelta(weeks=11 - j)
        assert sequence[j, CH["log_c"]] == math.log1p(count_of(week)), j


def test_one_step_by_hand():
    sequence = gru(panel(), [(CODE, T)]).sequence[0]
    week = T - timedelta(weeks=2)  # index 9
    upper = week_ordinal(week) % 7 + 1  # constant history: U1e = c(ordinal)
    temperature, precipitation, humidity = climate_of(week)

    assert sequence[9, CH["pos"]] == pytest.approx(math.log1p(count_of(week)) - math.log1p(upper), rel=1e-12, abs=1e-15)
    assert sequence[9, CH["anom_temp"]] == pytest.approx(temperature - 22.0, rel=1e-12, abs=1e-12)
    assert sequence[9, CH["anom_prec"]] == pytest.approx(precipitation - 30.0, rel=1e-12, abs=1e-12)
    assert sequence[9, CH["anom_hum"]] == pytest.approx(humidity - 64.0, rel=1e-12, abs=1e-12)


def test_one_hot_follows_the_channel_zone_in_every_step():
    # Spread history at the ordinal of T gives a channel with central < U1e; C_T between them is "alerta".
    history = {}
    for k, value in enumerate((2, 30, 5, 60, 9)):
        first = date(2007 + k, 1, 1)
        history[first + timedelta(days=(-first.weekday()) % 7) + timedelta(weeks=week_ordinal(T) - 1)] = value
    table = panel(overrides=history)
    counts = dict(zip(table.column("week_start").to_pylist(), table.column("confirmed_records_screening").to_pylist()))
    central, upper = channel_for_week(counts, T, n_years=5, shift=1, alpha=0.025)
    table = panel(overrides={**history, T: math.floor(central) + 1})
    counts[T] = math.floor(central) + 1
    sequence = gru(table, [(CODE, T)]).sequence[0]

    seen = set()
    for j in range(12):
        week = T - timedelta(weeks=11 - j)
        zone = channel_zone(counts[week], *channel_for_week(counts, week, n_years=5, shift=1, alpha=0.025))
        expected = [1.0 if zone == name else 0.0 for name in ZONES]
        assert list(sequence[j, 2:5]) == expected, j
        seen.add(zone)
    assert "alerta" in seen


def test_static_inputs():
    static = gru(panel(), [(CODE, T)]).static[0]
    target = T + timedelta(weeks=8)
    angle = 2 * math.pi * (week_ordinal(target) - 1) / 52.18
    upper_t, upper_target = week_ordinal(T) % 7 + 1, week_ordinal(target) % 7 + 1

    assert static[0] == pytest.approx(math.log1p(upper_target) - math.log1p(upper_t), rel=1e-12, abs=1e-15)
    assert (static[1], static[2]) == (math.sin(angle), math.cos(angle))


def test_twelve_weeks_of_channel_are_required():
    # t-11 = 2011-11-21 has no five-year history: the GRU needs the channel in all twelve weeks.
    with pytest.raises(EpidemiologyValidationError, match="Channel"):
        gru(panel(), [(CODE, date(2012, 2, 6))])


def test_empty_origins_give_empty_arrays():
    result = gru(panel(), [])

    assert result.sequence.shape == (0, 12, 8) and result.static.shape == (0, 3) and len(result.keys) == 0


def test_examples_have_the_same_rows_as_tabular_examples():
    product = build_channel_table(panel(), n_years=5, shift=1, alpha=0.025, first_year=FIRST_LABEL_YEAR)
    product = product.replace_schema_metadata(
        {b"denguealertas.channel": json.dumps({"n_years": 5, "shift": 1, "alpha": 0.025}).encode()})
    tabular = build_examples(product, h=8, canal=CANAL, features=FEATURES_CFG)
    sequences = build_gru_examples(product, h=8, canal=CANAL, features=FEATURES_CFG)

    assert sequences.keys.equals(tabular[["municipality_code", "t", "target_week", "y"]])
    assert sequences.sequence.shape == (len(tabular), 12, 8)
    assert np.isfinite(sequences.sequence).all()
