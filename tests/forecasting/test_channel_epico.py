"""Unit tests of the epiCo channel module with synthetic data (contrato §1; U1.2)."""

from datetime import date, timedelta
import math

import pyarrow as pa
import pytest

from src.epidemiology.channel_epico import (
    CHANNEL_SCHEMA, build_channel_table, channel_zone, epico_channel, t_critical,
)
from src.epidemiology.config import FIRST_LABEL_YEAR, EpidemiologyValidationError
from src.epidemiology.ewt import week_ordinal
from src.epidemiology.validation import SOURCE_SCHEMA

PARAMS = {"n_years": 5, "shift": 1, "alpha": 0.025}
VILLAVICENCIO_2019_01_28 = (35, 18, 15, 1, 3)


def make_panel(end, counts=lambda code, start: 4, codes=("08001",)):
    """Full W-SUN calendar from 2007-01-01, as in tests/epidemiology/conftest.py."""
    rows = []
    for code in codes:
        current = date(2007, 1, 1)
        while current <= end:
            value = counts(code, current)
            nominal = current + timedelta(days=6)
            effective = min(nominal, date(2025, 12, 31))
            days = (effective - current).days + 1
            rows.append({"municipality_code": code, "municipality": "Synthetic " + code,
                         "week_start": current, "population": 1000, "area_km2": 10.0,
                         "population_density_km2": 100.0, "temperature_c": 25.0,
                         "precipitation_mm": 0.0, "relative_humidity_pct": 70.0,
                         "confirmed_records_screening": value, "no_eligible_records_observed": value == 0,
                         "week_end_nominal": nominal, "week_end_effective": effective,
                         "n_days": days, "partial_week": days < 7})
            current += timedelta(days=7)
    return pa.Table.from_pylist(rows, schema=SOURCE_SCHEMA)


def build(table):
    return build_channel_table(table, first_year=FIRST_LABEL_YEAR, **PARAMS)


def rows_by_start(table):
    return {row["week_start"]: row for row in table.to_pylist()}


# --------------------------------------------------------------------------- formula
def test_parameters_are_keyword_only_without_defaults():
    with pytest.raises(TypeError):
        epico_channel(VILLAVICENCIO_2019_01_28, 5, 1, 0.025)
    with pytest.raises(TypeError):
        epico_channel(VILLAVICENCIO_2019_01_28, n_years=5, shift=1)
    with pytest.raises(TypeError):
        t_critical(5, 0.025)
    with pytest.raises(TypeError):
        t_critical(n_years=5)
    with pytest.raises(TypeError):
        build_channel_table(make_panel(date(2012, 12, 31)), n_years=5, shift=1, alpha=0.025)


def test_central_and_gsd_are_rounded_to_five_decimals():
    central, gsd, _ = epico_channel(VILLAVICENCIO_2019_01_28, **PARAMS)

    # Unrounded values: central 8.737628780..., gsd 1.193735311...
    assert central == 8.73763
    assert gsd == 1.19374
    for history in [(1, 2, 3, 4, 5), (0, 9, 2, 0, 31), (100, 250, 80, 410, 12)]:
        central, gsd, _ = epico_channel(history, **PARAMS)
        assert central == round(central, 5)
        assert gsd == round(gsd, 5)


def test_upper_uses_rounded_central_and_gsd():
    central, gsd, upper = epico_channel(VILLAVICENCIO_2019_01_28, **PARAMS)

    expected = central * math.exp(2.7764451051977996 * gsd / math.sqrt(5))
    assert upper == pytest.approx(expected, rel=1e-14)
    assert upper == pytest.approx(38.46939411375516, rel=1e-12)


def test_zero_central_gives_zero_upper():
    assert epico_channel((0, 0, 0, 0, 0), **PARAMS) == (0.0, 0.0, 0.0)


def test_constant_history_and_its_zone():
    central, gsd, upper = epico_channel((4, 4, 4, 4, 4), **PARAMS)

    assert (central, gsd, upper) == (4.0, 0.0, 4.0)
    assert channel_zone(4, central, upper) == "seguridad"
    assert channel_zone(5, central, upper) == "epidemia"


def test_zone_boundaries_are_strict_above():
    central, upper = 2.0, 5.5

    assert channel_zone(2, central, upper) == "seguridad"
    assert channel_zone(3, central, upper) == "alerta"
    assert channel_zone(5, central, 5.0) == "alerta"
    assert channel_zone(6, central, upper) == "epidemia"
    with pytest.raises(EpidemiologyValidationError):
        channel_zone(3, 5.0, 2.0)


def test_t_critical_uses_n_minus_one_degrees_of_freedom():
    assert t_critical(n_years=5, alpha=0.025) == pytest.approx(2.7764451051977996, abs=1e-12)
    assert t_critical(n_years=6, alpha=0.025) == pytest.approx(2.570582, abs=1e-6)


@pytest.mark.parametrize(
    ("history", "params"),
    [
        ((1, 2, 3, 4), PARAMS),
        ((1, 2, 3, 4, -1), PARAMS),
        ((1, 2, 3, 4, True), PARAMS),
        ((1, 2, 3, 4, 2.0), PARAMS),
        ((1, 2, 3, 4, 5), {**PARAMS, "alpha": 0.5}),
        ((1, 2, 3, 4, 5), {**PARAMS, "alpha": 0.0}),
        ((1, 2, 3, 4, 5), {**PARAMS, "shift": 0}),
        ((1, 2, 3, 4, 5), {**PARAMS, "n_years": 1}),
    ],
    ids=["short", "negative", "bool", "float", "alpha_half", "alpha_zero", "shift_zero", "n_one"],
)
def test_invalid_inputs_fail_clearly(history, params):
    with pytest.raises(ValueError):
        epico_channel(history, **params)


# --------------------------------------------------------------------------- panel
def test_week_53_falls_back_to_ordinal_52():
    # 2018-12-31 is ordinal 53; none of 2013-2017 has an ordinal 53.
    table = build(make_panel(date(2018, 12, 31),
                             counts=lambda code, start: 7 if week_ordinal(start) == 52 else 1))
    row = rows_by_start(table)[date(2018, 12, 31)]

    assert week_ordinal(date(2018, 12, 31)) == 53
    assert row["channel_eligible"] is True
    assert (row["epico_central"], row["epico_upper"]) == (7.0, 7.0)
    assert (row["target_u1e"], row["channel_zone"]) == (0, "seguridad")


def test_eligibility_nulls_and_types():
    source = make_panel(date(2025, 12, 29))
    table = build(source)
    rows = rows_by_start(table)

    assert table.select(SOURCE_SCHEMA.names).equals(source.replace_schema_metadata(None))
    assert table.schema.names == SOURCE_SCHEMA.names + CHANNEL_SCHEMA.names
    for field in CHANNEL_SCHEMA:
        assert table.schema.field(field.name).type == field.type
    for start in (date(2007, 1, 1), date(2011, 12, 26), date(2025, 12, 29)):
        row = rows[start]
        assert row["channel_eligible"] is False
        assert [row[name] for name in ("epico_central", "epico_upper", "target_u1e", "channel_zone")] == [None] * 4
    assert rows[date(2025, 12, 29)]["partial_week"] is True
    first = rows[date(2012, 1, 2)]
    assert first["channel_eligible"] is True
    assert (first["epico_central"], first["epico_upper"], first["target_u1e"], first["channel_zone"]) == (
        4.0, 4.0, 0, "seguridad")
    eligible = sum(row["channel_eligible"] for row in rows.values())
    assert eligible == table.num_rows - sum(start.year < FIRST_LABEL_YEAR for start in rows) - 1


def test_only_prior_years_enter_the_channel():
    def base(code, start):
        return 1 + (start.year * 7 + week_ordinal(start) * 13) % 23

    def altered(code, start):
        return base(code, start) * (10 if start.year >= 2016 else 1)

    original = rows_by_start(build(make_panel(date(2018, 12, 31), counts=base)))
    changed = rows_by_start(build(make_panel(date(2018, 12, 31), counts=altered)))

    for start, row in original.items():
        if start.year == 2016:
            assert (changed[start]["epico_central"], changed[start]["epico_upper"]) == (
                row["epico_central"], row["epico_upper"])
    assert any(changed[start]["epico_central"] != row["epico_central"]
               for start, row in original.items() if start.year == 2017)


def test_panel_requires_five_history_years():
    with pytest.raises(EpidemiologyValidationError):
        build_channel_table(make_panel(date(2012, 12, 31)), n_years=6, shift=1, alpha=0.025,
                            first_year=FIRST_LABEL_YEAR)
