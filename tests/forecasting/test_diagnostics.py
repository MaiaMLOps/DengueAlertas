"""Unit tests of src/forecasting/diagnostics.py beyond the F7 acceptance (synthetic data only)."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.epidemiology.config import EpidemiologyValidationError
from src.forecasting import diagnostics as d

WEEK = timedelta(weeks=1)


def _weeks(first: date, n: int) -> list[date]:
    return [first + i * WEEK for i in range(n)]


def test_episode_at_the_very_start_of_the_series_counts():
    weeks = _weeks(date(2016, 1, 4), 160)
    exceed = [1, 1, 1] + [0] * 157
    assert d.year_regimes(weeks, exceed, years=[2017]) == {2017: d.POST_EPIDEMICA}


def test_long_episode_spanning_two_years_and_post_epidemic_after():
    weeks = _weeks(date(2016, 1, 4), 260)
    exceed = np.zeros(260, dtype=int)
    exceed[90:130] = 1   # from 2017-09 to 2018-06
    assert d.year_regimes(weeks, exceed, years=[2017, 2018, 2019]) == {
        2017: d.EPIDEMICA, 2018: d.EPIDEMICA, 2019: d.POST_EPIDEMICA}


@pytest.mark.parametrize("years", [[], [2017.0], [True]])
def test_years_must_be_integers(years):
    weeks = _weeks(date(2016, 1, 4), 160)
    with pytest.raises(EpidemiologyValidationError):
        d.year_regimes(weeks, [0] * 160, years=years)


def test_exceed_must_be_binary():
    weeks = _weeks(date(2016, 1, 4), 160)
    with pytest.raises(EpidemiologyValidationError):
        d.year_regimes(weeks, [0] * 159 + [2], years=[2017])


def _table_and_predictions():
    """Two municipalities, three candidates of one family, three years: an out-of-year common choice exists."""
    codes, years = ("00001", "00002"), (2019, 2020, 2021)
    gains = {"c0": (0.10, 0.10, 0.10), "c1": (0.30, 0.00, 0.30), "c2": (0.00, 0.30, 0.00)}
    table, predictions, baselines = [], [], []
    for code in codes:
        for year_index, year in enumerate(years):
            targets = _weeks(date(year, 1, 7) if year == 2019 else date(year, 1, 6) if year == 2020
                             else date(year, 1, 4), 3)
            t = [w - 8 * WEEK for w in targets]
            for order, (name, gain) in enumerate(gains.items()):
                table.append({"municipality_code": code, "familia": "logistica", "candidato": name, "orden": order,
                              "parametros": f'{{"C": {order}}}', "anio": year, "logloss": 1 - gain[year_index],
                              "mejora_b2": gain[year_index]})
                predictions.append(pd.DataFrame({"municipality_code": code, "t": t, "target_week": targets,
                                                 "familia": "logistica", "candidato": name, "anio": year,
                                                 "y": [0, 1, 0], "p": [0.1 * (order + 1)] * 3}))
            for system in ("B2", "B1"):
                baselines.append(pd.DataFrame({"municipality_code": code, "t": t, "target_week": targets,
                                               "familia": system, "candidato": system, "anio": year,
                                               "y": [0, 1, 0], "p": [0.5] * 3}))
    return (pd.DataFrame(table), pd.concat(predictions, ignore_index=True),
            pd.concat(baselines, ignore_index=True))


def test_development_predictions_use_the_out_of_year_common_choice():
    table, predictions, baselines = _table_and_predictions()
    long = d.development_predictions(table, predictions, baselines, familia="logistica")
    assert list(long.columns) == ["municipality_code", "t", "target_week", "anio", "y", "sistema", "candidato", "p"]
    model = long[long["sistema"] == "modelo"]
    # Without 2019: c1 = (0 + 0.3)/2 = c2 = (0.3 + 0)/2 = 0.15 > c0 = 0.10, an exact tie that goes to the smaller
    # order (c1). Without 2020: c1 = 0.30. Without 2021: the same tie as without 2019, so c1 again.
    assert {year: set(model.loc[model["anio"] == year, "candidato"]) for year in (2019, 2020, 2021)} == {
        2019: {"c1"}, 2020: {"c1"}, 2021: {"c1"}}
    for system in ("modelo", "B2", "B1"):
        own = long[long["sistema"] == system]
        assert len(own) == 18 and not own.duplicated(["municipality_code", "t"]).any()


def test_development_predictions_require_the_baselines_of_every_cell():
    table, predictions, baselines = _table_and_predictions()
    missing = baselines.drop(baselines.index[(baselines["familia"] == "B1") & (baselines["anio"] == 2020)][:1])
    with pytest.raises(EpidemiologyValidationError):
        d.development_predictions(table, predictions, missing, familia="logistica")


def test_write_diagnostics_never_overwrites(tmp_path):
    frames = {name: pd.DataFrame({"x": [1, 2]}) for name in d.INPUTS}
    diagnostics = d.Diagnostics(**frames)
    paths = d.write_diagnostics(diagnostics, tmp_path / "salida")
    assert [path.name for path in paths] == [f"diag_{name}.csv" for name in d.INPUTS]
    assert all(path.is_file() for path in paths)
    with pytest.raises(EpidemiologyValidationError):
        d.write_diagnostics(diagnostics, tmp_path / "salida")
