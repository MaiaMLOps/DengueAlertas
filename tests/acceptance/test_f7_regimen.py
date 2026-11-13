"""F7 acceptance with synthetic series: regime of a municipality-year (contrato §8.8 and §10.22, D9.1;
especificación §6.F7). Approved by Jesús on 2026-11-12 (U7.1).

Rule (contrato §8.8): an episode is a run of >= 3 consecutive weeks in the epidemic zone (target_u1e = 1, i.e.
C > U1e) of the whole eligible series of the municipality; episodes are identified on the whole series and then
each calendar year (year of the Monday, W-SUN) is classified:
- `epidemica`: at least one week of the year belongs to an episode;
- `post_epidemica`: not epidemic and the previous year is epidemic (same rule; it does not chain);
- `tranquila`: the rest.
The classification needs the year and its previous year covered by the series; otherwise it fails clearly.

API frozen here (`src.forecasting.diagnostics`):
- constants EPIDEMICA = "epidemica", POST_EPIDEMICA = "post_epidemica", TRANQUILA = "tranquila",
  EPISODE_MIN_RUN = 3;
- `year_regimes(weeks, exceed, *, years) -> dict[int, str]` for one municipality's consecutive eligible weeks;
- `regime_table(observed, *, years) -> DataFrame` with columns municipality_code, anio, regimen and
  semanas_en_episodio, from the channel product (only rows with channel_eligible = True).
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from src.epidemiology.config import EpidemiologyValidationError

pytestmark = pytest.mark.acceptance

WEEK = timedelta(weeks=1)


def _series(first: date, last: date, exceed_weeks=()):
    weeks, current = [], first
    while current <= last:
        weeks.append(current)
        current += WEEK
    marked = set(exceed_weeks)
    return weeks, [1 if week in marked else 0 for week in weeks]


def _run(start: date, length: int):
    return [start + i * WEEK for i in range(length)]


def test_constants():
    from src.forecasting import diagnostics as d

    assert (d.EPIDEMICA, d.POST_EPIDEMICA, d.TRANQUILA) == ("epidemica", "post_epidemica", "tranquila")
    assert d.EPISODE_MIN_RUN == 3


def test_episode_crossing_the_year_makes_both_years_epidemic():
    """especificación §6.F7: a 3-week episode across the change of year (2 weeks in 2018, 1 in 2019)."""
    from src.forecasting.diagnostics import year_regimes

    weeks, exceed = _series(date(2016, 1, 4), date(2020, 12, 28), _run(date(2018, 12, 24), 3))
    assert date(2018, 12, 31).weekday() == 0 and date(2018, 12, 31).year == 2018
    regimes = year_regimes(weeks, exceed, years=[2017, 2018, 2019, 2020])
    assert regimes == {2017: "tranquila", 2018: "epidemica", 2019: "epidemica", 2020: "post_epidemica"}


def test_post_epidemic_year_does_not_chain():
    """especificación §6.F7: a year without episode after an epidemic one is post-epidemic; the next one is quiet."""
    from src.forecasting.diagnostics import year_regimes

    weeks, exceed = _series(date(2016, 1, 4), date(2020, 12, 28), _run(date(2017, 6, 5), 3))
    regimes = year_regimes(weeks, exceed, years=[2017, 2018, 2019])
    assert regimes == {2017: "epidemica", 2018: "post_epidemica", 2019: "tranquila"}


def test_run_of_two_weeks_is_not_an_episode():
    """especificación §6.F7: a year with a 2-week run is not epidemic; neither are two 2-week runs one week apart."""
    from src.forecasting.diagnostics import year_regimes

    weeks, exceed = _series(date(2016, 1, 4), date(2020, 12, 28), _run(date(2018, 3, 5), 2))
    assert year_regimes(weeks, exceed, years=[2018, 2019]) == {2018: "tranquila", 2019: "tranquila"}

    split = _run(date(2018, 3, 5), 2) + _run(date(2018, 3, 26), 2)
    weeks, exceed = _series(date(2016, 1, 4), date(2020, 12, 28), split)
    assert year_regimes(weeks, exceed, years=[2018, 2019]) == {2018: "tranquila", 2019: "tranquila"}


def test_run_of_exactly_three_weeks_is_an_episode():
    from src.forecasting.diagnostics import year_regimes

    weeks, exceed = _series(date(2016, 1, 4), date(2020, 12, 28), _run(date(2018, 3, 5), 3))
    assert year_regimes(weeks, exceed, years=[2018, 2019]) == {2018: "epidemica", 2019: "post_epidemica"}


def test_episodes_use_the_whole_series_not_only_the_requested_years():
    """An episode in a year that is not requested still makes its next year post-epidemic."""
    from src.forecasting.diagnostics import year_regimes

    weeks, exceed = _series(date(2016, 1, 4), date(2020, 12, 28), _run(date(2018, 10, 1), 5))
    assert year_regimes(weeks, exceed, years=[2019]) == {2019: "post_epidemica"}


@pytest.mark.parametrize("years", [[2016], [2021], [2015]])
def test_year_or_previous_year_not_covered_fails(years):
    from src.forecasting.diagnostics import year_regimes

    weeks, exceed = _series(date(2016, 1, 4), date(2020, 12, 28))
    with pytest.raises(EpidemiologyValidationError):
        year_regimes(weeks, exceed, years=years)


def test_non_consecutive_weeks_fail():
    from src.forecasting.diagnostics import year_regimes

    weeks, exceed = _series(date(2016, 1, 4), date(2020, 12, 28))
    del weeks[100], exceed[100]
    with pytest.raises(EpidemiologyValidationError):
        year_regimes(weeks, exceed, years=[2018])


def test_regime_table_uses_only_eligible_weeks():
    """Ineligible weeks (as 2007-2011 in the product) are not part of the series, even if flagged."""
    from src.forecasting.diagnostics import regime_table

    frames = []
    for code, episode in (("00001", _run(date(2017, 12, 25), 3)), ("00002", [])):
        weeks, exceed = _series(date(2015, 1, 5), date(2019, 12, 30), episode)
        eligible = [week >= date(2016, 1, 4) for week in weeks]
        exceed = [1 if (not ok and week.year == 2015) else value for week, value, ok in zip(weeks, exceed, eligible)]
        frames.append(pd.DataFrame({"municipality_code": code, "week_start": weeks, "target_u1e": exceed,
                                    "channel_eligible": eligible}))
    observed = pd.concat(frames, ignore_index=True)

    table = regime_table(observed, years=[2017, 2018, 2019])
    assert list(table.columns) == ["municipality_code", "anio", "regimen", "semanas_en_episodio"]
    got = {(row.municipality_code, row.anio): (row.regimen, row.semanas_en_episodio) for row in table.itertuples()}
    assert got == {
        ("00001", 2017): ("epidemica", 1), ("00001", 2018): ("epidemica", 2), ("00001", 2019): ("post_epidemica", 0),
        ("00002", 2017): ("tranquila", 0), ("00002", 2018): ("tranquila", 0), ("00002", 2019): ("tranquila", 0)}
    with pytest.raises(EpidemiologyValidationError):
        regime_table(observed, years=[2016])   # 2015 is not eligible, so the previous year is not covered
