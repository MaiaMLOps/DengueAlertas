"""F4 acceptance with the real channel product (--channel-parquet): onsets, entries and evaluable onsets per
period (contrato §2.2, §6.1, §7.4 and §10.13; especificación §6.F1 and §6.F4).

Literal figures:
- whole eligible series: 31 onsets and 174 entries by municipality (especificación §6.F1, contrato §2.2);
- per period: 11 onsets in development, 10 evaluable (Villavicencio 2019-01-28 censored by the window), 5 of
  them in 2019; 10 of 10 in test, none in Cali; 45 and 41 entries; 24 and 18 municipality-years
  (contrato §6.1 and §7.4). Onset dates as printed by docs/forecasting/verificacion/eventos_d8.py.
Evaluable entries (44 and 41) are not frozen: the contract does not report them.

The emissions of a period are the evaluated rows of `splits.yearly_trials` (D7.1 only changes training).
Test-period onsets are counted without a recipe; detection and false alarms on test rows go through the
barrier. The channel's event performance is computed here for the first time under §7.4 (contrato §10.15);
no figure of it is frozen.
"""

from datetime import date
import functools

import pandas as pd
import pytest

pytestmark = pytest.mark.acceptance

H = 8
DEVELOPMENT = (2019, 2022)
TEST = (2023, 2025)

ONSETS_AND_ENTRIES = {  # code: (onsets (Umaña, U1e), entries 0->1), whole eligible series
    "08001": (5, 28), "08758": (7, 46), "41001": (6, 26),
    "50001": (7, 31), "54001": (3, 30), "76001": (3, 13),
}
EVALUABLE_DEVELOPMENT = {
    ("08001", date(2020, 1, 27)), ("08001", date(2021, 8, 23)),
    ("08758", date(2019, 5, 27)), ("08758", date(2019, 11, 4)), ("08758", date(2022, 2, 28)),
    ("41001", date(2019, 2, 18)),
    ("50001", date(2019, 11, 25)), ("50001", date(2022, 10, 31)),
    ("54001", date(2021, 6, 21)),
    ("76001", date(2019, 12, 16)),
}
EVALUABLE_TEST = {
    ("08001", date(2023, 4, 10)),
    ("08758", date(2023, 7, 3)), ("08758", date(2024, 12, 23)), ("08758", date(2025, 5, 26)),
    ("41001", date(2023, 7, 17)), ("41001", date(2023, 8, 28)), ("41001", date(2024, 6, 3)),
    ("50001", date(2025, 12, 1)),
    ("54001", date(2024, 1, 15)), ("54001", date(2025, 2, 24)),
}


def _config():
    from src.forecasting.config import load_config

    return load_config()


@functools.lru_cache(maxsize=None)
def _product(channel_file: str):
    from src.forecasting.dataset import load_channel_product

    return load_channel_product(channel_file, expected_md5=_config().datos.version_dvc_canal)


@functools.lru_cache(maxsize=None)
def _observed(channel_file: str) -> pd.DataFrame:
    return _product(channel_file).to_pandas()


@functools.lru_cache(maxsize=None)
def _period_rows(channel_file: str, first: int, last: int) -> pd.DataFrame:
    from src.forecasting.dataset import build_examples
    from src.forecasting.splits import yearly_trials

    config = _config()
    examples = build_examples(_product(channel_file), h=H, canal=config.canal, features=config.features)
    trials = yearly_trials(examples, range(first, last + 1),
                           first_train_year=config.validacion.primer_anio_entrenamiento)
    labels = trials[0].evaluate.append([trial.evaluate for trial in trials[1:]])
    return examples.loc[labels]


def _period_onsets(channel_file, period):
    from src.forecasting.events import period_onsets

    return period_onsets(_period_rows(channel_file, *period), _observed(channel_file), window=_config().decision.W)


def _pairs(table) -> set:
    return set(zip(table["municipality_code"], table["week"]))


# --------------------------------------------------------------------------- whole series (U4.2)
def test_onsets_and_entries_on_the_whole_eligible_series(channel_path):
    from src.forecasting.events import outbreak_onsets, zone_entries

    observed = _observed(str(channel_path))
    eligible = observed[observed["channel_eligible"]].sort_values(["municipality_code", "week_start"])
    found = {}
    for code, group in eligible.groupby("municipality_code", sort=True):
        weeks, exceed = list(group["week_start"]), group["target_u1e"].to_numpy()
        found[code] = (len(outbreak_onsets(weeks, exceed)), len(zone_entries(weeks, exceed)))
    assert found == ONSETS_AND_ENTRIES
    assert sum(o for o, _ in found.values()) == 31
    assert sum(e for _, e in found.values()) == 174


# --------------------------------------------------------------------------- periods (U4.3)
def test_development_onsets_ten_of_eleven(channel_path):
    table = _period_onsets(str(channel_path), DEVELOPMENT)
    onsets = table[table["kind"] == "inicio"]
    assert len(onsets) == 11
    evaluable = onsets[onsets["evaluable"]]
    assert _pairs(evaluable) == EVALUABLE_DEVELOPMENT
    censored = onsets[~onsets["evaluable"]]
    assert list(zip(censored["municipality_code"], censored["week"], censored["censoring"])) == [
        ("50001", date(2019, 1, 28), "ventana")]
    assert sum(1 for week in evaluable["week"] if week.year == 2019) == 5


def test_test_onsets_ten_of_ten_counted_without_recipe(channel_path):
    table = _period_onsets(str(channel_path), TEST)
    onsets = table[table["kind"] == "inicio"]
    assert len(onsets) == 10
    assert bool(onsets["evaluable"].all())
    assert _pairs(onsets) == EVALUABLE_TEST
    assert "76001" not in set(onsets["municipality_code"])


@pytest.mark.parametrize("period, entries", [(DEVELOPMENT, 45), (TEST, 41)])
def test_entries_in_each_period(channel_path, period, entries):
    table = _period_onsets(str(channel_path), period)
    assert int((table["kind"] == "entrada").sum()) == entries


@pytest.mark.parametrize("period, expected", [(DEVELOPMENT, 24), (TEST, 18)])
def test_municipality_years(channel_path, period, expected):
    from src.forecasting.events import municipality_years

    assert municipality_years(_period_rows(str(channel_path), *period)) == expected


# --------------------------------------------------------------------------- channel through the same code
def _channel_events(channel_file, period, access):
    from src.forecasting.events import channel_positive, evaluate_period

    rows = _period_rows(channel_file, *period)
    decision = _config().decision
    return evaluate_period(rows, channel_positive(rows["zona_t"]), _observed(channel_file),
                           k=decision.k, window=decision.W, access=access)


def _access(recipe_path):
    from src.forecasting.guard import RecipeAccess

    config = _config()
    return RecipeAccess(test_years=tuple(config.validacion.prueba), channel_version=config.datos.version_dvc_canal,
                        recipe_path=recipe_path)


def test_channel_events_on_test_rows_require_the_recipe(channel_path, tmp_path):
    from src.forecasting.guard import FrozenRecipeError

    with pytest.raises(FrozenRecipeError):
        _channel_events(str(channel_path), TEST, _access(tmp_path / "receta_inexistente.yaml"))


def test_channel_events_in_development_are_consistent(channel_path, tmp_path):
    result = _channel_events(str(channel_path), DEVELOPMENT, _access(tmp_path / "receta_inexistente.yaml"))
    summary = result.summary
    assert summary["evaluable_onsets"] == 10
    assert summary["municipality_years"] == 24
    assert summary["continuity_alerts"] == 0          # the channel alerts in the alert zone, never in epidemic
    assert 0 <= summary["warned_onsets"] <= 10
    onsets = result.onsets[result.onsets["kind"] == "inicio"]
    assert _pairs(onsets[onsets["evaluable"]]) == EVALUABLE_DEVELOPMENT
