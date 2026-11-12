"""F3 acceptance: yearly trials by the year of the target week (contrato §6.1 amended by D7.1, §10.11;
especificación §6.F3; desviaciones DV-01), with h = 8.

API frozen here: `yearly_trials(examples, years, *, first_train_year)` returns one `Trial` per
(municipality, year) with attributes `year`, `municipality_code`, `t0`, `train` and `evaluate`; `train`
and `evaluate` are index labels of `examples` (the rows of `build_examples`, never rebuilt here).

The year of a row is the calendar year of the Monday of its target week t + h. In the trial of year Y
and a municipality, the evaluated rows are those whose target week falls in Y; t0 is the earliest
emission t among them; training uses the rows with target year >= first_train_year and t + h <= t0
(D7.1: only labels that existed at the trial's first emission).

Literal figures:
- development 2019–2022: 208 evaluated rows per municipality, 1,248 in total, 319 positive;
- test 2023–2025: 156 per municipality, 936 in total, 322 positive (contrato §6.1, especificación §6.F3);
- trial 2019: t0 = 2018-11-12; last training emission 2018-09-17 (target 2018-11-12); emissions from
  2018-09-24 to 2018-11-05 excluded from the 2019 training and included in 2020; the emission of
  2018-11-26 (target 2019-01-21) is evaluated in 2019 (contrato §6.1, D7.1 example).
"""

from datetime import date, timedelta
import functools

import pytest

pytestmark = pytest.mark.acceptance

H = 8
CODES = ("08001", "08758", "41001", "50001", "54001", "76001")
DEVELOPMENT = (2019, 2022)
TEST = (2023, 2025)
FIRST_TRAIN_YEAR = 2012


def _config():
    from src.forecasting.config import load_config

    return load_config()


@functools.lru_cache(maxsize=None)
def _examples(channel_file: str):
    from src.forecasting.dataset import build_examples, load_channel_product

    config = _config()
    product = load_channel_product(channel_file, expected_md5=config.datos.version_dvc_canal)
    return build_examples(product, h=H, canal=config.canal, features=config.features)


@functools.lru_cache(maxsize=None)
def _trials(channel_file: str, first: int, last: int):
    from src.forecasting.splits import yearly_trials

    config = _config()
    return tuple(yearly_trials(_examples(channel_file), range(first, last + 1),
                               first_train_year=config.validacion.primer_anio_entrenamiento))


def _target_year(examples):
    return examples["target_week"].map(lambda week: week.year)


def _trial(trials, code, year):
    matches = [trial for trial in trials if trial.municipality_code == code and trial.year == year]
    assert len(matches) == 1, (code, year, len(matches))
    return matches[0]


def test_configured_periods_match_contract():
    validation = _config().validacion
    assert tuple(validation.desarrollo) == DEVELOPMENT
    assert tuple(validation.prueba) == TEST
    assert validation.primer_anio_entrenamiento == FIRST_TRAIN_YEAR


@pytest.mark.parametrize("period, per_municipality, total, positives",
                         [(DEVELOPMENT, 208, 1248, 319), (TEST, 156, 936, 322)], ids=["desarrollo", "prueba"])
def test_evaluated_rows_per_period(channel_path, period, per_municipality, total, positives):
    examples = _examples(str(channel_path))
    trials = _trials(str(channel_path), *period)
    years = range(period[0], period[1] + 1)

    assert sorted((trial.municipality_code, trial.year) for trial in trials) == \
        sorted((code, year) for code in CODES for year in years)
    evaluated = [label for trial in trials for label in trial.evaluate]
    assert len(evaluated) == len(set(evaluated)) == total
    rows = examples.loc[evaluated]
    assert rows.groupby("municipality_code").size().to_dict() == {code: per_municipality for code in CODES}
    assert int(rows["y"].sum()) == positives


@pytest.mark.parametrize("period", [DEVELOPMENT, TEST], ids=["desarrollo", "prueba"])
def test_no_training_target_in_or_after_evaluated_year(channel_path, period):
    examples = _examples(str(channel_path))
    for trial in _trials(str(channel_path), *period):
        years = _target_year(examples.loc[trial.train])
        assert len(years) > 0, (trial.municipality_code, trial.year)
        assert (years < trial.year).all(), (trial.municipality_code, trial.year)


@pytest.mark.parametrize("period", [DEVELOPMENT, TEST], ids=["desarrollo", "prueba"])
def test_training_closes_at_first_emission_and_evaluation_is_the_year(channel_path, period):
    examples = _examples(str(channel_path))
    year_of_target = _target_year(examples)
    for trial in _trials(str(channel_path), *period):
        code, year = trial.municipality_code, trial.year
        own = examples["municipality_code"] == code
        expected_evaluate = set(examples.index[own & (year_of_target == year)])
        t0 = min(examples.loc[list(expected_evaluate), "t"])
        expected_train = set(examples.index[own & (year_of_target >= FIRST_TRAIN_YEAR)
                                            & (examples["target_week"] <= t0)])

        assert trial.t0 == t0, (code, year)
        assert set(trial.evaluate) == expected_evaluate, (code, year)
        assert set(trial.train) == expected_train, (code, year)
        assert len(trial.train) == len(set(trial.train)), (code, year)
        assert (examples.loc[trial.train, "target_week"] <= trial.t0).all(), (code, year)


@pytest.mark.parametrize("code", CODES)
def test_contract_example_for_2019(channel_path, code):
    examples = _examples(str(channel_path))
    trials = _trials(str(channel_path), *DEVELOPMENT)
    first, second = _trial(trials, code, 2019), _trial(trials, code, 2020)
    own = examples[examples["municipality_code"] == code]

    def label(t):
        rows = own.index[own["t"] == t]
        assert len(rows) == 1, (code, t)
        return rows[0]

    assert first.t0 == date(2018, 11, 12)
    last_train = examples.loc[first.train].sort_values("t").iloc[-1]
    assert (last_train["t"], last_train["target_week"]) == (date(2018, 9, 17), date(2018, 11, 12))

    gap = [date(2018, 9, 24) + timedelta(weeks=k) for k in range(7)]
    assert gap[-1] == date(2018, 11, 5)
    for t in gap:
        assert label(t) not in set(first.train), (code, t)
        assert label(t) in set(second.train), (code, t)

    late = label(date(2018, 11, 26))
    assert examples.loc[late, "target_week"] == date(2019, 1, 21)
    assert late in set(first.evaluate) and late not in set(first.train)
