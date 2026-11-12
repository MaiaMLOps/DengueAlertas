"""F3 acceptance: baselines B2 and B1 (contrato §4.1, §6.1 amended by D7.1, §10.8; especificación §6.F3).

API frozen here (the arrays follow the order of `evaluate`):
- `b2_predict(train, evaluate, *, access)`: per municipality and zone observed at t (`zona_t`),
  (positives + 1)/(cases + 2) over the training rows;
- `b1_predict(train, evaluate, *, half_window, access)`: per municipality, (positives + 1)/(cases + 2)
  over the training rows whose target-week ordinal falls in the circular window of the evaluated row's
  target-week ordinal; `half_window` comes from `lineas_base.b1_semiventana` (= 2, contrato §4.1).
`access` is a `guard.RecipeAccess`; here every evaluated row is outside the test years, so no recipe exists.

Window edges (approved by Jesús in U3.1): ordinal 53 counts as 52, for training rows and for the evaluated
row; the window is circular over 52 weeks, {((o - 1 + d) mod 52) + 1 : d = -2..2}. Ordinal 1 takes
51, 52 (with 53), 1, 2, 3; ordinal 53 takes the window of 52.

Literal case (Jesús, computed outside the repository with docs/forecasting/verificacion/caso_f3_hueco.py):
Villavicencio 50001, trial 2019 (t0 = 2018-11-12, 340 training rows), row t = 2018-12-03 (zona_t epidemia,
target 2019-01-28 with ordinal 4): B2 = 21/43 and B1 = 5/32. Two implementations are compared, so
pytest.approx(rel=1e-9, abs=1e-12) (especificación §6.F2, tolerance convention).
"""

from datetime import date, timedelta
import functools

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.acceptance

H = 8
TOLERANCE = {"rel": 1e-9, "abs": 1e-12}
COLUMNS = ["municipality_code", "t", "target_week", "y", "zona_t"]


def _config():
    from src.forecasting.config import load_config

    return load_config()


def _access(tmp_path):
    from src.forecasting.guard import RecipeAccess

    config = _config()
    return RecipeAccess(test_years=tuple(config.validacion.prueba), channel_version=config.datos.version_dvc_canal,
                        recipe_path=tmp_path / "receta_inexistente.yaml")


def _half_window():
    return _config().lineas_base.b1_semiventana


def _b2(train, evaluate, tmp_path):
    from src.forecasting.baselines import b2_predict

    return np.asarray(b2_predict(train, evaluate, access=_access(tmp_path)), dtype=float)


def _b1(train, evaluate, tmp_path):
    from src.forecasting.baselines import b1_predict

    return np.asarray(b1_predict(train, evaluate, half_window=_half_window(), access=_access(tmp_path)), dtype=float)


def _monday(year: int, ordinal: int) -> date:
    """Monday with the given ordinal: Mondays of the year counted from one (W-SUN convention)."""
    first = date(year, 1, 1)
    return first + timedelta(days=(-first.weekday()) % 7) + timedelta(weeks=ordinal - 1)


def _frame(rows):
    """rows: (code, target_week, y, zona_t); t = target_week - h."""
    return pd.DataFrame([(code, week - timedelta(weeks=H), week, y, zone) for code, week, y, zone in rows],
                        columns=COLUMNS).astype({"y": "int8"})


def _laplace(positives, cases):
    return (positives + 1) / (cases + 2)


# --------------------------------------------------------------------------- calendar of the synthetic data
def test_synthetic_calendar_uses_repository_ordinals():
    from src.epidemiology.ewt import week_ordinal

    assert _monday(2018, 53) == date(2018, 12, 31) and week_ordinal(date(2018, 12, 31)) == 53
    assert _monday(2018, 52) == date(2018, 12, 24) and week_ordinal(date(2018, 12, 24)) == 52
    assert _monday(2019, 1) == date(2019, 1, 7) and week_ordinal(date(2019, 1, 7)) == 1
    assert _monday(2019, 4) == date(2019, 1, 28)
    for year in range(2012, 2021):
        for ordinal in (1, 2, 3, 25, 50, 51, 52):
            assert week_ordinal(_monday(year, ordinal)) == ordinal


def test_b1_half_window_is_contract_value():
    assert _half_window() == 2


# --------------------------------------------------------------------------- B2
def _b2_train():
    rows = []
    for k in range(7):                                    # 08001 alerta: 3 positives of 7
        rows.append(("08001", _monday(2015, 10 + k), int(k < 3), "alerta"))
    for k in range(2):                                    # 08001 epidemia: 2 of 2
        rows.append(("08001", _monday(2016, 10 + k), 1, "epidemia"))
    for k in range(5):                                    # 50001 alerta: 0 of 5
        rows.append(("50001", _monday(2015, 30 + k), 0, "alerta"))
    return _frame(rows)


def _b2_evaluate():
    week = _monday(2019, 20)
    return _frame([("08001", week, 1, "alerta"), ("08001", week, 0, "epidemia"), ("08001", week, 1, "seguridad"),
                   ("50001", week, 1, "alerta"), ("50001", week, 0, "epidemia")])


def test_b2_laplace_by_municipality_and_zone(tmp_path):
    observed = _b2(_b2_train(), _b2_evaluate(), tmp_path)
    expected = [4 / 9, 3 / 4, 1 / 2, 1 / 7, 1 / 2]       # zone without cases: (0 + 1)/(0 + 2)
    assert observed.shape == (5,)
    assert observed == pytest.approx(expected, **TOLERANCE)


def test_b2_depends_only_on_training(tmp_path):
    train, evaluate = _b2_train(), _b2_evaluate()
    base = _b2(train, evaluate, tmp_path)

    flipped = evaluate.assign(y=(1 - evaluate["y"]).astype("int8"))
    assert _b2(train, flipped, tmp_path) == pytest.approx(base, **TOLERANCE)

    changed = train.copy()
    changed.loc[(changed["municipality_code"] == "08001") & (changed["zona_t"] == "alerta"), "y"] = 1
    assert _b2(changed, evaluate, tmp_path)[0] == pytest.approx(8 / 9, **TOLERANCE)

    other = _frame([("54001", _monday(2016, 5), 1, "alerta"), ("54001", _monday(2016, 6), 1, "epidemia")])
    assert _b2(pd.concat([train, other], ignore_index=True), evaluate, tmp_path) == pytest.approx(base, **TOLERANCE)


# --------------------------------------------------------------------------- B1, mid-year window
def _b1_mid_train():
    rows = []
    for year in (2016, 2017, 2018):
        for ordinal in range(17, 26):
            positive = (ordinal in (18, 24)                                 # outside 19–23
                        or (year == 2016 and ordinal in (19, 23)) or (year == 2017 and ordinal == 21))
            rows.append(("08001", _monday(year, ordinal), int(positive), "alerta"))
            rows.append(("50001", _monday(year, ordinal), 1, "seguridad"))
    return _frame(rows)


def test_b1_laplace_in_mid_year_window(tmp_path):
    evaluate = _frame([("08001", _monday(2019, 21), 0, "alerta"), ("50001", _monday(2019, 21), 0, "alerta")])
    observed = _b1(_b1_mid_train(), evaluate, tmp_path)
    assert observed == pytest.approx([_laplace(3, 15), _laplace(15, 15)], **TOLERANCE)   # 4/17 and 16/17


def test_b1_depends_only_on_training(tmp_path):
    train = _b1_mid_train()
    evaluate = _frame([("08001", _monday(2019, 21), 0, "alerta"), ("08001", _monday(2020, 21), 1, "epidemia")])
    base = _b1(train, evaluate, tmp_path)
    assert base == pytest.approx([4 / 17, 4 / 17], **TOLERANCE)

    flipped = evaluate.assign(y=(1 - evaluate["y"]).astype("int8"))
    assert _b1(train, flipped, tmp_path) == pytest.approx(base, **TOLERANCE)

    changed = train.copy()
    changed.loc[(changed["municipality_code"] == "08001") & (changed["target_week"] == _monday(2018, 20)), "y"] = 1
    assert _b1(changed, evaluate, tmp_path) == pytest.approx([5 / 17, 5 / 17], **TOLERANCE)

    other = _frame([("54001", _monday(2017, ordinal), 1, "alerta") for ordinal in range(19, 24)])
    assert _b1(pd.concat([train, other], ignore_index=True), evaluate, tmp_path) == pytest.approx(base, **TOLERANCE)


# --------------------------------------------------------------------------- B1, year edges and week 53
# Training of 08001: ordinal -> (target weeks, positives). Each ordinal has its own years so that
# (municipality, t) is unique; ordinal 53 is 2012-12-31.
EDGE_TRAIN = {50: (1, 0), 51: (1, 1), 52: (2, 0), 53: (1, 1), 1: (3, 1), 2: (1, 1), 3: (2, 2), 4: (1, 1), 5: (4, 0)}


def _edge_train():
    rows = []
    for ordinal, (cases, positives) in EDGE_TRAIN.items():
        years = [2012] if ordinal == 53 else list(range(2013, 2013 + cases))
        assert len(years) == cases
        rows += [("08001", _monday(year, ordinal), int(k < positives), "alerta") for k, year in enumerate(years)]
    return _frame(rows)


def _edge_expected(ordinals):
    cases = sum(EDGE_TRAIN[o][0] for o in ordinals if o in EDGE_TRAIN)
    positives = sum(EDGE_TRAIN[o][1] for o in ordinals if o in EDGE_TRAIN)
    return _laplace(positives, cases)


@pytest.mark.parametrize("label, target_week, window", [
    ("ordinal 1", _monday(2020, 1), (51, 52, 53, 1, 2, 3)),     # 7/12
    ("ordinal 2", _monday(2020, 2), (52, 53, 1, 2, 3, 4)),      # 7/12
    ("ordinal 3", _monday(2020, 3), (1, 2, 3, 4, 5)),           # 6/13, without 52
    ("ordinal 51", _monday(2019, 51), (49, 50, 51, 52, 53, 1)), # 4/10, with 1 and without 2
    ("ordinal 52", _monday(2019, 52), (50, 51, 52, 53, 1, 2)),  # 5/11
    ("ordinal 53", date(2018, 12, 31), (50, 51, 52, 53, 1, 2)), # 53 takes the window of 52
])
def test_b1_circular_window_at_year_edges(tmp_path, label, target_week, window):
    observed = _b1(_edge_train(), _frame([("08001", target_week, 0, "alerta")]), tmp_path)
    assert observed[0] == pytest.approx(_edge_expected(window), **TOLERANCE), label


def test_b1_edge_rule_literals(tmp_path):
    """Literal values of the approved rule; alternatives give 6/11 (no 53 -> 52), 4/10 or 5/10 (cycle of 53)."""
    evaluate = _frame([("08001", _monday(2020, 1), 0, "alerta"), ("08001", _monday(2020, 3), 0, "alerta"),
                       ("08001", _monday(2019, 51), 0, "alerta"), ("08001", _monday(2019, 52), 0, "alerta"),
                       ("08001", date(2018, 12, 31), 0, "alerta")])
    observed = _b1(_edge_train(), evaluate, tmp_path)
    assert observed == pytest.approx([7 / 12, 6 / 13, 4 / 10, 5 / 11, 5 / 11], **TOLERANCE)


# --------------------------------------------------------------------------- literal case with real data
@functools.lru_cache(maxsize=None)
def _examples(channel_file: str):
    from src.forecasting.dataset import build_examples, load_channel_product

    config = _config()
    product = load_channel_product(channel_file, expected_md5=config.datos.version_dvc_canal)
    return build_examples(product, h=H, canal=config.canal, features=config.features)


def test_villavicencio_2019_literal_case(channel_path, tmp_path):
    from src.forecasting.splits import yearly_trials

    config = _config()
    examples = _examples(str(channel_path))
    trials = [trial for trial in yearly_trials(examples, [2019],
                                               first_train_year=config.validacion.primer_anio_entrenamiento)
              if trial.municipality_code == "50001"]
    assert len(trials) == 1
    trial = trials[0]
    assert trial.t0 == date(2018, 11, 12) and len(trial.train) == 340

    train = examples.loc[trial.train]
    row = examples[(examples["municipality_code"] == "50001") & (examples["t"] == date(2018, 12, 3))]
    assert len(row) == 1 and row.index[0] in set(trial.evaluate)
    assert row.iloc[0]["zona_t"] == "epidemia" and row.iloc[0]["target_week"] == date(2019, 1, 28)

    assert _b2(train, row, tmp_path)[0] == pytest.approx(21 / 43, **TOLERANCE)
    assert _b1(train, row, tmp_path)[0] == pytest.approx(5 / 32, **TOLERANCE)
