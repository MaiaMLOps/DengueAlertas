"""Unit tests of src/forecasting/selection.py beyond the frozen F6 acceptance: input validation, report columns
and edge cases (synthetic data only)."""

from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest

from src.epidemiology.config import EpidemiologyValidationError
from src.forecasting import selection

CODES = ("08001", "76001")
YEARS = (2019, 2020, 2021, 2022)


def _table(loss, candidates=None, codes=CODES):
    candidates = candidates or {"logistica": ["r0", "r1"], "lightgbm": ["g0"]}
    rows = []
    for family, names in candidates.items():
        for position, name in enumerate(names):
            for code in codes:
                for year in YEARS:
                    value = float(loss(code, family, name, year))
                    rows.append({"municipality_code": code, "familia": family, "candidato": name, "orden": position,
                                 "parametros": json.dumps({"id": name}), "anio": year, "logloss": value,
                                 "logloss_b2": 1.0, "mejora_b2": 1 - value})
    return pd.DataFrame(rows)


def _constant(code, family, name, year):
    return {"r0": 0.5, "r1": 0.6, "g0": 0.7}.get(name, 0.8)


# --------------------------------------------------------------------------- table validation
def test_table_without_a_family_is_rejected():
    table = _table(_constant)
    with pytest.raises(EpidemiologyValidationError, match="lacks the families"):
        selection.compare_families(table[table["familia"] == "logistica"], alpha=0.05)


def test_table_with_an_unknown_family_is_rejected():
    table = _table(_constant, {"logistica": ["r0"], "lightgbm": ["g0"], "gru": ["x"]})
    with pytest.raises(EpidemiologyValidationError, match="outside the selection"):
        selection.compare_families(table, alpha=0.05)


def test_table_missing_a_column_is_rejected():
    with pytest.raises(EpidemiologyValidationError, match="lacks columns"):
        selection.common_configuration(_table(_constant).drop(columns="mejora_b2"), familia="logistica")


def test_candidate_lists_must_match_between_municipalities():
    table = _table(_constant)
    table = table[~((table["municipality_code"] == "76001") & (table["candidato"] == "r1"))]
    with pytest.raises(EpidemiologyValidationError, match="candidate list differs"):
        selection.municipality_choices(table, familia_comun="logistica")


def test_unknown_common_family_is_rejected():
    with pytest.raises(EpidemiologyValidationError, match="Unknown family"):
        selection.municipality_choices(_table(_constant), familia_comun="gru")


# --------------------------------------------------------------------------- report columns
def test_comparisons_have_one_row_per_municipality_rule_and_year():
    choices, comparisons = selection.municipality_choices(_table(_constant), familia_comun="logistica")
    assert len(choices) == len(CODES)
    assert len(comparisons) == len(CODES) * 2 * len(YEARS)
    assert set(comparisons["regla"]) == {selection.FAMILY_RULE, selection.HYPERPARAMETER_RULE}
    assert (comparisons["gana"] == (comparisons["diferencia"] < 0)).all()
    family = comparisons[comparisons["regla"] == selection.FAMILY_RULE]
    assert set(family["familia_opcion"]) == {"lightgbm"} and set(family["familia_referencia"]) == {"logistica"}
    assert list(choices["candidato"]) == ["r0", "r0"]
    assert list(choices["orden"]) == [0, 0]


def test_index_labels_of_the_table_do_not_matter():
    table = _table(_constant)
    relabeled = table.set_index(pd.Index(np.arange(len(table)) * 7 + 3))
    a = selection.municipality_choices(table, familia_comun="logistica")
    b = selection.municipality_choices(relabeled, familia_comun="logistica")
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])


def test_family_comparison_reports_the_threshold_of_its_cells():
    result = selection.compare_families(_table(_constant), alpha=0.05)
    assert result.casillas == 8
    assert result.umbral == 7        # P(X >= 7 | 8, 1/2) = 9/256 < 0.05 <= 37/256
    assert result.victorias == 0 and result.familia_comun == "logistica" and not result.reemplaza


# --------------------------------------------------------------------------- tau rule validation
def _curve(points):
    return pd.DataFrame(points, columns=["tau", "warned_onsets", "false_alarms"])


@pytest.mark.parametrize("curve, channel, message", [
    (_curve([]), 3, "must not be empty"),
    (_curve([(0.5, 1, 1), (0.5, 2, 2)]), 3, "repeated tau"),
    (_curve([(0.5, 1, 1)]), -1, "non-negative integer"),
    (_curve([(0.5, 1, 1)]), 2.0, "non-negative integer"),
    (pd.DataFrame({"tau": [0.5], "false_alarms": [1]}), 3, "lacks columns"),
])
def test_choose_tau_validates_its_inputs(curve, channel, message):
    with pytest.raises(EpidemiologyValidationError, match=message):
        selection.choose_tau(curve, channel_false_alarms=channel)


def test_choose_tau_with_zero_channel_false_alarms():
    choice = selection.choose_tau(_curve([(0.3, 5, 1), (0.6, 2, 0), (0.9, 2, 0)]), channel_false_alarms=0)
    assert choice == selection.TauChoice(tau=0.9, restriccion_cumplida=True)


# --------------------------------------------------------------------------- predictions
def _predictions(table):
    weeks = [date(2018, 11, 12) + timedelta(weeks=i) for i in range(3)]
    rows = []
    for (code, family, name), _ in table.groupby(["municipality_code", "familia", "candidato"]):
        for t in weeks:
            rows.append({"municipality_code": code, "t": t, "target_week": t + timedelta(weeks=8),
                         "familia": family, "candidato": name, "anio": 2019, "y": 0, "p": 0.1})
    return pd.DataFrame(rows)


def test_chosen_predictions_keep_only_the_chosen_configuration():
    table = _table(_constant)
    choices = pd.DataFrame({"municipality_code": list(CODES), "familia": ["logistica", "lightgbm"],
                            "candidato": ["r1", "g0"]})
    chosen = selection.chosen_predictions(_predictions(table), choices)
    assert len(chosen) == 6
    assert set(zip(chosen["municipality_code"], chosen["candidato"])) == {("08001", "r1"), ("76001", "g0")}
    assert list(chosen["t"]) == sorted(chosen["t"][:3]) * 2


def test_chosen_predictions_fail_when_a_configuration_has_no_predictions():
    table = _table(_constant)
    choices = pd.DataFrame({"municipality_code": list(CODES), "familia": ["logistica", "logistica"],
                            "candidato": ["r1", "inexistente"]})
    with pytest.raises(EpidemiologyValidationError, match="No predictions"):
        selection.chosen_predictions(_predictions(table), choices)


def test_duplicated_rows_are_rejected():
    rows = pd.DataFrame({"municipality_code": ["08001"] * 2, "t": [date(2018, 11, 12)] * 2,
                         "target_week": [date(2019, 1, 7)] * 2, "zona_t": ["seguridad"] * 2, "y": [0, 0]})
    predictions = rows[["municipality_code", "t", "target_week", "y"]].assign(p=0.2).iloc[:1]
    with pytest.raises(EpidemiologyValidationError, match="rows have duplicated"):
        selection._aligned_probabilities(rows, predictions)
