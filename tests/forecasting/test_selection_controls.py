"""Negative controls of the frozen F6 selection checks (contrato §8.2 and §8.3; especificación §7).

Imports tests/acceptance/test_f6_selection.py by path, so what is checked is the frozen assertion, not a copy.
With `src.forecasting.selection` as it is, the frozen checks pass (positive control); with two deliberate
errors patched into the module, they must fail:
- wins counted with mejora_b2 instead of log-loss: in the frozen ulp design, 1 − a/c equals 1 − b/c in
  floating point while a < b, so the faulty count sees ties;
- the out-of-year choice for year y made with the 4 years, y included (winner's bias).
The patches rely on two names of selection.py: `_loss(scores, which)`, the loss column of out-of-year scores
used for every win ("comun" or "propio"), and `out_of_year_scores`, imported from comparison into its namespace.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ACCEPTANCE_FILE = Path(__file__).resolve().parents[1] / "acceptance" / "test_f6_selection.py"


def _load_frozen_checks():
    spec = importlib.util.spec_from_file_location("frozen_f6_selection_acceptance", ACCEPTANCE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frozen = _load_frozen_checks()


def _mejora_loss(scores, which):
    """Faulty: a win is a larger mejora_b2 (sign flipped so that a smaller value still wins)."""
    column = {"comun": "mejora_comun", "propio": "mejora_propia"}[which]
    return -scores[column].to_numpy(dtype=float)


def _in_sample_scores(table, *, familia):
    """Faulty: common and own configurations chosen with all the years, the scored year included."""
    from src.forecasting.comparison import _choose

    own = table[table["familia"] == familia]
    cells = own.set_index(["municipality_code", "anio", "candidato"])
    common = _choose(own)
    rows = []
    for code in sorted(set(own["municipality_code"])):
        mine = _choose(own[own["municipality_code"] == code])
        for year in sorted(set(own["anio"])):
            rows.append({"municipality_code": code, "anio": year, "candidato_comun": common, "candidato_propio": mine,
                         "logloss_comun": float(cells.loc[(code, year, common), "logloss"]),
                         "logloss_propio": float(cells.loc[(code, year, mine), "logloss"]),
                         "mejora_comun": float(cells.loc[(code, year, common), "mejora_b2"]),
                         "mejora_propia": float(cells.loc[(code, year, mine), "mejora_b2"])})
    return pd.DataFrame(rows)


def test_frozen_checks_pass_with_the_module():
    frozen.test_family_wins_use_log_loss_not_mejora_b2()
    frozen.test_family_comparison_uses_out_of_year_configurations()
    frozen.test_nominal_hyperparameter_exception()


def test_wins_counted_with_mejora_b2_are_detected(monkeypatch):
    from src.forecasting import selection

    monkeypatch.setattr(selection, "_loss", _mejora_loss)
    with pytest.raises(AssertionError):
        frozen.test_family_wins_use_log_loss_not_mejora_b2()


def test_out_of_year_choice_including_the_year_is_detected(monkeypatch):
    from src.forecasting import selection

    monkeypatch.setattr(selection, "out_of_year_scores", _in_sample_scores)
    with pytest.raises(AssertionError):
        frozen.test_family_comparison_uses_out_of_year_configurations()
    with pytest.raises(AssertionError):
        frozen.test_nominal_hyperparameter_exception()
