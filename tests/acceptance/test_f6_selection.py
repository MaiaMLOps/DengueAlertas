"""F6 acceptance with synthetic inputs: selection order of the pre-registration and the rule for tau
(contrato §4.3, §6.2, §7.5 and §8.3; especificación §6.F6 and §7, «Orden de selección para F6»).
Approved by Jesús on 2026-11-12 (U6.1).

Selection order (especificación §7), all with out-of-year scores (`comparison.out_of_year_scores`):
1. common configuration of each family: the candidate with the largest mean improvement over B2, first over
   the years of each municipality and then over the municipalities, chosen with the 4 years (contrato §6.2);
   an exact tie goes to the smallest `orden`;
2. family comparison with the out-of-year common configurations: LightGBM replaces logistic regression only if
   its mean out-of-year improvement is strictly greater and it wins at least win_threshold(24, alfa) = 17 of
   the 24 cells. A win is a strictly smaller log-loss (contrato §8.2: development wins use log-loss, not
   mejora_b2); a tie is not a win;
3. family exception per municipality: the alternative family's out-of-year common configuration beats the
   common family's in the 4 years of that municipality (4 of 4);
4. hyperparameter exception within the final family of each municipality: the own candidate chosen without y
   beats the common one chosen without y in the 4 years; then the municipality uses its own best candidate
   chosen with the 4 years. If that candidate equals the common one, the exception is nominal and reported.
Every comparison is reported with its four annual differences, logloss of the option minus logloss of the
reference (negative = win), for every municipality and both rules.

tau (contrato §7.5): over decision.tau_grid, with the development predictions of the chosen configuration of
each municipality, concatenated and with the six municipalities together, through events.evaluate_period:
maximum number of warned evaluable onsets among the tau whose false alarms do not exceed the channel's on the
same rows (equality is feasible); a tie goes to the larger tau; if no tau is feasible, the tau with fewest false
alarms (tie to the larger) and restriccion_cumplida = False.

API frozen here (`src.forecasting.selection`):
- `common_configuration(table, *, familia) -> str`;
- `compare_families(table, *, alpha) -> FamilyComparison` with fields familia_comun, mejora (dict family ->
  mean out-of-year improvement), victorias, casillas, umbral and reemplaza;
- `municipality_choices(table, *, familia_comun) -> (choices, comparisons)`: choices has one row per
  municipality with municipality_code, familia, candidato, orden, parametros, excepcion_familia,
  excepcion_hiperparametros and excepcion_hiperparametros_nominal; comparisons has one row per (municipality,
  regla in {"familia", "hiperparametros"}, anio) with familia_opcion, candidato_opcion, familia_referencia,
  candidato_referencia, logloss_opcion, logloss_referencia and diferencia;
- `tau_curve(rows, predictions, observed, *, tau_grid, k, window, access) -> (curve, channel)`: curve has one
  row per tau with tau, evaluable_onsets, warned_onsets, false_alarms, municipality_years, judgeable_alerts
  and continuity_alerts; channel is the summary of the channel on the same rows; predictions are aligned with
  rows by (municipality_code, t) and must match them exactly;
- `choose_tau(curve, *, channel_false_alarms) -> TauChoice(tau, restriccion_cumplida)`;
- `select(table, predictions, rows, observed, *, config, access) -> Selection` with fields familias,
  elecciones, comparaciones, predicciones, curva, canal and tau.
Input errors raise EpidemiologyValidationError.

Every synthetic table has logloss_b2 = 1, so mejora_b2 = 1 − logloss (the formula of metrics.improvement),
except the permutation test.
"""

from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.acceptance

CODES = ("08001", "08758", "41001", "50001", "54001", "76001")
YEARS = (2019, 2020, 2021, 2022)
FAMILIES = ("logistica", "lightgbm")
WEEK = timedelta(weeks=1)


def _config():
    from src.forecasting.config import load_config

    return load_config()


def _alpha():
    return _config().comparacion.alfa


# --------------------------------------------------------------------------- synthetic single table
def _table(candidates, loss, *, reference=lambda code, year: 1.0, codes=CODES):
    """candidates: family -> names in pre-registered order; loss(code, family, name, year) -> logloss."""
    rows = []
    for family, names in candidates.items():
        for position, name in enumerate(names):
            for code in codes:
                for year in YEARS:
                    value, b2 = float(loss(code, family, name, year)), float(reference(code, year))
                    rows.append({
                        "municipality_code": code, "familia": family, "candidato": name, "orden": position,
                        "parametros": json.dumps({"id": name}, sort_keys=True), "anio": year,
                        "logloss": value, "logloss_b2": b2, "mejora_b2": 1 - value / b2,
                        "n_filas": 52, "primera_semana_objetivo": date(year, 1, 7),
                        "ultima_semana_objetivo": date(year, 12, 30),
                    })
    return pd.DataFrame(rows)


def _cells():
    return [(code, year) for code in CODES for year in YEARS]


def _pair_table(gb_losses, rl_loss=0.5):
    """One candidate per family: logistic regression `r` constant, LightGBM `g` by cell (code-major order)."""
    values = dict(zip(_cells(), gb_losses))
    return _table({"logistica": ["r"], "lightgbm": ["g"]},
                  lambda code, family, name, year: rl_loss if family == "logistica" else values[(code, year)])


def _families(table):
    from src.forecasting.selection import compare_families

    return compare_families(table, alpha=_alpha())


def _choices(table, common):
    from src.forecasting.selection import municipality_choices

    choices, comparisons = municipality_choices(table, familia_comun=common)
    return choices.set_index("municipality_code"), comparisons


def _rule(comparisons, code, rule):
    rows = comparisons[(comparisons["municipality_code"] == code) & (comparisons["regla"] == rule)]
    return rows.sort_values("anio")


# --------------------------------------------------------------------------- step 1
@pytest.mark.parametrize("order, expected", [(["r0", "r1", "r2"], "r0"), (["r1", "r0", "r2"], "r1")])
def test_common_configuration_uses_the_four_years_and_ties_go_to_the_smallest_order(order, expected):
    from src.forecasting.selection import common_configuration

    def loss(code, family, name, year):
        if family == "lightgbm":
            return 0.9
        if name == "r0":
            return 0.75                                      # mejora 0.25 every year
        if name == "r1":
            return 0.5 if year in (2019, 2021) else 1.0      # mejora 0.5, 0, 0.5, 0: mean 0.25, exact tie
        return 0.8

    table = _table({"logistica": order, "lightgbm": ["g"]}, loss)
    assert common_configuration(table, familia="logistica") == expected
    assert common_configuration(table, familia="lightgbm") == "g"


# --------------------------------------------------------------------------- step 2
FAMILY_CASES = {
    # case: (LightGBM logloss per cell against logistic 0.5, expected family, wins)
    "16 victorias": ([0.4] * 16 + [0.52] * 8, "logistica", 16),
    "17 victorias": ([0.4] * 17 + [0.52] * 7, "lightgbm", 17),
    "17 victorias, mejora menor": ([0.49] * 17 + [0.8] * 7, "logistica", 17),
    "mejora igual": ([0.375, 0.375, 0.375, 0.875] * 6, "logistica", 18),   # 0.625·3 + 0.125 = 4 · 0.5, exact
    "empates": ([0.4] * 16 + [0.5] * 8, "logistica", 16),
}


@pytest.mark.parametrize("case", list(FAMILY_CASES))
def test_family_comparison_needs_greater_improvement_and_17_of_24(case):
    gb_losses, expected, wins = FAMILY_CASES[case]
    result = _families(_pair_table(gb_losses))

    assert result.casillas == 24
    assert result.umbral == 17
    assert result.victorias == wins
    assert result.familia_comun == expected
    assert result.reemplaza == (expected == "lightgbm")
    assert result.mejora["logistica"] == 0.5
    assert result.mejora["lightgbm"] == pytest.approx(np.mean([1 - v for v in gb_losses]), rel=1e-9, abs=1e-12)


def test_family_wins_use_log_loss_not_mejora_b2():
    """17 cells where LightGBM's log-loss is one ulp smaller: 1 − a/1 equals 1 − b/1 in floating point, so
    counting with mejora_b2 would see ties there (7 wins) instead of 24 wins."""
    tiny = float(np.nextafter(0.1, 0.0))
    gb_losses = [tiny] * 17 + [0.05] * 7
    table = _pair_table(gb_losses, rl_loss=0.1)
    assert tiny < 0.1 and 1 - tiny / 1.0 == 1 - 0.1 / 1.0          # precondition of the design

    result = _families(table)
    assert result.victorias == 24
    assert result.familia_comun == "lightgbm"


def test_family_comparison_uses_out_of_year_configurations():
    """Each LightGBM candidate g_j is good (0.3) except in year j (0.9). Chosen with the 4 years, g0 would win
    18 of 24 cells against logistic 0.5 with a greater mean; out of year, the choice for y is always g_y,
    which is bad in y, so LightGBM wins no cell."""
    names = [f"g{j}" for j in range(4)]

    def loss(code, family, name, year):
        if family == "logistica":
            return 0.5
        return 0.9 if YEARS[names.index(name)] == year else 0.3

    result = _families(_table({"logistica": ["r"], "lightgbm": names}, loss))
    assert result.victorias == 0
    assert result.familia_comun == "logistica"
    assert not result.reemplaza


# --------------------------------------------------------------------------- step 3
def test_family_exception_needs_four_of_four():
    special = {("08001", y): 0.4 for y in YEARS}
    special.update({("08758", y): 0.4 for y in (2019, 2020, 2021)} | {("08758", 2022): 0.6})
    special.update({("41001", y): 0.4 for y in (2019, 2020, 2021)} | {("41001", 2022): 0.5})   # tie in 2022
    table = _table({"logistica": ["r"], "lightgbm": ["g"]},
                   lambda code, family, name, year: 0.5 if family == "logistica" else special.get((code, year), 0.6))
    families = _families(table)
    assert families.familia_comun == "logistica"
    assert families.victorias == 10

    choices, comparisons = _choices(table, families.familia_comun)
    assert set(choices.index) == set(CODES)
    assert choices.loc["08001", "familia"] == "lightgbm"
    assert choices.loc["08001", "candidato"] == "g"
    assert bool(choices.loc["08001", "excepcion_familia"])
    for code in CODES[1:]:
        assert choices.loc[code, "familia"] == "logistica"
        assert not bool(choices.loc[code, "excepcion_familia"])

    assert len(comparisons) == len(CODES) * 2 * len(YEARS)
    rows = _rule(comparisons, "08001", "familia")
    assert list(rows["anio"]) == list(YEARS)
    assert set(rows["familia_opcion"]) == {"lightgbm"} and set(rows["familia_referencia"]) == {"logistica"}
    assert list(rows["diferencia"]) == [0.4 - 0.5] * 4
    assert list(_rule(comparisons, "08758", "familia")["diferencia"]) == [0.4 - 0.5] * 3 + [0.6 - 0.5]
    assert list(_rule(comparisons, "41001", "familia")["diferencia"]) == [0.4 - 0.5] * 3 + [0.0]


def test_family_exception_can_return_to_logistic_regression():
    table = _table({"logistica": ["r"], "lightgbm": ["g"]},
                   lambda code, family, name, year: 0.5 if family == "logistica" else (0.6 if code == "08001" else 0.4))
    families = _families(table)
    assert families.familia_comun == "lightgbm" and families.victorias == 20

    choices, _ = _choices(table, families.familia_comun)
    assert choices.loc["08001", "familia"] == "logistica" and bool(choices.loc["08001", "excepcion_familia"])
    assert all(choices.loc[code, "familia"] == "lightgbm" for code in CODES[1:])


# --------------------------------------------------------------------------- step 4
def test_hyperparameter_exception_needs_four_of_four():
    own = {("08001", y): 0.4 for y in YEARS}
    own.update({("08758", 2019): 0.6} | {("08758", y): 0.4 for y in (2020, 2021, 2022)})
    own.update({("41001", y): 0.4 for y in (2019, 2020, 2021)} | {("41001", 2022): 0.5})      # tie in 2022

    def loss(code, family, name, year):
        if family == "lightgbm":
            return 0.9
        return 0.5 if name == "ra" else own.get((code, year), 0.6)

    table = _table({"logistica": ["ra", "rb"], "lightgbm": ["g"]}, loss)
    families = _families(table)
    assert families.familia_comun == "logistica"
    choices, comparisons = _choices(table, "logistica")

    assert choices.loc["08001", "candidato"] == "rb"
    assert choices.loc["08001", "orden"] == 1
    assert choices.loc["08001", "parametros"] == json.dumps({"id": "rb"}, sort_keys=True)
    assert bool(choices.loc["08001", "excepcion_hiperparametros"])
    assert not bool(choices.loc["08001", "excepcion_hiperparametros_nominal"])
    for code in CODES[1:]:
        assert choices.loc[code, "candidato"] == "ra"
        assert not bool(choices.loc[code, "excepcion_hiperparametros"])
        assert not bool(choices.loc[code, "excepcion_hiperparametros_nominal"])
    assert not choices["excepcion_familia"].any()

    rows = _rule(comparisons, "08001", "hiperparametros")
    assert list(rows["candidato_opcion"]) == ["rb"] * 4 and list(rows["candidato_referencia"]) == ["ra"] * 4
    assert list(rows["diferencia"]) == [0.4 - 0.5] * 4
    assert list(_rule(comparisons, "08758", "hiperparametros")["diferencia"]) == [0.6 - 0.5] + [0.4 - 0.5] * 3
    assert list(_rule(comparisons, "41001", "hiperparametros")["diferencia"]) == [0.4 - 0.5] * 3 + [0.0]
    assert list(_rule(comparisons, "50001", "hiperparametros")["diferencia"]) == [0.0] * 4   # own = common: ties


def test_nominal_hyperparameter_exception():
    """Out of year, the common choice for y is g_y (good outside y in five municipalities, bad in y and in
    08001); 08001's own choice is a, which beats g_y there in the 4 years. With the 4 years the common
    configuration is a too, so 08001's exception is nominal: same candidate, reported."""
    names = [f"g{j}" for j in range(4)]

    def loss(code, family, name, year):
        if family == "lightgbm":
            return 1.5
        if name == "a":
            return 0.5
        if code == "08001" or YEARS[names.index(name)] == year:
            return 1.0
        return 0.3

    table = _table({"logistica": ["a", *names], "lightgbm": ["h"]}, loss)
    from src.forecasting.selection import common_configuration

    assert common_configuration(table, familia="logistica") == "a"
    families = _families(table)
    assert families.familia_comun == "logistica"
    choices, comparisons = _choices(table, "logistica")

    assert choices.loc["08001", "candidato"] == "a"
    assert bool(choices.loc["08001", "excepcion_hiperparametros"])
    assert bool(choices.loc["08001", "excepcion_hiperparametros_nominal"])
    rows = _rule(comparisons, "08001", "hiperparametros")
    assert list(rows["candidato_opcion"]) == ["a"] * 4
    assert list(rows["candidato_referencia"]) == names
    for code in CODES[1:]:
        assert choices.loc[code, "candidato"] == "a"
        assert not bool(choices.loc[code, "excepcion_hiperparametros"])
        assert not bool(choices.loc[code, "excepcion_hiperparametros_nominal"])


def test_hyperparameter_exception_is_evaluated_in_the_final_family():
    """08001 takes LightGBM by family exception; its hyperparameter exception is evaluated within LightGBM
    (g1 over g0), and the one that would apply within logistic regression (r1 over r0) is not used."""
    def loss(code, family, name, year):
        first = code == "08001"
        return {("logistica", "r0"): 0.5, ("logistica", "r1"): 0.45 if first else 0.6,
                ("lightgbm", "g0"): 0.4 if first else 0.7, ("lightgbm", "g1"): 0.3 if first else 0.9}[(family, name)]

    table = _table({"logistica": ["r0", "r1"], "lightgbm": ["g0", "g1"]}, loss)
    families = _families(table)
    assert families.familia_comun == "logistica"
    choices, comparisons = _choices(table, "logistica")

    assert choices.loc["08001", "familia"] == "lightgbm"
    assert choices.loc["08001", "candidato"] == "g1"
    assert bool(choices.loc["08001", "excepcion_familia"])
    assert bool(choices.loc["08001", "excepcion_hiperparametros"])
    rows = _rule(comparisons, "08001", "hiperparametros")
    assert set(rows["familia_opcion"]) == {"lightgbm"} and set(rows["familia_referencia"]) == {"lightgbm"}
    assert list(rows["candidato_opcion"]) == ["g1"] * 4 and list(rows["candidato_referencia"]) == ["g0"] * 4
    assert list(rows["diferencia"]) == [0.3 - 0.4] * 4
    family_rows = _rule(comparisons, "08001", "familia")
    assert list(family_rows["candidato_opcion"]) == ["g0"] * 4
    assert list(family_rows["candidato_referencia"]) == ["r0"] * 4
    for code in CODES[1:]:
        assert (choices.loc[code, "familia"], choices.loc[code, "candidato"]) == ("logistica", "r0")
        assert not bool(choices.loc[code, "excepcion_familia"])
        assert not bool(choices.loc[code, "excepcion_hiperparametros"])


# --------------------------------------------------------------------------- order of the table rows
def test_permuted_table_gives_the_same_selection():
    rng = np.random.default_rng(20261112)
    values = {key: rng.uniform(0.2, 0.9) for key in
              [(c, f, n, y) for c in CODES for f, names in (("logistica", ["r0", "r1", "r2"]),
                                                             ("lightgbm", ["g0", "g1", "g2", "g3"]))
               for n in names for y in YEARS]}
    references = {(c, y): rng.uniform(0.5, 1.0) for c in CODES for y in YEARS}
    table = _table({"logistica": ["r0", "r1", "r2"], "lightgbm": ["g0", "g1", "g2", "g3"]},
                   lambda c, f, n, y: values[(c, f, n, y)], reference=lambda c, y: references[(c, y)])
    shuffled = table.sample(frac=1.0, random_state=7)
    renumbered = shuffled.reset_index(drop=True)

    from src.forecasting.selection import municipality_choices

    base = _families(table)
    base_choices, base_comparisons = municipality_choices(table, familia_comun=base.familia_comun)
    for other in (shuffled, renumbered):
        assert _families(other) == base
        choices, comparisons = municipality_choices(other, familia_comun=base.familia_comun)
        pd.testing.assert_frame_equal(choices, base_choices, check_exact=True)
        pd.testing.assert_frame_equal(comparisons, base_comparisons, check_exact=True)


# --------------------------------------------------------------------------- tau rule
def _curve(points):
    return pd.DataFrame(points, columns=["tau", "warned_onsets", "false_alarms"])


@pytest.mark.parametrize("points, channel, expected", [
    # maximum detection among feasible tau; tie in detection -> larger tau
    ([(0.1, 8, 15), (0.2, 7, 10), (0.3, 7, 6), (0.4, 5, 2)], 10, (0.3, True)),
    # equality with the channel is feasible
    ([(0.1, 9, 10), (0.2, 3, 2)], 10, (0.1, True)),
    # nothing feasible -> fewest false alarms, tie -> larger tau, reported
    ([(0.1, 9, 20), (0.2, 8, 14), (0.3, 6, 12), (0.4, 2, 12)], 10, (0.4, False)),
    # order of the rows does not matter
    ([(0.4, 5, 2), (0.3, 7, 6), (0.1, 8, 15), (0.2, 7, 10)], 10, (0.3, True)),
])
def test_choose_tau(points, channel, expected):
    from src.forecasting.selection import choose_tau

    choice = choose_tau(_curve(points), channel_false_alarms=channel)
    assert (choice.tau, choice.restriccion_cumplida) == expected


# --------------------------------------------------------------------------- synthetic world for tau and select
WORLD_CODES = ("08001", "76001")
FIRST_EMISSION = date(2018, 11, 12)      # first target week 2019-01-07 with h = 8
N_ROWS = 208                             # targets 2019-01-07 .. 2022-12-26
H = 8


def _world(seed=20261112):
    from src.epidemiology.channel_epico import ZONES

    rng = np.random.default_rng(seed)
    first_observed = date(2017, 1, 2)
    observed_weeks = [first_observed + i * WEEK for i in range(6 * 52 + 2)]
    observed, rows = [], []
    for c, code in enumerate(WORLD_CODES):
        exceed = {week: int((i + 7 * c) % 26 in (10, 11, 12, 13)) for i, week in enumerate(observed_weeks)}
        observed += [{"municipality_code": code, "week_start": week, "target_u1e": exceed[week],
                      "channel_eligible": True} for week in observed_weeks]
        zones = rng.choice(list(ZONES), size=N_ROWS, p=[0.6, 0.3, 0.1])
        for i in range(N_ROWS):
            t = FIRST_EMISSION + i * WEEK
            target = t + H * WEEK
            rows.append({"municipality_code": code, "t": t, "target_week": target, "zona_t": str(zones[i]),
                         "y": exceed[target]})
    observed = pd.DataFrame(observed)
    observed["channel_eligible"] = observed["channel_eligible"].astype(bool)
    return pd.DataFrame(rows), observed, rng


def _access(tmp_path):
    from src.forecasting.guard import RecipeAccess

    config = _config()
    return RecipeAccess(test_years=tuple(config.validacion.prueba), channel_version=config.datos.version_dvc_canal,
                        recipe_path=tmp_path / "receta_inexistente.yaml")


def _model_predictions(rows, rng):
    p = np.clip(0.35 * rows["y"].to_numpy() + 0.65 * rng.uniform(size=len(rows)), 0.0, 1.0)
    return rows[["municipality_code", "t", "target_week", "y"]].assign(p=p)


SUMMARY_KEYS = ("evaluable_onsets", "warned_onsets", "false_alarms", "municipality_years", "judgeable_alerts",
                "continuity_alerts")


def test_tau_curve_runs_model_and_channel_through_evaluate_period(tmp_path):
    from src.forecasting.events import channel_positive, evaluate_period, model_positive
    from src.forecasting.selection import tau_curve

    rows, observed, rng = _world()
    predictions = _model_predictions(rows, rng)
    access, grid = _access(tmp_path), (0.2, 0.5, 0.8)
    curve, channel = tau_curve(rows, predictions.sample(frac=1.0, random_state=3), observed, tau_grid=grid,
                               k=2, window=12, access=access)

    assert list(curve["tau"]) == list(grid)
    expected_channel = evaluate_period(rows, channel_positive(rows["zona_t"]), observed, k=2, window=12,
                                       access=access).summary
    assert {key: channel[key] for key in SUMMARY_KEYS} == {key: expected_channel[key] for key in SUMMARY_KEYS}
    assert expected_channel["evaluable_onsets"] > 0 and expected_channel["false_alarms"] > 0
    for tau, (_, row) in zip(grid, curve.iterrows()):
        summary = evaluate_period(rows, model_positive(predictions["p"].to_numpy(), tau=tau), observed, k=2,
                                  window=12, access=access).summary
        assert {key: int(row[key]) for key in SUMMARY_KEYS} == {key: summary[key] for key in SUMMARY_KEYS}
    assert curve["warned_onsets"].nunique() > 1 or curve["false_alarms"].nunique() > 1


@pytest.mark.parametrize("damage", ["missing_row", "other_y", "extra_row", "duplicated_row"])
def test_tau_curve_rejects_predictions_that_do_not_match_the_rows(tmp_path, damage):
    from src.epidemiology.config import EpidemiologyValidationError
    from src.forecasting.selection import tau_curve

    rows, observed, rng = _world()
    predictions = _model_predictions(rows, rng)
    if damage == "missing_row":
        predictions = predictions.drop(index=100)
    elif damage == "other_y":
        predictions.loc[100, "y"] = 1 - predictions.loc[100, "y"]
    elif damage == "extra_row":
        extra = predictions.iloc[[0]].assign(t=FIRST_EMISSION - WEEK, target_week=FIRST_EMISSION + (H - 1) * WEEK)
        predictions = pd.concat([predictions, extra], ignore_index=True)
    else:
        predictions = pd.concat([predictions, predictions.iloc[[5]]], ignore_index=True)
    with pytest.raises(EpidemiologyValidationError):
        tau_curve(rows, predictions, observed, tau_grid=(0.5,), k=2, window=12, access=_access(tmp_path))


def _world_search(rows, rng):
    names = {"logistica": ["r0", "r1"], "lightgbm": ["g0", "g1"]}
    values = {(c, f, n, y): rng.uniform(0.3, 0.9) for f in names for n in names[f] for c in WORLD_CODES for y in YEARS}
    table = _table(names, lambda c, f, n, y: values[(c, f, n, y)], codes=WORLD_CODES,
                   reference=lambda c, y: 0.6)
    frames = []
    for family, candidates in names.items():
        for name in candidates:
            frame = _model_predictions(rows, rng).assign(familia=family, candidato=name,
                                                         anio=[w.year for w in rows["target_week"]])
            frames.append(frame[["municipality_code", "t", "target_week", "familia", "candidato", "anio", "y", "p"]])
    return table, pd.concat(frames, ignore_index=True)


def test_select_composes_the_steps(tmp_path):
    from src.forecasting.selection import choose_tau, municipality_choices, select, tau_curve

    rows, observed, rng = _world()
    table, predictions = _world_search(rows, rng)
    config, access = _config(), _access(tmp_path)
    result = select(table, predictions, rows, observed, config=config, access=access)

    assert result.familias == _families(table)
    choices, comparisons = municipality_choices(table, familia_comun=result.familias.familia_comun)
    pd.testing.assert_frame_equal(result.elecciones, choices, check_exact=True)
    pd.testing.assert_frame_equal(result.comparaciones, comparisons, check_exact=True)

    chosen = result.predicciones
    assert len(chosen) == len(rows)
    for _, choice in choices.iterrows():
        own = chosen[chosen["municipality_code"] == choice["municipality_code"]]
        assert len(own) == N_ROWS
        assert set(zip(own["familia"], own["candidato"])) == {(choice["familia"], choice["candidato"])}
        source = predictions[(predictions["municipality_code"] == choice["municipality_code"])
                             & (predictions["familia"] == choice["familia"])
                             & (predictions["candidato"] == choice["candidato"])]
        merged = own.merge(source, on=["municipality_code", "t"], suffixes=("", "_fuente"))
        assert len(merged) == N_ROWS and (merged["p"] == merged["p_fuente"]).all()

    decision = config.decision
    curve, channel = tau_curve(rows, chosen, observed, tau_grid=decision.tau_grid, k=decision.k,
                               window=decision.W, access=access)
    pd.testing.assert_frame_equal(result.curva, curve, check_exact=True)
    assert {key: result.canal[key] for key in SUMMARY_KEYS} == {key: channel[key] for key in SUMMARY_KEYS}
    assert list(result.curva["tau"]) == list(decision.tau_grid)
    assert result.tau == choose_tau(curve, channel_false_alarms=channel["false_alarms"])


@pytest.mark.parametrize("where", ["table", "predictions"])
def test_select_rejects_test_period_rows(tmp_path, where):
    from src.epidemiology.config import EpidemiologyValidationError
    from src.forecasting.selection import select

    rows, observed, rng = _world()
    table, predictions = _world_search(rows, rng)
    if where == "table":
        extra = table.iloc[[0]].assign(anio=2023, primera_semana_objetivo=date(2023, 1, 2),
                                       ultima_semana_objetivo=date(2023, 12, 25))
        table = pd.concat([table, extra], ignore_index=True)
    else:
        extra = predictions.iloc[[0]].assign(t=date(2022, 11, 7), target_week=date(2023, 1, 2), anio=2023)
        predictions = pd.concat([predictions, extra], ignore_index=True)
    with pytest.raises(EpidemiologyValidationError):
        select(table, predictions, rows, observed, config=_config(), access=_access(tmp_path))
