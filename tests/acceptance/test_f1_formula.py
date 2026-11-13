"""F1 acceptance: epiCo channel formula (contrato §1 and §10.1; especificación §6.F1).

epiCo endemic_channel, method="geometric", shift=1, n=5 prior years:
    central = round(exp(mean(log(x + 1))) - 1, 5)
    gsd     = round(sd(log(x + 1)), 5)                (ddof = 1)
    U1e     = central * exp(|qt(0.025; n - 1)| * gsd / sqrt(n))   (0 if central = 0)

Parameters are passed explicitly: the channel functions do not read configuration.
Synthetic literals were computed independently with the formula above; the
Villavicencio history is the one of the week starting 2019-01-28, computed outside
the implementation session. They test the written formula, not epiCo executed in R.
"""

import pytest

pytestmark = pytest.mark.acceptance

N_YEARS = 5
SHIFT = 1.0
ALPHA = 0.025


def _channel(history):
    from src.epidemiology.channel_epico import epico_channel

    return epico_channel(history, n_years=N_YEARS, shift=SHIFT, alpha=ALPHA)


def test_t_critical_matches_qt_0025_with_four_degrees_of_freedom():
    from src.epidemiology.channel_epico import t_critical

    assert round(t_critical(n_years=N_YEARS, alpha=ALPHA), 6) == 2.776445


@pytest.mark.parametrize(
    ("history", "central", "gsd", "upper"),
    [
        ((10, 20, 30, 40, 50), 26.23112, 0.60563, 55.64184128845796),
        ((3, 0, 7, 1, 12), 2.83729, 1.03239, 10.223928112533596),
        ((0, 0, 0, 0, 1), 0.1487, 0.30998, 0.21850783229334014),
        ((35, 18, 15, 1, 3), 8.73763, 1.19374, 38.46939411375516),
    ],
    ids=["increasing", "with_zero", "single_nonzero", "villavicencio_2019_01_28"],
)
def test_formula_reproduces_reference_values(history, central, gsd, upper):
    result_central, result_gsd, result_upper = _channel(history)

    assert result_central == central
    assert result_gsd == gsd
    assert result_upper == pytest.approx(upper, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(
    "history",
    [(10, 20, 30, 40, 50), (3, 0, 7, 1, 12), (0, 0, 0, 0, 1), (35, 18, 15, 1, 3), (2, 9, 0, 5, 17)],
)
def test_central_and_gsd_are_rounded_to_five_decimals(history):
    central, gsd, upper = _channel(history)

    assert central == round(central, 5)
    assert gsd == round(gsd, 5)
    assert upper >= central


def test_constant_history_gives_exact_central_and_no_spread():
    central, gsd, upper = _channel((4, 4, 4, 4, 4))

    assert central == 4.0
    assert gsd == 0.0
    assert upper == 4.0


def test_zero_central_gives_zero_upper():
    central, gsd, upper = _channel((0, 0, 0, 0, 0))

    assert central == 0.0
    assert gsd == 0.0
    assert upper == 0.0
