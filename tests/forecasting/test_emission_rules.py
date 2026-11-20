"""Rule B01 (análisis complementario) and the emission-rule parameter, on synthetic data."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.epidemiology.config import EpidemiologyValidationError
from src.forecasting.complementary import first_alert_leads, weeks_under_alert
from src.forecasting.config import load_config
from src.forecasting.events import (alerts, emission_positive, model_positive,
                                    model_positive_outside_epidemic)

MONDAY = date(2020, 1, 6)


def _weeks(n):
    return [MONDAY + timedelta(weeks=i) for i in range(n)]


def test_week_in_epidemic_zone_is_negative():
    p = [0.9, 0.9, 0.9]
    zones = ["alerta", "epidemia", "seguridad"]
    assert model_positive_outside_epidemic(p, zones, tau=0.5).tolist() == [True, False, True]
    assert model_positive(p, tau=0.5).tolist() == [True, True, True]


def test_tau_border_is_positive_as_in_the_contract_rule():
    p = [0.0499, 0.05, 0.0501]
    zones = ["seguridad"] * 3
    assert model_positive_outside_epidemic(p, zones, tau=0.05).tolist() == [False, True, True]
    assert (model_positive_outside_epidemic(p, zones, tau=0.05) == model_positive(p, tau=0.05)).all()


def test_run_is_split_when_the_zone_enters_epidemia():
    p = [0.9] * 6
    zones = ["alerta", "alerta", "epidemia", "epidemia", "alerta", "alerta"]
    weeks = _weeks(6)
    current = alerts(weeks, model_positive(p, tau=0.5), zones, k=2)
    b01 = alerts(weeks, model_positive_outside_epidemic(p, zones, tau=0.5), zones, k=2)
    assert [(a.start, a.end) for a in current] == [(weeks[0], weeks[5])]
    assert [(a.start, a.confirmation, a.end) for a in b01] == [(weeks[0], weeks[1], weeks[1]),
                                                               (weeks[4], weeks[5], weeks[5])]
    assert not any(a.continuity for a in b01)


def test_emission_rule_dispatch_and_default():
    p, zones = [0.9, 0.9], ["epidemia", "alerta"]
    assert emission_positive("umbral", p, zones, tau=0.5).tolist() == [True, True]
    assert emission_positive("umbral_fuera_de_epidemia", p, zones, tau=0.5).tolist() == [False, True]
    with pytest.raises(EpidemiologyValidationError):
        emission_positive("otra", p, zones, tau=0.5)
    with pytest.raises(EpidemiologyValidationError):
        model_positive_outside_epidemic(p, ["alerta"], tau=0.5)
    assert load_config().decision.regla_emision == "umbral"


def test_first_alert_lead_and_weeks_under_alert():
    weeks = _weeks(20)
    alert_table = pd.DataFrame({"municipality_code": ["A", "A", "A"], "confirmation": [weeks[2], weeks[8], weeks[9]],
                                "end": [weeks[3], weeks[8], weeks[11]], "continuity": [False, False, True]})
    onsets = pd.DataFrame({"municipality_code": ["A", "A"], "week": [weeks[10], weeks[1]]})
    leads = first_alert_leads(onsets, alert_table, window=12)
    assert leads.iloc[0] == 8.0 and np.isnan(leads.iloc[1])   # continuity alert ignored; nothing before week 1
    rows = pd.DataFrame({"municipality_code": ["A"] * 20, "t": weeks})
    covered = weeks_under_alert(rows, alert_table)
    assert [i for i, value in enumerate(covered) if value] == [2, 3, 8, 9, 10, 11]
