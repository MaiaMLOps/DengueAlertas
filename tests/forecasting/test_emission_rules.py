"""Rule B01 (análisis complementario) and the emission-rule parameter, on synthetic data."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.epidemiology.config import EpidemiologyValidationError
from src.forecasting.complementary import first_alert_leads, weeks_under_alert
from src.forecasting.config import DecisionSection, ProductSection, load_config
from src.forecasting.events import (alerts, emission_positive, model_positive,
                                    model_positive_outside_epidemic, product_alert_states)

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


def test_preregistered_and_product_rules_are_separate_settings():
    config = load_config()
    assert config.decision.regla_emision == "umbral"                       # selection and evaluation
    assert config.producto.regla_alerta == "umbral_fuera_de_epidemia"      # API (D10.1)
    with pytest.raises(ValidationError):
        DecisionSection.model_validate({"k": 2, "W": 12, "tau_grid": [0.05], "regla_emision": "umbral_fuera_de_epidemia"})
    assert ProductSection().regla_alerta == "umbral"
    with pytest.raises(ValidationError):
        ProductSection.model_validate({"regla_alerta": "otra"})


def test_product_alert_states():
    weeks = _weeks(8)
    p = [0.01, 0.9, 0.9, 0.9, 0.9, 0.9, 0.01, 0.9]
    zones = ["seguridad", "alerta", "alerta", "alerta", "epidemia", "alerta", "alerta", "alerta"]
    states = product_alert_states(weeks, p, zones, tau=0.05, k=2, rule="umbral_fuera_de_epidemia")
    assert states == ["sin_alerta", "senal_en_observacion", "alerta_inicio_confirmada", "alerta_inicio_confirmada",
                      "continuidad_epidemica", "senal_en_observacion", "sin_alerta", "senal_en_observacion"]
    # under the pre-registered rule the run continues through the epidemic week
    assert product_alert_states(weeks, p, zones, tau=0.05, k=2, rule="umbral")[5] == "alerta_inicio_confirmada"


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
