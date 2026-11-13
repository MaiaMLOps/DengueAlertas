"""F7 acceptance with the real development data (--channel-parquet and --search-dir): integrated diagnostic analysis
in development (contrato §8.8 and §10.22, D9.1; especificación §6.F7). Approved by Jesús on 2026-11-12 (U7.1).

Nothing is retrained: the inputs are the saved F5b predictions. In each development cell the diagnosed model is the
common out-of-year configuration of the chosen family (`comparison.out_of_year_scores`, candidato_comun of
`logistica`), labelled "usado_para_elegir"; B2 and B1 are the saved baseline predictions of each trial. Events use
tau = 0.10 of the frozen recipe with k = 2 and W = 12 (decision section). Because 2020 uses logistica_04, the model's
event figures differ from those of F6 (which used logistica_03 in the four years); they are reported as they are (D3).

Literal figures: computed with a disposable reference in U7.1 and reproduced, before freezing, by Jesús's independent
calculation docs/forecasting/verificacion/caso_f7_diagnostico.py (md5 f4a32dc83b4d6b330291e0ee105e3b92, commit 6e21cc9;
it does not import src/): regimes, cell differences and wins, transition strata, onsets with leads and delta_w, and
the alert, false-alarm and entry counts of model and channel. Tolerances: exact for
counts, labels and weeks; rel 1e-9 for differences and improvements (criterion of F2: two implementations).
The channel figures are a regression of events.py (F4, F6): 10/10 warned, 63 judgeable alerts, 48 false alarms,
12 of them followed by an epidemic week. Regimes of 2018-2025 depend only on the observed series (no system is
scored), so the test years are included.
"""

from datetime import date
import functools
import math
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.acceptance

TAU = 0.10
CODES = ("08001", "08758", "41001", "50001", "54001", "76001")
COMMON_OUT_OF_YEAR = {2019: "logistica_03", 2020: "logistica_04", 2021: "logistica_03", 2022: "logistica_03"}
E, P, T = "epidemica", "post_epidemica", "tranquila"
REGIMES = {  # 2018 .. 2025
    "08001": (T, T, E, E, E, E, E, P),
    "08758": (T, E, P, E, E, E, E, E),
    "41001": (T, E, P, T, T, E, E, P),
    "50001": (E, E, P, T, E, E, P, E),
    "54001": (E, E, P, E, P, T, E, E),
    "76001": (T, E, E, P, T, E, E, P),
}
DEVELOPMENT_REGIME_COUNTS = {E: 13, P: 6, T: 5}
# (municipality, year): (positives, model logloss - B2, model Brier - B2, model logloss - B1, model Brier - B1,
#                        win logloss vs B2, win Brier vs B2, win logloss vs B1, win Brier vs B1)
CELLS = {
    ("08001", 2019): (1, -0.0123759461662632, -0.0011196450955706, -0.0910824723344381, -0.0175766628922828, True, True, True, True),
    ("08001", 2020): (6, 0.099917814374619, 0.042572993409391, 0.0374094224764437, 0.0489568395524408, False, False, False, False),
    ("08001", 2021): (20, 0.0740246380441124, 0.0204139193830705, -0.1700027524175269, -0.0754355122140247, False, False, True, True),
    ("08001", 2022): (51, -0.3012393107131597, -0.1218425144265151, -1.6518671220770096, -0.6263976005755421, True, True, True, True),
    ("08758", 2019): (13, -0.0089813738850396, -0.0052338145000139, 0.0591069586151898, 0.0080436937503464, True, True, False, False),
    ("08758", 2020): (4, 0.1516694149491661, 0.0540051829711245, 0.146080349332146, 0.0727380276506486, False, False, False, False),
    ("08758", 2021): (27, -0.1819206955799756, -0.0526806810585873, -0.3401765759093291, -0.1407900431813895, True, True, True, True),
    ("08758", 2022): (43, -0.2355676978484568, -0.0904362225874103, -0.7550479586024054, -0.3296921033559166, True, True, True, True),
    ("41001", 2019): (35, -0.2228231698156303, -0.083967946012105, -0.8108540676004605, -0.2989554436062805, True, True, True, True),
    ("41001", 2020): (0, 0.1798856255655014, 0.0743646806675956, 0.0451068992506579, 0.049288691183519, False, False, False, False),
    ("41001", 2021): (0, 0.0198799197047978, 0.004445439555756, -0.1657705723235154, -0.0315817412887326, False, False, True, True),
    ("41001", 2022): (0, -0.0278693225527251, 0.0003705149347015, -0.0566536038788504, -0.0033077529511874, True, False, True, True),
    ("50001", 2019): (45, -0.131903036357262, -0.0594655272678492, -1.1804041915014654, -0.4273103798067373, True, True, True, True),
    ("50001", 2020): (0, 0.0039247334527159, 0.0072568756576132, -0.0892736018412477, 0.0046561238059666, False, False, True, False),
    ("50001", 2021): (0, -0.0054250923586354, -0.0007167305176218, -0.230319137264277, -0.0463626313842317, True, True, True, True),
    ("50001", 2022): (10, -0.1562336214660071, -0.0490380080392544, -0.2993854452796956, -0.0943055577823084, True, True, True, True),
    ("54001", 2019): (21, -0.0363398570093538, -0.017392175025436, -0.6996621649451319, -0.181658076116866, True, True, True, True),
    ("54001", 2020): (0, -0.0334285167498671, -0.0049371555578672, -0.1602754574807107, -0.0269518706647823, True, True, True, True),
    ("54001", 2021): (23, -0.118694577593051, -0.0450974109695993, -0.6089491580596824, -0.2106830553982781, True, True, True, True),
    ("54001", 2022): (0, 0.2063863900774671, 0.0921851560499105, 0.2677448401748125, 0.1219281991504331, False, False, False, False),
    ("76001", 2019): (4, 0.1114788515442555, 0.0096721779072704, 0.030200748643115, -0.0025103571817106, False, False, False, True),
    ("76001", 2020): (14, 0.2166396123661209, 0.0361852550432574, 0.1594293722161021, 0.0072829210264283, False, False, False, False),
    ("76001", 2021): (2, 0.0070746335807053, -0.0079981403709258, -0.0366426059383335, -0.0150239118402609, False, True, True, True),
    ("76001", 2022): (0, -0.0868604014927734, -0.0107283079220977, -0.1241056186026025, -0.0226401919671037, True, True, True, True),
}
WINS = {"gana_logloss_b2": 14, "gana_brier_b2": 14, "gana_logloss_b1": 17, "gana_brier_b1": 17}
WINS_BY_REGIME = {  # regime: (logloss vs B2, Brier vs B2, logloss vs B1, Brier vs B1)
    E: (9, 9, 9, 10), P: (1, 2, 3, 2), T: (4, 3, 5, 5)}
# (year, stratum): (rows, positives, log-loss improvement vs B2, BSS vs B2); "dentro" = zona_t "epidemia"
TRANSITION = {
    ("2019", "fuera"): (197, 30, -0.058472775485338, -0.0520090055143456),
    ("2019", "dentro"): (115, 89, 0.2347639568987212, 0.2955444855834197),
    ("2020", "fuera"): (271, 12, -0.5222692726397167, -0.7422807237125033),
    ("2020", "dentro"): (41, 12, -0.218786794432177, -0.1305135159682664),
    ("2021", "fuera"): (260, 31, 0.086494941486657, 0.1142508966402056),
    ("2021", "dentro"): (52, 41, 0.1100905667473103, 0.1166708545883984),
    ("2022", "fuera"): (210, 15, 0.2834048728482227, 0.2127374083929358),
    ("2022", "dentro"): (102, 89, 0.244376326368375, 0.2878460354677542),
    ("todos", "fuera"): (938, 88, -0.023731778959841, -0.0564236342534123),
    ("todos", "dentro"): (310, 231, 0.147211549422501, 0.1965502719132211),
}
# Onsets of the development period: (municipality, week): (evaluable, censoring, warned by model, warned by channel,
#                                                           model lead, channel lead, delta_w)
NAN = math.nan
ONSETS = {
    ("08001", date(2020, 1, 27)): (True, None, True, True, 7, 1, 6),
    ("08001", date(2021, 8, 23)): (True, None, True, True, 7, 7, 0),
    ("08758", date(2019, 5, 27)): (True, None, True, True, 6, 4, 2),
    ("08758", date(2019, 11, 4)): (True, None, True, True, 2, 4, -2),
    ("08758", date(2022, 2, 28)): (True, None, False, True, NAN, 2, NAN),
    ("41001", date(2019, 2, 18)): (True, None, True, True, 4, 4, 0),
    ("50001", date(2019, 1, 28)): (False, "ventana", True, True, 10, 1, NAN),
    ("50001", date(2019, 11, 25)): (True, None, False, True, NAN, 1, NAN),
    ("50001", date(2022, 10, 31)): (True, None, False, True, NAN, 4, NAN),
    ("54001", date(2021, 6, 21)): (True, None, True, True, 5, 1, 4),
    ("76001", date(2019, 12, 16)): (True, None, False, True, NAN, 2, NAN),
}
ENTRIES = {"total": 45, "evaluable": 44, "warned_model": 19, "warned_channel": 36}
# system: (alerts, continuity alerts, judgeable non-continuity alerts, false alarms, false alarms followed by an
#          epidemic week, censored non-continuity alerts)
ALERTS = {"modelo": (39, 6, 33, 25, 5, 0), "canal": (65, 0, 63, 48, 12, 2)}


def _config():
    from src.forecasting.config import load_config

    return load_config()


@functools.lru_cache(maxsize=None)
def _product(channel_file: str):
    from src.forecasting.dataset import load_channel_product

    return load_channel_product(channel_file, expected_md5=_config().datos.version_dvc_canal)


@functools.lru_cache(maxsize=None)
def _development_rows(channel_file: str) -> pd.DataFrame:
    from src.forecasting.dataset import build_examples
    from src.forecasting.splits import yearly_trials

    config = _config()
    examples = build_examples(_product(channel_file), h=config.horizonte.h, canal=config.canal,
                              features=config.features)
    trials = yearly_trials(examples, range(2019, 2023), first_train_year=config.validacion.primer_anio_entrenamiento)
    labels = trials[0].evaluate.append([trial.evaluate for trial in trials[1:]])
    return examples.loc[labels]


_RESULTS = {}


@pytest.fixture
def diagnostic(channel_path, search_dir, tmp_path):
    from src.forecasting.diagnostics import development_predictions, diagnostic_inputs
    from src.forecasting.guard import RecipeAccess

    key = (str(channel_path), str(search_dir))
    if key not in _RESULTS:
        config = _config()
        table = pd.read_parquet(search_dir / "tabla_unica_desarrollo.parquet")
        predictions = development_predictions(
            table, pd.read_parquet(search_dir / "predicciones_desarrollo.parquet"),
            pd.read_parquet(search_dir / "predicciones_lineas_base_desarrollo.parquet"), familia="logistica")
        access = RecipeAccess.from_config(config, recipe_path=tmp_path / "receta_inexistente.yaml")
        diag = diagnostic_inputs(_development_rows(str(channel_path)), predictions,
                                 _product(str(channel_path)).to_pandas(), tau=TAU, k=config.decision.k,
                                 window=config.decision.W, clip=config.metricas.recorte_log_loss, access=access,
                                 rotulo="usado_para_elegir")
        _RESULTS[key] = (table, predictions, diag)
    return _RESULTS[key]


def test_regimes_2018_2025(channel_path):
    from src.forecasting.diagnostics import regime_table

    table = regime_table(_product(str(channel_path)).to_pandas(), years=list(range(2018, 2026)))
    got = {code: tuple(table[table["municipality_code"] == code].sort_values("anio")["regimen"]) for code in CODES}
    assert got == REGIMES


def test_tau_matches_the_frozen_recipe():
    import yaml

    root = Path(__file__).resolve().parents[2]
    recipe = yaml.safe_load((root / "configs" / "receta_congelada.yaml").read_text(encoding="utf-8"))
    assert recipe["decision"]["tau"] == TAU and recipe["familia_comun"] == "logistica"


def test_diagnosed_configuration_is_the_common_out_of_year_one(diagnostic):
    _, predictions, diag = diagnostic
    model = predictions[predictions["sistema"] == "modelo"]
    assert {year: set(model.loc[model["anio"] == year, "candidato"]) for year in COMMON_OUT_OF_YEAR} == {
        year: {name} for year, name in COMMON_OUT_OF_YEAR.items()}
    cells = diag.casillas
    assert all(cells.loc[cells["anio"] == year, "candidato"].eq(name).all() for year, name in COMMON_OUT_OF_YEAR.items())
    assert set(cells["rotulo"]) == {"usado_para_elegir"}


def test_cells_agree_with_the_single_table(diagnostic):
    """Internal consistency (no literal): every cell score equals the single score table of F5b."""
    table, _, diag = diagnostic
    scores = table[table["familia"] == "logistica"].set_index(["municipality_code", "anio", "candidato"])
    for cell in diag.casillas.itertuples():
        row = scores.loc[(cell.municipality_code, cell.anio, cell.candidato)]
        assert cell.logloss_modelo == pytest.approx(row["logloss"], rel=1e-12)
        assert cell.brier_modelo == pytest.approx(row["brier"], rel=1e-12)
        assert cell.logloss_b2 == pytest.approx(row["logloss_b2"], rel=1e-12)
        assert cell.brier_b2 == pytest.approx(row["brier_b2"], rel=1e-12)
        assert cell.logloss_b1 == pytest.approx(row["logloss_b1"], rel=1e-12)
        assert cell.brier_b1 == pytest.approx(row["brier_b1"], rel=1e-12)


def test_cell_sizes_and_regimes(diagnostic):
    _, _, diag = diagnostic
    cells = diag.casillas
    assert len(cells) == 24 and (cells["filas"] == 52).all() and int(cells["positivas"].sum()) == 319
    assert cells["regimen"].value_counts().to_dict() == DEVELOPMENT_REGIME_COUNTS
    for cell in cells.itertuples():
        assert cell.regimen == REGIMES[cell.municipality_code][cell.anio - 2018]


def test_cell_differences_and_wins(diagnostic):
    _, _, diag = diagnostic
    cells = diag.casillas.set_index(["municipality_code", "anio"])
    assert sorted(cells.index) == sorted(CELLS)
    for key, (positives, d_ll_b2, d_br_b2, d_ll_b1, d_br_b1, w1, w2, w3, w4) in CELLS.items():
        cell = cells.loc[key]
        assert cell["positivas"] == positives
        assert cell["dif_logloss_b2"] == pytest.approx(d_ll_b2, rel=1e-9)
        assert cell["dif_brier_b2"] == pytest.approx(d_br_b2, rel=1e-9)
        assert cell["dif_logloss_b1"] == pytest.approx(d_ll_b1, rel=1e-9)
        assert cell["dif_brier_b1"] == pytest.approx(d_br_b1, rel=1e-9)
        assert (bool(cell["gana_logloss_b2"]), bool(cell["gana_brier_b2"]), bool(cell["gana_logloss_b1"]),
                bool(cell["gana_brier_b1"])) == (w1, w2, w3, w4)
    assert {column: int(diag.casillas[column].sum()) for column in WINS} == WINS
    by_regime = diag.casillas.groupby("regimen")[list(WINS)].sum()
    assert {regime: tuple(int(v) for v in by_regime.loc[regime]) for regime in WINS_BY_REGIME} == WINS_BY_REGIME
    summary = diag.resumenes[diag.resumenes["nivel"] == "regimen"].set_index("grupo")
    for regime, wins in WINS_BY_REGIME.items():
        assert tuple(int(summary.loc[regime, column]) for column in (
            "victorias_logloss_b2", "victorias_brier_b2", "victorias_logloss_b1", "victorias_brier_b1")) == wins


def test_transition_strata_by_year(diagnostic):
    _, _, diag = diagnostic
    strata = diag.transicion.set_index(["anio", "estrato"])
    assert sorted(strata.index) == sorted(TRANSITION)
    for key, (rows, positives, gain, bss) in TRANSITION.items():
        assert (strata.loc[key, "filas"], strata.loc[key, "positivas"]) == (rows, positives)
        assert strata.loc[key, "mejora_logloss_b2"] == pytest.approx(gain, rel=1e-9)
        assert strata.loc[key, "bss_b2"] == pytest.approx(bss, rel=1e-9)


def _same(value, expected):
    if isinstance(expected, float) and math.isnan(expected):
        return value is None or (isinstance(value, float) and math.isnan(value))
    return value == expected


def test_onsets_model_and_channel(diagnostic):
    _, _, diag = diagnostic
    onsets = diag.eventos[diag.eventos["tipo"] == "inicio"]
    assert len(onsets) == 11 and int(onsets["evaluable"].sum()) == 10
    got = {(r.municipality_code, r.semana): r for r in onsets.itertuples()}
    assert sorted(got) == sorted(ONSETS)
    for key, (evaluable, censoring, warned_model, warned_channel, lead_model, lead_channel, delta) in ONSETS.items():
        r = got[key]
        assert bool(r.evaluable) == evaluable
        assert (None if not isinstance(r.censura, str) else r.censura) == censoring
        assert (bool(r.avisado_modelo), bool(r.avisado_canal)) == (warned_model, warned_channel)
        assert _same(r.anticipacion_modelo, lead_model) and _same(r.anticipacion_canal, lead_channel)
        assert _same(r.delta_w, delta)
        assert r.regimen == REGIMES[r.municipality_code][r.anio - 2018]
    evaluable = onsets[onsets["evaluable"]]
    assert (int(evaluable["avisado_modelo"].sum()), int(evaluable["avisado_canal"].sum())) == (6, 10)


def test_entries(diagnostic):
    _, _, diag = diagnostic
    entries = diag.eventos[diag.eventos["tipo"] == "entrada"]
    evaluable = entries[entries["evaluable"]]
    assert {"total": len(entries), "evaluable": len(evaluable), "warned_model": int(evaluable["avisado_modelo"].sum()),
            "warned_channel": int(evaluable["avisado_canal"].sum())} == ENTRIES


def test_alerts_of_model_and_channel(diagnostic):
    _, _, diag = diagnostic
    for system, expected in ALERTS.items():
        own = diag.alertas[diag.alertas["sistema"] == system]
        non_continuity = own[~own["continuidad"]]
        judgeable = non_continuity[non_continuity["juzgable"]]
        got = (len(own), int(own["continuidad"].sum()), len(judgeable), int(judgeable["falsa_alarma"].sum()),
               int((judgeable["falsa_alarma"] & judgeable["seguida_de_epidemia"]).sum()),
               int((~non_continuity["juzgable"]).sum()))
        assert got == expected, system


def test_weekly_series(diagnostic, channel_path):
    _, _, diag = diagnostic
    serie = diag.serie
    assert len(serie) == 1248 and not serie.duplicated(["municipality_code", "t"]).any()
    observed = _product(str(channel_path)).to_pandas().set_index(["municipality_code", "week_start"])
    keys = list(zip(serie["municipality_code"], serie["target_week"]))
    assert serie["casos_objetivo"].tolist() == [int(observed.loc[k, "confirmed_records_screening"]) for k in keys]
    assert serie["central_objetivo"].tolist() == [float(observed.loc[k, "epico_central"]) for k in keys]
    assert serie["u1e_objetivo"].tolist() == [float(observed.loc[k, "epico_upper"]) for k in keys]
    assert serie["y"].tolist() == [int(observed.loc[k, "target_u1e"]) for k in keys]
