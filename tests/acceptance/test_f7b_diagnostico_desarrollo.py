"""F7b acceptance with the real development data (--channel-parquet and --sin-clima-dir): integrated diagnostic
analysis in development for the recipe of D7.2 (contrato §6.6, §8.8 and §10.22; especificación §6.F7; DV-02).
Replaces test_f7_diagnostico_desarrollo.py::test_tau_matches_the_frozen_recipe, superseded by DV-02; the other tests
of that file stay as the record of the full-set diagnostic. Approved by Jesús on 2026-11-12.

Nothing is retrained: the inputs are the saved predictions of the sin_clima search (ae6225180d37d025224b). In each
development cell the diagnosed model is the common out-of-year configuration of the recipe family (logistica) in the
recipe feature set (sin_clima), labelled "usado_para_elegir"; B2 and B1 are the saved baselines of that search
(identical to F5b). Events use tau = 0.05, k = 2 and W = 12 of the frozen recipe (de74d4ab…fda5, 6fc68f6). Because
the configuration changes by year (decision D3), the model's false alarms (24) differ from those of the tau curve of
F6b (22, logistica_05 in the four years); they are reported as they are.

Provenance of the literals: computed by diagnostics.py (caae344) from 2a1ab0b; evidence a1d1cc0 in
reports/forecasting/evidence/diagnostico_desarrollo_sin_clima_20261112/; reproduced by Jesús's independent check
docs/forecasting/verificacion/caso_f7_diagnostico.py (md5 f4a32dc83b4d6b330291e0ee105e3b92, commit 6e21cc9; it does
not import src/): the 96 cell differences (rel 1e-9), wins in total and by regime, the 10 transition strata, the 11
onsets with leads and delta_w, and the alert, false-alarm and entry counts of model and channel. Tolerances: exact for
counts, labels and weeks; rel 1e-9 for differences and improvements. The regimes are the observed-series literals of
test_f7_diagnostico_desarrollo.py (recipe-independent); the channel figures are the regression of F4, F6 and F6b.
"""

from datetime import date
import functools
import math
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytestmark = pytest.mark.acceptance

ROOT = Path(__file__).resolve().parents[2]
RECIPE_HASH = "de74d4abd33537433ffc9d1572c6cdd8620a2608fcc2b67ee4a1752aa865fda5"
TAU = 0.05
CODES = ("08001", "08758", "41001", "50001", "54001", "76001")
COMMON_OUT_OF_YEAR = {2019: "logistica_04", 2020: "logistica_05", 2021: "logistica_04", 2022: "logistica_04"}
E, P, T = "epidemica", "post_epidemica", "tranquila"
REGIMES = {  # 2019 .. 2022 (development cells)
    "08001": (T, E, E, E), "08758": (E, P, E, E), "41001": (E, P, T, T), "50001": (E, P, T, E),
    "54001": (E, P, E, P), "76001": (E, E, P, T)}
DEVELOPMENT_REGIME_COUNTS = {E: 13, P: 6, T: 5}
# (municipality, year): (positives, model logloss - B2, model Brier - B2, model logloss - B1, model Brier - B1,
#                        win logloss vs B2, win Brier vs B2, win logloss vs B1, win Brier vs B1)
CELLS = {
    ("08001", 2019): (1, -0.0097032871749716, -0.00124111967766, -0.0884098133431464, -0.0176981374743722, True, True, True, True),
    ("08001", 2020): (6, 0.0433462831687447, 0.0187963177786225, -0.0191621087294304, 0.0251801639216723, False, False, True, False),
    ("08001", 2021): (20, -0.1470570260925202, -0.0525743497045697, -0.3910844165541596, -0.1484237813016651, True, True, True, True),
    ("08001", 2022): (51, -0.2173706707216309, -0.0893127697350985, -1.5679984820854809, -0.5938678558841255, True, True, True, True),
    ("08758", 2019): (13, 0.081762976851866, 0.0184537904110142, 0.1498513093520955, 0.0317312986613746, False, False, False, False),
    ("08758", 2020): (4, 0.0547938134765247, 0.0285783192972346, 0.0492047478595046, 0.0473111639767587, False, False, False, False),
    ("08758", 2021): (27, -0.2562082106081409, -0.0834547994482165, -0.4144640909374945, -0.1715641615710187, True, True, True, True),
    ("08758", 2022): (43, -0.192279305143194, -0.0709271849373923, -0.7117595658971425, -0.3101830657058986, True, True, True, True),
    ("41001", 2019): (35, -0.2783048751068364, -0.0997247115400737, -0.8663357728916666, -0.3147122091342492, True, True, True, True),
    ("41001", 2020): (0, 0.0588033277679244, 0.0321318190720782, -0.075975398546919, 0.0070558295880016, False, False, True, False),
    ("41001", 2021): (0, -0.011619990714311, -0.0020432600411075, -0.1972704827426242, -0.0380704408855962, True, True, True, True),
    ("41001", 2022): (0, -0.0873143339086123, -0.0136797377030428, -0.1160986152347376, -0.0173580055889319, True, True, True, True),
    ("50001", 2019): (45, -0.4116800882300581, -0.1761324174565428, -1.4601812433742614, -0.5439772699954308, True, True, True, True),
    ("50001", 2020): (0, 0.0284789348378394, 0.0171695276554227, -0.0647194004561242, 0.0145687758037762, False, False, True, False),
    ("50001", 2021): (0, -0.0121428875711836, -0.0009289627895438, -0.2370369324768252, -0.0465748636561537, True, True, True, True),
    ("50001", 2022): (10, -0.1051738374424488, -0.0236915442802189, -0.2483256612561373, -0.0689590940232729, True, True, True, True),
    ("54001", 2019): (21, -0.1809800961531107, -0.0769755793252426, -0.8443024040888889, -0.2412414804166725, True, True, True, True),
    ("54001", 2020): (0, -0.0400493451137498, -0.0055836755357694, -0.1668962858445933, -0.0275983906426846, True, True, True, True),
    ("54001", 2021): (23, 0.0867462122983201, 0.0329020918725417, -0.4035083681683112, -0.132683552556137, False, False, True, True),
    ("54001", 2022): (0, -0.0856917012394598, -0.0198534925199557, -0.0243332511421145, 0.0098895505805668, True, True, True, False),
    ("76001", 2019): (4, -0.0390276628139304, -0.0088505769381121, -0.120305765715071, -0.0210331120270931, True, True, True, True),
    ("76001", 2020): (14, 0.076950462660015, 0.0261988505947245, 0.0197402225099961, -0.0027034834221044, False, False, False, True),
    ("76001", 2021): (2, 0.0149571471803355, 7.248882301621323e-05, -0.0287600923387034, -0.0069532826463188, False, False, True, True),
    ("76001", 2022): (0, -0.0736470344888335, -0.0088797262692556, -0.1108922515986627, -0.0207916103142615, True, True, True, True),
}
WINS = {"gana_logloss_b2": 16, "gana_brier_b2": 16, "gana_logloss_b1": 21, "gana_brier_b1": 18}
WINS_BY_REGIME = {  # regime: (logloss vs B2, Brier vs B2, logloss vs B1, Brier vs B1)
    E: (9, 9, 11, 11), P: (2, 2, 5, 2), T: (5, 5, 5, 5)}
# (year, stratum): (rows, positives, log-loss improvement vs B2, BSS vs B2); "dentro" = zona_t "epidemia"
TRANSITION = {
    ("2019", "fuera"): (197, 30, 0.1265052327149359, 0.1115267677576693),
    ("2019", "dentro"): (115, 89, 0.3871225505859396, 0.473808005726987),
    ("2020", "fuera"): (271, 12, -0.1711281255288712, -0.4610349150005841),
    ("2020", "dentro"): (41, 12, -0.1016707247201336, -0.0297296052919533),
    ("2021", "fuera"): (260, 31, 0.1735852763076691, 0.176339588091353),
    ("2021", "dentro"): (52, 41, 0.082124760912258, 0.0868818552606076),
    ("2022", "fuera"): (210, 15, 0.5369833832536256, 0.5393721604237717),
    ("2022", "dentro"): (102, 89, 0.1491269421934932, 0.1756302966506235),
    ("todos", "fuera"): (938, 88, 0.175127045784431, 0.1234032759992328),
    ("todos", "dentro"): (310, 231, 0.1950158383573021, 0.251653810722254),
}
NAN = math.nan
# (municipality, week): (evaluable, censoring, warned by model, warned by channel, model lead, channel lead, delta_w)
ONSETS = {
    ("08001", date(2020, 1, 27)): (True, None, True, True, 8, 1, 7),
    ("08001", date(2021, 8, 23)): (True, None, True, True, 7, 7, 0),
    ("08758", date(2019, 5, 27)): (True, None, True, True, 2, 4, -2),
    ("08758", date(2019, 11, 4)): (True, None, True, True, 2, 4, -2),
    ("08758", date(2022, 2, 28)): (True, None, False, True, NAN, 2, NAN),
    ("41001", date(2019, 2, 18)): (True, None, True, True, 9, 4, 5),
    ("50001", date(2019, 1, 28)): (False, "ventana", True, True, 10, 1, NAN),
    ("50001", date(2019, 11, 25)): (True, None, False, True, NAN, 1, NAN),
    ("50001", date(2022, 10, 31)): (True, None, True, True, 8, 4, 4),
    ("54001", date(2021, 6, 21)): (True, None, True, True, 2, 1, 1),
    ("76001", date(2019, 12, 16)): (True, None, True, True, 8, 2, 6),
}
ENTRIES = {"total": 45, "evaluable": 44, "warned_model": 21, "warned_channel": 36}
# system: (alerts, continuity alerts, judgeable non-continuity alerts, false alarms, false alarms followed by an
#          epidemic week, censored non-continuity alerts)
ALERTS = {"modelo": (43, 5, 38, 24, 6, 0), "canal": (65, 0, 63, 48, 12, 2)}


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
    from src.forecasting.run_selection import development_rows

    config = _config()
    examples = build_examples(_product(channel_file), h=config.horizonte.h, canal=config.canal,
                              features=config.features)
    return development_rows(examples, config)


_RESULTS = {}


@pytest.fixture
def diagnostic(channel_path, sin_clima_dir, tmp_path):
    from src.forecasting.diagnostics import development_predictions, diagnostic_inputs
    from src.forecasting.guard import RecipeAccess

    key = (str(channel_path), str(sin_clima_dir))
    if key not in _RESULTS:
        config = _config()
        table = pd.read_parquet(sin_clima_dir / "tabla_unica_desarrollo.parquet")
        predictions = development_predictions(
            table, pd.read_parquet(sin_clima_dir / "predicciones_desarrollo.parquet"),
            pd.read_parquet(sin_clima_dir / "predicciones_lineas_base_desarrollo.parquet"), familia="logistica")
        access = RecipeAccess.from_config(config, recipe_path=tmp_path / "receta_inexistente.yaml")
        diag = diagnostic_inputs(_development_rows(str(channel_path)), predictions,
                                 _product(str(channel_path)).to_pandas(), tau=TAU, k=config.decision.k,
                                 window=config.decision.W, clip=config.metricas.recorte_log_loss, access=access,
                                 rotulo="usado_para_elegir")
        _RESULTS[key] = (table, predictions, diag)
    return _RESULTS[key]


def test_recipe_is_the_one_of_d72():
    recipe = yaml.safe_load((ROOT / "configs" / "receta_congelada.yaml").read_text(encoding="utf-8"))
    assert recipe["hash_sha256"] == RECIPE_HASH
    assert recipe["decision"]["tau"] == TAU and recipe["familia_comun"] == "logistica"
    assert {entry["conjunto_variables"] for entry in recipe["municipios"].values()} == {"sin_clima"}


def test_diagnosed_configuration_is_the_common_out_of_year_one(diagnostic):
    _, predictions, diag = diagnostic
    model = predictions[predictions["sistema"] == "modelo"]
    assert {year: set(model.loc[model["anio"] == year, "candidato"]) for year in COMMON_OUT_OF_YEAR} == {
        year: {name} for year, name in COMMON_OUT_OF_YEAR.items()}
    cells = diag.casillas
    assert all(cells.loc[cells["anio"] == year, "candidato"].eq(name).all() for year, name in COMMON_OUT_OF_YEAR.items())
    assert set(cells["rotulo"]) == {"usado_para_elegir"}


def test_cells_agree_with_the_sin_clima_table(diagnostic):
    """Internal consistency (no literal): every cell score equals the single table of the sin_clima search."""
    table, _, diag = diagnostic
    scores = table[table["familia"] == "logistica"].set_index(["municipality_code", "anio", "candidato"])
    for cell in diag.casillas.itertuples():
        row = scores.loc[(cell.municipality_code, cell.anio, cell.candidato)]
        for mine, theirs in (("logloss_modelo", "logloss"), ("brier_modelo", "brier"), ("logloss_b2", "logloss_b2"),
                             ("brier_b2", "brier_b2"), ("logloss_b1", "logloss_b1"), ("brier_b1", "brier_b1")):
            assert getattr(cell, mine) == pytest.approx(row[theirs], rel=1e-12)


def test_cell_sizes_and_regimes(diagnostic):
    _, _, diag = diagnostic
    cells = diag.casillas
    assert len(cells) == 24 and (cells["filas"] == 52).all() and int(cells["positivas"].sum()) == 319
    assert cells["regimen"].value_counts().to_dict() == DEVELOPMENT_REGIME_COUNTS
    for cell in cells.itertuples():
        assert cell.regimen == REGIMES[cell.municipality_code][cell.anio - 2019]


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
    evaluable = onsets[onsets["evaluable"]]
    assert (int(evaluable["avisado_modelo"].sum()), int(evaluable["avisado_canal"].sum())) == (8, 10)


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
    assert serie["y"].tolist() == [int(observed.loc[k, "target_u1e"]) for k in keys]


def test_evidence_matches_the_recomputation(diagnostic):
    """The committed evidence (a1d1cc0) is the output of this diagnostic."""
    _, _, diag = diagnostic
    folder = ROOT / "reports" / "forecasting" / "evidence" / "diagnostico_desarrollo_sin_clima_20261112"
    stored = pd.read_csv(folder / "diag_casillas.csv", dtype={"municipality_code": str})
    assert stored["candidato"].tolist() == diag.casillas["candidato"].tolist()
    for column in ("logloss_modelo", "brier_modelo", "dif_logloss_b2", "dif_brier_b2"):
        assert stored[column].tolist() == pytest.approx(diag.casillas[column].tolist(), rel=1e-12)
