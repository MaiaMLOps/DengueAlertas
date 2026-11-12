"""F6b acceptance with synthetic inputs: the feature set as step 0 of the selection (contrato §6.6, D7.2; §4.4, §8.3;
especificación §6.F6b). Approved by Jesús on 2026-11-12 (D7.2 and its implementation plan); validated with a
disposable reference and three negative controls before freezing.

Rules (contrato §6.6):
- two feature sets, dataset.FEATURE_SETS: "completo" (the 15 of §5.2) and "sin_clima" (the 6 without the climate
  block); the default is "sin_clima" (dataset.DEFAULT_FEATURE_SET);
- step 0, per family, with the common out-of-year configuration of each set (comparison.out_of_year_scores):
  "completo" replaces "sin_clima" only if its mean out-of-year improvement over B2 (years within municipality, then
  municipalities) is STRICTLY greater AND it wins at least win_threshold(cells, alfa) cells (17 of 24) by log-loss;
  a tie is not a win; there is no per-municipality exception for the feature set;
- steps 1-4 and tau of F6 then run on the chosen set of each family; with the full set forced the result is the F6
  selection on the full-set table, exactly;
- the ablation (contrato §4.4) is, in each municipality, its final family with the OTHER set, configured with the
  same protocol: common configuration and hyperparameter exception (selection.hyperparameter_choices on that set);
- the recipe draft adds conjunto_variables (per family) and ablacion (per municipality) to the F6 fields, and each
  municipality names its feature set; the provenance names the search of each set.

API frozen here (`src.forecasting.selection`):
- `feature_set_cells(tables, *, familia) -> DataFrame` with CELL_COLUMNS below (one row per municipality-year);
  `tables` maps "completo" and "sin_clima" to the single score table of each search;
- `choose_feature_set(tables, *, familia, alpha) -> FeatureSetChoice(familia, conjunto, mejora, victorias, casillas,
  umbral, reemplaza)`, mejora mapping each set to its mean out-of-year improvement;
- `select_with_feature_sets(tables, predictions, rows, observed, *, config, access, forced_set=None) -> SetSelection`
  with fields conjuntos (family -> FeatureSetChoice), celdas_conjuntos (feature_set_cells of both families),
  seleccion (the F6 `Selection` on the chosen sets), conjuntos_municipio (code -> feature set of its model),
  ablacion (ABLATION_COLUMNS) and comparaciones_ablacion (hyperparameter comparisons of the ablation);
- `build_recipe_with_feature_sets(set_selection, *, config, provenance) -> dict`; provenance has commit_codigo,
  commit_contrato, version_dvc_canal and busquedas {set: {busqueda_id, insumos_md5}}.
"""

from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest

from src.epidemiology.config import EpidemiologyValidationError

pytestmark = pytest.mark.acceptance

CODES = ("08001", "08758", "41001", "50001", "54001", "76001")
YEARS = (2019, 2020, 2021, 2022)
FAMILIES = ("logistica", "lightgbm")
SETS = ("completo", "sin_clima")
NO_CLIMATE = ("zona_t", "posicion_4s", "crecimiento_4s", "cambio_umbral", "sem_sin", "sem_cos")
CELL_COLUMNS = ["familia", "municipality_code", "anio", "candidato_completo", "candidato_sin_clima",
                "logloss_completo", "logloss_sin_clima", "mejora_completo", "mejora_sin_clima", "gana_completo"]
ABLATION_COLUMNS = ["municipality_code", "familia", "conjunto_variables", "candidato", "orden", "parametros",
                    "excepcion_hiperparametros", "excepcion_hiperparametros_nominal"]
F6_TOP_LEVEL = {"familia_comun", "comparacion_familias", "municipios", "decision", "configuracion", "procedencia"}
WEEK = timedelta(weeks=1)


def _config():
    from src.forecasting.config import load_config

    return load_config()


def _alpha():
    return _config().comparacion.alfa


def _table(candidates, loss, *, codes=CODES, reference=1.0):
    """candidates: family -> names in pre-registered order; loss(code, family, name, year) -> logloss."""
    rows = []
    for family, names in candidates.items():
        for position, name in enumerate(names):
            for code in codes:
                for year in YEARS:
                    value = float(loss(code, family, name, year))
                    rows.append({"municipality_code": code, "familia": family, "candidato": name, "orden": position,
                                 "parametros": json.dumps({"id": name}, sort_keys=True), "anio": year,
                                 "logloss": value, "logloss_b2": reference, "mejora_b2": 1 - value / reference,
                                 "n_filas": 52, "primera_semana_objetivo": date(year, 1, 7),
                                 "ultima_semana_objetivo": date(year, 12, 30)})
    return pd.DataFrame(rows)


def _cells():
    return [(code, year) for code in CODES for year in YEARS]


def _set_tables(completo_losses, sin_loss=0.5):
    """One candidate per family; logistic regression of the full set by cell (code-major order), the rest constant."""
    values = dict(zip(_cells(), completo_losses))
    completo = _table({"logistica": ["r"], "lightgbm": ["g"]},
                      lambda code, family, name, year: values[(code, year)] if family == "logistica" else 0.9)
    sin = _table({"logistica": ["r"], "lightgbm": ["g"]},
                 lambda code, family, name, year: sin_loss if family == "logistica" else 0.9)
    return {"completo": completo, "sin_clima": sin}


def _spread(wins, win_loss, lose_loss):
    """`wins` cells with win_loss, spread over the municipalities, the rest with lose_loss."""
    order = sorted(range(24), key=lambda i: (i % 4, i // 4))   # year-major spread over municipalities
    chosen = set(order[:wins])
    return [win_loss if i in chosen else lose_loss for i in range(24)]


def _choose(tables, familia="logistica"):
    from src.forecasting.selection import choose_feature_set

    return choose_feature_set(tables, familia=familia, alpha=_alpha())


# --------------------------------------------------------------------------- constants
def test_feature_sets():
    from src.forecasting.dataset import DEFAULT_FEATURE_SET, FEATURE_SETS, FEATURES

    assert DEFAULT_FEATURE_SET == "sin_clima"
    assert set(FEATURE_SETS) == set(SETS)
    assert FEATURE_SETS["completo"] == FEATURES and len(FEATURES) == 15
    assert FEATURE_SETS["sin_clima"] == NO_CLIMATE


# --------------------------------------------------------------------------- step 0: parsimony
def test_full_set_with_better_mean_and_16_of_24_does_not_replace():
    tables = _set_tables(_spread(16, 0.40, 0.55))   # mean improvement 0.55 > 0.50, 16 wins
    choice = _choose(tables)
    assert choice.mejora["completo"] == pytest.approx(0.55) and choice.mejora["sin_clima"] == pytest.approx(0.5)
    assert (choice.victorias, choice.casillas, choice.umbral) == (16, 24, 17)
    assert (choice.conjunto, choice.reemplaza, choice.familia) == ("sin_clima", False, "logistica")


def test_full_set_with_better_mean_and_17_of_24_replaces():
    choice = _choose(_set_tables(_spread(17, 0.40, 0.55)))
    assert choice.mejora["completo"] > choice.mejora["sin_clima"]
    assert (choice.victorias, choice.umbral, choice.conjunto, choice.reemplaza) == (17, 17, "completo", True)


def test_equal_mean_does_not_replace_even_with_18_wins():
    # per municipality: three years with improvement 0.625 and one with 0.125 -> mean exactly 0.5, like sin_clima
    losses = [0.875 if year == 2022 else 0.375 for _, year in _cells()]
    choice = _choose(_set_tables(losses))
    assert choice.mejora["completo"] == choice.mejora["sin_clima"] == 0.5
    assert choice.victorias == 18
    assert (choice.conjunto, choice.reemplaza) == ("sin_clima", False)


def test_worse_mean_does_not_replace_even_with_17_wins():
    choice = _choose(_set_tables(_spread(17, 0.49, 0.90)))
    assert choice.victorias == 17 and choice.mejora["completo"] < choice.mejora["sin_clima"]
    assert choice.conjunto == "sin_clima"


def test_a_tie_is_not_a_win():
    choice = _choose(_set_tables([0.5] * 24))
    assert choice.victorias == 0 and choice.conjunto == "sin_clima"


def test_cells_of_the_comparison():
    from src.forecasting.selection import feature_set_cells

    tables = _set_tables(_spread(17, 0.40, 0.55))
    cells = feature_set_cells(tables, familia="logistica")
    assert list(cells.columns) == CELL_COLUMNS and len(cells) == 24
    assert set(zip(cells["municipality_code"], cells["anio"])) == set(_cells())
    assert (cells["gana_completo"] == (cells["logloss_completo"] < cells["logloss_sin_clima"])).all()
    assert int(cells["gana_completo"].sum()) == 17
    assert set(cells["candidato_completo"]) == {"r"} and set(cells["familia"]) == {"logistica"}


def test_each_family_chooses_its_own_set():
    from src.forecasting.selection import choose_feature_set

    completo = _table({"logistica": ["r"], "lightgbm": ["g"]},
                      lambda code, family, name, year: 0.6 if family == "logistica" else 0.4)
    sin = _table({"logistica": ["r"], "lightgbm": ["g"]}, lambda code, family, name, year: 0.5)
    tables = {"completo": completo, "sin_clima": sin}
    assert choose_feature_set(tables, familia="logistica", alpha=_alpha()).conjunto == "sin_clima"
    assert choose_feature_set(tables, familia="lightgbm", alpha=_alpha()).conjunto == "completo"


# --------------------------------------------------------------------------- synthetic world for the whole selection
WORLD_CODES = ("08001", "76001")
FIRST_EMISSION = date(2018, 11, 12)
N_ROWS = 208
H = 8


def _world(seed=20261112):
    from src.epidemiology.channel_epico import ZONES

    rng = np.random.default_rng(seed)
    observed_weeks = [date(2017, 1, 2) + i * WEEK for i in range(6 * 52 + 2)]
    observed, rows = [], []
    for c, code in enumerate(WORLD_CODES):
        exceed = {week: int((i + 7 * c) % 26 in (10, 11, 12, 13)) for i, week in enumerate(observed_weeks)}
        observed += [{"municipality_code": code, "week_start": week, "target_u1e": exceed[week],
                      "channel_eligible": True} for week in observed_weeks]
        zones = rng.choice(list(ZONES), size=N_ROWS, p=[0.6, 0.3, 0.1])
        for i in range(N_ROWS):
            t = FIRST_EMISSION + i * WEEK
            rows.append({"municipality_code": code, "t": t, "target_week": t + H * WEEK, "zona_t": str(zones[i]),
                         "y": exceed[t + H * WEEK]})
    observed = pd.DataFrame(observed)
    observed["channel_eligible"] = observed["channel_eligible"].astype(bool)
    return pd.DataFrame(rows), observed, rng


def _access(tmp_path):
    from src.forecasting.guard import RecipeAccess

    config = _config()
    return RecipeAccess(test_years=tuple(config.validacion.prueba), channel_version=config.datos.version_dvc_canal,
                        recipe_path=tmp_path / "receta_inexistente.yaml")


NAMES = {"logistica": ["r0", "r1"], "lightgbm": ["g0", "g1"]}


def _world_search(rows, rng, losses):
    """losses[set](code, family, name, year) -> logloss; predictions differ by set, family and candidate."""
    tables, predictions = {}, {}
    for feature_set in SETS:
        tables[feature_set] = _table(NAMES, losses[feature_set], codes=WORLD_CODES, reference=0.6)
        frames = []
        for family, names in NAMES.items():
            for name in names:
                p = np.clip(0.35 * rows["y"].to_numpy() + 0.65 * rng.uniform(size=len(rows)), 0.0, 1.0)
                frames.append(rows[["municipality_code", "t", "target_week", "y"]].assign(
                    familia=family, candidato=name, anio=[w.year for w in rows["target_week"]], p=p))
        predictions[feature_set] = pd.concat(frames, ignore_index=True)[
            ["municipality_code", "t", "target_week", "familia", "candidato", "anio", "y", "p"]]
    return tables, predictions


def _constant(values):
    return lambda code, family, name, year: values[(family, name)]


MIXED = {  # logistic regression keeps sin_clima, LightGBM takes completo (8 of 8, better mean) and wins the families
    "completo": _constant({("logistica", "r0"): 0.50, ("logistica", "r1"): 0.55,
                           ("lightgbm", "g0"): 0.40, ("lightgbm", "g1"): 0.45}),
    "sin_clima": _constant({("logistica", "r0"): 0.48, ("logistica", "r1"): 0.45,
                            ("lightgbm", "g0"): 0.60, ("lightgbm", "g1"): 0.65}),
}


def _combined(frames, chosen):
    return pd.concat([frames[chosen[family]][frames[chosen[family]]["familia"] == family] for family in FAMILIES],
                     ignore_index=True)


def _same_selection(first, second):
    assert first.familias == second.familias and first.tau == second.tau
    pd.testing.assert_frame_equal(first.elecciones, second.elecciones, check_exact=True)
    pd.testing.assert_frame_equal(first.comparaciones, second.comparaciones, check_exact=True)
    pd.testing.assert_frame_equal(first.curva, second.curva, check_exact=True)
    pd.testing.assert_frame_equal(first.predicciones, second.predicciones, check_exact=True)


_RUNS = {}


def _run(tmp_path, losses_key, forced_set=None):
    from src.forecasting.selection import select_with_feature_sets

    key = (losses_key, forced_set)
    if key not in _RUNS:
        rows, observed, rng = _world()
        tables, predictions = _world_search(rows, rng, WORLDS[losses_key])
        result = select_with_feature_sets(tables, predictions, rows, observed, config=_config(),
                                          access=_access(tmp_path), forced_set=forced_set)
        _RUNS[key] = (rows, observed, tables, predictions, result)
    return _RUNS[key]


def test_steps_1_to_4_and_tau_run_on_the_chosen_set_of_each_family(tmp_path):
    from src.forecasting.selection import select

    rows, observed, tables, predictions, result = _run(tmp_path, "mixed")
    assert result.conjuntos["logistica"].conjunto == "sin_clima"
    assert result.conjuntos["lightgbm"].conjunto == "completo"
    assert result.conjuntos["lightgbm"].victorias == 8 and result.conjuntos["lightgbm"].umbral == 7
    chosen = {"logistica": "sin_clima", "lightgbm": "completo"}
    expected = select(_combined(tables, chosen), _combined(predictions, chosen), rows, observed, config=_config(),
                      access=_access(tmp_path))
    _same_selection(result.seleccion, expected)
    assert result.seleccion.familias.familia_comun == "lightgbm"
    assert result.conjuntos_municipio == {code: "completo" for code in WORLD_CODES}
    cells = result.celdas_conjuntos
    assert list(cells.columns) == CELL_COLUMNS and len(cells) == 2 * len(WORLD_CODES) * len(YEARS)


def test_forced_full_set_reproduces_the_f6_selection(tmp_path):
    from src.forecasting.selection import select

    rows, observed, tables, predictions, result = _run(tmp_path, "mixed", forced_set="completo")
    expected = select(tables["completo"], predictions["completo"], rows, observed, config=_config(),
                      access=_access(tmp_path))
    _same_selection(result.seleccion, expected)
    assert set(result.conjuntos_municipio.values()) == {"completo"}


NO_SET_EXCEPTION = {  # full set wins the 4 years of 08001 and loses those of 76001: 4 of 8, so sin_clima everywhere
    "completo": lambda code, family, name, year: (0.9 if family == "lightgbm"
                                                  else (0.40 if code == "08001" else 0.60) + (0.05 if name == "r1" else 0)),
    "sin_clima": lambda code, family, name, year: 0.9 if family == "lightgbm" else 0.50 + (0.05 if name == "r1" else 0),
}
ABLATION_EXCEPTION = {  # logistic regression keeps sin_clima; in the full set 76001 has its own best r1 (4 of 4)
    "completo": lambda code, family, name, year: (0.9 if family == "lightgbm" else
                                                  {("08001", "r0"): 0.40, ("08001", "r1"): 0.60,
                                                   ("76001", "r0"): 0.55, ("76001", "r1"): 0.45}[(code, name)]),
    "sin_clima": lambda code, family, name, year: 0.9 if family == "lightgbm" else (0.30 if name == "r0" else 0.35),
}
WORLDS = {"mixed": MIXED, "no_set_exception": NO_SET_EXCEPTION, "ablation_exception": ABLATION_EXCEPTION}


def test_no_per_municipality_exception_for_the_feature_set(tmp_path):
    _, _, _, _, result = _run(tmp_path, "no_set_exception")
    cells = result.celdas_conjuntos
    own = cells[(cells["familia"] == "logistica") & (cells["municipality_code"] == "08001")]
    assert own["gana_completo"].all()                       # the full set wins the 4 years of 08001 ...
    assert result.conjuntos["logistica"].victorias == 4     # ... but only 4 of 8 cells
    assert result.seleccion.familias.familia_comun == "logistica"
    assert result.conjuntos_municipio == {"08001": "sin_clima", "76001": "sin_clima"}
    assert set(result.ablacion["conjunto_variables"]) == {"completo"}


def test_ablation_is_the_final_family_with_the_other_set_and_its_own_protocol(tmp_path):
    from src.forecasting.selection import hyperparameter_choices

    _, _, tables, _, result = _run(tmp_path, "ablation_exception")
    assert result.conjuntos["logistica"].conjunto == "sin_clima"
    ablation = result.ablacion.set_index("municipality_code")
    assert list(result.ablacion.columns) == ABLATION_COLUMNS
    assert sorted(ablation.index) == list(WORLD_CODES)
    expected, comparisons = hyperparameter_choices(tables["completo"], familia="logistica")
    expected = expected.set_index("municipality_code")
    for code in WORLD_CODES:
        assert (ablation.loc[code, "familia"], ablation.loc[code, "conjunto_variables"]) == ("logistica", "completo")
        for column in ("candidato", "orden", "parametros", "excepcion_hiperparametros",
                       "excepcion_hiperparametros_nominal"):
            assert ablation.loc[code, column] == expected.loc[code, column]
    assert (ablation.loc["08001", "candidato"], ablation.loc["76001", "candidato"]) == ("r0", "r1")
    assert bool(ablation.loc["76001", "excepcion_hiperparametros"]) is True
    pd.testing.assert_frame_equal(result.comparaciones_ablacion.reset_index(drop=True),
                                  comparisons.reset_index(drop=True), check_exact=True)
    model = result.seleccion.elecciones.set_index("municipality_code")
    assert set(model["candidato"]) == {"r0"}                 # the recipe itself has no exception in sin_clima


def test_both_sets_are_required(tmp_path):
    from src.forecasting.selection import select_with_feature_sets

    rows, observed, rng = _world()
    tables, predictions = _world_search(rows, rng, MIXED)
    with pytest.raises(EpidemiologyValidationError):
        select_with_feature_sets({"completo": tables["completo"]}, predictions, rows, observed, config=_config(),
                                 access=_access(tmp_path))
    with pytest.raises(EpidemiologyValidationError):
        select_with_feature_sets(tables, predictions, rows, observed, config=_config(), access=_access(tmp_path),
                                 forced_set="parcial")


# --------------------------------------------------------------------------- recipe draft
def _provenance(**changes):
    provenance = {"commit_codigo": "a" * 40, "commit_contrato": "b" * 40,
                  "version_dvc_canal": _config().datos.version_dvc_canal,
                  "busquedas": {"completo": {"busqueda_id": "7f34b69832eea69407c6",
                                             "insumos_md5": {"tabla_unica_desarrollo.parquet": "c" * 32}},
                                "sin_clima": {"busqueda_id": "0123456789abcdef0123",
                                              "insumos_md5": {"tabla_unica_desarrollo.parquet": "d" * 32}}}}
    provenance.update(changes)
    return provenance


def test_recipe_draft_names_the_feature_sets_and_the_ablation(tmp_path):
    from src.forecasting.dataset import FEATURE_SETS
    from src.forecasting.selection import build_recipe_with_feature_sets, write_recipe_draft

    _, _, _, _, result = _run(tmp_path, "mixed")
    recipe = build_recipe_with_feature_sets(result, config=_config(), provenance=_provenance())
    assert set(recipe) == F6_TOP_LEVEL | {"conjunto_variables", "ablacion"}
    assert "aprobacion" not in recipe and "hash_sha256" not in recipe
    sets = recipe["conjunto_variables"]
    assert set(sets) == set(FAMILIES)
    for family in FAMILIES:
        choice = result.conjuntos[family]
        assert sets[family] == {"conjunto": choice.conjunto, "features": list(FEATURE_SETS[choice.conjunto]),
                                "mejora_media_fuera_de_anio": {name: float(value) for name, value in choice.mejora.items()},
                                "victorias": choice.victorias, "casillas": choice.casillas, "umbral": choice.umbral,
                                "reemplaza": choice.reemplaza}
    assert sets["logistica"]["conjunto"] == "sin_clima" and sets["lightgbm"]["conjunto"] == "completo"
    assert recipe["familia_comun"] == "lightgbm"
    for code in WORLD_CODES:
        entry, ablation = recipe["municipios"][code], recipe["ablacion"][code]
        assert entry["conjunto_variables"] == "completo"
        assert (ablation["familia"], ablation["conjunto_variables"]) == ("lightgbm", "sin_clima")
        assert ablation["features"] == list(NO_CLIMATE)
        assert {"candidato", "orden", "hiperparametros", "excepcion_hiperparametros"} <= set(ablation)
        assert set(ablation["excepcion_hiperparametros"]) == {"aplica", "nominal", "diferencias_logloss"}
        assert sorted(ablation["excepcion_hiperparametros"]["diferencias_logloss"]) == [str(y) for y in YEARS]
    provenance = recipe["procedencia"]
    assert set(provenance["busquedas"]) == set(SETS)
    assert provenance["busquedas"]["sin_clima"]["busqueda_id"] == "0123456789abcdef0123"

    again = build_recipe_with_feature_sets(_run(tmp_path, "mixed")[4], config=_config(), provenance=_provenance())
    assert again == recipe
    path = write_recipe_draft(recipe, tmp_path / "borrador.yaml")
    import yaml

    assert yaml.safe_load(path.read_text(encoding="utf-8")) == recipe


@pytest.mark.parametrize("damage", ["missing_set", "bad_commit", "other_channel"])
def test_recipe_draft_rejects_an_invalid_provenance(tmp_path, damage):
    from src.forecasting.selection import build_recipe_with_feature_sets

    if damage == "missing_set":
        provenance = _provenance(busquedas={"completo": _provenance()["busquedas"]["completo"]})
    elif damage == "bad_commit":
        provenance = _provenance(commit_codigo="abc")
    else:
        provenance = _provenance(version_dvc_canal="0" * 32)
    with pytest.raises(EpidemiologyValidationError):
        build_recipe_with_feature_sets(_run(tmp_path, "mixed")[4], config=_config(), provenance=provenance)
