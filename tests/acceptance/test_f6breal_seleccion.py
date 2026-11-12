"""F6b acceptance with the real development searches (--channel-parquet, --search-dir for the full set of F5b and
--sin-clima-dir for the sin_clima search): regression of the selection of D7.2 (contrato §4.4, §6.6, §7.5 and §8.3;
especificación §6.F6b). Approved by Jesús on 2026-11-12, after the PARADA 2.

File name: the F6b regression was requested in a new test_f6b_* file, but the deny rule
Edit(/tests/acceptance/test_f6b_*.py) of 50e35f5 (after freezing test_f6b_conjuntos.py) forbids creating it. As with
test_f6receta_* in F6, it uses its own pattern, test_f6breal_*, with its own deny rule after freezing.

Provenance of the literals: computed by selection.select_with_feature_sets (409f38c; evidence ced43ff in
reports/forecasting/evidence/seleccion_conjuntos_20261112/) and reproduced by Jesús's independent check
docs/forecasting/verificacion/caso_f6b.py (md5 ff2cf73278e4e5b2d77a4d3bd34c0850, commit 8f328b5; it uses caso_f6.py and
does not import src/): step 0 of both families, common configurations, the 6 of 24 LightGBM wins, the wins of the
per-municipality exceptions, the 19 rows of the tau curve, tau = 0.05 and the definition of the ablation.
Tolerances: exact for choices, counts and curve rows; rel 1e-9 for mean improvements (two implementations, criterion
of F2). The channel figures are the regression of F4/F6 (10 of 10, 48 false alarms over 24 municipality-years).

Structural tests (no score read): the five sin_clima inputs have the md5 of their DVC descriptors (c61f007) and the
manifest is search ae6225180d37d025224b, variant sin_clima, paired with F5b.
"""

import functools
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytestmark = pytest.mark.acceptance

ROOT = Path(__file__).resolve().parents[2]
SIN_CLIMA_DVC = ROOT / "data" / "processed" / "forecasting" / "sin_clima"
EVIDENCE = ROOT / "reports" / "forecasting" / "evidence" / "seleccion_conjuntos_20261112"
SEARCH_ID = "ae6225180d37d025224b"
F5B_SEARCH_ID = "7f34b69832eea69407c6"
INPUTS = {  # file: md5 of its .dvc (c61f007)
    "tabla_unica_desarrollo.parquet": "8e18c06857c0b840d6c08dc8cef8320a",
    "predicciones_desarrollo.parquet": "75d29d5396488b28f67908750cb7986e",
    "lineas_base_desarrollo.parquet": "a826f3661510f9e32afbdbe0682bd22a",
    "predicciones_lineas_base_desarrollo.parquet": "7fb8390c139e63ca847af3f0bd6e2ac5",
    "busqueda_manifest.json": "8d1df8e3beb107e78864a2aa5bb14ba6",
}
CODES = ("08001", "08758", "41001", "50001", "54001", "76001")
NO_CLIMATE = ("zona_t", "posicion_4s", "crecimiento_4s", "cambio_umbral", "sem_sin", "sem_cos")
REL = 1e-9

STEP_0 = {  # family: (chosen set, mean improvement completo, mean improvement sin_clima, wins of completo)
    "logistica": ("sin_clima", -0.023968181424018162, 0.21090958293979553, 9),
    "lightgbm": ("sin_clima", -0.028082585531237696, 0.08395871267355431, 11),
}
COMMON = {"logistica": "logistica_05", "lightgbm": "lightgbm_20"}   # chosen with the 4 years, sin_clima
FAMILY_WINS = {"08001": 1, "08758": 1, "41001": 2, "50001": 0, "54001": 1, "76001": 1}
HYPERPARAMETER_WINS = {"08001": 2, "08758": 2, "41001": 2, "50001": 1, "54001": 1, "76001": 0}
ABLATION_WINS = {"08001": 0, "08758": 2, "41001": 0, "50001": 3, "54001": 0, "76001": 3}
TAU_GRID = [round(0.05 * i, 2) for i in range(1, 20)]
CURVE = {  # per tau of the grid
    "warned_onsets": [8, 5, 4, 3, 3, 1, 2, 2, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
    "false_alarms": [22, 18, 9, 8, 8, 6, 6, 4, 3, 2, 2, 3, 3, 4, 3, 1, 1, 0, 0],
    "judgeable_alerts": [34, 24, 13, 11, 10, 6, 7, 5, 3, 2, 2, 3, 3, 4, 4, 2, 1, 0, 0],
    "continuity_alerts": [6, 7, 8, 10, 13, 15, 13, 14, 16, 17, 20, 22, 13, 11, 13, 14, 13, 11, 1],
}


def _config():
    from src.forecasting.config import load_config

    return load_config()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- structural (no score read)
@pytest.mark.parametrize("name", list(INPUTS))
def test_sin_clima_inputs_match_their_dvc_descriptors(sin_clima_dir, name):
    descriptor = yaml.safe_load((SIN_CLIMA_DVC / f"{name}.dvc").read_text(encoding="utf-8"))
    (out,) = descriptor["outs"]
    assert out["path"] == name and out["hash"] == "md5" and out["md5"] == INPUTS[name]
    assert _md5(sin_clima_dir / name) == INPUTS[name]


def test_sin_clima_manifest_is_the_paired_variant(sin_clima_dir, search_dir):
    from src.forecasting.run_selection import check_paired_searches

    manifest = json.loads((sin_clima_dir / "busqueda_manifest.json").read_text(encoding="utf-8"))
    assert manifest["busqueda_id"] == SEARCH_ID
    assert manifest["contenido"]["variante"] == {"nombre": "sin_clima", "features": list(NO_CLIMATE)}
    check_paired_searches(search_dir, sin_clima_dir)


# --------------------------------------------------------------------------- selection with real data
@functools.lru_cache(maxsize=None)
def _inputs(channel_file: str, full_dir: str, simple_dir: str):
    from src.forecasting.dataset import build_examples, load_channel_product
    from src.forecasting.run_selection import development_rows

    config = _config()
    product = load_channel_product(channel_file, expected_md5=config.datos.version_dvc_canal)
    examples = build_examples(product, h=config.horizonte.h, canal=config.canal, features=config.features)
    folders = {"completo": Path(full_dir), "sin_clima": Path(simple_dir)}
    tables = {name: pd.read_parquet(folder / "tabla_unica_desarrollo.parquet") for name, folder in folders.items()}
    predictions = {name: pd.read_parquet(folder / "predicciones_desarrollo.parquet")
                   for name, folder in folders.items()}
    return development_rows(examples, config), product.to_pandas(), tables, predictions


def _access(tmp_path):
    from src.forecasting.guard import RecipeAccess

    return RecipeAccess.from_config(_config(), recipe_path=tmp_path / "receta_inexistente.yaml")


_RESULTS = {}


@pytest.fixture
def result(channel_path, search_dir, sin_clima_dir, tmp_path):
    from src.forecasting.selection import select_with_feature_sets

    key = (str(channel_path), str(search_dir), str(sin_clima_dir))
    if key not in _RESULTS:
        rows, observed, tables, predictions = _inputs(*key)
        _RESULTS[key] = select_with_feature_sets(tables, predictions, rows, observed, config=_config(),
                                                 access=_access(tmp_path))
    return _RESULTS[key]


def test_seleccion_b_real_step_0(result):
    for family, (chosen, full, simple, wins) in STEP_0.items():
        choice = result.conjuntos[family]
        assert (choice.conjunto, choice.reemplaza, choice.victorias, choice.casillas, choice.umbral) == (
            chosen, False, wins, 24, 17)
        assert choice.mejora["completo"] == pytest.approx(full, rel=REL)
        assert choice.mejora["sin_clima"] == pytest.approx(simple, rel=REL)
    cells = result.celdas_conjuntos
    assert len(cells) == 48 and cells.groupby("familia")["gana_completo"].sum().to_dict() == {
        family: values[3] for family, values in STEP_0.items()}
    assert result.conjuntos_municipio == {code: "sin_clima" for code in CODES}


def test_seleccion_b_real_common_configurations_and_families(result, sin_clima_dir):
    from src.forecasting.selection import common_configuration

    table = pd.read_parquet(sin_clima_dir / "tabla_unica_desarrollo.parquet")
    assert {family: common_configuration(table, familia=family) for family in COMMON} == COMMON
    families = result.seleccion.familias
    assert (families.familia_comun, families.victorias, families.casillas, families.umbral, families.reemplaza) == (
        "logistica", 6, 24, 17, False)
    assert families.mejora["logistica"] == pytest.approx(STEP_0["logistica"][2], rel=REL)
    assert families.mejora["lightgbm"] == pytest.approx(STEP_0["lightgbm"][2], rel=REL)


def test_seleccion_b_real_municipalities_and_exceptions(result):
    choices = result.seleccion.elecciones.set_index("municipality_code")
    assert sorted(choices.index) == list(CODES)
    assert set(choices["familia"]) == {"logistica"} and set(choices["candidato"]) == {"logistica_05"}
    assert {json.dumps(json.loads(p), sort_keys=True) for p in choices["parametros"]} == {
        json.dumps({"C": 3.0, "max_iter": 5000}, sort_keys=True)}
    assert not choices["excepcion_familia"].any() and not choices["excepcion_hiperparametros"].any()
    wins = result.seleccion.comparaciones.groupby(["regla", "municipality_code"])["gana"].sum()
    assert wins["familia"].to_dict() == FAMILY_WINS
    assert wins["hiperparametros"].to_dict() == HYPERPARAMETER_WINS


def test_seleccion_b_real_tau_curve_and_tau(result):
    selection = result.seleccion
    curve = selection.curva
    assert list(curve["tau"]) == TAU_GRID == list(_config().decision.tau_grid)
    for column, values in CURVE.items():
        assert curve[column].astype(int).tolist() == values, column
    assert (curve["evaluable_onsets"] == 10).all() and (curve["municipality_years"] == 24).all()
    assert (selection.tau.tau, selection.tau.restriccion_cumplida) == (0.05, True)
    channel = selection.canal
    assert (channel["warned_onsets"], channel["evaluable_onsets"], channel["false_alarms"],
            channel["municipality_years"]) == (10, 10, 48, 24)


def test_seleccion_b_real_ablation(result):
    ablation = result.ablacion.set_index("municipality_code")
    assert sorted(ablation.index) == list(CODES)
    assert set(ablation["familia"]) == {"logistica"} and set(ablation["conjunto_variables"]) == {"completo"}
    assert set(ablation["candidato"]) == {"logistica_03"}
    assert not ablation["excepcion_hiperparametros"].any()
    assert result.comparaciones_ablacion.groupby("municipality_code")["gana"].sum().to_dict() == ABLATION_WINS


def test_seleccion_b_real_draft_matches_the_evidence(result):
    """The draft rebuilt now equals the committed draft of the evidence (same provenance)."""
    from src.forecasting.selection import build_recipe_with_feature_sets

    stored = yaml.safe_load((EVIDENCE / "receta_borrador.yaml").read_text(encoding="utf-8"))
    rebuilt = build_recipe_with_feature_sets(result, config=_config(), provenance=stored["procedencia"])
    assert json.loads(json.dumps(rebuilt)) == json.loads(json.dumps(stored))
    assert stored["procedencia"]["busquedas"]["completo"]["busqueda_id"] == F5B_SEARCH_ID
    assert stored["procedencia"]["busquedas"]["sin_clima"]["busqueda_id"] == SEARCH_ID
    for name in ("tabla_unica_desarrollo.parquet", "predicciones_desarrollo.parquet", "busqueda_manifest.json"):
        assert stored["procedencia"]["busquedas"]["sin_clima"]["insumos_md5"][name] == INPUTS[name]


def test_seleccion_b_real_full_set_forced_is_the_f6_selection(channel_path, search_dir, sin_clima_dir, tmp_path):
    from src.forecasting.selection import select, select_with_feature_sets

    rows, observed, tables, predictions = _inputs(str(channel_path), str(search_dir), str(sin_clima_dir))
    access = _access(tmp_path)
    forced = select_with_feature_sets(tables, predictions, rows, observed, config=_config(), access=access,
                                      forced_set="completo").seleccion
    f6 = select(tables["completo"], predictions["completo"], rows, observed, config=_config(), access=access)
    assert forced.familias == f6.familias and forced.tau == f6.tau
    for field in ("elecciones", "comparaciones", "predicciones", "curva"):
        pd.testing.assert_frame_equal(getattr(forced, field), getattr(f6, field), check_exact=True)
    assert f6.tau.tau == 0.1 and set(f6.elecciones["candidato"]) == {"logistica_03"}
