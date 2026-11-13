"""F6 acceptance of the recipe draft (U6.4): fields of especificación §8, compatibility with the test barrier,
writing without overwriting, and regression of the real selection (contrato §6.1, §6.2, §7.5 and §8.3;
especificación §4 «Barrera de prueba: detalle», §6.F6 and §8). Approved by Jesús on 2026-11-12.

The draft has no `aprobacion` and no `hash_sha256`: those are added when Jesús approves it, in a separate unit,
and only then is configs/receta_congelada.yaml written. A draft therefore never opens the test period.

API frozen here:
- `src.forecasting.selection.build_recipe(selection, *, config, provenance) -> dict` with top-level keys
  familia_comun, comparacion_familias, municipios, decision, configuracion and procedencia:
  - comparacion_familias: mejora_media_fuera_de_anio (family -> float), victorias, casillas, umbral, reemplaza;
  - municipios: municipality code (5-character text) -> familia, candidato, orden, hiperparametros (the
    parameters of the candidate), excepcion_familia {aplica, diferencias_logloss} and excepcion_hiperparametros
    {aplica, nominal, diferencias_logloss}; diferencias_logloss maps the year as text to logloss option −
    logloss reference (selection.comparaciones);
  - decision: tau, k, W, restriccion_cumplida, inicios_evaluables, inicios_avisados, falsas_alarmas_modelo,
    falsas_alarmas_canal, inicios_avisados_canal and municipio_anios (the curve row of tau and the channel);
  - configuracion: h and the sections canal, features, validacion, lineas_base and metricas of the
    configuration, as JSON values;
  - procedencia: the given provenance (commit_codigo and commit_contrato of 40 hexadecimal characters,
    version_dvc_canal equal to the configuration, busqueda_id, insumos_md5); an invalid one is an error.
- `src.forecasting.selection.write_recipe_draft(recipe, path) -> Path`: YAML that loads back to the same
  mapping; never overwrites; refuses configs/receta_congelada.yaml and a recipe with aprobacion or hash_sha256.
- `src.forecasting.run_selection.search_provenance(search_dir, *, config) -> dict`: commit_codigo (HEAD),
  commit_contrato (last commit of docs/forecasting/contrato_predictivo.md), version_dvc_canal, busqueda_id of the
  manifest and insumos_md5 of the single table, the predictions and the manifest.

Real-data literals (regression of U6.3, 2026-11-12): logistic regression common to the six municipalities,
logistica_03 (C = 0.3, max_iter = 5000) everywhere, no exception; LightGBM 9 of 24 cells (threshold 17); tau =
0.10 with the constraint met, 6 of 10 evaluable onsets warned and 24 false alarms against the channel's 48 over
24 municipality-years. Computed by selection.py (U6.3) and reproduced by Jesús's independent check
docs/forecasting/verificacion/caso_f6.py (selection, exceptions, tau curve and tau).
"""

from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest
import yaml

pytestmark = pytest.mark.acceptance

ROOT = Path(__file__).resolve().parents[2]
YEARS = (2019, 2020, 2021, 2022)
TOP_LEVEL = {"familia_comun", "comparacion_familias", "municipios", "decision", "configuracion", "procedencia"}
CODES = ("08001", "08758", "41001", "50001", "54001", "76001")
SEARCH_ID = "7f34b69832eea69407c6"


def _config():
    from src.forecasting.config import load_config

    return load_config()


# --------------------------------------------------------------------------- synthetic selection
def _selection():
    from src.forecasting.selection import FamilyComparison, Selection, TauChoice

    codes = ("08001", "76001")
    choices = pd.DataFrame([
        {"municipality_code": "08001", "familia": "lightgbm", "candidato": "lightgbm_04", "orden": 4,
         "parametros": json.dumps({"max_depth": 3, "num_leaves": 8, "learning_rate": 0.05}, sort_keys=True),
         "excepcion_familia": True, "excepcion_hiperparametros": False, "excepcion_hiperparametros_nominal": False},
        {"municipality_code": "76001", "familia": "logistica", "candidato": "logistica_02", "orden": 2,
         "parametros": json.dumps({"C": 0.1, "max_iter": 5000}, sort_keys=True),
         "excepcion_familia": False, "excepcion_hiperparametros": True, "excepcion_hiperparametros_nominal": True},
    ])
    comparisons = []
    for i, code in enumerate(codes):
        for j, rule in enumerate(("familia", "hiperparametros")):
            for year in YEARS:
                a, b = 0.30 + 0.01 * (i + j) + 0.001 * (year - 2019), 0.35
                comparisons.append({"municipality_code": code, "regla": rule, "anio": year,
                                    "familia_opcion": "x", "candidato_opcion": "y", "familia_referencia": "x",
                                    "candidato_referencia": "z", "logloss_opcion": a, "logloss_referencia": b,
                                    "diferencia": a - b, "gana": a < b})
    curve = pd.DataFrame([{"tau": 0.05, "evaluable_onsets": 7, "warned_onsets": 5, "false_alarms": 9,
                           "municipality_years": 8, "judgeable_alerts": 14, "continuity_alerts": 1},
                          {"tau": 0.10, "evaluable_onsets": 7, "warned_onsets": 4, "false_alarms": 6,
                           "municipality_years": 8, "judgeable_alerts": 10, "continuity_alerts": 2}])
    channel = {"evaluable_onsets": 7, "warned_onsets": 6, "false_alarms": 11, "municipality_years": 8}
    families = FamilyComparison(familia_comun="logistica", mejora={"logistica": 0.03, "lightgbm": 0.01},
                                victorias=3, casillas=8, umbral=7, reemplaza=False)
    return Selection(familias=families, elecciones=choices, comparaciones=pd.DataFrame(comparisons),
                     predicciones=pd.DataFrame(), curva=curve, canal=channel,
                     tau=TauChoice(tau=0.05, restriccion_cumplida=True))


def _provenance(**changes):
    provenance = {"commit_codigo": "a" * 40, "commit_contrato": "b" * 40,
                  "version_dvc_canal": _config().datos.version_dvc_canal, "busqueda_id": "0123456789abcdef0123",
                  "insumos_md5": {"tabla_unica_desarrollo.parquet": "c" * 32}}
    provenance.update(changes)
    return provenance


def _recipe(selection=None, provenance=None):
    from src.forecasting.selection import build_recipe

    return build_recipe(selection or _selection(), config=_config(), provenance=provenance or _provenance())


# --------------------------------------------------------------------------- fields (§8)
def test_draft_has_the_fields_of_section_8_and_no_approval_or_hash():
    recipe = _recipe()
    assert set(recipe) == TOP_LEVEL
    assert "aprobacion" not in recipe and "hash_sha256" not in recipe
    assert recipe["familia_comun"] == "logistica"
    assert recipe["comparacion_familias"] == {"mejora_media_fuera_de_anio": {"logistica": 0.03, "lightgbm": 0.01},
                                              "victorias": 3, "casillas": 8, "umbral": 7, "reemplaza": False}


def test_municipalities_keep_their_code_parameters_and_annual_differences():
    selection = _selection()
    recipe = _recipe(selection)
    assert list(recipe["municipios"]) == ["08001", "76001"]
    first, second = recipe["municipios"]["08001"], recipe["municipios"]["76001"]
    assert (first["familia"], first["candidato"], first["orden"]) == ("lightgbm", "lightgbm_04", 4)
    assert first["hiperparametros"] == {"max_depth": 3, "num_leaves": 8, "learning_rate": 0.05}
    assert first["excepcion_familia"]["aplica"] is True
    assert first["excepcion_hiperparametros"] == {
        "aplica": False, "nominal": False, "diferencias_logloss": first["excepcion_hiperparametros"]["diferencias_logloss"]}
    assert second["hiperparametros"] == {"C": 0.1, "max_iter": 5000}
    assert second["excepcion_hiperparametros"]["aplica"] is True and second["excepcion_hiperparametros"]["nominal"] is True

    comparisons = selection.comparaciones
    for code in ("08001", "76001"):
        for key, rule in (("excepcion_familia", "familia"), ("excepcion_hiperparametros", "hiperparametros")):
            differences = recipe["municipios"][code][key]["diferencias_logloss"]
            assert list(differences) == [str(year) for year in YEARS]
            rows = comparisons[(comparisons["municipality_code"] == code) & (comparisons["regla"] == rule)]
            assert [differences[str(year)] for year in YEARS] == list(rows.sort_values("anio")["diferencia"])


def test_decision_takes_tau_from_the_selection_and_k_w_from_the_configuration():
    decision = _recipe()["decision"]
    config = _config()
    assert decision == {"tau": 0.05, "k": config.decision.k, "W": config.decision.W, "restriccion_cumplida": True,
                        "inicios_evaluables": 7, "inicios_avisados": 5, "falsas_alarmas_modelo": 9,
                        "falsas_alarmas_canal": 11, "inicios_avisados_canal": 6, "municipio_anios": 8}
    assert decision["tau"] in config.decision.tau_grid


def test_configuration_is_copied_into_the_recipe():
    config = _config()
    copied = _recipe()["configuracion"]
    assert copied == {"h": config.horizonte.h,
                      "canal": config.canal.model_dump(mode="json"),
                      "features": config.features.model_dump(mode="json"),
                      "validacion": config.validacion.model_dump(mode="json"),
                      "lineas_base": config.lineas_base.model_dump(mode="json"),
                      "metricas": config.metricas.model_dump(mode="json")}


def test_provenance_is_kept_and_validated():
    from src.epidemiology.config import EpidemiologyValidationError

    assert _recipe()["procedencia"] == _provenance()
    for bad in (_provenance(commit_codigo="abc"), _provenance(commit_contrato="B" * 40),
                _provenance(version_dvc_canal="0" * 32)):
        with pytest.raises(EpidemiologyValidationError):
            _recipe(provenance=bad)


def test_recipe_is_deterministic():
    from src.forecasting.guard import recipe_hash

    assert _recipe() == _recipe()
    assert recipe_hash(_recipe()) == recipe_hash(_recipe())


# --------------------------------------------------------------------------- writing and the barrier
def test_draft_round_trips_and_is_never_overwritten(tmp_path):
    from src.epidemiology.config import EpidemiologyValidationError
    from src.forecasting.selection import write_recipe_draft

    recipe = _recipe()
    path = write_recipe_draft(recipe, tmp_path / "receta_borrador.yaml")
    text = path.read_text(encoding="utf-8")
    assert yaml.safe_load(text) == recipe
    # DIVIPOLA codes and years are quoted, so no YAML reader (1.1 or 1.2) turns 08001 into the number 8001
    for key in ("08001", "76001", *map(str, YEARS)):
        assert f"'{key}':" in text or f'"{key}":' in text
    with pytest.raises(EpidemiologyValidationError):
        write_recipe_draft({**recipe, "familia_comun": "lightgbm"}, path)
    assert path.read_text(encoding="utf-8") == text


@pytest.mark.parametrize("extra", ["aprobacion", "hash_sha256"])
def test_draft_with_approval_or_hash_is_refused(tmp_path, extra):
    from src.epidemiology.config import EpidemiologyValidationError
    from src.forecasting.selection import write_recipe_draft

    recipe = {**_recipe(), extra: {"nombre": "x", "fecha": "2026-11-12"} if extra == "aprobacion" else "0" * 64}
    with pytest.raises(EpidemiologyValidationError):
        write_recipe_draft(recipe, tmp_path / "receta_borrador.yaml")
    assert not (tmp_path / "receta_borrador.yaml").exists()


def test_draft_is_never_written_as_the_frozen_recipe():
    from src.epidemiology.config import EpidemiologyValidationError
    from src.forecasting.guard import FROZEN_RECIPE_PATH
    from src.forecasting.selection import write_recipe_draft

    frozen = ROOT / FROZEN_RECIPE_PATH
    existed = frozen.exists()
    try:
        for target in (frozen, ROOT / "configs" / ".." / "configs" / FROZEN_RECIPE_PATH.name):
            with pytest.raises(EpidemiologyValidationError):
                write_recipe_draft(_recipe(), target)
        assert frozen.exists() == existed
    finally:
        if not existed and frozen.exists():
            frozen.unlink()


def _access(path):
    from src.forecasting.guard import RecipeAccess

    config = _config()
    return RecipeAccess(test_years=tuple(config.validacion.prueba), channel_version=config.datos.version_dvc_canal,
                        recipe_path=path)


def test_the_draft_does_not_open_the_test_period(tmp_path):
    from src.forecasting.guard import FrozenRecipeError, check_test_access, load_frozen_recipe
    from src.forecasting.selection import write_recipe_draft

    path = write_recipe_draft(_recipe(), tmp_path / "receta_borrador.yaml")
    with pytest.raises(FrozenRecipeError):
        load_frozen_recipe(path, channel_version=_config().datos.version_dvc_canal)
    with pytest.raises(FrozenRecipeError):
        check_test_access([date(2023, 1, 2)], _access(path))


def test_approved_and_hashed_draft_satisfies_the_barrier(tmp_path):
    from src.forecasting.guard import check_test_access, load_frozen_recipe, recipe_hash

    recipe = {**_recipe(), "aprobacion": {"nombre": "Aprobación sintética", "fecha": date(2026, 11, 12)}}
    recipe["hash_sha256"] = recipe_hash(recipe)
    path = tmp_path / "receta_congelada.yaml"
    path.write_text(yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True), encoding="utf-8")
    loaded = load_frozen_recipe(path, channel_version=_config().datos.version_dvc_canal)
    assert loaded["municipios"] == recipe["municipios"]
    check_test_access([date(2023, 1, 2)], _access(path))


# --------------------------------------------------------------------------- real data
def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _dvc_md5(name: str) -> str:
    descriptor = yaml.safe_load((ROOT / "data" / "processed" / "forecasting" / f"{name}.dvc").read_text(encoding="utf-8"))
    return descriptor["outs"][0]["md5"]


def _real_recipe(channel_file: str, search_dir: Path, tmp_path):
    from src.forecasting.dataset import build_examples, load_channel_product
    from src.forecasting.run_selection import search_provenance
    from src.forecasting.selection import build_recipe, select
    from src.forecasting.splits import yearly_trials

    config = _config()
    product = load_channel_product(channel_file, expected_md5=config.datos.version_dvc_canal)
    examples = build_examples(product, h=config.horizonte.h, canal=config.canal, features=config.features)
    first, last = config.validacion.desarrollo
    trials = yearly_trials(examples, range(first, last + 1),
                           first_train_year=config.validacion.primer_anio_entrenamiento)
    labels = trials[0].evaluate.append([trial.evaluate for trial in trials[1:]])
    result = select(pd.read_parquet(search_dir / "tabla_unica_desarrollo.parquet"),
                    pd.read_parquet(search_dir / "predicciones_desarrollo.parquet"), examples.loc[labels],
                    product.to_pandas(), config=config, access=_access(tmp_path / "sin_receta.yaml"))
    return build_recipe(result, config=config, provenance=search_provenance(search_dir, config=config))


_REAL = {}


@pytest.fixture
def real_recipe(channel_path, search_dir, tmp_path):
    key = (str(channel_path), str(search_dir))
    if key not in _REAL:
        _REAL[key] = _real_recipe(str(channel_path), search_dir, tmp_path)
    return _REAL[key]


def test_receta_real_regression_of_the_selection(real_recipe):
    recipe = real_recipe
    assert recipe["familia_comun"] == "logistica"
    families = recipe["comparacion_familias"]
    assert (families["victorias"], families["casillas"], families["umbral"], families["reemplaza"]) == (9, 24, 17, False)
    assert list(recipe["municipios"]) == list(CODES)
    for code in CODES:
        entry = recipe["municipios"][code]
        assert (entry["familia"], entry["candidato"]) == ("logistica", "logistica_03")
        assert entry["hiperparametros"] == {"C": 0.3, "max_iter": 5000}
        assert entry["excepcion_familia"]["aplica"] is False
        assert entry["excepcion_hiperparametros"]["aplica"] is False
        assert entry["excepcion_hiperparametros"]["nominal"] is False
    decision = recipe["decision"]
    assert decision["tau"] == 0.1 and decision["restriccion_cumplida"] is True
    assert (decision["inicios_avisados"], decision["inicios_evaluables"]) == (6, 10)
    assert (decision["falsas_alarmas_modelo"], decision["falsas_alarmas_canal"]) == (24, 48)
    assert (decision["inicios_avisados_canal"], decision["municipio_anios"]) == (10, 24)
    assert (decision["k"], decision["W"]) == (2, 12)


def test_receta_real_provenance(real_recipe, search_dir):
    provenance = real_recipe["procedencia"]
    assert provenance["busqueda_id"] == SEARCH_ID
    assert provenance["version_dvc_canal"] == _config().datos.version_dvc_canal
    assert provenance["commit_codigo"] == _git("rev-parse", "HEAD")
    assert provenance["commit_contrato"] == _git("log", "-1", "--format=%H", "--",
                                                 "docs/forecasting/contrato_predictivo.md")
    for name in ("tabla_unica_desarrollo.parquet", "predicciones_desarrollo.parquet", "busqueda_manifest.json"):
        assert provenance["insumos_md5"][name] == _md5(search_dir / name) == _dvc_md5(name)


def test_receta_real_is_reproduced_from_the_saved_table(real_recipe, channel_path, search_dir, tmp_path):
    from src.forecasting.guard import recipe_hash

    again = _real_recipe(str(channel_path), search_dir, tmp_path)
    assert again == real_recipe
    approval = {"aprobacion": {"nombre": "Aprobación sintética", "fecha": date(2026, 11, 12)}}
    assert recipe_hash({**again, **approval}) == recipe_hash({**real_recipe, **approval})
