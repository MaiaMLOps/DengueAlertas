"""F6 acceptance with the real development search (--search-dir, and --channel-parquet for the selection)
(contrato §6.2, §7.5 and §8.3; especificación §6.F5b, §6.F6 and §7). Approved by Jesús on 2026-11-12 (U6.1).

Two groups:
- Structural, without reading scores (only --search-dir): the five F5b inputs have the md5 of their .dvc
  descriptors (versioned in 1a1ad2d; prefixes and suffixes as in especificación §6.F5b) and the manifest
  belongs to search 7f34b69832eea69407c6.
- `test_seleccion_real_*` run the selection with real data. Until U6.3 they are not run with the real folder
  (validated with a fabricated folder of the same schema); in the step reports they appear as omitted,
  pending U6.3, by deselecting them with -k "not seleccion_real".

Literal figures of the channel in development, on the 1,248 development rows with k = 2 and W = 12:
48 false alarms over 24 municipality-years (2.0 per municipality-year) and 10 of 10 evaluable onsets warned.
They are a REGRESSION test: they come from events.py as computed in F4 (especificación §6.F4, «Canal en
desarrollo», only reported there) and here they become the constraint of tau (contrato §7.5). Jesús's
independent check docs/forecasting/verificacion/caso_f6.py (run with python -I in --solo-canal mode, without
importing src/) reproduces them too: 48 false alarms, 10/10, 63 judgeable alerts and alert ratio 9/63.
"""

from datetime import date
import functools
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytestmark = pytest.mark.acceptance

ROOT = Path(__file__).resolve().parents[2]
DVC_DIR = ROOT / "data" / "processed" / "forecasting"
SEARCH_ID = "7f34b69832eea69407c6"
INPUTS = {  # file: (md5 prefix, md5 suffix) as in especificación §6.F5b
    "tabla_unica_desarrollo.parquet": ("a38ab5ef", "0c"),
    "predicciones_desarrollo.parquet": ("ac5a28f3", "ca"),
    "lineas_base_desarrollo.parquet": ("a826f366", "2a"),
    "predicciones_lineas_base_desarrollo.parquet": ("7fb8390c", "c5"),
    "busqueda_manifest.json": ("48b930bf", "89"),
}
CODES = ("08001", "08758", "41001", "50001", "54001", "76001")
DEVELOPMENT = (2019, 2022)
H = 8
CHANNEL_FALSE_ALARMS = 48
MUNICIPALITY_YEARS = 24
EVALUABLE_ONSETS = 10


# --------------------------------------------------------------------------- structural (no scores read)
def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.mark.parametrize("name", list(INPUTS))
def test_search_inputs_match_their_dvc_descriptors(search_dir, name):
    descriptor = yaml.safe_load((DVC_DIR / f"{name}.dvc").read_text(encoding="utf-8"))
    (out,) = descriptor["outs"]
    assert out["path"] == name and out["hash"] == "md5"
    prefix, suffix = INPUTS[name]
    assert out["md5"].startswith(prefix) and out["md5"].endswith(suffix)
    assert _md5(search_dir / name) == out["md5"]


def test_search_manifest_identity(search_dir):
    manifest = json.loads((search_dir / "busqueda_manifest.json").read_text(encoding="utf-8"))
    assert manifest["busqueda_id"] == SEARCH_ID


# --------------------------------------------------------------------------- selection with real data
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
    examples = build_examples(_product(channel_file), h=H, canal=config.canal, features=config.features)
    trials = yearly_trials(examples, range(DEVELOPMENT[0], DEVELOPMENT[1] + 1),
                           first_train_year=config.validacion.primer_anio_entrenamiento)
    labels = trials[0].evaluate.append([trial.evaluate for trial in trials[1:]])
    return examples.loc[labels]


def _access(tmp_path):
    from src.forecasting.guard import RecipeAccess

    config = _config()
    return RecipeAccess(test_years=tuple(config.validacion.prueba), channel_version=config.datos.version_dvc_canal,
                        recipe_path=tmp_path / "receta_inexistente.yaml")


def _inputs(search_dir: Path):
    return (pd.read_parquet(search_dir / "tabla_unica_desarrollo.parquet"),
            pd.read_parquet(search_dir / "predicciones_desarrollo.parquet"))


def _run(channel_file: str, search_dir: Path, tmp_path):
    """Read the saved files and run the whole selection (nothing is retrained)."""
    from src.forecasting.selection import select

    table, predictions = _inputs(search_dir)
    return select(table, predictions, _development_rows(channel_file), _product(channel_file).to_pandas(),
                  config=_config(), access=_access(tmp_path))


_RESULTS = {}


@pytest.fixture
def selection(channel_path, search_dir, tmp_path):
    key = (str(channel_path), str(search_dir))
    if key not in _RESULTS:
        _RESULTS[key] = _run(str(channel_path), search_dir, tmp_path)
    return _RESULTS[key]


def test_seleccion_real_channel_reference(selection):
    """Regression of events.py (F4): the channel's development figures, now the constraint of tau."""
    channel = selection.canal
    assert channel["false_alarms"] == CHANNEL_FALSE_ALARMS
    assert channel["municipality_years"] == MUNICIPALITY_YEARS
    assert channel["false_alarms_per_municipality_year"] == 2.0
    assert channel["evaluable_onsets"] == EVALUABLE_ONSETS
    assert channel["warned_onsets"] == EVALUABLE_ONSETS


def test_seleccion_real_curve_covers_the_grid(selection):
    curve = selection.curva
    assert list(curve["tau"]) == list(_config().decision.tau_grid)
    assert len(curve) == 19
    assert (curve["evaluable_onsets"] == EVALUABLE_ONSETS).all()
    assert (curve["municipality_years"] == MUNICIPALITY_YEARS).all()


def test_seleccion_real_tau_follows_the_rule(selection):
    curve, choice = selection.curva, selection.tau
    assert choice.tau in _config().decision.tau_grid
    feasible = curve[curve["false_alarms"] <= CHANNEL_FALSE_ALARMS]
    if choice.restriccion_cumplida:
        best = feasible["warned_onsets"].max()
        assert choice.tau == feasible.loc[feasible["warned_onsets"] == best, "tau"].max()
    else:
        assert feasible.empty
        fewest = curve["false_alarms"].min()
        assert choice.tau == curve.loc[curve["false_alarms"] == fewest, "tau"].max()


def test_seleccion_real_chosen_tau_row_matches_evaluate_period(selection, channel_path, tmp_path):
    from src.forecasting.events import evaluate_period, model_positive

    rows = _development_rows(str(channel_path))
    chosen = rows[["municipality_code", "t"]].merge(selection.predicciones, on=["municipality_code", "t"],
                                                    how="left", validate="one_to_one")
    decision = _config().decision
    summary = evaluate_period(rows, model_positive(chosen["p"].to_numpy(), tau=selection.tau.tau),
                              _product(str(channel_path)).to_pandas(), k=decision.k, window=decision.W,
                              access=_access(tmp_path)).summary
    row = selection.curva[selection.curva["tau"] == selection.tau.tau].iloc[0]
    assert int(row["warned_onsets"]) == summary["warned_onsets"]
    assert int(row["false_alarms"]) == summary["false_alarms"]


def test_seleccion_real_choices_exist_and_cover_the_rows(selection, channel_path, search_dir):
    table, predictions = _inputs(search_dir)
    choices = selection.elecciones
    assert sorted(choices["municipality_code"]) == list(CODES)
    described = set(map(tuple, table[["familia", "candidato", "orden", "parametros"]].drop_duplicates().values))
    for _, choice in choices.iterrows():
        assert (choice["familia"], choice["candidato"], choice["orden"], choice["parametros"]) in described

    rows = _development_rows(str(channel_path))
    chosen = selection.predicciones
    assert len(chosen) == len(rows) == 1248
    assert set(zip(chosen["municipality_code"], chosen["t"])) == set(zip(rows["municipality_code"], rows["t"]))
    assert all(week < date(2023, 1, 1) for week in chosen["target_week"])
    for _, choice in choices.iterrows():
        own = chosen[chosen["municipality_code"] == choice["municipality_code"]]
        assert len(own) == 208
        source = predictions[(predictions["municipality_code"] == choice["municipality_code"])
                             & (predictions["familia"] == choice["familia"])
                             & (predictions["candidato"] == choice["candidato"])]
        merged = own.merge(source, on=["municipality_code", "t"], suffixes=("", "_fuente"))
        assert len(merged) == 208 and (merged["p"] == merged["p_fuente"]).all()


def test_seleccion_real_family_comparison_recomputed(selection, search_dir):
    from src.forecasting.comparison import count_wins, out_of_year_scores, win_threshold

    table, _ = _inputs(search_dir)
    simple = out_of_year_scores(table, familia="logistica")
    complex_ = out_of_year_scores(table, familia="lightgbm")
    assert list(zip(simple["municipality_code"], simple["anio"])) == list(
        zip(complex_["municipality_code"], complex_["anio"]))
    means = {family: scores.groupby("municipality_code")["mejora_comun"].mean().mean()
             for family, scores in (("logistica", simple), ("lightgbm", complex_))}
    wins = count_wins(complex_["logloss_comun"], simple["logloss_comun"])
    threshold = win_threshold(24, alpha=_config().comparacion.alfa)
    replaces = means["lightgbm"] > means["logistica"] and wins >= threshold

    result = selection.familias
    assert threshold == 17 and result.umbral == 17 and result.casillas == 24
    assert result.victorias == wins
    assert result.reemplaza == replaces
    assert result.familia_comun == ("lightgbm" if replaces else "logistica")
    for family in ("logistica", "lightgbm"):
        assert math.isclose(result.mejora[family], means[family], rel_tol=1e-9, abs_tol=1e-12)


def _same_summary(first: dict, second: dict) -> bool:
    if first.keys() != second.keys():
        return False
    return all((isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)) or a == b
               for a, b in ((first[key], second[key]) for key in first))


def test_seleccion_real_is_reproducible(selection, channel_path, search_dir, tmp_path):
    again = _run(str(channel_path), search_dir, tmp_path)
    assert again.familias == selection.familias
    for field in ("elecciones", "comparaciones", "predicciones", "curva"):
        pd.testing.assert_frame_equal(getattr(again, field), getattr(selection, field), check_exact=True)
    assert _same_summary(again.canal, selection.canal)
    assert again.tau == selection.tau
