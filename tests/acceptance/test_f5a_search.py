"""F5a acceptance with real development data (--channel-parquet): smoke search with two candidates per family
(contrato §4.2, §6.1–§6.3, §10.9 and §10.12; especificación §6.F5a and §9). Approved by Jesús on 2026-11-11
(groups A and B of U5.1, plus the per-row predictions table).

Rows come from `build_examples` and trials from `splits.yearly_trials` (D7.1); nothing is rebuilt here.
The candidate values below only exercise the harness: they are not the pre-registered search space (§7).

API frozen here (`src.forecasting.search`):
- `Candidate(name, params)`;
- `run_search(examples, candidates, *, config, output_dir, tracking_uri) -> DataFrame`, with `candidates`
  a mapping family -> sequence of Candidate in pre-registered order; it saves the single table in
  `output_dir / TABLE_FILENAME` and the per-row predictions in `output_dir / PREDICTIONS_FILENAME`, and logs
  one MLflow run per (municipality, family, candidate) in the experiment seguimiento.experimentos.desarrollo
  of `tracking_uri`, with tags municipio, familia and candidato;
- single table: one row per (municipality_code, familia, candidato, anio), minimum columns as in
  test_f5a_comparison.py;
- predictions: one row per evaluated row and candidate, columns municipality_code, t, target_week, familia,
  candidato, anio, y, p.

Literal figures: 6 × 2 × 4 = 48 single-table rows per family, years 2019–2022 only, 208 evaluated rows per
municipality and candidate (contrato §6.1), 1,248 predictions per (family, candidate), 12 runs per family.
"""

from datetime import date
import functools

import mlflow
import pandas as pd
import pytest

pytestmark = pytest.mark.acceptance

H = 8
CODES = ("08001", "08758", "41001", "50001", "54001", "76001")
DEVELOPMENT = (2019, 2020, 2021, 2022)
FAMILIES = ("logistica", "lightgbm")
SEED = 20261112


def _candidates():
    from src.forecasting.search import Candidate

    lightgbm = {"learning_rate": 0.05, "min_child_samples": 20, "reg_lambda": 1.0, "subsample": 0.8,
                "subsample_freq": 1, "colsample_bytree": 0.8, "random_state": SEED}
    return {
        "logistica": [Candidate("humo_rl_0", {"C": 0.1, "max_iter": 2000}),
                      Candidate("humo_rl_1", {"C": 1.0, "max_iter": 2000})],
        "lightgbm": [Candidate("humo_gb_0", {**lightgbm, "num_leaves": 4, "max_depth": 2, "n_estimators": 100}),
                     Candidate("humo_gb_1", {**lightgbm, "num_leaves": 8, "max_depth": 3, "n_estimators": 150})],
    }


def _config():
    from src.forecasting.config import load_config

    return load_config()


@functools.lru_cache(maxsize=None)
def _examples(channel_file: str):
    from src.forecasting.dataset import build_examples, load_channel_product

    config = _config()
    product = load_channel_product(channel_file, expected_md5=config.datos.version_dvc_canal)
    return build_examples(product, h=H, canal=config.canal, features=config.features)


_SMOKE = {}


def _smoke(channel_file: str, tmp_path_factory):
    """Run the smoke search once per channel file; outputs and MLflow store live in a temporary folder."""
    if channel_file not in _SMOKE:
        from src.forecasting import search

        root = tmp_path_factory.mktemp("f5a_smoke")
        uri = (root / "mlruns").resolve().as_uri()
        table = search.run_search(_examples(channel_file), _candidates(), config=_config(),
                                  output_dir=root / "out", tracking_uri=uri)
        _SMOKE[channel_file] = (table, root / "out", uri)
    return _SMOKE[channel_file]


@pytest.fixture
def smoke(channel_path, tmp_path_factory):
    return _smoke(str(channel_path), tmp_path_factory)


# --------------------------------------------------------------------------- group A, assertion 5
def test_single_table_size_and_years(smoke):
    table, _, _ = smoke
    for family in FAMILIES:
        own = table[table["familia"] == family]
        assert len(own) == 6 * 2 * 4
        assert not own.duplicated(["municipality_code", "candidato", "anio"]).any()
    assert set(table["familia"]) == set(FAMILIES)
    assert set(table["anio"]) == set(DEVELOPMENT)
    assert all(week < date(2023, 1, 1) for week in table["ultima_semana_objetivo"])
    assert all(week >= date(2019, 1, 1) for week in table["primera_semana_objetivo"])


def test_every_candidate_scores_208_rows_per_municipality(smoke):
    table, _, _ = smoke
    totals = table.groupby(["municipality_code", "familia", "candidato"])["n_filas"].sum()
    assert len(totals) == 6 * 2 * 2
    assert (totals == 208).all()


def test_same_candidates_in_every_municipality(smoke):
    from src.forecasting.search import validate_score_table

    table, _, _ = smoke
    lists = {code: set(map(tuple, table.loc[table["municipality_code"] == code,
                                            ["familia", "candidato", "orden", "parametros"]].values))
             for code in CODES}
    assert set(lists) == set(CODES)
    assert all(lists[code] == lists[CODES[0]] for code in CODES)
    assert {name for _, name, _, _ in lists[CODES[0]]} == {"humo_rl_0", "humo_rl_1", "humo_gb_0", "humo_gb_1"}
    validate_score_table(table, municipalities=CODES, years=DEVELOPMENT)


def test_saved_table_equals_returned(smoke):
    from src.forecasting.search import TABLE_FILENAME

    table, output_dir, _ = smoke
    pd.testing.assert_frame_equal(pd.read_parquet(output_dir / TABLE_FILENAME), table)


def _predictions(output_dir):
    from src.forecasting.search import PREDICTIONS_FILENAME

    return pd.read_parquet(output_dir / PREDICTIONS_FILENAME)


def test_predictions_cover_development_rows_only(smoke):
    _, output_dir, _ = smoke
    predictions = _predictions(output_dir)
    assert list(predictions.columns[:8]) == ["municipality_code", "t", "target_week", "familia", "candidato",
                                            "anio", "y", "p"]
    sizes = predictions.groupby(["familia", "candidato"]).size()
    assert len(sizes) == 4 and (sizes == 1248).all()
    assert all(week < date(2023, 1, 1) for week in predictions["target_week"])
    assert set(predictions["anio"]) == set(DEVELOPMENT)
    assert not predictions.duplicated(["municipality_code", "t", "familia", "candidato"]).any()


def test_log_loss_from_predictions_equals_single_table(smoke, tmp_path):
    from src.forecasting.guard import RecipeAccess
    from src.forecasting.metrics import log_loss

    table, output_dir, _ = smoke
    config = _config()
    access = RecipeAccess.from_config(config, recipe_path=tmp_path / "absent.yaml")
    predictions = _predictions(output_dir)
    indexed = table.set_index(["municipality_code", "familia", "candidato", "anio"])
    keys = ["municipality_code", "familia", "candidato", "anio"]
    for key, rows in predictions.groupby(keys):
        rows = rows.sort_values("t")
        value = log_loss(rows["y"].to_numpy(), rows["p"].to_numpy(), list(rows["target_week"]), access=access,
                         clip=config.metricas.recorte_log_loss)
        assert value == indexed.loc[key, "logloss"]
        assert len(rows) == indexed.loc[key, "n_filas"]


# --------------------------------------------------------------------------- group B, assertion 8
def test_b2_reference_is_paired_and_exact(smoke, channel_path, tmp_path):
    from src.forecasting.baselines import b2_predict
    from src.forecasting.guard import RecipeAccess
    from src.forecasting.metrics import improvement, log_loss
    from src.forecasting.splits import yearly_trials

    table, _, _ = smoke
    config = _config()
    access = RecipeAccess.from_config(config, recipe_path=tmp_path / "absent.yaml")
    examples = _examples(str(channel_path))
    trials = yearly_trials(examples, DEVELOPMENT, first_train_year=config.validacion.primer_anio_entrenamiento)
    assert len(trials) == 24

    for trial in trials:
        cell = table[(table["municipality_code"] == trial.municipality_code) & (table["anio"] == trial.year)]
        assert len(cell) == 4
        train, evaluate = examples.loc[trial.train], examples.loc[trial.evaluate]
        p = b2_predict(train, evaluate, access=access)
        expected = log_loss(evaluate["y"].to_numpy(), p, list(evaluate["target_week"]), access=access,
                            clip=config.metricas.recorte_log_loss)
        assert (cell["logloss_b2"] == expected).all()
        for _, row in cell.iterrows():
            assert row["mejora_b2"] == improvement(row["logloss"], row["logloss_b2"])


# --------------------------------------------------------------------------- group B, assertion 9
def test_one_finished_run_per_municipality_family_candidate(smoke):
    _, _, uri = smoke
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(_config().seguimiento.experimentos.desarrollo)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id], max_results=10000)
    finished = [run for run in runs if run.info.status == "FINISHED" and run.data.tags.get("familia") in FAMILIES]
    keys = [(run.data.tags["municipio"], run.data.tags["familia"], run.data.tags["candidato"]) for run in finished]
    assert len(keys) == len(set(keys)) == 6 * 2 * 2
    for family in FAMILIES:
        assert sum(1 for _, own, _ in keys if own == family) == 12
    assert {code for code, _, _ in keys} == set(CODES)
