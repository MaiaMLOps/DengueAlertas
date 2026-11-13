"""Model pipelines of the two searched families (contrato §4.2, §5.5 and §10.9; especificación §4).

Flow (rows come from `build_examples`; train and evaluate are the rows of one `splits.yearly_trials` trial):
    _check_rows: the fifteen FEATURES, numeric ones finite, zona_t in ZONES; in train, y in {0, 1} with both classes
      -> make_model(family, params, features=FEATURES): unfitted sklearn Pipeline
           logistica: ColumnTransformer[StandardScaler(14 numeric) | OneHotEncoder(zona_t, ZONES, drop seguridad)]
                      -> LogisticRegression(L2, class_weight=None, **params)
           lightgbm:  FunctionTransformer(zona_t -> pd.Categorical with fixed categories ZONES)
                      -> LGBMClassifier(binary, class_weight=None, verbose=-1, n_jobs=1, deterministic, **params)
      -> fit_model(family, params, train, features=FEATURES): every fitted quantity (scaler, coefficients, trees)
         sees only train
      -> predict_positive(model, rows): predict_proba[:, 1] in the order of rows

Columns are selected by name in the order of `dataset.FEATURES`, so the input column order does not matter.
Hyperparameter values always come from the candidate (pre-registration, especificación §7): each family
requires exactly its allowed names and the module supplies no default value. The fixed settings are
structural, not searched: no class weights (contrato §4.2), L2 penalty, binary objective, and one thread with
deterministic LightGBM so that two equal fits give the same p bit for bit. Nothing is imputed (§5.5).

`features` (keyword only, F7) restricts the model to a subset of FEATURES, always with zona_t: the climate
ablation (contrato §4.4) fits the same family without the nine anom_* columns. Without it the pipeline is exactly
the one of F5a: logistic regression scales the requested numeric columns; LightGBM gets a leading column selection
step only when a subset is requested. The input rows still carry the fifteen columns (`_check_rows`).

Fitting and predicting is not scoring: the test-period barrier (guard.py) is applied by the metrics.
"""

from collections.abc import Mapping, Sequence
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.epidemiology.channel_epico import ZONES
from src.epidemiology.config import EpidemiologyValidationError
from src.epidemiology.validation import require

from .dataset import FEATURES

FAMILIES = ("logistica", "lightgbm")
NUMERIC_FEATURES = tuple(name for name in FEATURES if name != "zona_t")
_REFERENCE_ZONE = ZONES[0]   # seguridad: reference category of the two indicators (contrato §5.5)

# Names of the hyperparameters each candidate must give (values come from the pre-registration).
HYPERPARAMETERS = {
    "logistica": ("C", "max_iter"),
    "lightgbm": ("num_leaves", "max_depth", "learning_rate", "n_estimators", "min_child_samples",
                 "reg_lambda", "subsample", "subsample_freq", "colsample_bytree", "random_state"),
}
# Settings that a candidate may not change: class weights (contrato §4.2), early stopping (§6.3: the number
# of trees is searched without early stopping) and the fixed settings of each estimator.
_FORBIDDEN = ("class_weight", "is_unbalance", "scale_pos_weight", "early_stopping_round", "early_stopping_rounds",
              "early_stopping", "objective", "penalty", "solver", "n_jobs", "deterministic", "verbose")


def _zone_as_category(frame: pd.DataFrame) -> pd.DataFrame:
    """zona_t as a categorical with the fixed categories ZONES, so codes do not depend on the trial."""
    out = frame.copy()
    out["zona_t"] = pd.Categorical(out["zona_t"], categories=list(ZONES))
    return out


def _select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Columns of a feature subset, in the order of FEATURES (module-level so the pipeline can be pickled)."""
    return frame[columns]


def _check_features(features) -> tuple[str, ...]:
    """A subset of FEATURES with zona_t, without repetitions, returned in the order of FEATURES."""
    require(not isinstance(features, str), "features must be a sequence of feature names, not a string.")
    chosen = list(features)
    require(len(set(chosen)) == len(chosen), f"features has repeated names: {chosen}.")
    unknown = sorted(set(chosen) - set(FEATURES))
    require(not unknown, f"Unknown features {unknown}; expected a subset of {list(FEATURES)}.")
    require("zona_t" in chosen, "features must include zona_t (the zone is the core of every model, contrato §5.2).")
    return tuple(name for name in FEATURES if name in chosen)


def _check_params(family: str, params) -> dict:
    require(family in FAMILIES, f"Unknown model family {family!r}; expected one of {FAMILIES}.")
    require(isinstance(params, Mapping), f"params must be a mapping; got {type(params).__name__}.")
    forbidden = sorted(set(params) & set(_FORBIDDEN))
    require(not forbidden, f"Hyperparameters {forbidden} are fixed by the contract or the module and cannot be "
                           f"set by a candidate (no class weights, no early stopping; contrato §4.2, §6.3).")
    allowed = HYPERPARAMETERS[family]
    unknown = sorted(set(params) - set(allowed))
    require(not unknown, f"Hyperparameters {unknown} are not allowed for {family}; allowed: {list(allowed)}.")
    missing = [name for name in allowed if name not in params]
    require(not missing, f"Candidate for {family} lacks hyperparameters {missing}; the module has no defaults.")
    return dict(params)


def make_model(family: str, params: Mapping, *, features: Sequence[str] = FEATURES) -> Pipeline:
    """Unfitted pipeline of a family with the candidate's hyperparameters, on `features` (all by default)."""
    params = _check_params(family, params)
    ordered = _check_features(features)
    numeric = [name for name in ordered if name != "zona_t"]
    if family == "logistica":
        prepare = ColumnTransformer(
            [("estandarizar", StandardScaler(), numeric),
             ("zona", OneHotEncoder(categories=[list(ZONES)], drop=[_REFERENCE_ZONE], sparse_output=False,
                                    handle_unknown="error"), ["zona_t"])],
            remainder="drop")
        return Pipeline([("preparar", prepare),
                         ("modelo", LogisticRegression(penalty="l2", class_weight=None, **params))])
    estimator = lgb.LGBMClassifier(objective="binary", class_weight=None, verbose=-1, n_jobs=1, deterministic=True,
                                   **params)
    steps = [("preparar", FunctionTransformer(_zone_as_category, validate=False)), ("modelo", estimator)]
    if ordered != tuple(FEATURES):
        steps.insert(0, ("seleccionar", FunctionTransformer(_select_columns, kw_args={"columns": list(ordered)},
                                                            validate=False)))
    return Pipeline(steps)


def _check_rows(rows, name: str, *, target: bool) -> pd.DataFrame:
    require(isinstance(rows, pd.DataFrame), f"{name} must be a pandas DataFrame (rows of build_examples).")
    require(len(rows) > 0, f"{name} has no rows.")
    missing = [column for column in FEATURES if column not in rows.columns]
    require(not missing, f"{name} lacks feature columns {missing}.")
    features = rows[list(FEATURES)]
    for column in NUMERIC_FEATURES:
        values = features[column]
        require(pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values),
                f"{name}.{column} must be numeric.")
        require(np.isfinite(values.to_numpy(dtype=float)).all(),
                f"{name}.{column} has missing or infinite values; nothing is imputed (contrato §5.5).")
    zones = features["zona_t"]
    require(not zones.isna().any(), f"{name}.zona_t has missing values.")
    unknown = sorted(set(zones) - set(ZONES))
    require(not unknown, f"{name}.zona_t has values outside {ZONES}: {unknown}.")
    if target:
        require("y" in rows.columns, f"{name} lacks the target column y.")
        y = rows["y"]
        require(not y.isna().any() and y.isin([0, 1]).all(), f"{name}.y must be 0 or 1 without missing values.")
        require(y.nunique() == 2, f"{name} has a single class (y = {int(y.iloc[0])}); a model needs both classes.")
    return features


def fit_model(family: str, params: Mapping, train: pd.DataFrame, *, features: Sequence[str] = FEATURES) -> Pipeline:
    """Fit the family's pipeline with the training rows only (the trial's train), on `features`."""
    model = make_model(family, params, features=features)
    features = _check_rows(train, "train", target=True)
    y = train["y"].to_numpy(dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(features, y)
        except ConvergenceWarning as exc:
            raise EpidemiologyValidationError(
                f"Logistic regression did not converge with C = {params['C']} and max_iter = {params['max_iter']}; "
                f"the pre-registered max_iter must be enough to converge ({exc}).") from exc
    return model


def predict_positive(model: Pipeline, rows: pd.DataFrame) -> np.ndarray:
    """p(y = 1) of each row, in the order of rows."""
    require(isinstance(model, Pipeline), f"model must be a fitted Pipeline of make_model; got {type(model).__name__}.")
    features = _check_rows(rows, "rows", target=False)
    require(list(model.classes_) == [0, 1], f"Model classes must be [0, 1]; got {list(model.classes_)}.")
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)
