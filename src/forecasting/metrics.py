"""Estimation metrics that decide and their basic report companions (contrato §6.2, §7.1, §7.2, §10.16).

Flow of every function that receives rows:
    check_test_access(target weeks of the scored rows)   (test-period barrier, before anything else)
      -> validate: same non-zero length, y in {0, 1}, p finite in [0, 1]
      -> the metric

- log_loss: natural logarithm, mean per row, with p clipped to [ε, 1 − ε]; ε = metricas.recorte_log_loss,
  the same for every system (approved: 1e-15, a numerical guard only). Own implementation: scikit-learn
  1.7.2 no longer accepts eps and would clip with its own value.
- brier: mean of (p − y)², without clipping.
- average_precision: sklearn.metrics.average_precision_score; NaN when the rows have no positive, because
  AP is undefined there (scikit-learn would return 0.0, which reads as no discrimination).
- improvement: 1 − score/reference, the selection metric against B2 (§6.2) and the BSS (§7.2). It works on
  scores already computed through the barrier, so it receives no rows.

Murphy decomposition, reliability, calibration by year and metrics at τ are report metrics for F7.
"""

from collections.abc import Iterable
import math

import numpy as np
from sklearn.metrics import average_precision_score

from src.epidemiology.validation import require

from .guard import RecipeAccess, check_test_access


def _scored_rows(y, p, target_weeks: Iterable, access: RecipeAccess) -> tuple[np.ndarray, np.ndarray]:
    weeks = list(target_weeks)
    check_test_access(weeks, access)
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    require(y.ndim == 1 and p.ndim == 1, "y and p must be one-dimensional.")
    require(len(y) > 0, "Metrics need at least one row.")
    require(len(y) == len(p) == len(weeks),
            f"y, p and target_weeks must have the same length; got {len(y)}, {len(p)} and {len(weeks)}.")
    require(np.issubdtype(y.dtype, np.number) and not np.isnan(y.astype(float)).any()
            and np.isin(y, (0, 1)).all(), "y must be 0 or 1 without missing values.")
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "p must be finite probabilities in [0, 1].")
    return y.astype(float), p


def log_loss(y, p, target_weeks: Iterable, *, access: RecipeAccess, clip: float) -> float:
    """Mean natural-log loss with p clipped to [clip, 1 - clip] (contrato §10.16)."""
    y, p = _scored_rows(y, p, target_weeks, access)
    require(isinstance(clip, float) and 0 < clip < 0.5, f"clip must be a float in (0, 0.5); got {clip!r}.")
    clipped = np.clip(p, clip, 1 - clip)
    return float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))


def brier(y, p, target_weeks: Iterable, *, access: RecipeAccess) -> float:
    """Mean squared difference between p and y (contrato §7.2)."""
    y, p = _scored_rows(y, p, target_weeks, access)
    return float(np.mean((p - y) ** 2))


def average_precision(y, p, target_weeks: Iterable, *, access: RecipeAccess) -> float:
    """Average precision (scikit-learn); NaN when there is no positive row."""
    y, p = _scored_rows(y, p, target_weeks, access)
    if not y.any():
        return math.nan
    return float(average_precision_score(y, p))


def improvement(score: float, reference: float) -> float:
    """1 − score/reference: improvement of a loss over a reference system (log-loss vs B2, BSS)."""
    for name, value in (("score", score), ("reference", reference)):
        require(isinstance(value, (int, float, np.floating)) and not isinstance(value, bool) and math.isfinite(value),
                f"{name} must be a finite number; got {value!r}.")
    require(reference > 0, f"reference must be positive; got {reference!r}.")
    return 1 - score / reference
