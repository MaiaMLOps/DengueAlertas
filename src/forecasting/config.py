"""Load and validate the forecasting configuration (especificación §8; contrato §10.5 and §10.14).

All methodological values live in configs/forecasting.yaml. This module only checks that they
are well formed and applies the FORECAST_HORIZON override; it never supplies contract values.
"""

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "forecasting.yaml"
HORIZON_ENV_VAR = "FORECAST_HORIZON"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataSection(_Section):
    panel: Path
    canal: Path
    version_dvc_canal: str | None = None


class ChannelSection(_Section):
    n: int = Field(ge=2)
    desplazamiento: int = Field(ge=0)
    cuantil_t: float = Field(gt=0, lt=0.5)


class HorizonSection(_Section):
    # `permitidos` is declared first so that the `h` validator can read it.
    permitidos: tuple[int, ...] = Field(min_length=1)
    h: int

    @field_validator("permitidos")
    @classmethod
    def _positive_unique(cls, values):
        if any(value < 1 for value in values) or len(set(values)) != len(values):
            raise ValueError("allowed horizons must be unique positive integers")
        return values

    @field_validator("h")
    @classmethod
    def _h_allowed(cls, value, info: ValidationInfo):
        allowed = info.data.get("permitidos")
        if allowed is None:
            raise ValueError("cannot check h because horizonte.permitidos is invalid")
        if value not in allowed:
            raise ValueError(f"horizonte.h = {value} is not one of the allowed horizons {list(allowed)}")
        return value


class FeatureSection(_Section):
    w: int = Field(ge=1)
    bloques: dict[str, tuple[int, int]] = Field(min_length=1)
    ventana_semanas: int = Field(ge=1)

    @field_validator("bloques")
    @classmethod
    def _blocks_end_at_or_before_t(cls, blocks):
        for name, (start, end) in blocks.items():
            if not start <= end <= 0:
                raise ValueError(f"block {name} = [{start}, {end}] must satisfy start <= end <= 0 (weeks relative to t)")
        return blocks


class ValidationSection(_Section):
    desarrollo: tuple[int, int]
    prueba: tuple[int, int]
    primer_anio_entrenamiento: int

    @model_validator(mode="after")
    def _chronological(self):
        for name, (first, last) in (("desarrollo", self.desarrollo), ("prueba", self.prueba)):
            if first > last:
                raise ValueError(f"validacion.{name} = [{first}, {last}] is not ordered")
        if self.prueba[0] <= self.desarrollo[1]:
            raise ValueError(
                f"validacion.prueba {list(self.prueba)} must start after validacion.desarrollo {list(self.desarrollo)} ends"
            )
        if self.primer_anio_entrenamiento > self.desarrollo[0]:
            raise ValueError("validacion.primer_anio_entrenamiento must not be after the first development year")
        return self


class BaselineSection(_Section):
    # Half window of B1 (contrato §4.1). At most 25: a window of 2k + 1 >= 52 weeks would wrap the
    # whole 52-week cycle and count weeks twice.
    b1_semiventana: int = Field(ge=1, le=25, strict=True)


class MetricsSection(_Section):
    # Log-loss clip ε (contrato §10.16): p is clipped to [ε, 1 − ε] for every system.
    recorte_log_loss: float = Field(gt=0, lt=0.5, strict=True)


class ComparisonSection(_Section):
    # Chance threshold of the D9 counting rules (contrato §8.2): one-sided exact binomial and McNemar.
    alfa: float = Field(gt=0, lt=1, strict=True)


class DecisionSection(_Section):
    k: int = Field(ge=1)
    W: int = Field(ge=1)
    tau_grid: tuple[float, ...] = Field(min_length=1)

    @field_validator("tau_grid")
    @classmethod
    def _tau_in_unit_interval_and_increasing(cls, grid):
        outside = [value for value in grid if not 0 < value < 1]
        if outside:
            raise ValueError(f"tau values must lie strictly between 0 and 1; invalid: {outside}")
        if any(later <= earlier for earlier, later in zip(grid, grid[1:])):
            raise ValueError("tau values must be strictly increasing without repetitions")
        return grid


class LogisticSearch(_Section):
    # Grid of C, in pre-registered order (contrato §6.3); max_iter must be enough to converge (§7).
    C: tuple[float, ...] = Field(min_length=1)
    max_iter: int = Field(ge=1, strict=True)

    @field_validator("C")
    @classmethod
    def _positive_increasing(cls, grid):
        if any(value <= 0 for value in grid):
            raise ValueError("busqueda.logistica.C values must be positive")
        if any(later <= earlier for earlier, later in zip(grid, grid[1:])):
            raise ValueError("busqueda.logistica.C must be strictly increasing without repetitions")
        return grid


class ParameterSpace(_Section):
    # One sampled LightGBM hyperparameter: uniform integer, uniform real or log-uniform real on [a, b].
    escala: Literal["entera", "real", "log"]
    rango: tuple[float, float]
    tope_por_profundidad: bool = False   # num_leaves <= 2^max_depth (pre-registration, especificación §7)

    @model_validator(mode="after")
    def _valid_range(self):
        low, high = self.rango
        if low > high:
            raise ValueError(f"rango [{low}, {high}] is not ordered")
        if self.escala == "log" and low <= 0:
            raise ValueError("a log scale needs a positive range")
        if self.escala == "entera" and not (float(low).is_integer() and float(high).is_integer()):
            raise ValueError("an integer scale needs integer bounds")
        return self


class LightGBMSearch(_Section):
    # Random search with a fixed budget and seeds (contrato §6.3); values come from the pre-registration.
    n_candidatos: int = Field(ge=1, strict=True)
    semilla_muestreo: int = Field(strict=True)
    semilla_modelo: int = Field(strict=True)
    fijos: dict[str, int | float] = Field(default_factory=dict)
    espacio: dict[str, ParameterSpace] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self):
        overlap = sorted(set(self.fijos) & set(self.espacio))
        if overlap:
            raise ValueError(f"busqueda.lightgbm parameters both fixed and sampled: {overlap}")
        if "random_state" in self.fijos or "random_state" in self.espacio:
            raise ValueError("random_state is semilla_modelo; do not set it in fijos or espacio")
        capped = [name for name, space in self.espacio.items() if space.tope_por_profundidad]
        if capped and capped != ["num_leaves"]:
            raise ValueError("tope_por_profundidad only applies to num_leaves")
        if capped and "max_depth" not in self.espacio and "max_depth" not in self.fijos:
            raise ValueError("tope_por_profundidad needs max_depth in espacio or fijos")
        return self


class SearchSection(_Section):
    # Empty until the pre-registration (especificación §7 and §8).
    logistica: LogisticSearch | None = None
    lightgbm: LightGBMSearch | None = None

    @model_validator(mode="after")
    def _at_least_one_family(self):
        if self.logistica is None and self.lightgbm is None:
            raise ValueError("busqueda must define at least one family")
        return self


class ExperimentNames(_Section):
    desarrollo: str = Field(min_length=1)
    prueba: str = Field(min_length=1)
    final: str = Field(min_length=1)


class TrackingSection(_Section):
    modo: Literal["servidor", "local"]
    uri_servidor: str = Field(min_length=1)
    directorio_local: Path
    experimentos: ExperimentNames


class ForecastConfig(_Section):
    datos: DataSection
    canal: ChannelSection
    horizonte: HorizonSection
    features: FeatureSection
    validacion: ValidationSection
    lineas_base: BaselineSection
    metricas: MetricsSection
    comparacion: ComparisonSection
    decision: DecisionSection
    seguimiento: TrackingSection
    busqueda: SearchSection | None = None


def _horizon_override(raw: str, allowed: tuple[int, ...]) -> int:
    """Parse FORECAST_HORIZON strictly: an integer literal among the allowed horizons."""
    if not re.fullmatch(r"[0-9]+", raw):
        raise ValueError(f"{HORIZON_ENV_VAR} must be an integer literal such as 8; got {raw!r}")
    value = int(raw)
    if value not in allowed:
        raise ValueError(f"{HORIZON_ENV_VAR} = {value} is not one of the allowed horizons {list(allowed)}")
    return value


def load_config(path: str | Path | None = None) -> ForecastConfig:
    """Read the YAML, validate it and apply the FORECAST_HORIZON override when it is set."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Forecasting configuration not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping of sections")

    config = ForecastConfig.model_validate(data)
    raw = os.environ.get(HORIZON_ENV_VAR)
    if raw is None:
        return config
    h = _horizon_override(raw, config.horizonte.permitidos)
    data["horizonte"] = {**data["horizonte"], "h": h}
    return ForecastConfig.model_validate(data)
