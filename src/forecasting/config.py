"""Load and validate the forecasting configuration (especificación §8; contrato §10.5 and §10.14).

All methodological values live in configs/forecasting.yaml. This module only checks that they
are well formed and applies the FORECAST_HORIZON override; it never supplies contract values.

The family contrast (configs/contraste_familias.yaml, load_contrast_config) is a complementary development
analysis with its own schema at the end of this module; it does not change ForecastConfig.
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
    # Emission rule of the pre-registered selection and test evaluation (contrato §7.4): p >= tau. Fixed: it cannot be
    # set to another rule here. The rule shown to the user is producto.regla_alerta (D10.1).
    regla_emision: Literal["umbral"] = "umbral"

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
    # Grid of C, in pre-registered order (contrato §6.3); max_iter must be enough to converge (especificación §7).
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
    # Pre-registration of especificación §7; optional only for configurations written before it.
    logistica: LogisticSearch | None = None
    lightgbm: LightGBMSearch | None = None

    @model_validator(mode="after")
    def _at_least_one_family(self):
        if self.logistica is None and self.lightgbm is None:
            raise ValueError("busqueda must define at least one family")
        return self


class ProductSection(_Section):
    # Alert rule of the product (API and frontend; enmienda D10.1, DV-07), read only by the code that serves the user.
    # "umbral_fuera_de_epidemia" is rule B01 (p >= tau and zona_t != epidemia); "umbral" is the pre-registered rule.
    # tau and k are always those of the frozen recipe; the selection and the evaluation never read this section.
    regla_alerta: Literal["umbral", "umbral_fuera_de_epidemia"] = "umbral"


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
    producto: ProductSection = Field(default_factory=ProductSection)


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


# --------------------------------------------------------------------------- family contrast (complementary analysis)
# configs/contraste_familias.yaml describes a development-only exploration of other model families (análisis
# complementario). It never feeds the selection, the frozen recipe or the test evaluation. Its tabular searches
# reuse search.run_search with the base configuration and a contrast search section (contrast_search_config);
# its sequential families (GRU, CNN-1D) are sampled by scripts/run_seq_dl.py.

CONTRAST_CONFIG_PATH = ROOT / "configs" / "contraste_familias.yaml"
CONTRAST_FAMILIES = ("catboost", "xgboost", "svm", "mlp")
SEQUENTIAL_FAMILIES = ("gru", "cnn1d")


class RandomSearch(_Section):
    # Random search of one contrast family with a fixed budget and seeds. Same sampling scheme as LightGBMSearch
    # (search.candidates_from_config: parameters drawn in alphabetical order, then fijos, then random_state); the
    # names each family requires are checked by models.make_model when the candidates are built.
    n_candidatos: int = Field(ge=1, strict=True)
    semilla_muestreo: int = Field(strict=True)
    semilla_modelo: int = Field(strict=True)
    fijos: dict[str, int | float] = Field(default_factory=dict)
    espacio: dict[str, ParameterSpace] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self):
        overlap = sorted(set(self.fijos) & set(self.espacio))
        if overlap:
            raise ValueError(f"random search parameters both fixed and sampled: {overlap}")
        if "random_state" in self.fijos or "random_state" in self.espacio:
            raise ValueError("random_state is semilla_modelo; do not set it in fijos or espacio")
        if any(space.tope_por_profundidad for space in self.espacio.values()):
            raise ValueError("tope_por_profundidad only applies to the num_leaves of busqueda.lightgbm")
        return self


class ContrastSearchSection(SearchSection):
    # The two contract families (same types as the pre-registration) plus the contrast families.
    catboost: RandomSearch | None = None
    xgboost: RandomSearch | None = None
    svm: RandomSearch | None = None
    mlp: RandomSearch | None = None

    @model_validator(mode="after")
    def _at_least_one_family(self):
        if all(getattr(self, family) is None for family in ("logistica", "lightgbm", *CONTRAST_FAMILIES)):
            raise ValueError("busqueda must define at least one family")
        return self


class SequentialSpace(_Section):
    # Sampling space of scripts/run_seq_dl.py. Draw order per candidate (fixed in the script, so that the
    # candidates of a seed never change): lr, weight_decay, dropout, epochs, batch_size, num_layers, then the
    # own choices of the family (gru: hidden_size; cnn1d: channels, kernel_size).
    lr: tuple[float, float]                 # log-uniform
    weight_decay: tuple[float, float]       # log-uniform
    dropout: tuple[float, float]            # uniform
    epochs: tuple[int, int]                 # uniform integer, both ends included
    batch_size: tuple[int, ...] = Field(min_length=1)    # uniform choice
    num_layers: tuple[int, ...] = Field(min_length=1)    # uniform choice
    gru_hidden_size: tuple[int, ...] = Field(min_length=1)
    cnn1d_channels: tuple[int, ...] = Field(min_length=1)
    cnn1d_kernel_size: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid(self):
        for name in ("lr", "weight_decay", "dropout", "epochs"):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"secuencial.espacio.{name} = [{low}, {high}] is not ordered")
        if self.lr[0] <= 0 or self.weight_decay[0] <= 0:
            raise ValueError("lr and weight_decay are log-uniform and need positive ranges")
        if not 0 <= self.dropout[0] <= self.dropout[1] < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if self.epochs[0] < 1:
            raise ValueError("epochs must be positive")
        if not set(self.num_layers) <= {1, 2}:
            raise ValueError("num_layers must be 1 or 2 (the sequential models define one or two layers)")
        return self


class SequentialSearch(_Section):
    familias: tuple[Literal["gru", "cnn1d"], ...] = Field(min_length=1)
    variantes: tuple[Literal["sin_clima", "con_clima"], ...] = Field(min_length=1)
    n_candidatos: int = Field(ge=1, strict=True)
    semilla: int = Field(strict=True)   # sampling, weight initialisation and batch order
    espacio: SequentialSpace


class ContrastRun(_Section):
    # One run of the contrast: a folder with a single development table and how it was produced. The run name
    # is the folder name (scripts/build_thesis_report.py labels each table by it).
    carpeta: Path
    busqueda: ContrastSearchSection | None = None
    busqueda_preregistrada: bool = False   # the pre-registered search of the base configuration (sin_clima)
    secuencial: SequentialSearch | None = None

    @model_validator(mode="after")
    def _one_origin(self):
        origins = [self.busqueda is not None, self.busqueda_preregistrada, self.secuencial is not None]
        if sum(origins) != 1:
            raise ValueError("a contrast run needs exactly one of busqueda, busqueda_preregistrada or secuencial")
        return self


class HorizonSensitivity(_Section):
    horizontes: tuple[int, ...] = Field(min_length=1)
    datos: Path              # folder of the per-cell tables of the horizon sensitivity
    tabla_ganadores: Path    # ranking table whose winning configurations are re-scored at every horizon

    @field_validator("horizontes")
    @classmethod
    def _positive_unique(cls, values):
        if any(value < 1 for value in values) or len(set(values)) != len(values):
            raise ValueError("horizons must be unique positive integers")
        return values


class ContrastConfig(_Section):
    configuracion_base: Path
    experimento: str = Field(min_length=1)                # MLflow experiment of new contrast runs
    experimento_registrado_id: str = Field(min_length=1)  # experiment that holds the runs already logged
    variante: Literal["sin_clima"]
    reporte: Path
    corridas: dict[str, ContrastRun] = Field(min_length=1)
    sensibilidad_horizonte: HorizonSensitivity

    @model_validator(mode="after")
    def _names_match_folders(self):
        different = sorted(name for name, run in self.corridas.items() if run.carpeta.name != name)
        if different:
            raise ValueError(f"contrast runs whose name differs from their folder name: {different}")
        sequential = [name for name, run in self.corridas.items() if run.secuencial is not None]
        if len(sequential) > 1:
            raise ValueError(f"at most one sequential contrast run is supported; got {sequential}")
        return self

    @property
    def secuencial(self) -> tuple[str, SequentialSearch]:
        """Name and search of the sequential run."""
        found = [(name, run.secuencial) for name, run in self.corridas.items() if run.secuencial is not None]
        if not found:
            raise ValueError("the contrast configuration has no sequential run")
        return found[0]


def resolve_path(path: str | Path) -> Path:
    """A configured path, relative to the repository root unless it is absolute."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_contrast_config(path: str | Path | None = None) -> ContrastConfig:
    """Read and validate configs/contraste_familias.yaml (or `path`)."""
    config_path = Path(path) if path is not None else CONTRAST_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Family contrast configuration not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping of sections")
    return ContrastConfig.model_validate(data)


def contrast_base_config(contrast: ContrastConfig) -> ForecastConfig:
    """The base forecasting configuration named by the contrast (data, channel, horizon, validation, metrics)."""
    return load_config(resolve_path(contrast.configuracion_base))


def contrast_search_config(base: ForecastConfig, contrast: ContrastConfig, corrida: str) -> ForecastConfig:
    """The base configuration with the search section of one contrast run and the contrast experiment.

    Everything else (data, channel, horizon, validation, metrics) is the base configuration, unchanged, so a
    contrast search uses the same rows, trials and baselines as the pre-registered one.
    """
    if corrida not in contrast.corridas:
        raise ValueError(f"Unknown contrast run {corrida!r}; expected one of {sorted(contrast.corridas)}")
    run = contrast.corridas[corrida]
    if run.busqueda is None:
        raise ValueError(f"Contrast run {corrida!r} has no tabular search section")
    experiments = base.seguimiento.experimentos.model_copy(update={"desarrollo": contrast.experimento})
    tracking = base.seguimiento.model_copy(update={"experimentos": experiments})
    return base.model_copy(update={"busqueda": run.busqueda, "seguimiento": tracking})
