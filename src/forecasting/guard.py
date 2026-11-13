"""Test-period barrier and minimal frozen-recipe format (especificación §1, rule 1, §4 «Barrera de prueba:
detalle» and §8; contrato §6.1 and §7.7).

Flow of `check_test_access(target_weeks, access)`, called by every function that scores rows (metrics,
baselines, events, evaluate, artifacts, reports and diagnostics) with the target weeks of the rows it scores:
    no target week in access.test_years -> return (development needs no recipe)
    otherwise -> load_frozen_recipe(access.recipe_path), read from disk on every call (no cache)
                   -> structure and formats of procedencia, aprobacion and hash_sha256
                   -> hash_sha256 == recipe_hash(recipe)
                   -> procedencia.version_dvc_canal == access.channel_version
                 any failure -> FrozenRecipeError naming the cause

Hash (approved in U3.1): SHA-256 of json.dumps(recipe without hash_sha256, sort_keys=True,
separators=(",", ":"), ensure_ascii=False, allow_nan=False) in UTF-8, dates in ISO 8601, so the order
of the YAML keys and quoting a date do not change it. Duplicate YAML keys are rejected, because the hash
is computed on the loaded content and a duplicate would hide one of the values.

The module only reads: it never writes or creates configs/receta_congelada.yaml (generated in F6 and,
after DV-02, in F6b; approved by the person responsible for the implementation). The log of test executions
belongs to evaluate.py (F7).
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import yaml

from src.epidemiology.validation import require

FROZEN_RECIPE_PATH = Path("configs/receta_congelada.yaml")   # especificación §4 and §8
HASH_FIELD = "hash_sha256"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_MD5 = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class FrozenRecipeError(RuntimeError):
    """A score on test-period rows was requested without a valid frozen recipe."""


@dataclass(frozen=True)
class RecipeAccess:
    """What the barrier needs: the test years, the configured channel version and the recipe path."""

    test_years: tuple[int, int]
    channel_version: str
    recipe_path: Path

    def __post_init__(self):
        years = tuple(self.test_years)
        require(len(years) == 2 and all(isinstance(y, int) and not isinstance(y, bool) for y in years)
                and years[0] <= years[1], f"test_years must be two ordered integers; got {self.test_years!r}.")
        require(isinstance(self.channel_version, str) and _MD5.fullmatch(self.channel_version) is not None,
                f"channel_version must be a 32-character lowercase md5; got {self.channel_version!r}.")
        object.__setattr__(self, "test_years", years)
        object.__setattr__(self, "recipe_path", Path(self.recipe_path))

    @classmethod
    def from_config(cls, config, recipe_path: str | Path = FROZEN_RECIPE_PATH) -> "RecipeAccess":
        return cls(test_years=tuple(config.validacion.prueba), channel_version=config.datos.version_dvc_canal,
                   recipe_path=Path(recipe_path))


# --------------------------------------------------------------------------- hash and loading
def _iso(value):
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"value of type {type(value).__name__} is not serializable")


def recipe_hash(recipe: Mapping) -> str:
    """SHA-256 of the recipe without its hash field (canonical JSON, dates in ISO, UTF-8)."""
    if not isinstance(recipe, Mapping):
        raise FrozenRecipeError(f"A recipe must be a mapping; got {type(recipe).__name__}.")
    content = {key: value for key, value in recipe.items() if key != HASH_FIELD}
    try:
        text = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False, default=_iso)
    except (TypeError, ValueError) as exc:
        raise FrozenRecipeError(f"Recipe content cannot be serialized for its hash: {exc}.") from exc
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate keys in any mapping."""


def _construct_unique_mapping(loader, node, deep=False):
    keys = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in keys:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
        keys.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _section(recipe: Mapping, name: str, path: Path) -> Mapping:
    section = recipe.get(name)
    if not isinstance(section, Mapping):
        raise FrozenRecipeError(f"Frozen recipe {path} lacks the section {name!r}.")
    return section


def _text(section: Mapping, section_name: str, field: str, path: Path, pattern: re.Pattern | None = None) -> str:
    value = section.get(field)
    if not isinstance(value, str) or not value.strip():
        raise FrozenRecipeError(f"Frozen recipe {path}: {section_name}.{field} is missing or empty.")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise FrozenRecipeError(f"Frozen recipe {path}: {section_name}.{field} has an invalid format.")
    return value


def _approval_date(section: Mapping, path: Path) -> None:
    value = section.get("fecha")
    if isinstance(value, date) and not isinstance(value, datetime):
        return
    if isinstance(value, str) and len(value) == 10:
        try:
            date.fromisoformat(value)
            return
        except ValueError:
            pass
    raise FrozenRecipeError(f"Frozen recipe {path}: aprobacion.fecha is missing or not a date (YYYY-MM-DD).")


def load_frozen_recipe(path: str | Path, *, channel_version: str) -> dict:
    """Read and validate the frozen recipe; raise FrozenRecipeError naming the cause if it is not valid."""
    path = Path(path)
    if not path.is_file():
        raise FrozenRecipeError(f"No frozen recipe at {path}: scores on test-period rows require the recipe "
                                f"approved by Jesús (CLAUDE.md rule 1).")
    try:
        recipe = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise FrozenRecipeError(f"Frozen recipe {path} is not valid YAML: {exc}") from exc
    if not isinstance(recipe, Mapping):
        raise FrozenRecipeError(f"Frozen recipe {path} must be a YAML mapping.")

    provenance = _section(recipe, "procedencia", path)
    _text(provenance, "procedencia", "commit_codigo", path, _COMMIT)
    _text(provenance, "procedencia", "commit_contrato", path, _COMMIT)
    recipe_channel = _text(provenance, "procedencia", "version_dvc_canal", path, _MD5)
    approval = _section(recipe, "aprobacion", path)
    _text(approval, "aprobacion", "nombre", path)
    _approval_date(approval, path)

    stored = recipe.get(HASH_FIELD)
    if not isinstance(stored, str) or _SHA256.fullmatch(stored) is None:
        raise FrozenRecipeError(f"Frozen recipe {path}: {HASH_FIELD} is missing or not a SHA-256.")
    computed = recipe_hash(recipe)
    if stored != computed:
        raise FrozenRecipeError(f"Frozen recipe {path}: stored hash {stored} differs from the content hash "
                                f"{computed}; the recipe changed after it was frozen.")
    if recipe_channel != channel_version:
        raise FrozenRecipeError(f"Frozen recipe {path} was frozen for channel version {recipe_channel}, "
                                f"but the configuration expects {channel_version}.")
    return dict(recipe)


# --------------------------------------------------------------------------- the barrier
def _years(target_weeks: Iterable) -> set[int]:
    years = set()
    for week in target_weeks:
        require(isinstance(week, date) and not pd.isna(week),
                f"Target weeks must be dates; got {week!r}.")
        years.add(week.year)
    return years


def check_test_access(target_weeks: Iterable, access: RecipeAccess) -> None:
    """Single verification: test-period target weeks require a valid frozen recipe (read on every call)."""
    require(isinstance(access, RecipeAccess), f"access must be a RecipeAccess; got {type(access).__name__}.")
    first, last = access.test_years
    if any(first <= year <= last for year in _years(target_weeks)):
        load_frozen_recipe(access.recipe_path, channel_version=access.channel_version)
