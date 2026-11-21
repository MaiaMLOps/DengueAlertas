"""Copies of the final models in modelos/ (course delivery structure; modelos/README.md).

modelos/ holds a byte-for-byte copy of the DVC output data/processed/forecasting/prueba/modelos_finales/, which stays
the source of truth. The comparison with that output runs only when it is present locally (after `dvc pull`) and is
skipped otherwise; the check against the frozen recipe uses versioned files only and always runs.
"""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COPY_DIR = ROOT / "modelos"
DVC_OUTPUT = ROOT / "data" / "processed" / "forecasting" / "prueba" / "modelos_finales"
RECIPE = ROOT / "configs" / "receta_congelada.yaml"
MUNICIPALITIES = ("08001", "08758", "41001", "50001", "54001", "76001")
FILES = tuple(f"{code}.joblib" for code in MUNICIPALITIES) + ("manifiesto.json",)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_copy_has_the_six_models_and_the_manifest():
    present = sorted(path.name for path in COPY_DIR.iterdir() if path.name != "README.md")
    assert present == sorted(FILES)


def test_copy_manifest_matches_the_frozen_recipe():
    manifest = json.loads((COPY_DIR / "manifiesto.json").read_text(encoding="utf-8"))
    recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    assert manifest["receta_hash"] == recipe["hash_sha256"]
    assert sorted(manifest["modelos"]) == list(MUNICIPALITIES)
    for code, entry in manifest["modelos"].items():
        spec = recipe["municipios"][code]
        assert entry["familia"] == spec["familia"]
        assert entry["candidato"] == spec["candidato"]
        assert entry["conjunto_variables"] == spec["conjunto_variables"]
        assert entry["hiperparametros"] == spec["hiperparametros"]


def test_copy_is_identical_to_the_dvc_output():
    missing = [name for name in FILES if not (DVC_OUTPUT / name).is_file()]
    if missing:
        pytest.skip(f"DVC output {DVC_OUTPUT.relative_to(ROOT)} not available locally (dvc pull); missing {missing}.")
    differing = [name for name in FILES if _md5(COPY_DIR / name) != _md5(DVC_OUTPUT / name)]
    assert not differing, f"modelos/ differs from the DVC output in {differing}."
