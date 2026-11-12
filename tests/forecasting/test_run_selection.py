"""Unit tests of src/forecasting/run_selection.py and of the recipe draft helpers (synthetic data only)."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.epidemiology.config import EpidemiologyValidationError
from src.forecasting import run_selection
from src.forecasting.config import load_config
from src.forecasting.selection import _check_provenance, write_recipe_draft


def _descriptor(folder: Path, name: str, md5: str) -> None:
    (folder / f"{name}.dvc").write_text(yaml.safe_dump({"outs": [{"md5": md5, "hash": "md5", "path": name}]}),
                                        encoding="utf-8")


def _search_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "busqueda"
    folder.mkdir()
    (folder / run_selection.TABLE_FILENAME).write_bytes(b"tabla")
    (folder / run_selection.PREDICTIONS_FILENAME).write_bytes(b"predicciones")
    (folder / run_selection.MANIFEST_FILENAME).write_text(json.dumps({"busqueda_id": "abc"}), encoding="utf-8")
    return folder


def test_search_provenance_reads_the_manifest_and_hashes_the_inputs(tmp_path):
    folder = _search_folder(tmp_path)
    provenance = run_selection.search_provenance(folder, config=load_config())
    assert provenance["busqueda_id"] == "abc"
    assert set(provenance["insumos_md5"]) == set(run_selection.PROVENANCE_INPUTS)
    assert provenance["insumos_md5"][run_selection.TABLE_FILENAME] == run_selection._md5(
        folder / run_selection.TABLE_FILENAME)
    assert len(provenance["commit_codigo"]) == 40 and len(provenance["commit_contrato"]) == 40


def test_search_provenance_fails_when_an_input_is_missing(tmp_path):
    folder = _search_folder(tmp_path)
    (folder / run_selection.MANIFEST_FILENAME).unlink()
    with pytest.raises(EpidemiologyValidationError, match="missing"):
        run_selection.search_provenance(folder, config=load_config())


def test_inputs_are_checked_against_their_descriptors(tmp_path):
    folder = _search_folder(tmp_path)
    provenance = run_selection.search_provenance(folder, config=load_config())
    descriptors = tmp_path / "dvc"
    descriptors.mkdir()
    for name, digest in provenance["insumos_md5"].items():
        _descriptor(descriptors, name, digest)
    run_selection.check_inputs_against_dvc(provenance, descriptors)

    _descriptor(descriptors, run_selection.TABLE_FILENAME, "0" * 32)
    with pytest.raises(EpidemiologyValidationError, match="DVC descriptor says"):
        run_selection.check_inputs_against_dvc(provenance, descriptors)
    (descriptors / f"{run_selection.MANIFEST_FILENAME}.dvc").unlink()
    with pytest.raises(EpidemiologyValidationError):
        run_selection.check_inputs_against_dvc(provenance, descriptors)


def test_evidence_is_never_overwritten(tmp_path):
    output = tmp_path / "evidencia"
    output.mkdir()
    (output / "curva_tau.csv").write_text("previo", encoding="utf-8")
    with pytest.raises(EpidemiologyValidationError, match="never overwritten"):
        run_selection.write_evidence(output, {}, {}, {})
    assert (output / "curva_tau.csv").read_text(encoding="utf-8") == "previo"
    assert sorted(path.name for path in output.iterdir()) == ["curva_tau.csv"]


def test_provenance_needs_md5_digests():
    config = load_config()
    base = {"commit_codigo": "a" * 40, "commit_contrato": "b" * 40, "version_dvc_canal": config.datos.version_dvc_canal,
            "busqueda_id": "x", "insumos_md5": {"tabla": "c" * 32}}
    assert _check_provenance(base, config)["insumos_md5"] == {"tabla": "c" * 32}
    for bad in ({}, {"tabla": "corto"}):
        with pytest.raises(EpidemiologyValidationError, match="insumos_md5"):
            _check_provenance({**base, "insumos_md5": bad}, config)
    with pytest.raises(EpidemiologyValidationError, match="busqueda_id"):
        _check_provenance({**base, "busqueda_id": " "}, config)


def test_digit_texts_are_quoted_at_any_depth(tmp_path):
    recipe = {"municipios": {"08758": {"diferencias": {"2019": -0.1}}}, "nombre": "logistica_03", "n": 8}
    path = write_recipe_draft(recipe, tmp_path / "borrador.yaml")
    text = path.read_text(encoding="utf-8")
    assert "'08758':" in text and "'2019':" in text and "n: 8" in text and "nombre: logistica_03" in text
    assert yaml.safe_load(text) == recipe


def test_draft_that_does_not_round_trip_is_refused(tmp_path):
    with pytest.raises(EpidemiologyValidationError, match="round-trip"):
        write_recipe_draft({"valor": float("nan")}, tmp_path / "borrador.yaml")
    assert not (tmp_path / "borrador.yaml").exists()


def test_selection_runs_are_compared_field_by_field():
    from src.forecasting.selection import FamilyComparison, Selection, TauChoice

    frame = pd.DataFrame({"a": [1.0]})
    families = FamilyComparison("logistica", {"logistica": 0.1, "lightgbm": 0.0}, 1, 2, 2, False)
    first = Selection(families, frame, frame, frame, frame, {"x": 1}, TauChoice(0.1, True))
    assert run_selection._same(first, Selection(families, frame.copy(), frame, frame, frame, {"x": 1},
                                                TauChoice(0.1, True)))
    assert not run_selection._same(first, Selection(families, frame, frame, frame, frame + 1e-15, {"x": 1},
                                                    TauChoice(0.1, True)))
    assert not run_selection._same(first, Selection(families, frame, frame, frame, frame, {"x": 2},
                                                    TauChoice(0.1, True)))


# --------------------------------------------------------------------------- D7.2 mode
def _manifest_folder(tmp_path: Path, name: str, **content) -> Path:
    from src.forecasting.dataset import FEATURE_SETS

    base = {"anios": [2019, 2020, 2021, 2022], "prueba": [2023, 2025], "primer_anio_entrenamiento": 2012,
            "municipios": ["08001"], "configuracion": {"h": 8}, "candidatos": [{"candidato": "r"}],
            "ejemplos_sha256": "e" * 64}
    if name == "sin_clima":
        base["variante"] = {"nombre": "sin_clima", "features": list(FEATURE_SETS["sin_clima"])}
    base.update(content)
    folder = tmp_path / name
    folder.mkdir()
    (folder / run_selection.MANIFEST_FILENAME).write_text(json.dumps({"busqueda_id": name, "contenido": base}),
                                                          encoding="utf-8")
    return folder


def test_paired_searches_are_accepted(tmp_path):
    run_selection.check_paired_searches(_manifest_folder(tmp_path, "completo"), _manifest_folder(tmp_path, "sin_clima"))


@pytest.mark.parametrize("change", [{"ejemplos_sha256": "f" * 64}, {"candidatos": [{"candidato": "s"}]},
                                    {"municipios": ["08001", "76001"]}, {"variante": {"nombre": "otra"}}])
def test_unpaired_searches_are_refused(tmp_path, change):
    with pytest.raises(EpidemiologyValidationError):
        run_selection.check_paired_searches(_manifest_folder(tmp_path, "completo"),
                                            _manifest_folder(tmp_path, "sin_clima", **change))


def test_the_full_set_search_must_have_no_variant(tmp_path):
    full = _manifest_folder(tmp_path, "completo", variante={"nombre": "sin_clima"})
    with pytest.raises(EpidemiologyValidationError):
        run_selection.check_paired_searches(full, _manifest_folder(tmp_path, "sin_clima"))


def test_forced_set_needs_the_sin_clima_search(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_selection, "_dirty_paths", lambda: [])
    assert run_selection.main(["--salida", str(tmp_path / "x"), "--conjunto-forzado", "completo"]) == 1
    assert "--busqueda-sin-clima" in capsys.readouterr().err


def test_dvc_status_of_a_search_without_descriptors(tmp_path):
    folder = _search_folder(tmp_path)
    provenance = run_selection.search_provenance(folder, config=load_config())
    assert run_selection._dvc_status(provenance, folder).startswith("pendiente")
    for name, digest in provenance["insumos_md5"].items():
        _descriptor(folder, name, digest)
    assert run_selection._dvc_status(provenance, folder).startswith("verificado")
