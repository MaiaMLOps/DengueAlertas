"""Hooks shared by the whole tests/ tree.

DV-02 (docs/forecasting/desviaciones.md): amendment D7.2 (contrato §6.6) made the feature set a selection step and
replaced the frozen recipe d2135a39…da98 before the test period was opened. Some frozen acceptance tests assert that
recipe (its hash, logistica_03, tau = 0.10), an explicit `ablation` argument or an ablation system named
"sin_clima". They are not edited (CLAUDE.md, «Pruebas»): they are marked here, explicitly, as `sustituida_dv02` and
skipped with a reason that names their replacement, so they show up as SKIPPED in every report (-rA) and are never
silently omitted. The frozen files that stay as a record of the full-set selection of F6 (test_f6_real.py,
test_f6receta_borrador.py and the other tests of test_f7_diagnostico_desarrollo.py) keep running and passing.

The list below is the one in DV-02; changing it requires a new entry in desviaciones.md.
"""

import pytest

_F7B_EVALUATION = ("run_test reads the ablation from the recipe (family, feature set and configuration per "
                   "municipality); the ablation system is no longer 'sin_clima' nor an explicit argument. "
                   "Replacement: tests/acceptance/test_f7b_evaluacion.py (U7.3).")
_F7B_CARD = ("the card compares the recipe with its ablation, whose feature set depends on D7.2; the synthetic "
             "fixture names it 'sin_clima'. Replacement: tests/acceptance/test_f7b_artefactos.py (U7.4).")
_F7B_REAL = ("asserts the superseded recipe (hash d2135a39…da98, logistica_03 with the full set, tau = 0.10). "
             "Replacement: tests/acceptance/test_f7b_prueba_real.py (U7.3).")

SUPERSEDED_DV02 = {
    "tests/acceptance/test_f7_evaluacion.py": _F7B_EVALUATION,
    "tests/acceptance/test_f7_artefactos.py::test_card_has_the_fields_of_api_section_7": _F7B_CARD,
    "tests/acceptance/test_f7_artefactos.py::test_global_levels_use_the_api_spelling_and_the_lower_level": _F7B_CARD,
    "tests/acceptance/test_f7_artefactos.py::test_municipality_cards_use_the_descriptive_level": _F7B_CARD,
    "tests/acceptance/test_f7_artefactos.py::test_card_events_and_the_note_for_a_municipality_without_onsets":
        _F7B_CARD,
    "tests/acceptance/test_f7_artefactos.py::test_final_models_use_all_rows_and_the_recipe":
        ("the recipe now names the feature set of each municipality; a recipe without it is not valid. "
         "Replacement: tests/acceptance/test_f7b_artefactos.py (U7.4)."),
    "tests/acceptance/test_f7_prueba_real.py::test_prueba_real_log_has_one_execution": _F7B_REAL,
    "tests/acceptance/test_f7_prueba_real.py::test_prueba_real_predictions_cover_the_test_rows": _F7B_REAL,
    "tests/acceptance/test_f7_prueba_real.py::test_prueba_real_card_fields_and_sizes": _F7B_REAL,
    "tests/acceptance/test_f7_prueba_real.py::test_prueba_real_history_has_only_rolling_origin_predictions": _F7B_REAL,
    "tests/acceptance/test_f7_prueba_real.py::test_prueba_real_final_models": _F7B_REAL,
    "tests/acceptance/test_f7_diagnostico_desarrollo.py::test_tau_matches_the_frozen_recipe":
        ("reads configs/receta_congelada.yaml, retired by DV-02. The development diagnostic of the new recipe goes "
         "to tests/acceptance/test_f7b_diagnostico_desarrollo.py, with an independent check before freezing."),
}


def pytest_configure(config):
    config.addinivalue_line("markers", "sustituida_dv02: frozen acceptance test superseded by DV-02 (D7.2); skipped "
                                       "with the reason that names its replacement")


def pytest_collection_modifyitems(config, items):
    for item in items:
        path, _, rest = item.nodeid.partition("::")
        name = rest.split("[", 1)[0]
        reason = SUPERSEDED_DV02.get(path) or SUPERSEDED_DV02.get(f"{path}::{name}")
        if reason is not None:
            item.add_marker(pytest.mark.sustituida_dv02)
            item.add_marker(pytest.mark.skip(reason=f"Sustituida por DV-02: {reason}"))
