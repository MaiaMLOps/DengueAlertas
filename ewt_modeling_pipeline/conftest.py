"""Aislamiento del aporte EWT respecto de la suite del repositorio.

Cuando se ejecuta `pytest` desde la raíz del repo, estas pruebas se omiten porque
dependen de `pythonpath = ["src"]` definido en el pyproject de esta carpeta.
Para correrlas: `cd ewt_modeling_pipeline && python -m pytest`.
"""
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CWD = Path.cwd().resolve()

# Solo se ignoran cuando NO se está ejecutando desde dentro de esta carpeta.
if _CWD == _HERE or _HERE in _CWD.parents:
    collect_ignore = []
else:
    collect_ignore = ["tests", "test_smoke.py"]
