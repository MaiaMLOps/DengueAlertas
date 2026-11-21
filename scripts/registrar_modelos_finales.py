"""Registration of the six final models in the MLflow model registry (especificación §9; F8).

Flow:
    verify_files: md5 of the six modelos/<code>.joblib and of manifiesto.json against the table of modelos/README.md;
      manifest consistent with the frozen recipe (hash and per-municipality configuration). Any mismatch stops.
    emission_rows: inputs of the emissions of the final-model window (the week after the last training emission up
      to the last complete week; the emissions of notebooks/03_uso_del_modelo.ipynb), built with
      dataset.build_features from the channel product. The last one is the input example of the signature.
    build_plan (read-only calls only): per municipality, the registered name DengueAlertas_<Municipio>, whether it
      already exists and with which versions, the version whose md5 tag equals the local file (skipped: idempotent),
      the tags of the registered model and of the version, the run parameters and the alias action.
    dry run (default): prints the plan; nothing is written to the server.
    --ejecutar: per municipality to register, one run in the dengue_h8_final experiment with parameters, tags and the
      original .joblib as an artifact; the model logged with mlflow.sklearn.log_model (signature and input example);
      the registered model created with its tags only when the name is new; the new version with its tags; the alias
      only when --alias is given and that alias is free in the registered model. Then every version of the plan is
      loaded from models:/<name>/<version> and its probabilities must equal those of the local .joblib bit for bit.

Nothing existing in MLflow is modified or deleted: tags of an existing registered model are reported, not changed;
an alias already assigned to another version is never moved (the run stops before writing). The tag `enabled` of a
new registered model is "true" only with --habilitar: src/api.py loads every DengueAlertas_* model with enabled=true,
by alias and, without one, by its latest version.

Authentication goes through src/forecasting/tracking.py (Cloudflare Access headers from .env); credentials are never
printed. The client version must equal the server version (README §4; especificación §9).

Run from the repository root:
    .venv/Scripts/python.exe -B -X utf8 scripts/registrar_modelos_finales.py                 (dry run)
    .venv/Scripts/python.exe -B -X utf8 scripts/registrar_modelos_finales.py --ejecutar [--alias staging] [--habilitar]
"""

import argparse
from dataclasses import dataclass, field
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "60")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import yaml  # noqa: E402

MODELS_DIR = ROOT / "modelos"
README_PATH = MODELS_DIR / "README.md"
MANIFEST_NAME = "manifiesto.json"
RECIPE_PATH = ROOT / "configs" / "receta_congelada.yaml"
NAME_PREFIX = "DengueAlertas_"
PHASE = "F8"
ARTIFACT_NAME = "modelo"
ORIGINAL_ARTIFACT_DIR = "joblib_original"
CHECKED_PATHS = ("src", "configs", "modelos", "scripts")


@dataclass(frozen=True)
class Municipality:
    """Registry name (ASCII, especificación §9) and the geographic tags read by src/api.py."""

    registro: str
    nombre: str
    departamento: str
    depto_id: str   # "CO" + ISO 3166-2 department code, as in src/api.py and the frontend map


MUNICIPALITIES = {
    "08001": Municipality("Barranquilla", "Barranquilla", "Atlántico", "COATL"),
    "08758": Municipality("Soledad", "Soledad", "Atlántico", "COATL"),
    "41001": Municipality("Neiva", "Neiva", "Huila", "COHUI"),
    "50001": Municipality("Villavicencio", "Villavicencio", "Meta", "COMET"),
    "54001": Municipality("Cucuta", "San José de Cúcuta", "Norte de Santander", "CONSA"),
    "76001": Municipality("Cali", "Cali", "Valle del Cauca", "COVAC"),
}
_README_ROW = re.compile(r"^\|\s*`(?P<file>[^`]+)`\s*\|[^|]*\|[^|]*\|\s*`(?P<md5>[0-9a-f]{32})`\s*\|\s*$")


class RegistrationError(RuntimeError):
    """The registration cannot proceed; the message names the cause."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistrationError(message)


def registered_name(code: str) -> str:
    return NAME_PREFIX + MUNICIPALITIES[code].registro


# --------------------------------------------------------------------------- local files
def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def readme_md5(readme: Path) -> dict[str, str]:
    """{file name: md5} from the table of modelos/README.md."""
    rows = {}
    for line in Path(readme).read_text(encoding="utf-8").splitlines():
        match = _README_ROW.match(line.strip())
        if match:
            _require(match["file"] not in rows, f"{readme} lists {match['file']} twice.")
            rows[match["file"]] = match["md5"]
    return rows


def verify_files(models_dir: Path, readme: Path, recipe: dict) -> tuple[dict, dict[str, str]]:
    """(manifest, {code: md5 of <code>.joblib}) after checking md5 against the README and the manifest against the recipe."""
    expected = readme_md5(readme)
    names = [f"{code}.joblib" for code in MUNICIPALITIES] + [MANIFEST_NAME]
    missing = sorted(set(names) - set(expected))
    _require(not missing, f"{readme} has no md5 for {missing}.")
    observed = {}
    for name in names:
        path = Path(models_dir) / name
        _require(path.is_file(), f"Missing {path}.")
        observed[name] = file_md5(path)
    differing = [name for name in names if observed[name] != expected[name]]
    _require(not differing, f"md5 differs from {readme.name} for {differing}; nothing is registered.")

    manifest = json.loads((Path(models_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))
    _require(manifest.get("receta_hash") == recipe.get("hash_sha256"),
             "manifiesto.json does not correspond to the frozen recipe (receta_hash).")
    _require(sorted(manifest.get("modelos", {})) == sorted(MUNICIPALITIES),
             f"manifiesto.json must list exactly {sorted(MUNICIPALITIES)}.")
    for code, entry in manifest["modelos"].items():
        spec = recipe["municipios"][code]
        for key in ("familia", "candidato", "conjunto_variables", "hiperparametros"):
            _require(entry.get(key) == spec.get(key),
                     f"manifiesto.json {code}.{key} differs from the frozen recipe.")
    return manifest, {code: observed[f"{code}.joblib"] for code in MUNICIPALITIES}


def _git(*args: str) -> str:
    result = subprocess.run(["git", "--no-optional-locks", *args], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def dirty_paths() -> list[str]:
    return _git("status", "--porcelain=v1", "--untracked-files=all", "--", *CHECKED_PATHS).splitlines()


# --------------------------------------------------------------------------- tags and parameters
def registered_model_tags(code: str, entry: dict, recipe_hash: str, *, enabled: bool) -> dict[str, str]:
    """Tags of the registered model (especificación §9); src/api.py reads enabled, municipality_code, department, depto_id."""
    municipality = MUNICIPALITIES[code]
    return {"enabled": "true" if enabled else "false", "municipality_code": code,
            "department": municipality.departamento, "depto_id": municipality.depto_id,
            "receta_hash": recipe_hash, "familia": entry["familia"]}


def version_tags(code: str, entry: dict, md5: str, recipe_hash: str, commit: str) -> dict[str, str]:
    """Tags of the model version; md5 identifies the source file and makes the registration idempotent."""
    return {"municipality_code": code, "municipio": MUNICIPALITIES[code].nombre, "receta_hash": recipe_hash,
            "familia": entry["familia"], "candidato": entry["candidato"],
            "conjunto_variables": entry["conjunto_variables"], "md5": md5, "commit": commit}


def run_tags(code: str, entry: dict, md5: str, recipe: dict, commit: str) -> dict[str, str]:
    """Run tags of especificación §9 (municipio, familia, candidato, fase, commit, contrato_commit, datos_dvc,
    receta_hash), plus the DANE code, the name and the md5 of the source file."""
    return {"municipio": code, "municipio_nombre": MUNICIPALITIES[code].nombre, "municipality_code": code,
            "familia": entry["familia"], "candidato": entry["candidato"], "fase": PHASE, "commit": commit,
            "contrato_commit": recipe["procedencia"]["commit_contrato"],
            "datos_dvc": recipe["procedencia"]["version_dvc_canal"], "receta_hash": recipe["hash_sha256"],
            "md5": md5}


def run_params(entry: dict) -> dict[str, str]:
    params = {"familia": entry["familia"], "candidato": entry["candidato"],
              "conjunto_variables": entry["conjunto_variables"], "filas": entry["filas"],
              "ultima_emision": entry["ultima_emision"], "ultima_objetivo": entry["ultima_objetivo"]}
    params.update(entry["hiperparametros"])
    return {key: str(value) for key, value in params.items()}


# --------------------------------------------------------------------------- plan (read-only)
@dataclass
class PlanItem:
    code: str
    name: str
    md5: str
    exists: bool
    versions: list[dict]                 # [{"version", "md5", "aliases"}] of the existing registered model
    existing_tags: dict[str, str]
    action: str                          # "registrar" or "omitir"
    duplicate_version: str | None
    model_tags: dict[str, str]
    version_tags: dict[str, str]
    run_tags: dict[str, str]
    params: dict[str, str]
    alias_action: str                    # "ninguno", "asignar", "ya_asignado"
    notes: list[str] = field(default_factory=list)


def _registered_model(client, name: str):
    matches = [model for model in client.search_registered_models(filter_string=f"name = '{name}'")
               if model.name == name]
    return matches[0] if matches else None


def build_plan(client, manifest: dict, md5s: dict[str, str], recipe: dict, *, commit: str, alias: str | None,
               enabled: bool) -> list[PlanItem]:
    """The registration plan from read-only registry calls; raises when an alias would have to be moved."""
    plan = []
    for code in MUNICIPALITIES:
        entry, name = manifest["modelos"][code], registered_name(code)
        model = _registered_model(client, name)
        versions, existing_tags, aliases = [], {}, {}
        if model is not None:
            existing_tags = dict(model.tags or {})
            aliases = dict(model.aliases or {})
            for version in client.search_model_versions(filter_string=f"name = '{name}'"):
                versions.append({"version": str(version.version), "md5": (version.tags or {}).get("md5"),
                                 "aliases": sorted(a for a, v in aliases.items() if str(v) == str(version.version))})
            versions.sort(key=lambda item: int(item["version"]))
        duplicate = next((item["version"] for item in versions if item["md5"] == md5s[code]), None)
        model_tags = registered_model_tags(code, entry, recipe["hash_sha256"], enabled=enabled)
        notes = []
        if model is not None:
            differing = {key: existing_tags.get(key) for key, value in model_tags.items()
                         if existing_tags.get(key) != value}
            if differing:
                notes.append(f"El modelo registrado ya existe; sus etiquetas no se modifican (difieren: {differing}).")
        alias_action = "ninguno"
        if alias:
            holder = aliases.get(alias)
            if holder is None:
                alias_action = "asignar"
            elif duplicate is not None and str(holder) == duplicate:
                alias_action = "ya_asignado"
            else:
                raise RegistrationError(f"Alias {alias!r} of {name} already points to version {holder}; it is never "
                                        f"moved by this script.")
        plan.append(PlanItem(code=code, name=name, md5=md5s[code], exists=model is not None, versions=versions,
                             existing_tags=existing_tags, action="omitir" if duplicate else "registrar",
                             duplicate_version=duplicate, model_tags=model_tags,
                             version_tags=version_tags(code, entry, md5s[code], recipe["hash_sha256"], commit),
                             run_tags=run_tags(code, entry, md5s[code], recipe, commit), params=run_params(entry),
                             alias_action=alias_action, notes=notes))
    return plan


def experiment_state(client, name: str) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        return "no existe (se crea al ejecutar)"
    if experiment.lifecycle_stage != "active":
        raise RegistrationError(f"Experiment {name} exists with lifecycle_stage={experiment.lifecycle_stage}; "
                                f"it is not restored by this script.")
    return f"existe (id {experiment.experiment_id})"


def format_plan(plan: list[PlanItem], *, experiment: str, experiment_name: str, alias: str | None,
                signature=None, example=None) -> str:
    lines = [f"Experimento {experiment_name}: {experiment}",
             f"Alias: {alias or 'ninguno (no se asigna)'}"]
    if signature is not None:
        lines.append(f"Firma: entradas {signature.inputs.to_dict()}; salidas {signature.outputs.to_dict()}")
    if example is not None:
        lines.append("Ejemplo de entrada (última emisión):\n" + example.to_string())
    for item in plan:
        lines.append("")
        lines.append(f"[{item.code}] {item.name}: {item.action.upper()}"
                     + (f" (la versión {item.duplicate_version} ya tiene md5 {item.md5})" if item.duplicate_version else ""))
        if item.exists:
            listed = ", ".join(f"v{v['version']} md5={v['md5']} alias={v['aliases']}" for v in item.versions) or "sin versiones"
            lines.append(f"  En el registro: sí; versiones: {listed}")
        else:
            lines.append("  En el registro: no; se crea el modelo registrado con sus etiquetas")
        lines.append(f"  Etiquetas del modelo registrado: {item.model_tags}")
        lines.append(f"  Etiquetas de la versión: {item.version_tags}")
        lines.append(f"  Run: tags {item.run_tags}")
        lines.append(f"       params {item.params}")
        lines.append(f"  Alias: {item.alias_action}")
        lines.extend(f"  Nota: {note}" for note in item.notes)
    return "\n".join(lines)


# --------------------------------------------------------------------------- inputs (signature and verification)
def emission_rows(config, manifest: dict):
    """(rows with the fifteen features of every final-model emission, last complete week)."""
    import pyarrow.compute as pc

    from src.epidemiology.validation import SOURCE_SCHEMA
    from src.forecasting.dataset import build_features, load_channel_product

    product = load_channel_product(config.datos.canal, expected_md5=config.datos.version_dvc_canal)
    panel = product.select(SOURCE_SCHEMA.names)
    last = pc.max(pc.filter(panel["week_start"], pc.invert(panel["partial_week"]))).as_py()
    first_dates = {date.fromisoformat(entry["ultima_emision"]) for entry in manifest["modelos"].values()}
    _require(len(first_dates) == 1, "The final models do not share their last training emission.")
    first = first_dates.pop() + timedelta(weeks=1)
    weeks = [first + timedelta(weeks=i) for i in range((last - first).days // 7 + 1)]
    _require(len(weeks) >= 1, "No emission after the last training emission.")
    rows = build_features(panel, [(code, week) for code in MUNICIPALITIES for week in weeks], h=config.horizonte.h,
                          canal=config.canal, features=config.features)
    return rows, last


def signature_and_example(model, rows, code: str, last, features: tuple[str, ...]):
    from mlflow.models import infer_signature

    example = rows[(rows["municipality_code"] == code) & (rows["t"] == last)][list(features)].reset_index(drop=True)
    _require(len(example) == 1, f"No input row for {code} at {last}.")
    return infer_signature(example, model.predict(example)), example


# --------------------------------------------------------------------------- server
def server_version(uri: str) -> str:
    import requests
    from mlflow.tracking.request_header.registry import resolve_request_headers

    response = requests.get(uri.rstrip("/") + "/version", headers=resolve_request_headers(), timeout=30)
    _require(response.status_code == 200, f"The server does not expose /version (HTTP {response.status_code}).")
    return response.text.strip()


def check_versions(client_version: str, server: str) -> None:
    _require(client_version == server, f"MLflow client {client_version} differs from the server {server} "
                                       f"(requirements/forecasting.txt must match the server).")


def register(plan: list[PlanItem], *, client, experiment_name: str, alias: str | None, rows, last,
             features: tuple[str, ...], models_dir: Path = MODELS_DIR) -> dict[str, str]:
    """Writes the plan; returns {name: version} of every municipality (new or already registered)."""
    import joblib
    import mlflow
    import mlflow.sklearn
    import numpy
    import pandas
    import sklearn
    import cloudpickle

    requirements = [f"scikit-learn=={sklearn.__version__}", f"numpy=={numpy.__version__}",
                    f"pandas=={pandas.__version__}", f"cloudpickle=={cloudpickle.__version__}"]
    mlflow.set_experiment(experiment_name)
    registered = {}
    for item in plan:
        if item.action == "omitir":
            registered[item.name] = item.duplicate_version
        else:
            path = Path(models_dir) / f"{item.code}.joblib"
            _require(file_md5(path) == item.md5, f"{path} changed after the plan was built.")
            model = joblib.load(path)
            signature, example = signature_and_example(model, rows, item.code, last, features)
            with mlflow.start_run(run_name=f"final_{item.code}", tags=item.run_tags) as run:
                mlflow.log_params(item.params)
                mlflow.log_artifact(str(path), artifact_path=ORIGINAL_ARTIFACT_DIR)
                # cloudpickle, not the skops default of MLflow 3.17: skops is not a dependency of the project or of
                # the API, and cloudpickle ships with MLflow.
                info = mlflow.sklearn.log_model(model, name=ARTIFACT_NAME, signature=signature, input_example=example,
                                                pip_requirements=requirements,
                                                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
            if not item.exists:
                municipality = MUNICIPALITIES[item.code]
                client.create_registered_model(
                    item.name, tags=item.model_tags,
                    description=(f"P(zona epidémica del canal de epiCo en t + 8) para {municipality.nombre} "
                                 f"({item.code}). Regresión logística L2 de la receta congelada "
                                 f"{item.model_tags['receta_hash'][:8]}; probabilidad = predict_proba[:, 1] con las "
                                 f"variables sin_clima de src.forecasting.dataset.build_features."))
            version = mlflow.register_model(info.model_uri, item.name, tags=item.version_tags)
            client.update_model_version(item.name, version.version,
                                        description=f"Run {run.info.run_id}; archivo modelos/{item.code}.joblib "
                                                    f"md5 {item.md5}.")
            registered[item.name] = str(version.version)
            print(f"[{item.code}] {item.name}: versión {version.version} registrada (run {run.info.run_id}).")
        if alias and item.alias_action == "asignar":
            client.set_registered_model_alias(item.name, alias, registered[item.name])
            print(f"[{item.code}] {item.name}: alias {alias} -> versión {registered[item.name]}.")
    return registered


def verify_registered(registered: dict[str, str], rows, *, features: tuple[str, ...],
                      models_dir: Path = MODELS_DIR) -> None:
    """Probabilities from models:/<name>/<version> (sklearn flavor and the pyfunc path of src/api.py) equal the
    local .joblib on every final-model emission."""
    import joblib
    import mlflow.pyfunc
    import mlflow.sklearn
    import numpy

    from src.forecasting.dataset import FEATURES
    from src.forecasting.models import predict_positive

    for code in MUNICIPALITIES:
        name = registered_name(code)
        version = registered[name]
        own = rows[rows["municipality_code"] == code].sort_values("t")
        local = predict_positive(joblib.load(Path(models_dir) / f"{code}.joblib"), own[list(FEATURES)])
        uri = f"models:/{name}/{version}"
        remote = mlflow.sklearn.load_model(uri).predict_proba(own[list(FEATURES)])[:, 1]
        api_path = mlflow.pyfunc.load_model(uri)._model_impl.predict_proba(own[list(features)])[:, 1]
        _require(numpy.array_equal(local, remote) and numpy.array_equal(local, api_path),
                 f"{uri}: probabilities differ from the local .joblib.")
        print(f"[{code}] {uri}: {len(own)} emisiones, probabilidades idénticas al .joblib local.")


# --------------------------------------------------------------------------- command line
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Registra los seis modelos finales en el registro de MLflow.")
    parser.add_argument("--ejecutar", action="store_true", help="Registra en el servidor; sin esta opción, ensayo.")
    parser.add_argument("--alias", default=None, help="Alias que se asigna a cada versión (por ejemplo, staging).")
    parser.add_argument("--habilitar", action="store_true",
                        help="Etiqueta enabled=true en los modelos registrados nuevos (src/api.py los carga).")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import mlflow
    from mlflow import MlflowClient

    from src.forecasting.config import load_config
    from src.forecasting.dataset import FEATURE_SETS
    from src.forecasting.tracking import configure_tracking

    config = load_config()
    _require(config.seguimiento.modo == "servidor", "seguimiento.modo must be servidor to use the model registry.")
    recipe = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))
    manifest, md5s = verify_files(MODELS_DIR, README_PATH, recipe)
    print("md5 de los seis modelos y del manifiesto: coinciden con modelos/README.md; manifiesto coherente con la receta.")

    commit = _git("rev-parse", "HEAD")
    _require(bool(commit), "No git commit found.")
    dirty = dirty_paths()
    if args.ejecutar:
        _require(not dirty, f"Uncommitted changes in {CHECKED_PATHS}: {dirty}. The registration records the commit.")
    elif dirty:
        print(f"Aviso: cambios sin commit en {CHECKED_PATHS} ({len(dirty)}); --ejecutar los rechaza.")

    sets = {entry["conjunto_variables"] for entry in manifest["modelos"].values()}
    _require(len(sets) == 1, "The final models do not share one feature set.")
    features = FEATURE_SETS[sets.pop()]
    rows, last = emission_rows(config, manifest)

    uri = configure_tracking(config)
    print(f"Servidor: {uri.split('//')[-1].split('/')[0]}")
    server = server_version(uri)
    check_versions(mlflow.__version__, server)
    print(f"Cliente MLflow {mlflow.__version__} = servidor {server}")

    client = MlflowClient()
    experiment_name = config.seguimiento.experimentos.final
    state = experiment_state(client, experiment_name)
    plan = build_plan(client, manifest, md5s, recipe, commit=commit, alias=args.alias, enabled=args.habilitar)

    import joblib
    example_model = joblib.load(MODELS_DIR / f"{plan[0].code}.joblib")
    signature, example = signature_and_example(example_model, rows, plan[0].code, last, features)
    print(format_plan(plan, experiment=state, experiment_name=experiment_name, alias=args.alias, signature=signature,
                      example=example))

    if not args.ejecutar:
        print("\nEnsayo: no se escribió nada en el servidor. El registro real requiere --ejecutar.")
        return 0
    registered = register(plan, client=client, experiment_name=experiment_name, alias=args.alias, rows=rows,
                          last=last, features=features)
    verify_registered(registered, rows, features=features)
    print("\nRegistrados: " + ", ".join(f"{name} v{version}" for name, version in registered.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
