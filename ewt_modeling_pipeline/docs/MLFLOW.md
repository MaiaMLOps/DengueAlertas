# Registro en MLflow sin Staging

## Objetivo

Registrar trazabilidad experimental y versiones ganadoras en el MLflow del equipo, sin promoverlas a ninguna etapa y sin crear alias de ciclo de vida.

## Variables de entorno

```bash
MLFLOW_TRACKING_URI=https://platform.arkoiot.com
MLFLOW_EXPERIMENT_NAME=dengue-ews-ewt-h8-final
CF_ACCESS_CLIENT_ID=<secreto>
CF_ACCESS_CLIENT_SECRET=<secreto>
```

Use un `.env` local o el gestor de secretos de CI. No confirme valores reales en Git, ZIP, logs, capturas o reportes.

## Qué se registra

### Runs candidatos

Uno por combinación tarea–municipio–algoritmo–conjunto de variables:

- tags: tarea, DIVIPOLA, versión metodológica, `lifecycle=candidate`;
- parámetros: algoritmo, feature set, horizonte, purga y calibración;
- métricas de validación rolling-origin.

### Runs ganadores

Uno por combinación evaluable tarea–municipio:

- parámetros del ganador y umbral;
- métricas de prueba bloqueada;
- bundle pyfunc;
- predicciones, importancias, folds, métricas y candidatos como evidencia;
- nombre registrado `DengueAlertas_EWT_<tarea>_<DIVIPOLA>_h8`.

## Política de ciclo de vida

El código pasa `registered_model_name` al log del modelo, pero no llama APIs de transición de etapa ni de alias. Por eso una versión nueva queda en etapa `None`. Además, la carga de configuración falla si `transition_stage` o `set_alias` son distintos de `null`.

## Comprobación

Tras la ejecución, revise `outputs/final_run/mlflow_status.json`. Si el estado es `published`, contiene los run IDs, nombres registrados y URI. En la interfaz de MLflow verifique que cada versión nueva figure con etapa `None`.

Estados locales posibles:

- `published`: publicación completa;
- `skipped_dependency`: falta instalar MLflow;
- `skipped_credentials`: faltan credenciales Cloudflare;
- `failed`: publicación parcial o error remoto; los resultados locales siguen intactos.

## Reintento sin reentrenar

La función `publish_to_mlflow` consume los artefactos existentes. Puede invocarse desde Python después de instalar MLflow y exportar credenciales:

```bash
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from ewt_final.pipeline import load_config
from ewt_final.mlflow_io import publish_to_mlflow
print(publish_to_mlflow(Path('outputs/final_run'), load_config('configs/pipeline.yaml')))
PY
```
