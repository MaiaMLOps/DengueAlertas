# Cuadernos

El análisis, el entrenamiento y la evaluación del proyecto **no se hicieron en cuadernos**, sino con módulos y scripts reproducibles: cada salida corresponde a un commit, ninguna orden sobrescribe una salida existente, la barrera del periodo de prueba (`src/forecasting/guard.py`) impide calcular métricas de 2023–2025 sin la receta congelada, y las cifras de aceptación se comprueban con pruebas automáticas (README raíz, §6). Los cuadernos de esta carpeta son una **vía de lectura**: muestran el canal y la variable objetivo, los resultados versionados y el uso del modelo final, sin producir resultados nuevos.

## Etapas, código y documentación

| Etapa | Módulos o scripts | Documentación | Salidas y evidencia |
|---|---|---|---|
| Datos | `src/data/` (`python -m src.data`) | [`docs/data_pipeline.md`](../docs/data_pipeline.md), [`docs/data_dictionary.md`](../docs/data_dictionary.md) | `data/processed/master_weekly_screening.parquet` (DVC); `reports/data_quality/` |
| Selección de pilotos | `src/pilot_selection/` (`python -m src.pilot_selection`; `src.pilot_selection.subset` para el panel de los seis pilotos) | [`docs/pilot_selection.md`](../docs/pilot_selection.md) | `data/processed/pilot_municipalities_weekly.parquet` (DVC); `reports/pilot_selection/evidence/` |
| Canal y etiqueta | `src/epidemiology/channel_epico.py` (fórmula de epiCo), `src/forecasting/build_channel.py` (F1, producto del canal con `target_u1e`), `scripts/describe_channel_epico.py`. Legado congelado: `src/epidemiology/ewt.py` y `labels.py` (etiqueta EWT, que el pronóstico no usa) | [`docs/forecasting/contrato_predictivo.md`](../docs/forecasting/contrato_predictivo.md) §1–§2; [`docs/epidemiology_labels.md`](../docs/epidemiology_labels.md) (legado) | `data/processed/epidemiology/pilot_weekly_channel.parquet` (DVC); `reports/epidemiology/evidence/` |
| Ejemplos, cortes, líneas base y eventos (F2–F4) | `src/forecasting/dataset.py`, `splits.py`, `baselines.py`, `metrics.py`, `comparison.py`, `events.py`, `guard.py` | Contrato §3–§5 y §7; [`docs/forecasting/especificacion_implementacion.md`](../docs/forecasting/especificacion_implementacion.md) §6 | Funciones usadas por las fases siguientes; sin salidas propias |
| Búsqueda y selección (F5–F6) | `src/forecasting/models.py`, `search.py`, `selection.py`, `run_selection.py`, `diagnostics.py`; pre-registro en `configs/forecasting.yaml` (`busqueda`) | Contrato §6; [`docs/forecasting/registro_decisiones.md`](../docs/forecasting/registro_decisiones.md), [`docs/forecasting/desviaciones.md`](../docs/forecasting/desviaciones.md) | `data/processed/forecasting/` y `sin_clima/` (DVC); `configs/receta_congelada.yaml`; `reports/forecasting/evidence/seleccion_*`, `diagnostico_desarrollo_*` |
| Prueba única (F7) y reportes (U7.5) | `src/forecasting/evaluate.py`, `artifacts.py`, `reports.py`, `figures_reports.py`; verificación independiente en `docs/forecasting/verificacion/` | Contrato §8; [`docs/forecasting/analisis_resultados.md`](../docs/forecasting/analisis_resultados.md) | `data/processed/forecasting/prueba/` (DVC, incluye los modelos finales copiados en [`modelos/`](../modelos/README.md)); `reports/forecasting/evidence/prueba_20261112/`, `reportes_20261112/`; `reports/forecasting/bitacora_prueba.jsonl` |
| Análisis complementario | `src/forecasting/complementary.py`; `scripts/analisis_b01.py`, `analisis_continuidad.py`, `analisis_m0.py`, `analisis_regularizacion.py`; contraste de familias con `scripts/run_contrast_search.py`, `run_seq_dl.py`, `run_horizon_sensitivity.py`, `run_horizon_climate_mlflow.py`, `log_seq_dl_mlflow.py` y `build_thesis_report.py`, configurado en `configs/contraste_familias.yaml` | [`docs/forecasting/analisis_complementario.md`](../docs/forecasting/analisis_complementario.md), [`reports/tesis_contraste_modelos/README.md`](../reports/tesis_contraste_modelos/README.md) | `reports/forecasting/evidence/complementario_20261120/`, `continuidad_20261120/`, `m0_posicion_20261120/`, `regularizacion_c_ampliada_20261120/`, `reproduccion_r0_20261120/`; `reports/tesis_contraste_modelos/` |

El estado de cada carpeta de datos y de evidencia (vigente, registro o sustituida) está en [`reports/forecasting/INDICE.md`](../reports/forecasting/INDICE.md).

## Cuadernos

| Cuaderno | Contenido | Datos que necesita |
|---|---|---|
| [`01_canal_y_variable_objetivo.ipynb`](01_canal_y_variable_objetivo.ipynb) | El canal de epiCo y sus zonas para un municipio (Neiva, 2019–2025) y la prevalencia de la zona epidémica por municipio y partición | Producto del canal (`dvc pull`); datos agregados municipio-semana |
| [`02_resultados_principales.ipynb`](02_resultados_principales.ipynb) | Tablas de la evaluación preregistrada en prueba frente a B2 y B1, con sus niveles; eventos de la regla preregistrada, la regla del producto (B01) y el canal; continuidad; M0; sensibilidad a la regularización. Cada tabla cita su archivo de origen | Solo evidencia versionada en git (`reports/forecasting/evidence/`) |
| [`03_uso_del_modelo.ipynb`](03_uso_del_modelo.ipynb) | Carga de un modelo de `modelos/`, variables de la última emisión disponible (2025-12-22) con `build_features`, probabilidad y estado de alerta del producto con `events.product_alert_states`, y lectura del resultado | Producto del canal (`dvc pull`) y `modelos/` |

Los cuadernos se versionan **ejecutados de arriba abajo, con sus salidas guardadas**, para leerlos en GitHub sin ejecutarlos. No escriben en `data/` ni en `reports/`, no usan MLflow y resuelven las rutas desde la raíz del repositorio, tanto si se abren desde la raíz como desde `notebooks/`.

## Ejecución

Requiere las dependencias de [`requirements/notebooks.txt`](../requirements/notebooks.txt) en el `.venv` del repositorio:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements/notebooks.txt
```

El kernel se registra dentro del `.venv`, para que los cuadernos se ejecuten con su intérprete y no con otro Python del sistema:

```powershell
.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (ipykernel)"
```

Para ejecutar un cuaderno y guardar sus salidas en el mismo archivo, desde la raíz del repositorio (`JUPYTER_PATH` da prioridad al kernel del `.venv` sobre otro kernel `python3` instalado; `PYTHONUTF8` evita que Windows lea el cuaderno con la codificación cp1252):

```powershell
$env:JUPYTER_PATH = "$PWD\.venv\share\jupyter"; $env:PYTHONUTF8 = "1"; .venv\Scripts\jupyter-execute.exe --inplace notebooks\01_canal_y_variable_objetivo.ipynb
```

En Linux o macOS, la ruta del intérprete es `.venv/bin/python` y las variables se pasan como prefijo de la orden (`JUPYTER_PATH=.venv/share/jupyter .venv/bin/jupyter-execute --inplace …`). Una nueva ejecución solo cambia los metadatos de tiempo de las celdas mientras los datos y el código no cambien.
