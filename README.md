# DengueAlertas

Alerta temprana de excesos en canales endémicos de dengue en Colombia mediante *machine learning*. Prototipo académico retrospectivo de la Maestría en Inteligencia Artificial de la Universidad de los Andes.

## 1. Problema

Las secretarías de salud vigilan el dengue con el **canal endémico**: una banda construida con los conteos de los años anteriores para cada semana del año. Cuando los casos de una semana superan el límite superior del canal, el municipio está en **zona epidémica**. El canal describe la semana que ya ocurrió; no dice qué va a pasar en las semanas siguientes.

La pregunta del proyecto es: **¿qué tan probable es que un municipio esté en zona epidémica dentro de 8 semanas?**, y si esa probabilidad, convertida en alertas, avisa los inicios de brote mejor o antes que la alerta descriptiva del propio canal.

## 2. Solución

- **Datos.** Registros confirmados del evento 210 (dengue) de SIVIGILA por municipio de ocurrencia y semana de inicio de síntomas (lunes a domingo), con covariables climáticas semanales, 2007–2025. Seis municipios piloto: Barranquilla `08001`, Soledad `08758`, Neiva `41001`, Villavicencio `50001`, Cúcuta `54001` y Cali `76001`.
- **Variable objetivo.** `target_u1e` = 1 si los casos de la semana superan el límite superior U1e del canal endémico calculado como en epiCo (cinco años previos, media geométrica, intervalo t). El modelo predice ese estado 8 semanas después de la última semana con datos completos.
- **Modelo (receta congelada).** Una regresión logística con penalización L2 por municipio, entrenada con seis variables sin clima: la zona actual del canal (`zona_t`), la posición reciente frente al umbral (`posicion_4s`), el crecimiento del último mes (`crecimiento_4s`), el cambio del umbral hasta la semana objetivo (`cambio_umbral`) y la época del año de la semana objetivo (`sem_sin`, `sem_cos`). Cada semana emite una probabilidad de zona epidémica en t + 8. Se compara con dos líneas base: B2 (probabilidad según la zona actual del canal) y B1 (estacionalidad de la semana objetivo). Familia, hiperparámetros, variables, τ, k y W están en [`configs/receta_congelada.yaml`](configs/receta_congelada.yaml).
- **Regla de la evaluación preregistrada.** Una emisión es positiva si p̂ ≥ τ, y k = 2 emisiones positivas consecutivas confirman una alerta, que se compara con la alerta del canal (contrato §7.4). Es la regla de la selección y de la prueba única, y la única que sostiene las conclusiones del contrato; queda fija en `decision.regla_emision` de `configs/forecasting.yaml`.
- **Regla de alerta del producto (enmienda D10.1).** Es la que verá el usuario en la API y el frontend: una emisión es positiva si p̂ ≥ τ **y** la zona observada en t no es epidémica, de modo que el modelo no emite alertas en zona epidémica. Cada emisión toma uno de cuatro estados: `sin_alerta`, `senal_en_observacion`, `alerta_inicio_confirmada` (k emisiones positivas consecutivas) y `continuidad_epidemica` (el municipio ya está en zona epidémica). La probabilidad es siempre la misma p̂, pero se etiqueta según el estado: probabilidad de **seguir** en zona epidémica en `continuidad_epidemica` y de **entrar** en ella en los demás. Se fija en `producto.regla_alerta`, usa el τ y el k de la receta y no cambia la receta ni la evaluación preregistrada (DV-07; esquema en [`docs/forecasting/api_salida_d10.md`](docs/forecasting/api_salida_d10.md)).
- **Validación.** Origen móvil con reentrenamiento anual: desarrollo 2019–2022 para elegir; prueba 2023–2025 evaluada una sola vez, después de congelar la receta.
- **Análisis complementario.** Posterior a la prueba y descriptivo: contraste de familias de modelos en desarrollo, modelo mínimo de posición (M0), sensibilidad a la regularización, regla de alerta del producto (B01) y calidad de la probabilidad en continuidad epidémica. Está en [`docs/forecasting/analisis_complementario.md`](docs/forecasting/analisis_complementario.md); no modifica la receta ni las conclusiones preregistradas.
- **Entrega.** Modelos finales en [`modelos/`](modelos/README.md), cuadernos de lectura en [`notebooks/`](notebooks/README.md) y, pendientes de F8, la API FastAPI y el frontend React con cuatro vistas: pronóstico vigente, historial fuera de muestra, carga de datos y desempeño en prueba.

La metodología completa, sus decisiones (D0–D10) y la secuencia de fases están en [`docs/forecasting/README.md`](docs/forecasting/README.md). Este README no reporta resultados: están en la evidencia indexada en [`reports/forecasting/INDICE.md`](reports/forecasting/INDICE.md). El análisis integrado de los resultados está en [`docs/forecasting/analisis_resultados.md`](docs/forecasting/analisis_resultados.md), con versiones en Word y presentación en [`reports/forecasting/entregables/`](reports/forecasting/entregables/).

## 3. Estructura del repositorio

```
configs/                  Parámetros del pronóstico (forecasting.yaml), receta congelada (receta_congelada.yaml)
                          y contraste de familias del análisis complementario (contraste_familias.yaml)
data/processed/           Descriptores .dvc y manifiestos; el contenido está en el remoto DVC
docs/                     Documentación de datos, selección de pilotos y etiquetas epidemiológicas
docs/forecasting/         Contrato predictivo, especificación, decisiones, desviaciones, API, análisis de resultados,
                          análisis complementario (analisis_complementario.md) y scripts de verificación
frontend/                 Aplicación React/Vite (Cloudflare Pages)
modelos/                  Modelos finales: seis .joblib y su manifiesto, copia de la salida DVC de la prueba (README propio)
notebooks/                Cuadernos de lectura con salidas guardadas: canal, resultados y uso del modelo (README propio)
reports/                  Evidencia versionada: selección de pilotos, etiquetas y pronóstico (tablas, figuras, bitácora, entregables)
reports/tesis_contraste_modelos/   Tablas y figuras del contraste de familias (desarrollo 2019–2022)
requirements/             Dependencias por etapa; opcionales: contraste.txt (contraste de familias) y notebooks.txt (cuadernos)
scripts/                  Utilidades (descripción del canal, comprobación de MLflow en solo lectura) y scripts del análisis
                          complementario: analisis_b01.py, analisis_continuidad.py, analisis_m0.py, analisis_regularizacion.py
                          y el contraste de familias (run_contrast_search.py, run_seq_dl.py, run_horizon_sensitivity.py,
                          run_horizon_climate_mlflow.py, log_seq_dl_mlflow.py, build_thesis_report.py); registro de
                          los modelos finales en MLflow (registrar_modelos_finales.py)
src/data/                 Integración de SIVIGILA y covariables (panel maestro)
src/pilot_selection/      Selección descriptiva de los seis pilotos y derivado territorial
src/epidemiology/         Canal de epiCo (channel_epico.py) y etiquetas EWT de legado
src/forecasting/          Pipeline de pronóstico: ejemplos, cortes, líneas base, modelos, búsqueda, selección, prueba
src/api.py                API FastAPI de inferencia (se adapta en F8)
src/webhook_api.py, src/resource_monitor.py   Servicios auxiliares de la infraestructura del equipo
tests/                    Pruebas: unitarias por paquete y de aceptación por fase (tests/acceptance/)
.github/workflows/        CI y despliegue del frontend
Dockerfile                Imagen de la API
```

## 4. Dependencias y entorno

- **Python 3.11** (el mismo del `Dockerfile`).
- Entorno virtual en `.venv/` en la raíz del repositorio. Los comandos de este README usan la ruta de Windows `.venv/Scripts/python.exe`; en Linux o macOS es `.venv/bin/python`.
- Archivos de dependencias:

  | Archivo | Uso |
  |---|---|
  | `requirements/data.txt` | Integración de datos |
  | `requirements/pilot-selection.txt` | Selección de pilotos y figuras; incluye `data.txt` |
  | `requirements/forecasting.txt` | Modelado (scikit-learn, LightGBM, SciPy, `mlflow-skinny`, Pydantic); incluye `pilot-selection.txt` |
  | `requirements/forecasting.lock.txt` | Entorno completo verificado del modelado, con DVC (`pip freeze`) |
  | `requirements/data-test.txt` | Pruebas de la integración de datos |
  | `requirements/serve.txt` | Dependencias de la API |
  | `requirements/contraste.txt` | Opcional: familias del contraste (CatBoost, XGBoost, PyTorch); incluye `forecasting.txt` |
  | `requirements/notebooks.txt` | Opcional: ejecución de los cuadernos (nbformat, nbclient, ipykernel); incluye `forecasting.txt` |

Instalación del entorno de modelado:

```bash
py -3.11 -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements/forecasting.lock.txt
```

`requirements/forecasting.lock.txt` reproduce las versiones exactas con las que se generaron los resultados; `requirements/forecasting.txt` fija solo las directas.

**Dependencias opcionales.** El pipeline principal, la API y las pruebas no las necesitan; se instalan en el mismo `.venv`, después del entorno de modelado.

- Contraste de familias del análisis complementario (`configs/contraste_familias.yaml` y sus scripts). Solo las familias secuenciales (GRU y CNN-1D) usan PyTorch; para usarlo con GPU, el archivo indica instalar la rueda de CUDA desde el índice del sitio oficial de PyTorch, y sin GPU basta la de CPU:

  ```bash
  .venv/Scripts/python.exe -m pip install -r requirements/contraste.txt
  ```

- Cuadernos de `notebooks/`. La ejecución con salidas guardadas, incluido el registro del kernel del `.venv`, se describe en [`notebooks/README.md`](notebooks/README.md):

  ```bash
  .venv/Scripts/python.exe -m pip install -r requirements/notebooks.txt
  ```

## 5. Datos con DVC

Los Parquet y los artefactos del pronóstico no están en git: git guarda sus descriptores `.dvc` y DVC guarda el contenido en el remoto `vps-storage` (WebDAV, configurado en `.dvc/config`).

El remoto exige credenciales, que entrega el **responsable de infraestructura y API**. Se configuran solo en el archivo local `.dvc/config.local`, que git no versiona, y nunca se escriben en código, mensajes ni commits:

```bash
.venv/Scripts/dvc.exe remote modify --local vps-storage user <usuario>
```

```bash
.venv/Scripts/dvc.exe remote modify --local vps-storage ask_password true
```

Con `ask_password`, DVC pide la contraseña por teclado, así que los comandos contra el remoto se ejecutan en una terminal interactiva. Para descargar los datos:

```bash
.venv/Scripts/dvc.exe pull
```

Productos principales (cada uno con su `.dvc` en git):

| Ruta | Contenido |
|---|---|
| `data/processed/master_weekly_screening.parquet` | Panel maestro: 78 municipios × 992 semanas |
| `data/processed/pilot_municipalities_weekly.parquet` | Panel de los seis pilotos (5.952 × 15), insumo del canal |
| `data/processed/epidemiology/pilot_weekly_channel.parquet` | Producto del canal de epiCo con `target_u1e` |
| `data/processed/epidemiology/pilot_weekly_labels.parquet` | Etiquetas EWT de legado (congeladas, no las usa el pronóstico) |
| `data/processed/forecasting/` | Búsquedas de desarrollo (raíz, `sin_clima/`, `ablacion_clima/`) y salidas de la prueba única (`prueba/`) |

Los datos originales de SIVIGILA y las covariables no están en el repositorio ni en DVC; ver [`docs/data_pipeline.md`](docs/data_pipeline.md). El estado de cada carpeta (vigente, registro o sustituida) está en [`reports/forecasting/INDICE.md`](reports/forecasting/INDICE.md).

## 6. Reproducción del pipeline de modelado

Todos los comandos se ejecutan desde la raíz del repositorio. Principios que el código hace cumplir:

- **Nunca sobrescribe.** Cada comando se niega a escribir sobre una salida existente. Para reproducir, se escribe en una carpeta nueva (`--salida`; `--output` en `build_channel`; en `artifacts`, una copia de `prueba/` sin la ficha, el historial ni `modelos_finales/`, pasada con `--prueba`) y se compara con la versionada (md5 del `.dvc` o bytes de la evidencia; los JSON pueden diferir solo en el fin de línea entre Windows y Linux).
- **Árbol limpio.** La búsqueda, la selección, la prueba, los artefactos y los reportes se niegan a correr con cambios sin commit en `src/` o `configs/`, para que cada salida corresponda a un commit. La búsqueda, la selección, los artefactos y los reportes admiten `--permitir-cambios` para corridas de humo; la prueba no lo admite, y solo el manifiesto de los reportes lo registra (`cambios_sin_commit`). El diagnóstico (`diagnostics`) no comprueba el árbol y `build_channel` lo registra como `src_dirty` en su manifiesto.
- **Barrera de prueba.** Ninguna métrica sobre semanas objetivo de 2023–2025 se calcula sin una receta congelada válida (`src/forecasting/guard.py`).
- **Seguimiento.** La búsqueda y la prueba registran runs en MLflow según `seguimiento.modo` de `configs/forecasting.yaml` (modo `servidor` con credenciales de Cloudflare Access en `.env`, o `local` en `mlruns/`).

| Fase | Qué produce | Comando |
|---|---|---|
| F1 | Producto del canal de epiCo | `.venv/Scripts/python.exe -m src.forecasting.build_channel --output <nuevo.parquet>` |
| F2–F4 | Ejemplos, cortes, líneas base y eventos: funciones que llaman las fases siguientes; no tienen línea de comandos | — |
| F5b | Búsqueda de desarrollo, conjunto completo | `.venv/Scripts/python.exe -B -X utf8 -m src.forecasting.search --salida <carpeta nueva>` |
| F6b | Búsqueda de desarrollo, conjunto sin clima (D7.2) | `.venv/Scripts/python.exe -B -X utf8 -m src.forecasting.search --variante sin_clima --salida <carpeta nueva>` |
| F6 / F6b | Selección y borrador de receta, sin reentrenar | `.venv/Scripts/python.exe -B -X utf8 -m src.forecasting.run_selection --busqueda data/processed/forecasting --busqueda-sin-clima data/processed/forecasting/sin_clima --salida <carpeta nueva>` |
| F7 | Diagnóstico de desarrollo (contrato §8.8) | `.venv/Scripts/python.exe -B -X utf8 -m src.forecasting.diagnostics --busqueda data/processed/forecasting/sin_clima --salida <carpeta nueva>` |
| F7 | **Prueba única** 2023–2025 | `.venv/Scripts/python.exe -B -X utf8 -m src.forecasting.evaluate --salida <carpeta>` — ya se ejecutó; ver la advertencia |
| F7 | Ficha de desempeño, historial y modelos finales | `.venv/Scripts/python.exe -B -X utf8 -m src.forecasting.artifacts --prueba <carpeta con las salidas de la prueba>` |
| U7.5 | Reportes derivables y figuras | `.venv/Scripts/python.exe -B -X utf8 -m src.forecasting.reports --salida <carpeta nueva> --figuras <carpeta nueva>` |

> **Advertencia sobre la prueba única.** `src.forecasting.evaluate` ya se ejecutó una vez y está registrada en `reports/forecasting/bitacora_prueba.jsonl`. Volver a ejecutarla exige `--razon` y añade una entrada a la bitácora; no se hace sin una decisión del equipo. Sus salidas se obtienen con `dvc pull` (`data/processed/forecasting/prueba/`).

La selección, el diagnóstico, los artefactos y los reportes se pueden recalcular desde las salidas guardadas, sin repetir la búsqueda ni la prueba. Los pasos previos al pronóstico (panel maestro, pilotos y etiquetas de legado) están en [`docs/data_pipeline.md`](docs/data_pipeline.md), [`docs/pilot_selection.md`](docs/pilot_selection.md) y [`docs/epidemiology_labels.md`](docs/epidemiology_labels.md).

**Pruebas.**

```bash
.venv/Scripts/python.exe -B -m pytest tests/ -q -p no:cacheprovider
```

Las pruebas de aceptación con datos reales necesitan `dvc pull` y se ejecutan pasando la carpeta y las rutas de los datos:

```bash
.venv/Scripts/python.exe -B -m pytest tests/acceptance -q -rA -p no:cacheprovider --panel-parquet data/processed/pilot_municipalities_weekly.parquet --channel-parquet data/processed/epidemiology/pilot_weekly_channel.parquet --legacy-labels-parquet data/processed/epidemiology/pilot_weekly_labels.parquet --search-dir data/processed/forecasting --sin-clima-dir data/processed/forecasting/sin_clima --prueba-dir data/processed/forecasting/prueba
```

Sin las opciones, las pruebas con datos se omiten; con una opción y sin el archivo, fallan. Las pruebas `*_real_*` de F6 y F7 ya no se deseleccionan con `-k` (sus docstrings congelados lo piden solo hasta U6.3 y U7.3, ya cerradas): se ejecutan con el comando anterior.

## 7. El modelo

- **Familia.** Regresión logística con penalización L2, un modelo por municipio, entrenada con el conjunto de variables *sin clima* (estado y dinámica frente al canal y semana del año). Familia, hiperparámetros, conjunto de variables, τ, k y W están en [`configs/receta_congelada.yaml`](configs/receta_congelada.yaml), con su hash SHA-256 y su aprobación. Cómo se eligió: [`docs/forecasting/README.md`](docs/forecasting/README.md).
- **Tamaño.** Seis archivos `joblib` de unos 4 KB cada uno, más un manifiesto: unos 28 KB en total.
- **Dónde están.**
  - **DVC (fuente de verdad):** `data/processed/forecasting/prueba/modelos_finales/` (`<código>.joblib` y `manifiesto.json`, con la familia, el candidato, el conjunto de variables, las filas y el hash de la receta). Los modelos son salidas del pipeline y se versionan con DVC junto a las salidas de la prueba de la que dependen (especificación §4, «Ubicación de resultados»). Copia del manifiesto en git: `reports/forecasting/evidence/prueba_20261112/manifiesto_modelos_finales.json`.
  - **`modelos/` (copia de entrega):** los seis `.joblib` y el manifiesto, idénticos byte a byte a la salida DVC, para consultarlos sin acceso al remoto. Su [`README`](modelos/README.md) describe cada archivo (municipio, tamaño y md5), la entrada que esperan (las variables que construye `src.forecasting.dataset.build_features`), un ejemplo de carga y predicción y su relación con DVC y MLflow. `tests/forecasting/test_modelos_entrega.py` comprueba que la copia coincide con la salida DVC cuando está disponible y que el manifiesto corresponde a la receta congelada. El uso completo, con el estado de alerta del producto, está en [`notebooks/03_uso_del_modelo.ipynb`](notebooks/03_uso_del_modelo.ipynb).
  - **MLflow:** la prueba única quedó registrada en el experimento `dengue_h8_prueba` del servidor del equipo; la búsqueda de desarrollo, en `dengue_h8_desarrollo`. Los seis modelos finales están en el registro de modelos, que es lo que leerá la API, como `DengueAlertas_Barranquilla`, `DengueAlertas_Soledad`, `DengueAlertas_Neiva`, `DengueAlertas_Villavicencio`, `DengueAlertas_Cucuta` y `DengueAlertas_Cali`, **todos en versión 1** (2026-11-20).
    - **Ruta de registro.** El servidor no admite escrituras en el registro con el token del equipo, así que el registro sigue tres pasos (DV-08):
      1. `scripts/registrar_modelos_finales.py --solo-runs` crea un run por municipio en el experimento `dengue_h8_final`, con parámetros, etiquetas, el `.joblib` original y el modelo guardado con firma;
      2. cada modelo se registra manualmente desde la interfaz de MLflow;
      3. `--verificar`, de solo lectura, comprueba la versión, el run (por id y md5) y que las probabilidades cargadas desde `models:/` son idénticas a las del `.joblib`.

      Los run id y el detalle están en [`modelos/README.md`](modelos/README.md), «Registro en MLflow».
    - **Registrados sin habilitar.** Están sin la etiqueta `enabled` y sin alias, para que la API anterior, que descubre solo los modelos con `enabled = true`, no los cargue antes de F8. La habilitación y los alias forman parte del despliegue de F8: el webhook (`src/webhook_api.py`) promueve la versión a la etapa del alias y sincroniza la API correspondiente.
- **Habilitación en F8.** El responsable del despliegue pone en cada modelo registrado estas etiquetas, que lee `src/api.py`:

  | Modelo registrado | `enabled` | `municipality_code` | `department` | `depto_id` |
  |---|---|---|---|---|
  | `DengueAlertas_Barranquilla` | `true` | `08001` | `Atlántico` | `COATL` |
  | `DengueAlertas_Soledad` | `true` | `08758` | `Atlántico` | `COATL` |
  | `DengueAlertas_Neiva` | `true` | `41001` | `Huila` | `COHUI` |
  | `DengueAlertas_Villavicencio` | `true` | `50001` | `Meta` | `COMET` |
  | `DengueAlertas_Cucuta` | `true` | `54001` | `Norte de Santander` | `CONSA` |
  | `DengueAlertas_Cali` | `true` | `76001` | `Valle del Cauca` | `COVAC` |

  Luego sigue este orden:
  1. alias `staging` en la versión 1;
  2. prueba en el ambiente de pruebas;
  3. alias `production`.

  `src/api.py` carga cada modelo por el alias igual a `STAGE`. Si no lo encuentra, recurre a la etapa clásica y después a la última versión. Por eso, con `enabled = true`, la API de producción cargaría la versión 1 aunque aún no tenga el alias `production`; F8 decide cómo tratar ese último recurso antes de habilitar (detalle en [`modelos/README.md`](modelos/README.md)).
- **Dependencias de la API (nota para F8).** `requirements/serve.txt` debe fijar `scikit-learn==1.7.2`, la versión con la que se serializaron los modelos, y `mlflow==3.17.0`, la versión del servidor. Hoy fija `scikit-learn>=1.3.0` y `mlflow>=2.0.0`. Este cambio y el del `Dockerfile` corresponden al responsable del despliegue.
- **Requerimientos de hardware.** Solo CPU; ni el entrenamiento ni la inferencia de la receta usan GPU. Cada modelo es una regresión logística sobre seis variables, y los seis ocupan unos 28 KB en disco. La inferencia requiere scikit-learn 1.7.2, la versión con la que se serializaron (`requirements/forecasting.txt`). Solo las familias secuenciales del contraste de familias (GRU y CNN-1D, fuera de la receta) se entrenaron con PyTorch y admiten GPU.
- **Tiempos de inferencia.** Medición local del 2026-11-20 en un portátil con Intel Core i7-12650H (10 núcleos, 16 hilos), 64 GB de RAM y Windows 11, con Python 3.11.0 y scikit-learn 1.7.2, en un proceso de Python sin MLflow ni API. No es una medición de la API desplegada, que añadirá la carga desde el registro de MLflow, la red y la serialización de la respuesta.

  | Operación | Tiempo |
  |---|---|
  | Carga de los seis modelos desde `modelos/` (`joblib.load`) | 7 ms la primera vez; mediana de 6,5 ms en 50 repeticiones |
  | Predicción de una emisión para un municipio (`predict_positive`, una fila) | Mediana de 2,5 ms en 1.000 repeticiones (percentil 95: 3,2 ms) |
  | Predicción de una emisión para los seis municipios | Mediana de 19 ms en 200 repeticiones |
  | Construcción de las variables de una emisión (`build_features` sobre la base completa) | Mediana de 0,10 s en 5 repeticiones |

  La importación de pandas, scikit-learn y joblib al iniciar el proceso tomó 1,3 s y ocurre una sola vez.

## 8. Parametrización

| Dónde | Qué fija |
|---|---|
| `configs/forecasting.yaml` | Rutas de datos y versión DVC del canal; parámetros del canal; horizonte h; ventanas de las variables; años de desarrollo y prueba; líneas base; recorte del log-loss; α de las comparaciones; k, W y rejilla de τ; pre-registro de la búsqueda (candidatos y semillas); seguimiento en MLflow. Se valida con Pydantic al cargarse (`src/forecasting/config.py`) |
| `configs/forecasting.yaml`, `decision.regla_emision` | Regla de emisión de la selección y la evaluación preregistrada: `umbral` (p̂ ≥ τ, contrato §7.4). Es fija: la validación de la configuración no admite otro valor |
| `configs/forecasting.yaml`, `producto.regla_alerta` | Regla de alerta del producto (enmienda D10.1, DV-07): `umbral_fuera_de_epidemia` (p̂ ≥ τ y zona observada en t distinta de epidemia), con τ y k de la receta. La lee la API (F8), que calcula el estado con `events.product_alert_states`; la evaluación nunca la usa |
| `configs/receta_congelada.yaml` | Receta elegida en desarrollo y aprobada antes de la prueba. No se edita: cualquier cambio invalida su hash y la barrera de prueba la rechaza |
| `configs/contraste_familias.yaml` | Contraste de familias del análisis complementario (solo desarrollo 2019–2022): corridas, espacios de búsqueda, semillas y sensibilidad al horizonte. Se valida con `load_contrast_config` (`src/forecasting/config.py`); no participa en la receta ni en la prueba |

Variables de entorno existentes:

| Variable | Uso |
|---|---|
| `FORECAST_HORIZON` | Sobrescribe h del pipeline (por defecto 8); debe ser 2, 4, 8 o 12 (`horizonte.permitidos`), y solo h = 8 tiene receta y modelos |
| `MLFLOW_TRACKING_URI` | Servidor MLflow (por defecto, el de `configs/forecasting.yaml`) |
| `CF_ACCESS_CLIENT_ID`, `CF_ACCESS_CLIENT_SECRET` | Credenciales de Cloudflare Access para MLflow. Van en `.env` (plantilla: `.env.example`), que git ignora; las entrega el responsable de infraestructura y API |
| `STAGE`, `MODEL_NAME`, `MODEL_THRESHOLD` | Configuración de la API actual (`src/api.py`, `Dockerfile`); se revisan en F8 |

Las variables nuevas de la API (`ALERT_K`, `DATA_VERSION` y la URL de la API en el frontend) se definen en F8; su diseño está en la especificación §10.

## 9. Despliegue

Estado al 2026-11-23: F8 está desplegado de forma parcial. La API corre en una VM, el frontend se publica en Cloudflare Pages y el panel se materializa en el servidor. El contrato predictivo (§9.6) y [`docs/forecasting/api_salida_d10.md`](docs/forecasting/api_salida_d10.md) siguen siendo la referencia de la salida; las secciones a continuación distinguen lo que ya existe de lo que queda pendiente.

Estrategia contractual (no todo implementado): modelos pequeños cargados al arrancar; inferencia en CPU con τ y k congelados de la receta; `/predict` para el pronóstico vigente; y vistas precalculadas `/historial` y `/desempeno`. Hoy están operativos la carga de modelos y `/predict` v0.2; `/predict/upload`, `/historial` y `/desempeno` siguen pendientes (§9.8).

### 9.1 Ambientes y arranque de la API

Servidor: `ubuntu-mlops-01` (`116.203.114.120`). Código de la API en `/opt/dengue-api`, entorno Python en `/opt/mlflow-env`. **Docker no se usa**: el comando `docker` no está instalado en la VM; el `Dockerfile` del repositorio no es la vía de despliegue actual.

| Ambiente | Unidad systemd | Puerto local | Proceso |
|---|---|---|---|
| Producción | `dengue-api-prod.service` | `127.0.0.1:8000` | `uvicorn src.api:app --host 127.0.0.1 --port 8000` |
| Staging | `dengue-api-staging.service` | `127.0.0.1:8001` | `uvicorn src.api:app --host 127.0.0.1 --port 8001` |

Ambas unidades tienen `User=admin`, `WorkingDirectory=/opt/dengue-api`, `EnvironmentFile=/opt/dengue-api/.env` y `Restart=always`. Se arrancan o reinician con `sudo systemctl restart dengue-api-prod` / `dengue-api-staging`; el estado se revisa con `systemctl status <unidad>` y `journalctl -u <unidad>`.

NGINX (`/etc/nginx/sites-enabled/dengue-api`) escucha en `127.0.0.1:8080`, con `server_name api.arkoiot.com`, y enruta según la cabecera `Origin`: orígenes de Producción (`https://demo.arkoiot.com`, `https://denguealertas.pages.dev`, localhost) van al puerto 8000; `https://staging.denguealertas.pages.dev` y los preview `*.denguealertas.pages.dev` van al 8001; cualquier otro origen recibe 403. El mismo servidor publica `/webhook/` (puerto 8005), `/dvc/` (WebDAV) y, en otro virtual host, `platform.arkoiot.com` → MLflow (5000). El túnel `cloudflared.service` está activo; su configuración de ingress no está versionada en este repositorio.

Servicios relacionados en la VM: `mlflow.service` (5000), `dengue-webhook.service` (8005), `nginx.service`, `cloudflared.service` y `dengue-resource-monitor.service` (§9.9).

### 9.2 URLs y acceso

| Ambiente | Frontend | API pública | Upstream |
|---|---|---|---|
| Producción | `https://demo.arkoiot.com` (dominio propio, tras Cloudflare Access) y `https://denguealertas.pages.dev` (Pages) | `https://api.arkoiot.com`, con `Origin: https://demo.arkoiot.com` o `https://denguealertas.pages.dev` | `127.0.0.1:8000` |
| Staging | `https://staging.denguealertas.pages.dev` (tras Cloudflare Access) | `https://api.arkoiot.com`, con `Origin: https://staging.denguealertas.pages.dev` | `127.0.0.1:8001` |
| Desarrollo local | `http://localhost:5173` / `http://127.0.0.1:5173` | `https://api.arkoiot.com` (o el proxy `/api` de `frontend/vite.config.js`) | Según el `Origin`, hoy Producción |

Cloudflare Access protege `demo.arkoiot.com` y `platform.arkoiot.com` (aplicaciones `dengue` y `mlflow`). Para una persona: abrir el host e iniciar sesión con correo `@uniandes.edu.co` desde Colombia. Para un servicio o script: enviar las cabeceras `CF-Access-Client-Id` y `CF-Access-Client-Secret` con un service token; los valores se entregan fuera del repositorio. NGINX reenvía la identidad autenticada en `Cf-Access-Authenticated-User-Email` y la API la registra en su telemetría. El frontend de desarrollo lee esas dos variables del entorno local para el proxy; no las versiona.

### 9.3 Regla de Cloudflare en Producción (`?new=false`)

Regla **prevista, pendiente de aplicar** en el host `demo.arkoiot.com`: cuando llegue una petición cuyo query incluya `new=false`, responder `302` a la misma ruta sin ese parámetro (conservando cualquier otro parámetro). No bloquea ni muestra un error; simplemente devuelve al usuario a `/`.

Motivo: desde DV-11, el código abre el modo v0.2 por defecto en `/`, pero todavía respeta un `?new=false` explícito (útil en Staging y desarrollo). En Producción se quiere que el usuario no pueda volver al modo manual de DV-09 escribiendo la URL. Esa es la diferencia entre el código y lo que verá el usuario de `demo.arkoiot.com` una vez aplicada la regla. Al cierre de este PR la regla está documentada en DV-10, **no creada**: no debe afirmarse que ya está activa en Cloudflare.

### 9.4 Panel y datos en el servidor

- Remoto DVC `vps-storage`: `webdavs://api.arkoiot.com/dvc` (`.dvc/config`). En el servidor se sirve como WebDAV en `/dvc/`, con autenticación basic y `alias /opt/dvc-storage/`.
- Panel versionado: `data/processed/pilot_municipalities_weekly.parquet.dvc`, md5 `82c092dd57f8a316c6992732664d75ce`.
- Copia materializada: `/opt/dengue-api/data/processed/pilot_municipalities_weekly.parquet`, con el mismo md5. El objeto content-addressed está en `/opt/dvc-storage/files/md5/82/c092dd57f8a316c6992732664d75ce`.
- `FORECAST_PANEL_PATH` (`src/forecasting/api_v2.py`) permite sustituir la ruta del panel; si no se define, se usa `datos.panel` de `configs/forecasting.yaml`. La semana emisible se deriva del panel y excluye `partial_week` y `n_days < 7`.
- No hay CLI de DVC instalada en la VM: la materialización observada es copiar el objeto del almacén WebDAV a la ruta indicada y verificar el md5. No se alteran los descriptores `.dvc` versionados.

### 9.5 Variables de entorno

Solo nombres; los valores se entregan por el responsable de infraestructura y viven en archivos ignorados por git.

| Variable | Dónde | Uso |
|---|---|---|
| `STAGE` | API (`systemd`) | `Staging` o `Production`; define el alias de MLflow |
| `MODEL_NAME` | API (`systemd`) | Modelo por defecto/retrocompatible |
| `MLFLOW_TRACKING_URI` | API y webhook | URI del servidor de MLflow |
| `MODEL_THRESHOLD` | API | Umbral por defecto del modo manual |
| `CF_ACCESS_CLIENT_ID`, `CF_ACCESS_CLIENT_SECRET` | API y frontend local | Service token de Cloudflare Access |
| `FORECAST_PANEL_PATH` | API (opcional) | Ruta alternativa del panel |
| `WEBHOOK_SECRET`, `LOG_DIR` | Webhook | Firma HMAC y directorio de logs |
| `CF_ACCOUNT_ID`, `CF_APP_ID`, `CF_API_TOKEN` | `.env` de la API en la VM | Integración de Cloudflare; el código vigente en `src/` no las referencia directamente |
| `VITE_API_URL` | Frontend, al compilar | URL base de la API; el código usa `https://api.arkoiot.com` si no se define |
| `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `GITHUB_TOKEN` | GitHub Actions (secrets) | Despliegue a Pages; solo nombres aquí |

La unidad `mlflow.service` además define, entre otras, `MLFLOW_DISABLE_TELEMETRY`, `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS`, `MLFLOW_SERVER_ENABLE_JOB_EXECUTION`, `MLFLOW_TRACING_ENABLED`, `MLFLOW_WEBHOOK_ALLOWED_SCHEMES`, `MLFLOW_WEBHOOK_ALLOW_PRIVATE_IPS` y `MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY`.

### 9.6 MLflow y promoción de modelos

MLflow corre en `mlflow.service` en `127.0.0.1:5000`, con backend SQLite en `/opt/mlflow/mlflow.db` y artefactos en `/opt/mlflow/artifacts`; la URL pública es `https://platform.arkoiot.com`.

Hoy existen seis modelos registrados `DengueAlertas_Barranquilla`, `DengueAlertas_Cali`, `DengueAlertas_Cucuta`, `DengueAlertas_Neiva`, `DengueAlertas_Soledad` y `DengueAlertas_Villavicencio`. Estado observado en el registro: versión 1, tag `enabled=true` en los seis, alias `staging` → versión 1 en los seis y `current_stage=Staging`. **No hay alias `production` en estos seis modelos**, así que la promoción a Producción sigue pendiente. Por eso `/health` de Producción muestra `model_name=ModeloPrueba` (configuración heredada) aunque los seis modelos municipales ya están cargados para `/predict`.

Promoción: al crear el alias `production` (o `champion`) en MLflow, el webhook `POST /webhook/mlflow` valida la firma HMAC con `WEBHOOK_SECRET`, llama a `transition_model_version_stage(..., stage=Production, archive_existing_versions=True)` y hace `POST http://127.0.0.1:8000/sync` para recargar la API de Producción. Los alias `staging`/`challenger` apuntan al puerto 8001; un alias borrado lleva el modelo a `Archived`. `/sync` también se puede invocar manualmente. El repositorio fija esta secuencia en [`modelos/README.md`](modelos/README.md) y en DV-08.

### 9.7 Frontend y CI/CD

- `ci-frontend.yml`: compila (`npm run build`) en push o PR a `main` cuando cambia `frontend/**`, y también por `workflow_dispatch`. Sube el artefacto `frontend-build` desde `frontend/dist/` (retención de 7 días).
- `cd-frontend.yml`: se lanza manualmente con `workflow_dispatch`; toma `environment` (`staging` o `production`, por defecto `production`) y un `run_id` opcional. Descarga el artefacto y ejecuta `wrangler pages deploy ./frontend-deploy --project-name=denguealertas`, con rama `main` para Producción y rama `staging` para Staging.
- Producción: proyecto `denguealertas`, rama `main`, URL de Pages `https://denguealertas.pages.dev` y dominio propio `https://demo.arkoiot.com` (Cloudflare Access). Staging: misma rama `staging`, URL `https://staging.denguealertas.pages.dev` (también con Access).
- `VITE_API_URL` se lee al compilar. El workflow de CI no lo define, así que el artefacto usa el valor por defecto del código salvo que se construya con otro valor.
- Desde DV-11, `/` abre el modo v0.2; `?new=false` conserva el modo manual. La regla de §9.3 evitará ese parámetro solo en `demo.arkoiot.com`.

### 9.8 Endpoints existentes y pendientes

Existen hoy en la API (`src/api.py`) y en el servicio de webhook (`src/webhook_api.py`):

| Método | Ruta | Uso |
|---|---|---|
| `GET` | `/health` | Estado, etapa, modelos cargados |
| `GET` | `/municipios` | Códigos y modelos con `enabled=true` |
| `GET` | `/schema?municipio=<DIVIPOLA>` | Esquema de entrada del modelo |
| `GET` | `/predict?municipio=&semana_emision=` | Vista v0.2: la API deriva las seis variables del panel |
| `POST` | `/predict` | Modo manual: seis variables enviadas por el usuario |
| `POST` | `/sync` | Recarga en caliente desde MLflow |
| `POST` | `/telemetry/event` | Evento de clickstream del frontend |
| `POST` | `/webhook/mlflow`, `/webhook` | Receptor de eventos de alias de MLflow (puerto 8005) |

Ejemplo real (la respuesta declara `version_esquema: "0.2"` e incluye `probabilidad`, `decision`, `trayectoria`, `serie_canal` y `referencias`):

```bash
curl -sS \
  -H "Origin: https://demo.arkoiot.com" \
  -H "CF-Access-Client-Id: <service-token-id>" \
  -H "CF-Access-Client-Secret: <service-token-secret>" \
  "https://api.arkoiot.com/predict?municipio=54001&semana_emision=2025-12-22"
```

Desde la VM se puede probar sin service token apuntando a NGINX, por ejemplo con `curl -H "Origin: https://staging.denguealertas.pages.dev" "http://127.0.0.1:8080/predict?municipio=54001&semana_emision=2025-12-22"`.

**Pendiente** (contratado, no implementado): `POST /predict/upload`, `GET /historial` y `GET /desempeno`. No deben considerarse disponibles. El detalle de campos y errores de `/predict` está en [`docs/forecasting/api_salida_d10.md`](docs/forecasting/api_salida_d10.md) §4–§7.

### 9.9 Telemetría y monitoreo

Son dos cosas distintas: la telemetría registra el uso de la API (peticiones, errores e interacción de la interfaz), mientras que el monitor de recursos solo mide el consumo de la infraestructura de los servicios que se levantaron en la VM. El monitor no registra peticiones, predicciones ni métricas de negocio.

- API: middleware JSONL en `/opt/dengue-api/logs/api_<stage>.jsonl` con `X-Request-ID`/`CF-Ray`, correo de Access, latencia, estado y errores; filtra el polling de `/schema`, `/health` y `/telemetry/event` salvo error. `POST /telemetry/event` registra `action`, `target`, `section` y `metadata` del frontend en el mismo logger. Rotación y respaldos en `/opt/dengue-api/backups/telemetry/`.
- NGINX: `/var/log/nginx/dengue_telemetry.json` registra tiempo, `request_id`, correo de Access, IP de Cloudflare, `Origin`, método, URI, estado, bytes, latencia y latencia del upstream; excluye las rutas de polling.
- Webhook: `/opt/dengue-api/logs/webhook_events.jsonl` registra entregas y validación de firma.
- Monitor de recursos: `src/resource_monitor.py` muestrea cada 10 s CPU/RAM/carga, E/S de disco y red, almacenamiento y estado de **los servicios levantados en la VM** —`mlflow.service`, `dengue-api-prod.service`, `dengue-api-staging.service` y `dengue-webhook.service`— en `/opt/dengue-api/logs/system_resources.jsonl`. Es información operativa de esos procesos, no telemetría de la API ni del modelo. Estado al 2026-11-23: **operativo**. Entre el 2026-10-21 y esta fecha estuvo caído por `ModuleNotFoundError: No module named 'src'`; se corrigió con el drop-in `/etc/systemd/system/dengue-resource-monitor.service.d/10-pythonpath.conf`, que define `Environment=PYTHONPATH=/opt/dengue-api` sin modificar la unidad original.

## 10. Documentación

- [`docs/forecasting/README.md`](docs/forecasting/README.md): guía metodológica del pronóstico (fases, módulos, reproducibilidad, gobernanza, figuras y glosario).
- [`reports/forecasting/INDICE.md`](reports/forecasting/INDICE.md): mapa de los datos, la evidencia, las figuras y la configuración, con su estado.
- [`docs/forecasting/contrato_predictivo.md`](docs/forecasting/contrato_predictivo.md): qué se construye y cómo se evalúa (vinculante).
- [`docs/forecasting/especificacion_implementacion.md`](docs/forecasting/especificacion_implementacion.md): cómo se implementa.
- [`docs/forecasting/registro_decisiones.md`](docs/forecasting/registro_decisiones.md) y [`docs/forecasting/desviaciones.md`](docs/forecasting/desviaciones.md): por qué, y qué cambió durante la implementación.
- [`docs/forecasting/analisis_complementario.md`](docs/forecasting/analisis_complementario.md): contraste de familias, M0, regularización, regla de alerta del producto y continuidad, posteriores a la prueba.
- [`modelos/README.md`](modelos/README.md) y [`notebooks/README.md`](notebooks/README.md): artefactos del modelo final y cuadernos de lectura, con el mapa de etapas, módulos y documentación.
- [`docs/contexto_metodologico.md`](docs/contexto_metodologico.md), [`docs/data_dictionary.md`](docs/data_dictionary.md), [`docs/data_pipeline.md`](docs/data_pipeline.md), [`docs/pilot_selection.md`](docs/pilot_selection.md) y [`docs/epidemiology_labels.md`](docs/epidemiology_labels.md): etapas anteriores al pronóstico.
- [`AGENTS.md`](AGENTS.md) y [`CLAUDE.md`](CLAUDE.md): reglas de trabajo en el repositorio.

---

En la escritura del código y de la documentación se usaron herramientas de asistencia a la programación basadas en inteligencia artificial. Las decisiones metodológicas, la revisión del código y la aprobación de los resultados corresponden al equipo del proyecto.
