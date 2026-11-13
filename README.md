# DengueAlertas

Alerta temprana de excesos en canales endémicos de dengue en Colombia mediante *machine learning*. Prototipo académico retrospectivo de la Maestría en Inteligencia Artificial de la Universidad de los Andes.

## 1. Problema

Las secretarías de salud vigilan el dengue con el **canal endémico**: una banda construida con los conteos de los años anteriores para cada semana del año. Cuando los casos de una semana superan el límite superior del canal, el municipio está en **zona epidémica**. El canal describe la semana que ya ocurrió; no dice qué va a pasar en las semanas siguientes.

La pregunta del proyecto es: **¿qué tan probable es que un municipio esté en zona epidémica dentro de 8 semanas?**, y si esa probabilidad, convertida en alertas, avisa los inicios de brote mejor o antes que la alerta descriptiva del propio canal.

## 2. Solución

- **Datos.** Registros confirmados del evento 210 (dengue) de SIVIGILA por municipio de ocurrencia y semana de inicio de síntomas (lunes a domingo), con covariables climáticas semanales, 2007–2025. Seis municipios piloto: Barranquilla `08001`, Soledad `08758`, Neiva `41001`, Villavicencio `50001`, Cúcuta `54001` y Cali `76001`.
- **Variable objetivo.** `target_u1e` = 1 si los casos de la semana superan el límite superior U1e del canal endémico calculado como en epiCo (cinco años previos, media geométrica, intervalo t). El modelo predice ese estado 8 semanas después de la última semana con datos completos.
- **Modelo.** Un modelo independiente por municipio que emite una probabilidad semanal. Se compara con dos líneas base: B2 (probabilidad según la zona actual del canal) y B1 (estacionalidad de la semana objetivo). Una regla fija (dos emisiones consecutivas sobre un umbral τ) convierte las probabilidades en alertas, que se comparan con la alerta del canal.
- **Validación.** Origen móvil con reentrenamiento anual: desarrollo 2019–2022 para elegir; prueba 2023–2025 evaluada una sola vez, después de congelar la receta.
- **Entrega.** API FastAPI y frontend React con cuatro vistas: pronóstico vigente, historial fuera de muestra, carga de datos y desempeño en prueba (pendiente de F8).

La metodología completa, sus decisiones (D0–D10) y la secuencia de fases están en [`docs/forecasting/README.md`](docs/forecasting/README.md). Este README no reporta resultados: están en la evidencia indexada en [`reports/forecasting/INDICE.md`](reports/forecasting/INDICE.md). El análisis integrado de los resultados está en [`docs/forecasting/analisis_resultados.md`](docs/forecasting/analisis_resultados.md), con versiones en Word y presentación en [`reports/forecasting/entregables/`](reports/forecasting/entregables/).

## 3. Estructura del repositorio

```
configs/                  Parámetros del pronóstico (forecasting.yaml) y receta congelada (receta_congelada.yaml)
data/processed/           Descriptores .dvc y manifiestos; el contenido está en el remoto DVC
docs/                     Documentación de datos, selección de pilotos y etiquetas epidemiológicas
docs/forecasting/         Contrato predictivo, especificación, decisiones, desviaciones, API y scripts de verificación
frontend/                 Aplicación React/Vite (Cloudflare Pages)
reports/                  Evidencia versionada: selección de pilotos, etiquetas y pronóstico (tablas, figuras, bitácora, entregables)
requirements/             Dependencias por etapa
scripts/                  Utilidades (descripción del canal, comprobación de MLflow en solo lectura)
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

Instalación del entorno de modelado:

```bash
py -3.11 -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements/forecasting.lock.txt
```

`requirements/forecasting.lock.txt` reproduce las versiones exactas con las que se generaron los resultados; `requirements/forecasting.txt` fija solo las directas.

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
  - **DVC:** `data/processed/forecasting/prueba/modelos_finales/` (`<código>.joblib` y `manifiesto.json`, con la familia, el candidato, el conjunto de variables, las filas y el hash de la receta). Copia del manifiesto en git: `reports/forecasting/evidence/prueba_20261112/manifiesto_modelos_finales.json`.
  - **MLflow:** la prueba única quedó registrada en el experimento `dengue_h8_prueba` del servidor del equipo; la búsqueda de desarrollo, en `dengue_h8_desarrollo`. El registro de los modelos finales como `DengueAlertas_<Municipio>` (nombre del municipio; propuesta de la especificación §9) en el registro de modelos, que es lo que lee la API, es parte de F8.
- **Por qué no hay carpeta `modelos/`.** Los modelos son salidas del pipeline, como las predicciones y la ficha: se versionan con DVC junto a las salidas de la prueba de la que dependen (especificación §4, «Ubicación de resultados»), y git solo guarda su descriptor. El punto de distribución para la API es el registro de modelos de MLflow, no una carpeta del repositorio.

## 8. Parametrización

| Dónde | Qué fija |
|---|---|
| `configs/forecasting.yaml` | Rutas de datos y versión DVC del canal; parámetros del canal; horizonte h; ventanas de las variables; años de desarrollo y prueba; líneas base; recorte del log-loss; α de las comparaciones; k, W y rejilla de τ; pre-registro de la búsqueda (candidatos y semillas); seguimiento en MLflow. Se valida con Pydantic al cargarse (`src/forecasting/config.py`) |
| `configs/receta_congelada.yaml` | Receta elegida en desarrollo y aprobada antes de la prueba. No se edita: cualquier cambio invalida su hash y la barrera de prueba la rechaza |

Variables de entorno existentes:

| Variable | Uso |
|---|---|
| `FORECAST_HORIZON` | Sobrescribe h del pipeline (por defecto 8); debe ser 2, 4, 8 o 12 (`horizonte.permitidos`), y solo h = 8 tiene receta y modelos |
| `MLFLOW_TRACKING_URI` | Servidor MLflow (por defecto, el de `configs/forecasting.yaml`) |
| `CF_ACCESS_CLIENT_ID`, `CF_ACCESS_CLIENT_SECRET` | Credenciales de Cloudflare Access para MLflow. Van en `.env` (plantilla: `.env.example`), que git ignora; las entrega el responsable de infraestructura y API |
| `STAGE`, `MODEL_NAME`, `MODEL_THRESHOLD` | Configuración de la API actual (`src/api.py`, `Dockerfile`); se revisan en F8 |

Las variables nuevas de la API (`ALERT_K`, `DATA_VERSION` y la URL de la API en el frontend) se definen en F8; su diseño está en la especificación §10.

## 9. Despliegue

### 9.1 Aplicación

Pendiente de F8.

### 9.2 Uso de la API

Pendiente de F8. El contrato de la salida (`/predict`, `/predict/upload`, `/historial` y `/desempeno`, errores y variables de entorno) está fijado en [`docs/forecasting/api_salida_d10.md`](docs/forecasting/api_salida_d10.md).

### 9.3 Ejemplos de entrada y salida

Pendiente de F8. Los campos de cada respuesta están en [`docs/forecasting/api_salida_d10.md`](docs/forecasting/api_salida_d10.md) §4–§7, y los datos simulados para el frontend, en §11.

### 9.4 Credenciales de ejemplo

Pendiente de F8.

## 10. Documentación

- [`docs/forecasting/README.md`](docs/forecasting/README.md): guía metodológica del pronóstico (fases, módulos, reproducibilidad, gobernanza, figuras y glosario).
- [`reports/forecasting/INDICE.md`](reports/forecasting/INDICE.md): mapa de los datos, la evidencia, las figuras y la configuración, con su estado.
- [`docs/forecasting/contrato_predictivo.md`](docs/forecasting/contrato_predictivo.md): qué se construye y cómo se evalúa (vinculante).
- [`docs/forecasting/especificacion_implementacion.md`](docs/forecasting/especificacion_implementacion.md): cómo se implementa.
- [`docs/forecasting/registro_decisiones.md`](docs/forecasting/registro_decisiones.md) y [`docs/forecasting/desviaciones.md`](docs/forecasting/desviaciones.md): por qué, y qué cambió durante la implementación.
- [`docs/contexto_metodologico.md`](docs/contexto_metodologico.md), [`docs/data_dictionary.md`](docs/data_dictionary.md), [`docs/data_pipeline.md`](docs/data_pipeline.md), [`docs/pilot_selection.md`](docs/pilot_selection.md) y [`docs/epidemiology_labels.md`](docs/epidemiology_labels.md): etapas anteriores al pronóstico.
- [`AGENTS.md`](AGENTS.md) y [`CLAUDE.md`](CLAUDE.md): reglas de trabajo en el repositorio.

---

La implementación se desarrolló con asistencia de una herramienta de IA para programación, bajo revisión y aprobación del equipo.
