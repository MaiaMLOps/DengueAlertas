# Instrucciones para agentes — DengueAlertas

**Proyecto:** «Alerta temprana de excesos en canales endémicos de dengue en Colombia mediante Machine Learning».
**Ámbito:** repositorio `MaiaMLOps/DengueAlertas` · Maestría en Inteligencia Artificial, Universidad de los Andes.
**Lecturas principales:** `README.md`, `docs/forecasting/README.md`, `docs/contexto_metodologico.md` (documento histórico: estado al 2026-10-21), `docs/epidemiology_labels.md`, `docs/pilot_selection.md`, `docs/data_dictionary.md` y, cuando corresponda, `docs/data_pipeline.md`.

## 1. Alcance y estado

**Prototipo académico retrospectivo** para anticipar estados futuros de exceso de dengue, con **modelos independientes por municipio**. No se exige cobertura nacional, conexión operacional con SIVIGILA, adquisición en tiempo real ni análisis causal del clima.

**Etapas terminadas:**

- Integración de SIVIGILA evento 210 y covariables semanales (`src/data/`): panel maestro `data/processed/master_weekly_screening.parquet` (78 municipios × 992 semanas = 77.376 filas, 15 columnas).
- Selección descriptiva de **seis pilotos**: Cúcuta `54001`, Barranquilla `08001`, Neiva `41001`, Villavicencio `50001`, Cali `76001` y Soledad `08758` (`src/pilot_selection/`, `docs/pilot_selection.md`). Derivado `data/processed/pilot_municipalities_weekly.parquet` (5.952 × 15, sin etiquetas ni rezagos).
- Etapa epidemiológica (`src/epidemiology/`): etiqueta EWT de legado en `data/processed/epidemiology/pilot_weekly_labels.parquet` (5.952 × 18), con su manifiesto `run_manifest.json` y evidencia en `reports/epidemiology/evidence/labels_20261021/`.
- MVP de pronóstico (`src/forecasting/`, en `main` desde el PR #8): fases F0–F7, reportes derivables (U7.5) y análisis integrado de §8.8 (`docs/forecasting/analisis_resultados.md`) cerrados; prueba única ejecutada. Pendiente: F8 (registro de modelos, API y frontend). Guía: `docs/forecasting/README.md`; estado de cada salida: `reports/forecasting/INDICE.md`.

El repositorio también contiene el frontend React/Vite, la API FastAPI y la infraestructura MLOps del equipo.

**Legado congelado.** `ewt`, `excess_label`, `pilot_weekly_labels.parquet`, `src/epidemiology/ewt.py` y `src/epidemiology/labels.py` son **legado**: no se modifican y se conservan para reproducir el informe de avance y su auditoría. El objetivo predictivo es el canal de epiCo (`target_u1e`), no `excess_label`, y `src/forecasting/` no lee ese Parquet. Los descriptores DVC existentes tampoco se modifican.

**Contrato epidemiológico de legado:** conteo retrospectivo `confirmed_records_screening`; EWT municipal = `exp(mean(log(C + 0.001))) - 0.001` con los cinco años anteriores, incluidos años epidémicos; semanas W-SUN con el año del lunes inicial y ordinal de sus lunes desde uno; ordinal 53 histórico ausente sustituido por 52; etiqueta estricta `conteo > EWT`; 2007–2011 y la semana parcial 2025-12-29 no elegibles. Detalles en `docs/epidemiology_labels.md`.

**Contrato predictivo:** emisión y horizonte, variables, particiones, líneas base, métricas y regla de decisión (D0–D10) están fijados en `docs/forecasting/contrato_predictivo.md`, y su implementación en `docs/forecasting/especificacion_implementacion.md`. Si este archivo los contradice, prevalecen ellos. Esos dos documentos y `docs/forecasting/registro_decisiones.md` son pre-registro: no se reescriben, se corrigen con una errata fechada al final o con una DV, y `docs/forecasting/desviaciones.md` solo crece (`CLAUDE.md`, regla 8).

## 2. Decisiones

Distinguir siempre **hechos comprobados**, **decisiones del equipo**, **hipótesis provisionales**, **propuestas antiguas** y **decisiones abiertas**. La propuesta académica original no prevalece automáticamente sobre decisiones posteriores.

- **Se puede resolver sin consulta:** diseño interno de módulos y funciones, validaciones técnicas, pruebas, organización de código y manejo de errores, dentro de una tarea autorizada.
- **Se consulta antes de cambiar lo aprobado:** definición de casos, canal y etiqueta, referencia temporal, manejo de ceros, años epidémicos, semanas parciales y anomalías; horizonte y emisión, variables, cortes de validación, líneas base, métricas y umbrales de decisión. Un cambio aprobado se registra en `docs/forecasting/desviaciones.md` antes de continuar.
- **Se advierte y se detiene la parte afectada** ante una posible fuga temporal, contradicciones científicas, pérdida de trazabilidad o supuestos sin respaldo. No se corrige en silencio ni se optimizan decisiones metodológicas sobre el conjunto de prueba.

Priorizar explicaciones comprensibles de los módulos, funciones principales y transformaciones; evitar complejidad, automatización y documentación innecesarias para el MVP.

## 3. Datos, protección e integridad

- Los originales de SIVIGILA, covariables y referencias están fuera del repositorio y son de **solo lectura**. Usar rutas configurables, nunca rutas personales en git.
- No copiar registros individuales de salud a git, al almacenamiento DVC compartido, a logs ni a resultados publicados. Para publicar derivados, verificar la autorización y las condiciones de las fuentes.
- El conteo histórico cuenta registros del **evento 210** con `confirmados = 1`, municipio de ocurrencia (`COD_DPTO_O` + `COD_MUN_O`), fecha `INI_SIN` y semana **lunes–domingo (W-SUN)**. No certifica personas o episodios únicos ni brotes oficiales.
- DIVIPOLA es texto de cinco caracteres, con ceros iniciales. Una semana sin registros no prueba cero infecciones. Los estados de confirmación son finales y retrospectivos.
- No se ha validado la equivalencia exacta con el calendario epidemiológico oficial del INS. La última semana (2025-12-29) tiene **tres días**.
- Preservar las anomalías conocidas (incluida la discrepancia de fechas de notificación y síntomas en 2013) y la incertidumbre de duplicados y retrasos; no corregirlas automáticamente.
- Validar antes y después de cada transformación: claves, fechas, esquema, filas, nulos, totales, duplicados, exclusiones y hash o procedencia. No alterar datos para reproducir cifras esperadas.

## 4. Ciencia predictiva y evaluación

Separar expresamente: **panel histórico** → **estados epidemiológicos** → **ejemplos supervisados** → **modelos**. El artículo de Umaña et al. (2026) estudia la parametrización y el comportamiento de los canales endémicos; **no valida una herramienta de pronóstico**.

- Ningún umbral ni variable usa información posterior a la emisión t; la prueba de truncamiento lo verifica.
- Lo conocido al emitir es lo que hay en la base hasta t: los conteos finales de SIVIGILA y el clima retrospectivo no equivalen a datos disponibles en tiempo real.
- Particiones cronológicas; imputación, escalamiento, selección y calibración solo con entrenamiento.
- Evaluar por municipio y horizonte con definiciones y cortes comparables. No suponer valor añadido del clima por estar disponible ni llamar «probabilidad calibrada» a un puntaje sin evaluación.
- Ninguna métrica se calcula sobre semanas objetivo de 2023–2025 sin una receta congelada aprobada; la barrera `src/forecasting/guard.py` lo impone y no se desactiva.

## 5. Ingeniería, Git, DVC y colaboración

- Python 3.11 modular, reproducible, probado y con manejo de errores; cambios mínimos en el módulo pertinente. Antes de editar, revisar `git status`, la rama y el código existente.
- Código, pruebas, documentación y archivos `.dvc` se versionan en git. **El contenido Parquet se gestiona con DVC** (remoto `vps-storage`, WebDAV). Las credenciales de DVC y de MLflow son locales y nunca se escriben en mensajes, código, logs ni commits.
- `.gitignore` ignora `*.csv` y `*.parquet`; la evidencia final elegible para git se habilita con excepciones explícitas (`reports/epidemiology/evidence/labels_20261021/`, `reports/forecasting/evidence/`, manifiestos adyacentes de los datasets).
- No hacer `git commit`, `git push`, merges, PR, `dvc add`, `dvc push`, instalaciones ni cambios de infraestructura, API, frontend o MLflow sin aprobación explícita. No sobrescribir fuentes ni salidas existentes.
- Al cerrar una tarea, comunicar con brevedad **qué cambió, por qué, cómo se verificó, qué no se hizo y qué queda abierto**. Al explicar, recorrer el flujo real de los módulos, no solo enumerar archivos.

## 6. Jerarquía de referencias

1. Decisiones explícitas y recientes del equipo; evidencia de ejecuciones y archivos actuales.
2. `docs/forecasting/` (contrato, especificación, decisiones y desviaciones) para el pronóstico; `docs/contexto_metodologico.md`, `docs/epidemiology_labels.md` y `docs/pilot_selection.md`, con su evidencia versionada, para las etapas anteriores.
3. `docs/data_pipeline.md`, `docs/data_dictionary.md`, informes técnicos de SIVIGILA y covariables, y el artículo de Umaña et al. (2026).
4. Propuesta inicial y estado del arte, como antecedentes y no como parámetros aprobados.

Si una fuente no está disponible o no puede leerse, informarlo; no inventar su contenido.
