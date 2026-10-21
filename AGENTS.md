# Instrucciones para agentes — DengueAlertas

**Proyecto:** «Alerta temprana de excesos en canales endémicos de dengue en Colombia mediante Machine Learning».  
**Ámbito:** repositorio `MaiaMLOps/DengueAlertas` · Maestría en Inteligencia Artificial, Universidad de los Andes · acompañamiento BIOMAC.  
**Estado contextual:** 2026-10-21 (hora de Colombia), cierre local de etiquetas y caracterización; versionamiento manual pendiente.
**Lecturas principales:** `docs/contexto_metodologico.md`, `docs/epidemiology_labels.md`, `docs/pilot_selection.md`, `docs/data_dictionary.md` y, cuando corresponda, `docs/data_pipeline.md`.

## 1. Alcance y estado vigente

El proyecto desarrolla un **prototipo académico retrospectivo** para anticipar estados futuros de alerta/exceso de dengue, con **modelos independientes por municipio**. No se exige cobertura nacional, conexión operacional con SIVIGILA, adquisición en tiempo real ni análisis causal del clima.

**Etapas terminadas:**

- Integración de SIVIGILA evento 210 y covariables semanales: `src/data/`; panel maestro `data/processed/master_weekly_screening.parquet` (**78 municipios × 992 semanas = 77.376 filas, 15 columnas**).
- Selección descriptiva documentada de **seis pilotos**: **Cúcuta `54001`, Barranquilla `08001`, Neiva `41001`, Villavicencio `50001`, Cali `76001` y Soledad `08758`**. Código y evidencia: `src/pilot_selection/`, `docs/pilot_selection.md` y `reports/pilot_selection/evidence/selection_20261020/`.
- Derivado territorial `data/processed/pilot_municipalities_weekly.parquet`: **5.952 filas × 15 columnas**, sin etiquetas ni rezagos. Ambos Parquet tienen descriptores DVC versionados; la carga del derivado al remoto fue reportada, sin descarga independiente documentada.
- Integración a `main` mediante los PR #3 y #4. El repositorio también dispone de frontend React/Vite, API FastAPI e infraestructura MLOps del equipo; esto no implica que exista ya un predictor de dengue operativo.
- Etapa epidemiológica implementada y caracterizada en `feature/epidemiology-labels`: `src/epidemiology/` y `tests/epidemiology/`. Producto consolidado `data/processed/epidemiology/pilot_weekly_labels.parquet` (**5.952 × 18**), copia binaria del preliminar validado; manifiesto adyacente `run_manifest.json`. **4.380** elegibles, **2.870** positivas, **1.510** negativas y **1.572** no elegibles; **271** rachas positivas, **20** de al menos 52 semanas; **569** elegibles y **530** positivas con algún cero histórico. Evidencia final: `reports/epidemiology/evidence/labels_20261021/`. Preparado localmente para revisión y posterior versionamiento manual: no implica commit, integración en `main`, `dvc add` ni transferencia al remoto.

**Contrato epidemiológico aprobado:** conteo retrospectivo `confirmed_records_screening`; EWT municipal = `exp(mean(log(C + 0.001))) - 0.001` con los cinco años anteriores, incluidos años epidémicos. W-SUN; año del lunes inicial y ordinal de sus lunes desde uno; ordinal 53 histórico ausente sustituido por 52. Etiqueta estricta `conteo > EWT`, sin redondeo ni tolerancia. 2007–2011 y la semana parcial 2025-12-29 no elegibles, con EWT y etiqueta nulos. El control de residuos numéricos reconoce únicamente identidades enteras exactas; no introduce un mínimo epidemiológico. Detalles y reproducción en `docs/epidemiology_labels.md`.

**Etapa científica siguiente, aún no resuelta:** fecha de emisión y horizontes, disponibilidad histórica simulada, características/rezagos, particiones cronológicas, baselines, entrenamiento, evaluación y despliegue. Las rachas calculadas son descriptivas: no constituyen una nueva variable objetivo ni episodios validados. Los seis pilotos siguen siendo el ámbito adoptado; no supone entrenarlos simultáneamente.

## 2. Autonomía y decisiones

Distinguir siempre **hechos comprobados**, **decisiones del equipo**, **hipótesis provisionales**, **propuestas antiguas** y **decisiones abiertas**. La propuesta académica original no prevalece automáticamente sobre decisiones posteriores.

- **Puede resolver:** diseño interno de módulos y funciones, validaciones técnicas, pruebas, organización de código y errores, dentro de una tarea autorizada.
- **Debe consultar antes de cambiar lo aprobado o fijar decisiones pendientes:** definición de casos, algoritmo/parámetros del EWT, etiqueta, referencia temporal, manejo de ceros/años epidémicos/semanas parciales/anomalías; horizonte y emisión, rezagos, predictores, cortes de validación, baselines, métricas y umbrales de decisión. Las convenciones epidemiológicas anteriores ya fueron aprobadas; no reabrirlas ni optimizarlas automáticamente.
- **Debe advertir y detener la parte afectada** si detecta fuga temporal de información (*data leakage*), contradicciones científicas, pérdida de trazabilidad o supuestos sin respaldo. No corregir silenciosamente ni optimizar decisiones metodológicas sobre el conjunto de prueba.

El usuario emplea ChatGPT para discutir el fundamento científico y Codex para implementar y verificar software. **Priorizar explicaciones comprensibles de los módulos, funciones principales y transformaciones realizadas**; evitar complejidad, automatización y documentación innecesarias para el MVP. No hace falta comentar línea a línea salvo que se solicite.

## 3. Datos, protección e integridad

- Los originales SIVIGILA, covariables y referencias están fuera del repositorio, son de **solo lectura** y pueden encontrarse en unidades distintas del computador. Usar rutas configurables, no rutas personales en Git.
- No copiar registros individuales de salud a Git, almacenamiento DVC compartido, logs o resultados publicados. Para publicar derivados, verificar autorización y condiciones de las fuentes.
- El screening histórico cuenta registros del **evento 210** con `confirmados = 1`, municipio de ocurrencia (`COD_DPTO_O` + `COD_MUN_O`), fecha `INI_SIN` y semana **lunes–domingo (W-SUN)**. Se adoptó ese conteo para esta etapa retrospectiva; no certifica personas/episodios únicos ni brotes oficiales.
- Preservar DIVIPOLA como texto de cinco caracteres, incluidos ceros iniciales. Las semanas sin registros elegibles no prueban cero infecciones ni notificación negativa. Los estados de confirmación son finales y retrospectivos.
- No se ha validado equivalencia exacta con el calendario epidemiológico oficial del INS. La última semana inicia el **2025-12-29** y comprende **tres días** hasta el 31 de diciembre. El derivado de seis pilotos la conserva; el análisis descriptivo de selección la excluyó.
- Preservar anomalías conocidas, incluida la discrepancia de fechas notificación/síntomas en 2013, y la incertidumbre de duplicados y retrasos; no corregir automáticamente.
- Validar antes y después de cada transformación: claves, fechas, esquema, filas, nulos, totales, duplicados, exclusiones y hash/procedencia cuando aplique. No alterar datos para reproducir cifras esperadas.

## 4. Ciencia predictiva y evaluación

Separar expresamente: **panel histórico** → **estados/etiquetas epidemiológicas** → **ejemplos supervisados** → **modelos**. El artículo de Umaña et al. (2026) estudia la parametrización y comportamiento de canales endémicos; **no valida una herramienta de pronóstico**.

Para el futuro modelamiento:

- Calcular umbrales históricos sin utilizar información posterior a la fecha pertinente; discutir el tratamiento de años, ceros, semanas iniciales y revisiones retrospectivas.
- Definir qué era realmente conocido al emitir una predicción: los conteos finales de SIVIGILA y el clima retrospectivo no equivalen automáticamente a datos disponibles en tiempo real.
- Utilizar **particiones cronológicas**, con ajustes de imputación, escalamiento, selección y calibración exclusivamente dentro de entrenamiento.
- Evaluar separadamente por municipio y horizonte bajo definiciones y cortes comparables. No suponer valor añadido del clima por estar disponible ni llamar «probabilidad calibrada» a un score sin evaluación.

## 5. Ingeniería, Git, DVC y colaboración

- Python modular, reproducible, probado y con manejo de errores; cambios mínimos en el módulo pertinente. Antes de editar, revisar `git status`, la rama y el código existente.
- Código, pruebas, documentación y archivos `.dvc` se versionan en Git. **El contenido Parquet se gestiona mediante DVC**, remoto `vps-storage` WebDAV configurado; credenciales únicamente locales, nunca en mensajes, código o commits.
- Los CSV/PNG/SVG y manifiestos finales de `reports/epidemiology/evidence/labels_20261021/`, y el manifiesto adyacente del dataset, son elegibles para Git. Las corridas temporales y todos los Parquet permanecen ignorados. El descriptor DVC del nuevo Parquet está pendiente de creación manual; no fabricarlo ni ejecutar DVC sin autorización.
- No hacer `git commit`, `git push`, merges, PR, `dvc push`, instalaciones no autorizadas ni cambios de infraestructura, API, frontend o MLflow sin aprobación explícita. No sobrescribir fuentes ni salidas existentes.
- En tareas relevantes: leer contexto, confirmar alcance, implementar lo autorizado, probar y comunicar con brevedad **qué cambió, por qué, cómo se verificó, qué no se hizo y qué permanece abierto**. Al solicitar explicación, recorrer el flujo real de los módulos, no solo enumerar archivos.

## 6. Jerarquía de referencias

1. Decisiones explícitas y recientes del equipo; evidencia de ejecuciones y archivos actuales.
2. `docs/contexto_metodologico.md`, `docs/epidemiology_labels.md` y evidencia final epidemiológica local; `docs/pilot_selection.md`, reporte final de selección territorial y evidencia versionada. Distinguir siempre qué artefactos siguen pendientes de versionamiento.
3. `docs/data_pipeline.md`, `docs/data_dictionary.md`, informes técnicos SIVIGILA/covariables y el artículo de Umaña et al. (2026).
4. Propuesta inicial y estado del arte como antecedentes, no como parámetros automáticamente aprobados.

Si una fuente no está disponible o no puede leerse, informarlo; no inventar su contenido.
