# Instrucciones para agentes — DengueAlertas

**Proyecto:** «Alerta temprana de excesos en canales endémicos de dengue en Colombia mediante Machine Learning».  
**Ámbito:** repositorio `MaiaMLOps/DengueAlertas` · Maestría en Inteligencia Artificial, Universidad de los Andes · acompañamiento BIOMAC.  
**Estado contextual:** 2026-10-20 (hora de Colombia), después de la integración y de la selección territorial.  
**Lecturas principales:** `docs/contexto_metodologico.md`, `docs/pilot_selection.md`, `docs/data_dictionary.md` y, cuando corresponda, `docs/data_pipeline.md`.

## 1. Alcance y estado vigente

El proyecto desarrolla un **prototipo académico retrospectivo** para anticipar estados futuros de alerta/exceso de dengue, con **modelos independientes por municipio**. No se exige cobertura nacional, conexión operacional con SIVIGILA, adquisición en tiempo real ni análisis causal del clima.

**Etapas terminadas:**

- Integración de SIVIGILA evento 210 y covariables semanales: `src/data/`; panel maestro `data/processed/master_weekly_screening.parquet` (**78 municipios × 992 semanas = 77.376 filas, 15 columnas**).
- Selección descriptiva documentada de **seis pilotos**: **Cúcuta `54001`, Barranquilla `08001`, Neiva `41001`, Villavicencio `50001`, Cali `76001` y Soledad `08758`**. Código y evidencia: `src/pilot_selection/`, `docs/pilot_selection.md` y `reports/pilot_selection/evidence/selection_20261020/`.
- Derivado territorial `data/processed/pilot_municipalities_weekly.parquet`: **5.952 filas × 15 columnas**, sin etiquetas ni rezagos. Ambos Parquet tienen descriptores DVC versionados; la carga del derivado al remoto fue reportada, sin descarga independiente documentada.
- Integración a `main` mediante los PR #3 y #4. El repositorio también dispone de frontend React/Vite, API FastAPI e infraestructura MLOps del equipo; esto no implica que exista ya un predictor de dengue operativo.

**Etapa científica siguiente, aún no resuelta:** definición epidemiológica de alerta/exceso y construcción de la variable objetivo. El canal endémico, las etiquetas, los horizontes, las ventanas, las particiones temporales y los modelos **no están aprobados ni implementados**. Los seis pilotos son el ámbito territorial adoptado; no supone que los seis deban entrenarse simultáneamente.

## 2. Autonomía y decisiones

Distinguir siempre **hechos comprobados**, **decisiones del equipo**, **hipótesis provisionales**, **propuestas antiguas** y **decisiones abiertas**. La propuesta académica original no prevalece automáticamente sobre decisiones posteriores.

- **Puede resolver:** diseño interno de módulos y funciones, validaciones técnicas, pruebas, organización de código y errores, dentro de una tarea autorizada.
- **Debe consultar antes de fijar:** definición final de casos; algoritmo y parámetros del canal endémico; etiqueta objetivo (estado, cruce de umbral o inicio de episodio); periodos de referencia; horizonte y fecha de emisión; manejo de ceros, años epidémicos, semanas parciales y anomalías; rezagos, predictores, cortes de validación, baselines, métricas y umbrales de decisión.
- **Debe advertir y detener la parte afectada** si detecta fuga temporal de información (*data leakage*), contradicciones científicas, pérdida de trazabilidad o supuestos sin respaldo. No corregir silenciosamente ni optimizar decisiones metodológicas sobre el conjunto de prueba.

El usuario emplea ChatGPT para discutir el fundamento científico y Codex para implementar y verificar software. **Priorizar explicaciones comprensibles de los módulos, funciones principales y transformaciones realizadas**; evitar complejidad, automatización y documentación innecesarias para el MVP. No hace falta comentar línea a línea salvo que se solicite.

## 3. Datos, protección e integridad

- Los originales SIVIGILA, covariables y referencias están fuera del repositorio, son de **solo lectura** y pueden encontrarse en unidades distintas del computador. Usar rutas configurables, no rutas personales en Git.
- No copiar registros individuales de salud a Git, almacenamiento DVC compartido, logs o resultados publicados. Para publicar derivados, verificar autorización y condiciones de las fuentes.
- El screening histórico cuenta registros del **evento 210** con `confirmados = 1`, municipio de ocurrencia (`COD_DPTO_O` + `COD_MUN_O`), fecha `INI_SIN` y semana **lunes–domingo (W-SUN)**; esto **no determina** la definición definitiva de caso o de etiqueta.
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
- No hacer `git commit`, `git push`, merges, PR, `dvc push`, instalaciones no autorizadas ni cambios de infraestructura, API, frontend o MLflow sin aprobación explícita. No sobrescribir fuentes ni salidas existentes.
- En tareas relevantes: leer contexto, confirmar alcance, implementar lo autorizado, probar y comunicar con brevedad **qué cambió, por qué, cómo se verificó, qué no se hizo y qué permanece abierto**. Al solicitar explicación, recorrer el flujo real de los módulos, no solo enumerar archivos.

## 6. Jerarquía de referencias

1. Decisiones explícitas y recientes del equipo; evidencia de ejecuciones y archivos actuales.
2. `docs/contexto_metodologico.md` y `docs/pilot_selection.md`; reporte final de selección territorial y evidencia versionada.
3. `docs/data_pipeline.md`, `docs/data_dictionary.md`, informes técnicos SIVIGILA/covariables y el artículo de Umaña et al. (2026).
4. Propuesta inicial y estado del arte como antecedentes, no como parámetros automáticamente aprobados.

Si una fuente no está disponible o no puede leerse, informarlo; no inventar su contenido.
