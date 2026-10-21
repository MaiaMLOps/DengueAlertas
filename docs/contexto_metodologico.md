# Contexto metodológico y estado de decisiones — DengueAlertas

**Actualizado:** 2026-10-20 (hora de Colombia; PR #4 integrado el 21-10-2026 UTC).  
**Proyecto:** «Alerta temprana de excesos en canales endémicos de dengue en Colombia mediante Machine Learning».  
**Equipo:** Maestría en Inteligencia Artificial, Universidad de los Andes; proyecto colaborativo con acompañamiento BIOMAC.  
**Estado:** integración retrospectiva y selección de **seis municipios piloto** concluidas; definición del desenlace y modelamiento pendientes.  
**Documentación asociada:** `../AGENTS.md`, `data_pipeline.md`, `data_dictionary.md`, `pilot_selection.md`.

## 1. Propósito y alcance

El proyecto busca construir un prototipo académico que **anticipe estados futuros de alerta/exceso de dengue** frente a una referencia histórica local. Un **canal endémico** permite interpretar estados epidemiológicos observados; un **modelo predictivo** debe anticiparlos sin utilizar información indisponible al emitir el pronóstico. Son componentes diferentes.

El alcance vigente fue reducido respecto de la propuesta inicial: **modelos independientes por municipio**, en territorios seleccionados, con validación histórica y posterior integración en un prototipo. No se exige un único modelo nacional, cobertura de todos los municipios, conexión operativa con SIVIGILA, inferencia en tiempo real ni demostración causal de efectos climáticos.

Esta documentación separa **resultados documentados**, **decisiones vigentes del equipo** y **aspectos científicos abiertos**. Parámetros exploratorios de la propuesta original no se adoptan automáticamente como diseño final.

## 2. Productos de datos disponibles

### 2.1. Panel maestro integrado

El pipeline `src/data/` integra la auditoría del evento 210 de SIVIGILA (archivos nominales 2007–2025) con las covariables semanales de ERA5-Land, CHIRPS y DANE. Se agrupa por municipio de ocurrencia y semana de inicio de síntomas (`INI_SIN`), y se une por `municipality_code` + `week_start`.

**Producto:** `data/processed/master_weekly_screening.parquet`; descriptor `data/processed/master_weekly_screening.parquet.dvc`.

| Propiedad | Resultado documentado |
|---|---:|
| Registros individuales examinados en 19 cachés anuales | 1.705.604 |
| Municipios candidatos / departamentos | 78 / 26 |
| Semanas por municipio (2007–2025) | 992 |
| Filas / columnas | 77.376 / 15 |
| Registros confirmados elegibles según screening | 637.271 |
| Posiciones con al menos un registro elegible | 52.831 |
| Posiciones sin registros elegibles observados | 24.545 |
| Posiciones de semana parcial final | 78 |

El proceso es reproducible y cuenta con pruebas automatizadas (46 aprobadas en el cierre de integración). Las nueve columnas de covariables originales permanecen y se añaden conteo semanal de screening, indicador de ausencia de registros elegibles y cuatro campos de calendario. Las definiciones de las 15 columnas están en `data_dictionary.md`; transformaciones y controles en `data_pipeline.md`.

**Regla aplicada:** evento 210, `confirmados = 1` (estado final 3 o 5 en archivos auditados), municipio de ocurrencia, semana lunes–domingo derivada de `INI_SIN` y ventana de fecha del 01-01-2007 al 31-12-2025. Esa regla reproduce el **screening retrospectivo**, pero no determina todavía la definición final de casos ni de exceso. La agregación suma aportes de diferentes archivos nominales. Los ceros son ausencia de registros elegibles observados, no notificaciones negativas verificadas.

### 2.2. Selección territorial finalizada

La etapa `src/pilot_selection/` examinó descriptivamente los **78 candidatos**, su continuidad, volumen, rachas sin registros, perfiles interanuales y concentración de casos. Contrastó grupos de alta, media y baja persistencia, series gráficas semanales y una sensibilidad restringida a **2007–2022**. La selección es **intencional del equipo**, no un ranking automático ni una estratificación oficial del riesgo de transmisión.

**Pilotos adoptados:**

| Municipio | DIVIPOLA | Semanas activas en 2007–2022 (%) |
|---|---|---:|
| Cúcuta | `54001` | 98,08 |
| Barranquilla | `08001` | 97,96 |
| Neiva | `41001` | 97,72 |
| Villavicencio | `50001` | 96,77 |
| Cali | `76001` | 96,65 |
| Soledad | `08758` | 93,41 |

Los primeros cinco estuvieron en el grupo de mayor persistencia histórica; **Soledad se incorporó por decisión explícita** tras la exploración. La actividad semanal se refiere a presencia de al menos un registro elegible, no a calidad/completitud de vigilancia. La sensibilidad 2007–2022 reduce la dependencia descriptiva de picos recientes, pero no borra la selección retrospectiva con conocimiento de 2007–2025.

**Producto:** `data/processed/pilot_municipalities_weekly.parquet`, descriptor `data/processed/pilot_municipalities_weekly.parquet.dvc`.

| Propiedad del derivado | Resultado documentado |
|---|---:|
| Municipios / semanas por municipio | 6 / 992 |
| Filas / columnas | 5.952 / 15 |
| Confirmados elegibles, todas las semanas | 231.406 |
| Filas / elegibles en semanas completas | 5.946 / 231.281 |
| Posiciones de semana parcial (3 días) | 6 |
| Claves duplicadas / nulos reportados | 0 / 0 |

Este producto **solo filtra territorialmente** el maestro y conserva las 15 columnas, su esquema y sus valores. No contiene canales endémicos, etiquetas, ventanas, rezagos, particiones ni modelos. El filtro usa la tupla `PILOTS` de `src/pilot_selection/config.py`. El informe de cierre, `pilot_selection.md` y la evidencia en `reports/pilot_selection/evidence/selection_20261020/` sustentan la decisión. PR #4 fusionado en `main` (commit de fusión `5cae9f8a`).

## 3. Fuentes y limitaciones que no deben perderse

**SIVIGILA.** Se utilizaron los 19 Parquet anuales de la caché del evento 210, no la totalidad de las 69 variables de los libros originales. Los estados de confirmación son finales y retrospectivos. No se reconstruyeron secuencias de revisiones, disponibilidad histórica de cada registro ni deduplicación de personas/episodios entre años. En la auditoría se identificaron **58.876** pares con notificación anterior a síntomas, de los cuales **58.869** corresponden al archivo nominal 2013; no se corrigieron automáticamente. La selección original de 78 y la adopción de seis pilotos utilizaron información de todo el histórico, lo que condiciona la interpretación de futuras evaluaciones temporales.

**Calendario.** La referencia es `week_start`, lunes inicial de semanas **W-SUN**. La equivalencia con las semanas oficiales INS no se verificó. El último periodo comienza el **2025-12-29** y contiene **tres días** observados hasta 2025-12-31. Se **excluyó de las estadísticas de selección**, pero **se conservó en ambos datasets Parquet**. La cantidad de días del intervalo no demuestra completitud epidemiológica ni climática.

**Clima y demografía.** Temperatura del aire y humedad relativa estimada proceden de **ERA5-Land**; precipitación de **CHIRPS v2.0 Final**; población anual y densidad de **DANE/MGN 2025**. La humedad es derivada a partir de temperatura y punto de rocío diarios, no medida directa; las medias son zonales municipales, no necesariamente representativas de zonas pobladas. Población/densidad no equivalen a población oficial en riesgo. ONI/ENSO y MODIS LST **no fueron incorporados** al núcleo actual. Ninguna covariable tiene aporte predictivo probado todavía.

**Interpretación de la selección.** Alta persistencia de registros no garantiza suficientes episodios positivos bajo la futura definición de exceso. Los picos graficados no son por sí mismos brotes definidos. Los seis pilotos no son una muestra probabilística ni necesariamente equivalentes en dificultad predictiva.

## 4. Definiciones científicas todavía abiertas

| Tema | Estado |
|---|---|
| Definición de caso del estudio | El screening `confirmados = 1` no se ha adoptado como criterio universal del desenlace |
| Canal endémico | Sin método/parametrización aprobados: referencia histórica, años, tratamiento de ceros y periodos epidémicos, umbrales, historia insuficiente |
| Variable objetivo | Falta distinguir y elegir entre estado de alerta, exceso epidémico o inicio de episodio sostenido; estudiar su frecuencia por municipio |
| Disponibilidad retrospectiva | Falta fijar fecha de emisión y reconocer retrasos/ajustes de SIVIGILA y latencias climáticas |
| Predicción supervisada | Horizonte, ventanas, rezagos y matriz de entrada aún no fijados |
| Entrenamiento y evaluación | Baselines, modelos, cortes cronológicos, métricas, calibración y umbral de decisión aún no fijados |
| Despliegue del predictor | API/frontend existen, pero todavía no hay un modelo de dengue validado conectado al producto |

El artículo de **Umaña et al. (2026)** (*Endemic Channel Parametrization in Dengue Surveillance*) es una referencia central sobre definición y sensibilidad de umbrales, ventanas históricas y tratamiento de ceros. **El artículo no valida pronósticos de ML**. Las opciones históricas de ventanas de 3/5/7 años, desplazamiento mínimo `ε = 0,001`, exclusión de años epidémicos, rachas y horizontes `h = 1, 2, 4` se deben estudiar, **no ejecutar por defecto**.

En la etapa siguiente debe distinguirse la **asignación retrospectiva de un estado epidemiológico** de su **anticipación futura**. No utilizar umbrales calculados con años posteriores a la fecha relevante ni características contemporáneas/futuras de la semana objetivo. Una vez aprobada la formulación del desenlace, caracterizar su prevalencia y distribución temporal antes de entrenar modelos.

## 5. Código, versionamiento y forma de trabajo

- Git/GitHub aloja el código, pruebas, documentación y los descriptores DVC; el repositorio `MaiaMLOps/DengueAlertas` integra las etapas de datos y selección territorial mediante los **PR #3 y #4**.
- DVC referencia ambos Parquet en `data/processed/`. El contenido se almacena en el remoto WebDAV **`vps-storage`**; la transferencia del maestro fue comprobada y la del derivado está **reportada** por el equipo. No está documentado un `dvc pull` independiente desde otro equipo. Credenciales locales quedan fuera de Git.
- `src/data/` integra fuentes; `src/pilot_selection/` genera análisis territorial y derivado; `tests/` contiene comprobaciones; `docs/` y `reports/pilot_selection/evidence/` recogen metodología y evidencia. No se exige volver a generar las fuentes de origen para estudiar el desenlace.
- MLflow y los servicios de aplicación forman parte de la infraestructura del equipo, pero no hay aún entrenamiento/registro de un predictor epidemiológico definitivo que pueda inferirse de su existencia.
- ChatGPT apoya discusión científica y tutoría; Codex ejecuta tareas autorizadas sobre el repositorio local. El usuario supervisa las decisiones científicas, desea comprender los propósitos y flujo de los módulos y prefiere **prompts breves, implementación modular y sin sobreingeniería**.

## 6. Fuentes y prioridad de evidencia

- `docs/pilot_selection.md` y el informe **«Informe_final_seleccion_municipios_piloto_DengueAlertas.docx»**: decisión y resultados territoriales.
- `docs/data_pipeline.md`, `docs/data_dictionary.md` y los informes técnicos de **SIVIGILA** y **covariables**: construcción e interpretación de las 15 columnas.
- Umaña et al. (2026): soporte científico para estudiar la parametrización del canal endémico, sin trasladar automáticamente sus métodos a las etiquetas del proyecto.
- `Base científica y estado del arte.docx` y propuesta académica original: antecedentes; algunas opciones reflejan un alcance anterior.

Ante discrepancias, distinguir lo observado de lo inferido y usar primero las decisiones expresas más recientes del equipo y la evidencia reproducible actual. Cambios científicos importantes requieren justificación y aprobación.
