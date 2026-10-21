# Contexto metodológico y estado de decisiones — DengueAlertas

**Actualizado:** 2026-10-21 (hora de Colombia; cierre epidemiológico local, pendiente de versionamiento manual).
**Proyecto:** «Alerta temprana de excesos en canales endémicos de dengue en Colombia mediante Machine Learning».  
**Equipo:** Maestría en Inteligencia Artificial, Universidad de los Andes; proyecto colaborativo con acompañamiento BIOMAC.  
**Estado:** integración, selección de seis pilotos y construcción/caracterización de etiquetas concluidas localmente; modelamiento pendiente.
**Documentación asociada:** `../AGENTS.md`, `epidemiology_labels.md`, `data_pipeline.md`, `data_dictionary.md`, `pilot_selection.md`.

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

**Regla aplicada:** evento 210, `confirmados = 1` (estado final 3 o 5 en archivos auditados), municipio de ocurrencia, semana lunes–domingo derivada de `INI_SIN` y ventana de fecha del 01-01-2007 al 31-12-2025. Esa regla reproduce el **screening retrospectivo**; posteriormente se adoptó su conteo como insumo de las etiquetas de esta etapa. La agregación suma aportes de diferentes archivos nominales. Los ceros son ausencia de registros elegibles observados, no notificaciones negativas verificadas.

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

### 2.3. Producto epidemiológico consolidado

`data/processed/epidemiology/pilot_weekly_labels.parquet` contiene **5.952 filas y
18 columnas**: conserva exactamente los 15 campos del derivado territorial y añade
`ewt`, `excess_label` y `label_eligible`. Es una copia binaria del preliminar validado,
sin recalcular ni alterar sus etiquetas. Su manifiesto adyacente documenta la promoción;
los metadatos internos originales, incluido el estado preliminar de la ejecución de
origen, se conservan para mantener la igualdad binaria.

| Control | Resultado |
|---|---:|
| Municipios / semanas por municipio | 6 / 992 |
| Elegibles / no elegibles | 4.380 / 1.572 |
| Positivas / negativas | 2.870 / 1.510 |
| Inicialización / posiciones parciales no etiquetadas | 1.566 / 6 |
| Rachas positivas / rachas de al menos 52 semanas | 271 / 20 |
| Elegibles / positivas con algún cero histórico | 569 / 530 |

Evidencias finales: `reports/epidemiology/evidence/labels_20261021/`. Los porcentajes
de clase usan solo semanas elegibles. Las rachas son secuencias descriptivas, no
episodios epidemiológicos validados. El 18,47 % de las positivas tiene algún cero
histórico y el 81,53 % no; las 20 rachas prolongadas incluyen 1.881 semanas, de las
cuales 288 tienen algún cero. Los cuartiles municipales usados para describir
referencias bajas/conteos altos se calculan con 2012–2025 y no son cortes aprobados
para entrenamiento ni nuevos umbrales epidemiológicos.

El código, datos y evidencias de esta etapa están preparados en
`feature/epidemiology-labels`; no se afirma su integración a `main` ni la creación
de un descriptor DVC o transferencia remota del nuevo Parquet.

## 3. Fuentes y limitaciones que no deben perderse

**SIVIGILA.** Se utilizaron los 19 Parquet anuales de la caché del evento 210, no la totalidad de las 69 variables de los libros originales. Los estados de confirmación son finales y retrospectivos. No se reconstruyeron secuencias de revisiones, disponibilidad histórica de cada registro ni deduplicación de personas/episodios entre años. En la auditoría se identificaron **58.876** pares con notificación anterior a síntomas, de los cuales **58.869** corresponden al archivo nominal 2013; no se corrigieron automáticamente. La selección original de 78 y la adopción de seis pilotos utilizaron información de todo el histórico, lo que condiciona la interpretación de futuras evaluaciones temporales.

**Calendario.** La referencia es `week_start`, lunes inicial de semanas **W-SUN**. La equivalencia con las semanas oficiales INS no se verificó. El último periodo comienza el **2025-12-29** y contiene **tres días** observados hasta 2025-12-31. Se **excluyó de las estadísticas de selección**, pero **se conservó en ambos datasets Parquet**. La cantidad de días del intervalo no demuestra completitud epidemiológica ni climática.

**Clima y demografía.** Temperatura del aire y humedad relativa estimada proceden de **ERA5-Land**; precipitación de **CHIRPS v2.0 Final**; población anual y densidad de **DANE/MGN 2025**. La humedad es derivada a partir de temperatura y punto de rocío diarios, no medida directa; las medias son zonales municipales, no necesariamente representativas de zonas pobladas. Población/densidad no equivalen a población oficial en riesgo. ONI/ENSO y MODIS LST **no fueron incorporados** al núcleo actual. Ninguna covariable tiene aporte predictivo probado todavía.

**Interpretación de la selección y las etiquetas.** Alta persistencia de registros no garantiza episodios independientes suficientes para pronóstico. Se caracterizaron las etiquetas municipales, pero las semanas positivas y sus rachas no son brotes oficialmente declarados. Los seis pilotos no son una muestra probabilística ni necesariamente equivalentes en dificultad predictiva. Las asociaciones entre conteos, EWT y etiquetas son parcialmente mecánicas por su propia definición; no acreditan causalidad o valor predictivo.

## 4. Contrato epidemiológico aprobado y decisiones pendientes

| Tema | Estado |
|---|---|
| Conteo de esta etapa | Se adoptó `confirmed_records_screening`, evento 210 y confirmación final; no certifica personas/episodios únicos |
| Referencia EWT | Media geométrica desplazada con ε=0,001, cinco años anteriores, incluidos años epidémicos sin excluirlos ni sustituirlos |
| Calendario | W-SUN; año del lunes inicial; ordinal de lunes desde 1; ordinal 53 ausente en un año histórico sustituido por su ordinal 52; sin redistribuir cruces de año |
| Variable objetivo | `excess_label=1` solo si `conteo > EWT`, 0 si elegible y no lo supera; no exige persistencia ni exceder un límite epidémico superior |
| Elegibilidad | Inicialización 2007–2011 y semana parcial 2025-12-29 sin etiqueta; EWT/etiqueta nulos; 2012 primer año elegible |
| Disponibilidad retrospectiva | Falta fijar fecha de emisión y reconocer retrasos/ajustes de SIVIGILA y latencias climáticas |
| Predicción supervisada | Horizonte, ventanas, rezagos y matriz de entrada aún no fijados |
| Entrenamiento y evaluación | Baselines, modelos, cortes cronológicos, métricas, calibración y umbral de decisión aún no fijados |
| Despliegue del predictor | API/frontend existen, pero todavía no hay un modelo de dengue validado conectado al producto |

El EWT aprobado es `exp(mean(log(C + 0.001))) - 0.001` para las cinco semanas
históricas alineadas del mismo municipio. Se evalúa de forma estable con
`0.001 * expm1(mean(log1p(C/0.001)))`; historias constantes e identidades enteras
exactas evitan residuos que cambiarían una igualdad. No se redondea el EWT ni se
añade tolerancia o mínimo epidemiológico. Véase `epidemiology_labels.md`.

El artículo de **Umaña et al. (2026)** (*Endemic Channel Parametrization in Dengue Surveillance*) apoya el estudio de referencias históricas y ceros; **no valida pronósticos de ML**. Las alternativas de ventanas y horizontes de la propuesta inicial no se adoptan automáticamente. El contrato vigente de cinco años y ε=0,001 se implementó por decisión explícita del equipo.

La **asignación retrospectiva del estado epidemiológico** está implementada y
caracterizada. Su **anticipación futura** sigue pendiente: no se han elegido
horizontes, características, rezagos, cortes cronológicos, modelos ni métricas.
Al construir ejemplos supervisados habrá que verificar qué era conocido a la fecha
de emisión; usar solo años anteriores para el EWT no resuelve por sí mismo los
retrasos y revisiones de SIVIGILA ni la disponibilidad del clima.

## 5. Código, versionamiento y forma de trabajo

- Git/GitHub aloja el código, pruebas, documentación y los descriptores DVC; el repositorio `MaiaMLOps/DengueAlertas` integra las etapas de datos y selección territorial mediante los **PR #3 y #4**.
- DVC referencia ambos Parquet en `data/processed/`. El contenido se almacena en el remoto WebDAV **`vps-storage`**; la transferencia del maestro fue comprobada y la del derivado está **reportada** por el equipo. No está documentado un `dvc pull` independiente desde otro equipo. Credenciales locales quedan fuera de Git.
- `src/data/` integra fuentes; `src/pilot_selection/` genera análisis territorial y derivado; `tests/` contiene comprobaciones; `docs/` y `reports/pilot_selection/evidence/` recogen metodología y evidencia. No se exige volver a generar las fuentes de origen para estudiar el desenlace.
- `src/epidemiology/` calcula y caracteriza estados semanales; `tests/epidemiology/` incluye pruebas sintéticas y regresiones locales optativas. El nuevo Parquet se preparó para DVC, todavía sin descriptor ni transferencia. Los CSV/PNG/SVG, README y manifiestos finales se prepararon para Git; las corridas temporales permanecen ignoradas. El versionamiento y la integración se realizarán manualmente después de la revisión.
- MLflow y los servicios de aplicación forman parte de la infraestructura del equipo, pero no hay aún entrenamiento/registro de un predictor epidemiológico definitivo que pueda inferirse de su existencia.
- ChatGPT apoya discusión científica y tutoría; Codex ejecuta tareas autorizadas sobre el repositorio local. El usuario supervisa las decisiones científicas, desea comprender los propósitos y flujo de los módulos y prefiere **prompts breves, implementación modular y sin sobreingeniería**.

## 6. Fuentes y prioridad de evidencia

- `docs/pilot_selection.md` y el informe **«Informe_final_seleccion_municipios_piloto_DengueAlertas.docx»**: decisión y resultados territoriales.
- `docs/data_pipeline.md`, `docs/data_dictionary.md` y los informes técnicos de **SIVIGILA** y **covariables**: construcción e interpretación de las 15 columnas.
- Umaña et al. (2026): soporte científico para estudiar la parametrización del canal endémico, sin trasladar automáticamente sus métodos a las etiquetas del proyecto.
- `Base científica y estado del arte.docx` y propuesta académica original: antecedentes; algunas opciones reflejan un alcance anterior.

Ante discrepancias, distinguir lo observado de lo inferido y usar primero las decisiones expresas más recientes del equipo y la evidencia reproducible actual. Cambios científicos importantes requieren justificación y aprobación.
