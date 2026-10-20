# Contexto metodológico y estado de decisiones — DengueAlertas

**Actualizado:** 2026-10-20.

**Proyecto:** «Alerta temprana de excesos en canales endémicos de dengue en Colombia mediante Machine Learning».

**Equipo:** proyecto de grado MAIA, Universidad de los Andes, con acompañamiento de BIOMAC.

**Estado:** integración retrospectiva inicial completada y validada. El protocolo científico supervisado permanece abierto. Consultar también `../AGENTS.md` y `data_pipeline.md`.

## 1. Autoridad y alcance vigente

Distinguir resultados comprobados en archivos o ejecuciones, decisiones explícitas del equipo, propuestas históricas y recomendaciones técnicas. Una propuesta anterior no se convierte automáticamente en una decisión vigente. Las discrepancias se documentan sin modificar reglas para forzar resultados.

El alcance científico vigente es un prototipo de predicción retrospectiva para **uno o unos pocos municipios**, con **modelos independientes por municipio**. Los 78 municipios del panel son candidatos, no la selección final. No se exige cobertura nacional, inferencia en tiempo real, adquisición periódica de SIVIGILA ni integración institucional operacional.

Un canal endémico define una referencia para interpretar estados observados. Un modelo predictivo anticipa un estado futuro utilizando solo información admisible hasta la fecha de emisión. La integración completada no demuestra capacidad de pronóstico ni disponibilidad prospectiva de sus variables.

## 2. Estado comprobado de la integración

El pipeline modular de `src/data/` valida fuentes, normaliza códigos y fechas, reconstruye los conteos del screening y los integra con covariables mediante `municipality_code` + `week_start`. No genera etiquetas, canales endémicos, rezagos, horizontes, particiones de entrenamiento ni modelos.

La ejecución validada y las repeticiones con salidas nuevas reprodujeron:

| Control | Resultado comprobado |
|---|---:|
| Parquet anuales, evento 210, años nominales 2007–2025 | 19 |
| Registros fuente/caché según manifiesto y lectura | 1.705.604 |
| Municipios candidatos / prefijos departamentales | 78 / 26 |
| Posiciones semanales por municipio | 992 |
| Filas del panel integrado | 77.376 |
| Columnas del panel integrado | 15 |
| Registros confirmados elegibles de los candidatos | 637.271 |
| Posiciones con al menos un registro elegible | 52.831 |
| Posiciones sin registros elegibles observados | 24.545 |
| Posiciones finales parciales, de tres días | 78 |
| Claves duplicadas / filas multiplicadas por las uniones | 0 / 0 |

Se comprobó equivalencia exacta del esquema y los valores de las 15 columnas entre ejecuciones, ordenando por las claves. Los hashes de las fuentes permanecieron sin cambios. La suite cuenta con **46 pruebas aprobadas**, con entradas sintéticas y regresiones de errores de escritura, publicación, rutas y conteos nulos.

El Parquet conserva las nueve columnas de covariables y añade `confirmed_records_screening`, `no_eligible_records_observed`, `week_end_nominal`, `week_end_effective`, `n_days` y `partial_week`. También se generó una copia XLSX de inspección, cuya lectura de regreso coincidió exactamente en valores, códigos, fechas, números y booleanos. Esa copia no sustituye al Parquet maestro.

Los artefactos de la ejecución validada se guardaron en `data/processed/` y `reports/data_quality/`, fuera de Git. El reporte incluye versiones, configuración, hashes de fuentes y código, reglas, perfiles de calidad, exclusiones, resultados de aceptación y checksum de salida. Los comandos, dependencias, diccionario y detalles de publicación están en `data_pipeline.md`.

## 3. Fuentes y reglas de integridad

Las rutas de las fuentes son configurables y los originales son inmutables. El pipeline requiere los 19 archivos `sivigila_210_AAAA.parquet`, `cache_manifest.csv` y `covariates_weekly_78_2007_2025.xlsx`. Los nombres describen el contrato de entrada y deben contrastarse con los archivos recibidos en cada entorno. Un clon del código no incluye las fuentes ni los derivados.

### 3.1. Epidemiología

La auditoría histórica describió 69 campos comunes en los Excel originales. Las cachés utilizadas por el pipeline contienen una proyección de **12 campos fuente**, todos almacenados como texto:

`CONSECUTIVE`, `COD_EVE`, `FEC_NOT`, `SEMANA`, `ANO`, `COD_DPTO_O`, `COD_MUN_O`, `INI_SIN`, `TIP_CAS`, `confirmados`, `Estado_final_de_caso`, `nom_est_f_caso`.

Se añaden `_year_nominal` y `_excel_row` como enteros, y `_source_file` como texto. Validar las cachés no equivale a repetir la auditoría de las 69 columnas ni a reconstruir los Excel originales.

**Regla autorizada exclusivamente para reproducir el screening histórico:**

1. Evento 210 y `confirmados = 1`, correspondiente en las fuentes comprobadas a estados finales 3 o 5.
2. Municipio de **ocurrencia**, compuesto desde `COD_DPTO_O` + `COD_MUN_O`, como texto de cinco caracteres y preservando ceros iniciales.
3. Fecha de síntomas `INI_SIN`, con parser explícito día/mes/año y normalización de marcadores AM/PM. Conservar solo fechas interpretables dentro de 2007-01-01–2025-12-31, inclusive.
4. Agrupar por municipio y semana de síntomas. Sumar nuevamente entre archivos nominales: **542 claves** recibieron aportes de más de un archivo en la ejecución validada.
5. Completar la grilla con cero únicamente cuando no exista agregado elegible para una clave. Un conteo nulo en una clave coincidente es un error de integridad y se rechaza.

Esta regla es **provisional**, no la definición científica definitiva de caso ni del desenlace.

El balance secuencial y mutuamente excluyente de la ejecución validada fue:

| Resultado del filtrado | Registros |
|---|---:|
| No confirmados | 701.166 |
| Confirmados sin fecha de síntomas | 35 |
| Confirmados con fecha no interpretable | 0 |
| Confirmados fuera de la ventana | 883 |
| Confirmados con fecha válida fuera de los candidatos | 366.249 |
| Incluidos | 637.271 |
| Total | 1.705.604 |

Advertencias que deben preservarse:

- En el conjunto completo, `INI_SIN` tiene **133 vacíos** y ningún fallo de parseo en valores no vacíos bajo el parser validado. El año nominal, `ANO` y `SEMANA` no sustituyen la fecha de síntomas.
- Hay **58.876** pares con `FEC_NOT` anterior a `INI_SIN`: **58.869** en el archivo nominal 2013 y siete en 2007. No se corrigen ni se excluyen automáticamente por esta anomalía. No interpretar esos pares como retrasos válidos de notificación.
- `CONSECUTIVE` es único dentro de cada archivo comprobado. No se certificó deduplicación entre archivos, personas o episodios; no se aplica una estrategia de deduplicación.
- Los estados finales y la confirmación se conocen retrospectivamente. No se reconstruyó qué clasificación estaba disponible en una fecha histórica de alerta.
- La ausencia de registros elegibles no demuestra notificación negativa verificada, vigilancia activa ni ausencia de infecciones.
- El alcance auditado es el evento 210; no se presupone inclusión del evento 220 ni equivalencia con todo el conjunto institucional de dengue.
- Los 78 candidatos se seleccionaron con información del periodo completo 2007–2025. Esa selección retrospectiva puede incorporar información de periodos que después se destinen a evaluación.

### 3.2. Calendario

El calendario utiliza `week_start` y periodos **lunes–domingo (W-SUN)** derivados de síntomas:

- Primer inicio: **2007-01-01**; último inicio: **2025-12-29**.
- Cada municipio conserva **991 semanas completas y una parcial**.
- La última semana tiene fin nominal **2026-01-04**, fin efectivo **2025-12-31** y **tres días**. Se conserva sin extrapolar ni completar artificialmente.
- `n_days` mide la intersección del calendario con la ventana autorizada; no certifica observaciones de vigilancia o raster para cada día.
- No se validó equivalencia exacta con la numeración epidemiológica oficial del INS.
- La población anual y los resúmenes semanales se asignan por el año del lunes inicial. Una semana que cruza año no equivale a una agrupación individual de días calendario.

El tratamiento de esta semana parcial en futuros experimentos y la fecha de corte de información disponible siguen pendientes.

### 3.3. Covariables

El integrado de entrada contiene nueve columnas comprobadas:

| Columna | Significado |
|---|---|
| `municipality_code` | Código DIVIPOLA de cinco caracteres; clave territorial |
| `municipality` | Nombre descriptivo; no utilizar para unir |
| `week_start` | Lunes inicial; clave temporal |
| `population` | Población total DANE anual asignada por año del lunes inicial |
| `area_km2` | Área municipal fija del MGN DANE 2025 |
| `population_density_km2` | Población anual / área fija |
| `temperature_c` | Media semanal municipal de temperatura del aire a 2 m, ERA5-Land, °C |
| `precipitation_mm` | Acumulado de precipitación del intervalo y media zonal municipal, CHIRPS v2.0 Final, mm |
| `relative_humidity_pct` | Humedad relativa estimada desde temperatura y punto de rocío medios diarios, ERA5-Land, % |

Se comprobaron las 77.376 claves únicas del integrado y de las cinco capas de respaldo, sin faltantes en sus columnas y con valores numéricos concordantes dentro de precisión de punto flotante. Los nombres descriptivos difieren entre algunas capas; los códigos y las fechas sostienen la unión.

Límites físicos y temporales:

- Las medias climáticas corresponden a geometrías municipales, no a estaciones ni exclusivamente a cabeceras. La ausencia de nulos no certifica exactitud meteorológica ni cobertura de cada día y píxel.
- La humedad es derivada mediante una formulación tipo Magnus aplicada a entradas diarias; no es medición directa ni promedio exacto de humedad horaria.
- La población se conserva anual, sin interpolación semanal, y no equivale necesariamente a la población en riesgo oficial del INS.
- El área MGN 2025 se mantiene fija para todo el histórico; la densidad no es densidad urbana ni exposición efectiva al vector.
- La caída poblacional de **Montelíbano en 2008, −9,30 %**, se conserva sin corrección causal no validada.
- Los ceros de precipitación son valores observados y no se sustituyen por ausencias. No se imputan covariables ni se eliminan extremos automáticamente.
- ONI/ENSO y MODIS LST no forman parte del núcleo integrado. ONI queda como extensión con disponibilidad temporal pendiente; LST fue descartada de esta versión tras pruebas de calidad/cobertura.
- La disponibilidad de estas variables no prueba su aporte predictivo ni obliga a incluirlas en un modelo por municipio.

## 4. Decisiones científicas pendientes

| Asunto | Estado y precaución |
|---|---|
| Municipios finales | Abierto: uno o pocos municipios, sin selección oportunista por desempeño en test |
| Definición final de caso | Abierta: el screening restrictivo no constituye una definición institucional universal |
| Canal endémico | Abierto: algoritmo, ventana histórica, ceros, años epidémicos, umbrales y semanas sin historial suficiente |
| Variable objetivo | Abierta: estado futuro de alerta/exceso e inicio de episodio sostenido son objetivos distintos |
| Horizonte | Abierto: las opciones históricas de una, dos y cuatro semanas no son configuración adoptada |
| Emisión y disponibilidad | Abiertas: corte de información, retrasos de notificación y publicación de productos climáticos |
| Ventanas y rezagos | Abiertos: longitud de historia y prevención de información futura |
| Predictores finales | Abiertos: historia de casos, clima y contexto poblacional deben compararse, no incorporarse por disponibilidad |
| Modelos y baselines | Abiertos: comparación parsimoniosa y defendible bajo condiciones idénticas |
| Cortes y métricas | Abiertos: particiones cronológicas, métricas principales y eventos suficientes en evaluación |
| Umbral de decisión | Abierto: no fijarlo con test ni interpretar scores como probabilidades calibradas sin evaluación |

Las propuestas históricas de horizontes `h = 1, 2, 4`, ventanas del canal de 3/5/7 años, *minimal shift* `ε = 0,001`, rachas sostenidas y arquitecturas DLNM/INLA, boosting, LSTM o ST-GNN no deben implementarse automáticamente. Las posiciones sin confirmados tampoco equivalen a etiquetas negativas de exceso.

## 5. Principios para etapas posteriores

- Mantener separados el panel integrado, las etiquetas epidemiológicas y la matriz de ejemplos supervisados.
- Toda característica utilizada en una predicción emitida en `t` debe estar disponible en `t` bajo una convención aprobada. Evitar conteos o clima de la semana objetivo y estadísticas aprendidas con datos futuros.
- Usar validación temporal. Ajustar imputaciones, escaladores, selección de variables, calibración y parámetros aprendidos exclusivamente dentro del entrenamiento correspondiente.
- Calcular umbrales de referencia sin años posteriores a la fecha relevante. Conservar las limitaciones de ajustes retrospectivos de SIVIGILA y de selección territorial.
- Entrenar estimadores independientes por municipio dentro del alcance vigente. Comparar baselines y grupos de covariables con idéntico target, horizonte y cortes, una vez aprobados.
- Registrar resultados por municipio, horizonte y periodo, con prevalencia, eventos positivos, falsas alertas, errores en picos/transiciones e incertidumbre cuando corresponda.

## 6. Ingeniería, versionamiento y servicios

El código, las pruebas, la documentación y las dependencias son versionables en Git. Los datos individuales, derivados, reportes y temporales permanecen excluidos. Los artefactos actuales describen una integración retrospectiva validada, no un modelo supervisado.

El pipeline rechaza sobrescrituras, verifica hashes de fuentes y prepara Parquet/JSON en temporales antes de publicarlos. Requiere soporte de enlaces duros para publicación exclusiva y utiliza un reporte de éxito con checksum como señal de finalización. No existe una transacción indivisible entre ambos archivos ante una terminación abrupta; los detalles y la recuperación están en `data_pipeline.md`.

El seguimiento mediante DVC está pendiente de configuración en esta etapa. Antes de incorporarlo deben acordarse los artefactos y su provisión, revisar las exclusiones para permitir los metadatos necesarios y definir la reproducción conservando protección contra sobrescrituras. La disponibilidad remota de DVC o MLflow no se presume por una configuración reportada; no se ha comprobado en esta integración. No se han registrado experimentos ni promovido modelos.

El repositorio incluye frontend y backend, cuyo contrato debe inspeccionarse cuando una tarea explícita los involucre. No modificar API, frontend, CI/CD, secretos, infraestructura, remotos o despliegues como efecto lateral del trabajo de datos. Commits, pushes, publicaciones y operaciones remotas requieren autorización específica; nunca imprimir o persistir credenciales.

## 7. Referencias y prioridad de evidencia

Las referencias se consultan en la ubicación configurada, sin alterar sus originales. Fuentes principales:

- `Informe_integral_auditoria_SIVIGILA_2007_2025_consenso_78.docx`: auditoría, filtros, calendario y selección de candidatos.
- `Informe_final_covariables_semanales_78_municipios_2007_2025.docx`: origen, transformaciones y límites de las covariables.
- `Endemic Channel Parametrization in Dengue Surveillance.pdf`: referente sobre parametrización del canal; no certifica pronóstico ML.
- `Base científica y estado del arte.docx` y propuestas académicas: antecedentes sujetos al alcance vigente y a las decisiones posteriores del equipo.

Ante discrepancias prevalecen: instrucciones expresas y recientes del usuario/equipo; evidencias de ejecución y archivos inspeccionados; informes técnicos; propuestas anteriores. Toda decisión científica nueva debe registrar fecha, justificación, responsable y efectos sobre reproducibilidad, etiquetas y evaluación.
