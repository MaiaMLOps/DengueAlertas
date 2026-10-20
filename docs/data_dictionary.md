# Diccionario del dataset maestro municipio–semana

**Fecha de comprobación:** 2026-10-20.

**Producto:** panel retrospectivo integrado bajo la regla provisional de screening;
77.376 filas y 15 columnas. Cada fila representa un municipio candidato y un
intervalo semanal. No contiene etiquetas epidemiológicas ni ejemplos supervisados.

## Evidencia y alcance de las definiciones

Este diccionario traduce y desarrolla las definiciones existentes en los metadatos
`denguealertas.data_dictionary`, `denguealertas.rule` y `denguealertas.warnings` del
[Parquet maestro](../data/processed/master_weekly_screening.parquet), contrastadas
con las secciones `data_dictionary`, `rule`, `warnings`, `panel` y `output` del
[reporte de calidad validado](../reports/data_quality/integration_quality_fixed_20261020.json).
Las tres definiciones de metadatos coinciden con las del reporte.

El reporte corresponde al archivo `master_weekly_screening_fixed_20261020.parquet`.
Su SHA-256 coincide con el del Parquet maestro inspeccionado:

```text
ccc0ab5360e29cabb8b1f5d3a0feb7fe0f00dffba195c7f55ffe83c9d0a73c5e
```

Los datos y reportes son artefactos locales excluidos de Git; estos enlaces
requieren que estén presentes en las ubicaciones indicadas. La procedencia
científica y los límites de las covariables se describen en
[contexto_metodologico.md](contexto_metodologico.md); las transformaciones de
integración, en [data_pipeline.md](data_pipeline.md). Este documento no introduce
filtros, imputaciones ni decisiones científicas nuevas.

## Esquema físico y contrato de lectura

Los tipos siguientes son los comprobados en el esquema Arrow del **Parquet**, en
su orden de almacenamiento. `double` es un número de punto flotante de 64 bits;
`date32[day]` es una fecha de calendario, sin hora ni zona horaria. Los tipos de
Pandas registrados en `panel.types` describen la tabla en memoria antes de la
exportación y pueden diferir: por ejemplo, `datetime64[ns]` frente a `date32[day]`.

| N.º | Columna | Tipo Arrow | Unidad o dominio |
|---|---|---|---|
| 1 | `municipality_code` | `string` | Código territorial de cinco dígitos, almacenado como texto |
| 2 | `municipality` | `string` | Nombre descriptivo |
| 3 | `week_start` | `date32[day]` | Fecha del lunes inicial |
| 4 | `population` | `int64` | Habitantes |
| 5 | `area_km2` | `double` | km² |
| 6 | `population_density_km2` | `double` | Habitantes/km² |
| 7 | `temperature_c` | `double` | °C |
| 8 | `precipitation_mm` | `double` | mm acumulados en el intervalo |
| 9 | `relative_humidity_pct` | `double` | Porcentaje, escala 0–100 |
| 10 | `confirmed_records_screening` | `int64` | Registros elegibles; entero no negativo |
| 11 | `no_eligible_records_observed` | `bool` | `true` / `false` |
| 12 | `week_end_nominal` | `date32[day]` | Fecha final nominal, inclusive |
| 13 | `week_end_effective` | `date32[day]` | Fecha final efectiva, inclusive |
| 14 | `n_days` | `uint8` | Días de calendario; valores observados 3 y 7 |
| 15 | `partial_week` | `bool` | `true` / `false` |

Todos los campos tienen `nullable = true` en el esquema físico. Esto **no** implica
que se acepten faltantes en el producto validado: el reporte registra cero nulos
en las 15 columnas, cero cadenas vacías en códigos y nombres y cero valores
numéricos no finitos. La integración valida esos controles; no se debe interpretar
la nulabilidad física como una regla de imputación o permiso para omitir valores.

La clave compuesta es `municipality_code` + `week_start`, única en las 77.376
filas. El orden del producto es código municipal y fecha ascendente. Los nombres
no son claves. Las nueve primeras columnas proceden del integrado
`covariates_weekly_78_2007_2025.xlsx`; las seis restantes se incorporan durante la
integración. Las covariables numéricas se conservan sin imputar ni recalcular sus
productos climáticos o demográficos de origen.

## Definiciones por columna

### 1. `municipality_code`

- **Definición y fuente:** código DIVIPOLA de cinco caracteres del municipio
  candidato en el integrado de covariables; identifica el territorio de ocurrencia
  con el que se enlazan los registros SIVIGILA.
- **Transformación:** lectura como texto y validación de cinco dígitos. En
  SIVIGILA, normalización de `COD_DPTO_O` a dos dígitos y `COD_MUN_O` a tres,
  conservando ceros iniciales, y concatenación antes de agregar y unir.
- **Interpretación:** identificador categórico, sin unidad numérica. Mantenerlo
  como texto en exportaciones. No representa municipio de residencia ni certifica
  correspondencia histórica de todos los límites territoriales.

### 2. `municipality`

- **Definición y fuente:** nombre descriptivo suministrado por el integrado de
  covariables.
- **Transformación:** conservación del nombre del integrado; no se utiliza para
  agrupar ni unir registros epidemiológicos.
- **Interpretación:** texto sin unidad. Pueden existir diferencias de escritura
  entre capas de origen; usar siempre el código para la integración.

### 3. `week_start`

- **Definición y fuente:** lunes inicial del intervalo W-SUN, presente en la
  grilla de covariables y usado como clave temporal de integración.
- **Transformación:** en SIVIGILA, `INI_SIN` se interpreta con el parser explícito
  día/mes/año y normalización AM/PM, se reduce a fecha y se asigna al lunes de su
  semana. Se conservan síntomas dentro de 2007-01-01–2025-12-31, inclusive.
- **Interpretación:** fecha, no número de semana oficial. Rango observado:
  2007-01-01–2025-12-29, con 992 inicios por municipio. `FEC_NOT`, `ANO` y `SEMANA`
  no sustituyen `INI_SIN`. No se certifica equivalencia con el calendario
  epidemiológico oficial del INS.

### 4. `population`

- **Definición y fuente:** población total anual DANE incluida en las covariables,
  expresada en habitantes.
- **Transformación:** el insumo asigna la población del año de `week_start` a sus
  semanas, sin interpolación semanal; la integración conserva el valor como
  entero de 64 bits. En semanas que cruzan año se usa el año del lunes inicial.
- **Interpretación:** la repetición semanal no constituye mediciones semanales
  independientes. No es una población en riesgo certificada por el INS. Se
  conserva la discontinuidad de Montelíbano en 2008, sin corrección automática.

### 5. `area_km2`

- **Definición y fuente:** área municipal fija del Marco Geoestadístico Nacional
  DANE 2025, en kilómetros cuadrados, suministrada por las covariables.
- **Transformación:** conservación del área fija para todas las semanas del
  municipio; no se reconstruyen áreas históricas durante la integración.
- **Interpretación:** no mide superficie urbana ni exposición efectiva al vector.
  La geometría de 2025 no demuestra estabilidad de límites a lo largo del periodo.

### 6. `population_density_km2`

- **Definición y fuente:** densidad de población del integrado de covariables,
  en habitantes por kilómetro cuadrado.
- **Transformación:** en el insumo corresponde a `population / area_km2`; la
  integración conserva sus valores y valida esa relación con tolerancia numérica.
- **Interpretación:** combina población anual con área municipal fija de 2025.
  No es densidad urbana ni exposición al vector; hereda las discontinuidades de
  población y las limitaciones territoriales del área.

### 7. `temperature_c`

- **Definición y fuente:** temperatura media semanal municipal del aire a dos
  metros de altura, ERA5-Land, suministrada por las covariables en grados Celsius.
- **Transformación:** el producto de entrada ya contiene el resumen semanal y
  municipal; la integración conserva el valor, sin repetir conversiones de unidad
  ni procesar el raster de origen.
- **Interpretación:** media sobre geometrías municipales, no observación puntual
  de estación ni temperatura exclusiva de la cabecera. La ausencia de nulos no
  certifica exactitud meteorológica ni disponibilidad histórica para pronóstico.

### 8. `precipitation_mm`

- **Definición y fuente:** acumulado de precipitación de los días disponibles del
  intervalo, resumido mediante media zonal municipal, CHIRPS v2.0 Final, en mm.
- **Transformación:** acumulación temporal y resumen espacial realizados en el
  producto de covariables; la integración conserva el valor, incluidos los ceros.
  No aplica imputación ni extrapolación a siete días.
- **Interpretación:** profundidad acumulada de precipitación, no volumen total
  municipal ni tasa diaria. La última semana cubre tres días y tiene distinta
  duración que una completa. `n_days` no verifica cobertura de cada día o píxel;
  un cero de precipitación no se convierte en valor faltante.

### 9. `relative_humidity_pct`

- **Definición y fuente:** humedad relativa estimada a partir de temperatura del
  aire y punto de rocío medios diarios ERA5-Land; expresada en porcentaje.
- **Transformación:** la covariable de origen aplica una formulación tipo Magnus
  a entradas diarias y produce el resumen semanal municipal. La integración
  conserva ese resultado; no deriva humedad a partir de medias semanales.
- **Interpretación:** estimación derivada, no medición directa ni promedio exacto
  de humedad horaria. No usar la escala 0–1 sin conversión explícita. Comparte
  los límites espaciales, de calidad y disponibilidad de los productos climáticos.

### 10. `confirmed_records_screening`

- **Definición y fuente:** número de filas SIVIGILA evento 210 retrospectivamente
  confirmadas y elegibles bajo el screening histórico provisional; unidad:
  registros, no personas o episodios únicos certificados.
- **Transformación:** filtro `confirmados = 1`, municipio de ocurrencia candidato
  y `INI_SIN` interpretable dentro de la ventana inclusiva autorizada. Se agrupa
  por código y lunes de síntomas y se suman los aportes de los archivos nominales.
  No se deduplica entre archivos, personas o episodios. Se asigna cero solo a
  claves de la grilla sin agregado elegible; un conteo nulo en una clave
  coincidente es un error, no un cero imputable.
- **Interpretación:** suma comprobada de 637.271 registros; rango observado 0–914
  por posición. No es tasa de incidencia, etiqueta de alerta ni definición final
  del desenlace científico. Los estados finales 3 o 5 asociados al filtro son
  retrospectivos: su disponibilidad en una fecha histórica de predicción no está
  demostrada. Se preserva la anomalía de fechas de notificación/síntomas de 2013,
  sin corregir ni excluir por ella. El alcance no presupone inclusión del evento 220.

### 11. `no_eligible_records_observed`

- **Definición y fuente:** indicador lógico calculado desde el conteo de screening.
- **Transformación:** `confirmed_records_screening == 0`, exactamente; se
  comprueban 24.545 valores `true` y 52.831 valores `false`.
- **Interpretación:** `true` significa que no se observaron registros elegibles
  para esa posición en la extracción, no notificación negativa verificada ni
  ausencia de infecciones. `false` indica al menos un registro elegible; tampoco
  certifica completitud de vigilancia. No constituye una etiqueta negativa/positiva
  de exceso epidemiológico.

### 12. `week_end_nominal`

- **Definición y fuente:** domingo final nominal del intervalo W-SUN, calculado
  por el calendario del pipeline; fecha inclusive.
- **Transformación:** `week_start + 6 días`.
- **Interpretación:** rango observado 2007-01-07–2026-01-04. Puede superar el fin
  autorizado y no implica que se incluyan observaciones de enero de 2026.

### 13. `week_end_effective`

- **Definición y fuente:** último día incluido del intervalo dentro de la ventana
  autorizada, calculado por el calendario; fecha inclusive.
- **Transformación:** mínimo entre `week_end_nominal` y 2025-12-31.
- **Interpretación:** rango observado 2007-01-07–2025-12-31. Es un límite de
  calendario, no fecha de notificación, confirmación, publicación o disponibilidad
  de todas las fuentes.

### 14. `n_days`

- **Definición y fuente:** número de días de calendario de la intersección del
  intervalo nominal con 2007-01-01–2025-12-31, calculado por el pipeline.
- **Transformación:** diferencia inclusive entre el fin efectivo y el máximo
  entre `week_start` y 2007-01-01: días de diferencia más uno. En esta grilla
  todos los inicios están dentro de la ventana, por lo que equivale a
  `(week_end_effective - week_start).days + 1`.
- **Interpretación:** entero sin signo de ocho bits; hay 77.298 posiciones de
  siete días y 78 de tres días. No cuenta días con notificación, cobertura diaria
  del raster ni observaciones válidas. No autoriza corregir o escalar conteos.

### 15. `partial_week`

- **Definición y fuente:** indicador lógico de intervalo con menos de siete días
  dentro de la ventana autorizada, calculado desde `n_days`.
- **Transformación:** `n_days < 7`.
- **Interpretación:** hay 78 valores `true`, uno por municipio en el intervalo que
  inicia el 2025-12-29, termina nominalmente el 2026-01-04 y efectivamente el
  2025-12-31. Las semanas se conservan sin extrapolación. Un valor `false` no
  certifica completitud de vigilancia ni de las covariables.

## Controles comprobados y límites de uso

El reporte registra 78 municipios candidatos, 992 posiciones por municipio,
77.376 claves únicas y cero filas duplicadas por clave. Las uniones son uno a uno,
sin multiplicación de filas ni pérdida de la suma de registros elegibles. Los
controles contrastan cifras históricas; no se alteran datos para hacerlas coincidir.

Los 78 candidatos se seleccionaron retrospectivamente con el periodo completo
2007–2025; no constituyen la selección científica final. La comprobación del
manifiesto y de las cachés Parquet tampoco repite la auditoría de los Excel
epidemiológicos originales.

Este diccionario describe hechos del producto validado y reglas de integración
ya autorizadas. Continúan abiertas la definición final de caso y exceso, los
municipios finales, el tratamiento analítico de semanas parciales, la disponibilidad
histórica de información, etiquetas, horizontes, rezagos y diseño de evaluación.
La completitud de esta tabla no demuestra disponibilidad prospectiva ni utilidad
predictiva de sus columnas.
