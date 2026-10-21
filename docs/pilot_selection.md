# Selección territorial y evidencia descriptiva reproducible

**Decisión comunicada por el equipo:** 2026-10-20. **Estado:** la etapa descriptiva
se incorporó al commit local `10047de`; el procedimiento de derivación territorial
descrito abajo se prepara para revisión independiente, sin nuevo versionamiento.

## Propósito y alcance

El equipo adoptó seis municipios piloto después de explorar descriptivamente los
78 candidatos del panel retrospectivo: Cúcuta (`54001`), Barranquilla (`08001`),
Neiva (`41001`), Villavicencio (`50001`), Cali (`76001`) y Soledad (`08758`). La
decisión territorial es explícita y actualiza, **solo para este asunto**, el estado
abierto descrito en el [contexto metodológico](contexto_metodologico.md) y el
[diccionario](data_dictionary.md), anteriores a esta decisión. Las demás decisiones
científicas pendientes siguen abiertas.

El criterio comunicado por el equipo fue considerar persistencia, suficiencia
histórica, variabilidad temporal y factibilidad de un prototipo retrospectivo con
modelos independientes por municipio. Es una justificación descriptiva y de alcance
académico: no se aprobó un umbral numérico de suficiencia ni una función que
optimice la selección. El código no infiere ni recomienda la lista final.

`src/pilot_selection/` consume únicamente
`data/processed/master_weekly_screening.parquet`. No importa ni ejecuta el pipeline
`src/data/`, no reconstruye fuentes, no corrige registros y no genera canales
endémicos, etiquetas, matrices supervisadas ni modelos. Los cálculos generales
conservan los 78 municipios; `PILOTS` en `config.py` controla la tabla específica,
la figura final y el nuevo filtro territorial. Es la única lista de configuración
de los seis municipios.

## Definiciones y calendario

La unidad es una posición municipio–semana; la clave es `municipality_code` como
texto de cinco dígitos, junto con `week_start`. Los nombres son descriptivos.
`confirmed_records_screening` cuenta registros retrospectivamente elegibles bajo
el screening provisional del evento 210: no certifica personas o episodios únicos.
La definición final de caso del estudio no se adopta en este componente.

En los **análisis descriptivos** se excluyen las filas `partial_week = True`.
Esta exclusión no se aplica al dataset territorial descrito más adelante.
La semana del 2025-12-29, de tres
días, permanece intacta en el maestro pero no participa en los resultados.
Quedan 991 semanas completas por municipio, con inicios entre 2007-01-01 y
2025-12-22 y fin efectivo hasta 2025-12-28. Son intervalos lunes–domingo (W-SUN),
sin equivalencia certificada con la numeración epidemiológica oficial del INS.

Los años son los del lunes `week_start`; no se reparten las semanas entre años.
Hay 53 semanas en 2007, 2012, 2018 y 2024, 51 en 2025 y 52 en los demás años.
`n_days` describe duración de calendario, no completitud de vigilancia.

| Indicador | Definición |
|---|---|
| `total_records` | Suma de registros del periodo |
| `n_weeks` | Número de semanas completas del periodo |
| `active_weeks` / `active_pct` | Semanas con conteo > 0; porcentaje sobre `n_weeks` |
| `weekly_median` | Mediana de los conteos semanales, incluidos los ceros |
| `annual_active_median` / `annual_active_min` | Mediana y mínimo de semanas activas anuales; se incluyen años con cero actividad |
| `max_zero_run` | Mayor bloque consecutivo de conteos cero en el periodo |
| `records_per_week` | Total anual / semanas completas de ese año; no es incidencia poblacional |
| `active_proportion` | Semanas activas anuales / semanas completas anuales, en escala 0–1 |
| `peak_year` / `min_year` | Año de mayor / menor volumen bruto; se presenta el más temprano ante empate |
| `peak_years` / `min_years` | Todos los años empatados, separados por `\|` |
| `peak_concentration_pct` | 100 × mayor total anual / total del periodo |

La racha se calcula ordenando fechas: el contador aumenta con cada cero y vuelve
a cero con un conteo positivo. Puede cruzar años y se corta en los límites del
periodo. Los huecos de calendario se rechazan; no se convierten en ceros.
Con total cero, la concentración y el año máximo se dejan indefinidos; el CSV
utiliza campos vacíos. Un cero no demuestra ausencia de infecciones ni notificación
negativa verificada. La persistencia tampoco certifica calidad de vigilancia.

Los resúmenes entre municipios usan mínimo, P25, mediana, P75 y máximo con
interpolación lineal; los conteos originales pueden producir percentiles
fraccionarios. No se redondean los datos analíticos para ordenar ni calcular.

La comparación de 15 contiene cinco mayores porcentajes activos, cinco menores
distancias absolutas al porcentaje mediano y cinco menores porcentajes. Como las
series tienen igual denominador, se usan conteos activos para evitar efectos del
punto flotante. DIVIPOLA ascendente resuelve los empates. Si los grupos se
solaparan, se informa el problema sin inventar una regla alternativa; en el maestro
actual son 15 municipios distintos. Esta comparación es una muestra descriptiva,
no el procedimiento que decide los pilotos.

## Exploraciones y sensibilidad retrospectiva

La exploración previa examinó volumen, actividad, rachas, distribución anual,
concentración y trayectorias semanales. La implementación conserva esas reglas;
las figuras exploratorias anteriores no se sobrescriben.

Para comprobar la influencia de 2023–2025, cada municipio se vuelve a resumir
después de filtrar **su propio subconjunto** con años de `week_start` 2007–2022.
Son 835 semanas por municipio. No se usan estadísticas del periodo completo para
los indicadores históricos. La comparación posterior muestra ambos resultados y
diferencias histórico menos completo; en `active_pct` la diferencia está en puntos
porcentuales. `recent_share_pct` es el porcentaje del volumen completo que
corresponde a semanas iniciadas en 2023–2025.

La última semana histórica comienza 2022-12-26 y termina 2023-01-01. El Parquet
semanal no permite separar ese día: el corte es por inicio semanal, no por cada
día de síntomas. Este periodo es una sensibilidad retrospectiva, **no una
partición de entrenamiento o evaluación**.

## Resultados observados y decisión del equipo

Los siguientes valores corresponden al maestro actual; no son constantes para
construir o ajustar los datos.

| Control | Resultado |
|---|---:|
| Candidatos | 78 |
| Posiciones / registros en semanas completas | 77.298 / 636.865 |
| Perfiles municipio–año | 1.482 |
| Posiciones / registros históricos | 65.130 / 384.008 |
| Municipios con máximo anual bruto en 2024 | 34 |
| Posiciones / registros de los seis pilotos, periodo completo | 5.946 / 231.281 |

| Municipio | DIVIPOLA | Registros completos | Actividad completa (%) | Actividad histórica (%) | Mediana semanal histórica | Racha histórica |
|---|---|---:|---:|---:|---:|---:|
| Cúcuta | `54001` | 30.489 | 98,39 | 98,08 | 19 | 2 |
| Barranquilla | `08001` | 39.758 | 98,28 | 97,96 | 16 | 3 |
| Neiva | `41001` | 27.717 | 98,08 | 97,72 | 11 | 2 |
| Villavicencio | `50001` | 25.684 | 97,28 | 96,77 | 14 | 3 |
| Cali | `76001` | 93.894 | 97,17 | 96,65 | 39 | 9 |
| Soledad | `08758` | 13.739 | 94,45 | 93,41 | 5 | 5 |

**Observación:** los seis conservan más del 93 % de semanas activas antes de
2023; los primeros cinco superan el 96 %. La reducción respecto al periodo
completo es de 0,30–0,53 puntos en esos cinco y 1,04 puntos en Soledad. Las rachas
máximas de los seis son iguales en ambos periodos. Sus años máximos históricos
son diferentes: 2010 en Cúcuta y Cali, 2019 en Neiva y Villavicencio, y 2022 en
Barranquilla y Soledad.

**Interpretación descriptiva del equipo:** la persistencia observada no depende
exclusivamente de los incrementos recientes y ofrece una base de factibilidad para
explorar predicción retrospectiva. Las diferencias de volumen y trayectoria
permiten estudiar contextos temporales distintos. Esto respalda el alcance elegido,
pero no demuestra suficiencia de eventos positivos futuros, estabilidad de un
desenlace aún no definido ni mejor desempeño predictivo.

La exploración también mostró alternativas con continuidad histórica, por ejemplo
Bucaramanga. La tabla histórica de diez municipios se conserva como evidencia de
que la decisión territorial no es la salida automática de un ranking.

## Limitaciones

- Los candidatos iniciales y la selección del equipo usan información retrospectiva
  del periodo completo. La sensibilidad no elimina ese sesgo ni convierte el
  diseño en una selección prospectiva. La evaluación posterior deberá explicitarlo.
- Los incrementos de registros no son brotes o excesos epidemiológicos formalmente
  identificados. No se construyó una referencia endémica ni un umbral.
- Volumen y promedios semanales no ajustan por población ni miden riesgo de infección.
  La concentración por año usa volumen bruto, con años de distinta cantidad de semanas.
- Los estados finales SIVIGILA son retrospectivos; no está demostrada su
  disponibilidad en fechas históricas de predicción ni la de las covariables.
- Se conserva la anomalía documentada de notificación/síntomas en 2013; este
  componente no dispone de microdatos para reinterpretarla o corregirla.
- Siguen pendientes definición final de caso y exceso, target, horizontes,
  disponibilidad, rezagos, cortes de evaluación, baselines y métricas. La selección
  territorial no resuelve esos acuerdos.

## Ejecución desde la raíz

Python 3.11 y dependencias fijadas; el entorno debe contar con ellas. El Python del
sistema inspeccionado las tiene; `.venv` no las tenía en esta revisión. Para
preparar un entorno, sin mezclarlo con las dependencias de serving:

```powershell
python -m pip install -r requirements/pilot-selection.txt -r requirements/data-test.txt
```

Ejecución temporal completa (elegir un nombre nuevo si ya existe):

```powershell
python -m src.pilot_selection --input data/processed/master_weekly_screening.parquet --output-dir reports/pilot_selection/runs/exploration_v1 --figures
```

Sin `--figures` se generan solamente las tablas y el manifiesto. Las rutas se
resuelven desde la raíz del repositorio. No se accede a servicios remotos.

Evidencia preparada para revisión en esta etapa:

```powershell
python -m src.pilot_selection --input data/processed/master_weekly_screening.parquet --output-dir reports/pilot_selection/evidence/selection_20261020 --figures
```

El comando rechaza un directorio existente. Todas las validaciones, cálculos y
serializaciones terminan antes de crear la salida. Cada archivo se crea en modo
exclusivo y se verifica por checksum. El manifiesto se escribe al final: un
directorio sin él es una ejecución incompleta, que requiere inspección y otro
nombre de salida. No existe una transacción indivisible entre todos los archivos
ni recuperación automática frente a una interrupción abrupta.

## Artefactos y trazabilidad

Los enlaces siguientes apuntan a la evidencia descriptiva incorporada al commit
local `10047de`; no incluyen el nuevo Parquet territorial:

| Archivo | Filas / contenido |
|---|---|
| [municipal_indicators.csv](../reports/pilot_selection/evidence/selection_20261020/municipal_indicators.csv) | 156 filas: 78 municipios × dos periodos |
| [annual_profiles.csv](../reports/pilot_selection/evidence/selection_20261020/annual_profiles.csv) | 1.482 perfiles del periodo completo |
| [distribution_summary.csv](../reports/pilot_selection/evidence/selection_20261020/distribution_summary.csv) | Ocho indicadores × dos periodos |
| [comparison_15.csv](../reports/pilot_selection/evidence/selection_20261020/comparison_15.csv) | 15 filas, grupos y desempates reproducibles |
| [sensitivity_top10.csv](../reports/pilot_selection/evidence/selection_20261020/sensitivity_top10.csv) | 10 mayores porcentajes activos históricos |
| [sensitivity_comparison.csv](../reports/pilot_selection/evidence/selection_20261020/sensitivity_comparison.csv) | 78 comparaciones entre periodos |
| [peak_year_distribution.csv](../reports/pilot_selection/evidence/selection_20261020/peak_year_distribution.csv) | 19 años completos + 16 históricos, incluidos años sin máximos |
| [selected_pilots.csv](../reports/pilot_selection/evidence/selection_20261020/selected_pilots.csv) | Seis municipios configurados × dos periodos, en orden de la figura |
| [weekly_comparison_6.png](../reports/pilot_selection/evidence/selection_20261020/weekly_comparison_6.png) | 3840 × 2430 píxeles, 300 dpi |
| [weekly_comparison_6.svg](../reports/pilot_selection/evidence/selection_20261020/weekly_comparison_6.svg) | Figura vectorial equivalente |
| [run_manifest.json](../reports/pilot_selection/evidence/selection_20261020/run_manifest.json) | Insumo, versiones, configuración, controles y hashes |

La figura replica el diseño horizontal aprobado: grilla 3 × 2, orden Cúcuta,
Barranquilla, Neiva, Villavicencio, Cali y Soledad, colores sobrios distintos,
fechas reales, escalas Y propias, lineales y con origen cero. Cada línea conserva
los conteos originales, sin suavizado ni reducción de puntos. Los trazados del SVG
son vectoriales; las tipografías se convierten en trazados para facilitar su
portabilidad. Las escalas independientes deben leerse al comparar magnitudes.

CSV UTF-8, punto decimal, sin índice y con precisión de doble flotante; campos de
texto entre comillas. Para evitar inferencia numérica de DIVIPOLA al releer:

```python
pd.read_csv(path, dtype={"municipality_code": str})
```

El manifiesto registra el SHA-256 del insumo y de cada artefacto, hashes del código
y dependencias, pruebas, documentación y exclusiones, versión de Python y bibliotecas, commit de base, rama y estado
local sucio, periodos, selección explícita, reglas, perfiles agregados y
reconciliación de cada serie dibujada. No contiene rutas privadas absolutas ni
filas semanales exportadas. Un commit de base no identifica por sí solo código
sin commit; los hashes del componente permiten rastrear esa diferencia.

`runs/` está excluido de Git y contiene pruebas/ejecuciones temporales. Solo los
CSV agregados de `evidence/selection_20261020/` tienen una excepción acotada a la
exclusión general de CSV. Esa evidencia, sus figuras, manifiesto, código, pruebas
y documentación están propuestos para revisión y posterior versionamiento manual;
el pipeline no ejecuta `git add`, commits ni publicaciones. Las seis figuras
exploratorias anteriores permanecen sin alteraciones.

## Pruebas y regresión

Pruebas sintéticas ordinarias:

```powershell
python -m pytest tests/pilot_selection -q
```

Regresión local explícita; si el archivo solicitado falta, falla en vez de omitirla:

```powershell
python -m pytest tests/pilot_selection -q --master-parquet data/processed/master_weekly_screening.parquet
```

Las pruebas cubren ceros, total cero, rachas entre años y en extremos, fechas,
semanas parciales, denominadores anuales, tipos, DIVIPOLA textual, datos inválidos,
empates, percentiles y orden de la comparación. Alterar 2023–2025 debe dejar
idénticos los indicadores históricos. La exportación verifica lectura de códigos,
repetibilidad numérica, hashes, ausencia de rutas privadas y rechazo de
sobrescritura. Las pruebas gráficas verifican las filas, fechas, colores y orden
realmente dibujados, seis ejes lineales independientes, PNG a 300 dpi y SVG sin
imágenes raster incrustadas. No se exige identidad binaria entre versiones de
bibliotecas o sistemas de tipografía.

Las regresiones conocidas se mantienen **solo en pruebas**, nunca como valores
para forzar el cálculo. Incluyen todos los controles anteriores, la lista de 15,
los diez históricos y rachas de los 78 contrastadas con un cálculo independiente
por bloques. Cualquier diferencia debe investigarse; no se imputan ni alteran
datos para hacerla desaparecer.

La verificación local de esta implementación aprobó **80 pruebas**: 34 del nuevo
componente, incluidas las regresiones explícitas contra el maestro, y las 46
existentes de integración. El sandbox bloqueó inicialmente seis pruebas existentes
que publican archivos sintéticos mediante enlaces duros; la suite completa aprobó
al repetirse con autorización fuera del aislamiento. No se cambió `src/data/`
para eludir esa restricción. El componente descriptivo y la generación final se
ejecutan dentro del aislamiento y no requieren enlaces duros.

## Dataset derivado exclusivamente territorial

`data/processed/pilot_municipalities_weekly.parquet` es un subconjunto del maestro
para los seis municipios adoptados por el equipo. Su propósito es disponer del
panel de esos territorios para las siguientes etapas; **no es un dataset
supervisado ni contiene una variable objetivo epidemiológica**. Sigue siendo
retrospectivo, bajo el screening provisional SIVIGILA evento 210. Las definiciones
de sus 15 variables son las del [diccionario del maestro](data_dictionary.md).

La única transformación de filas es `municipality_code` perteneciente a los
códigos de `PILOTS` en `src/pilot_selection/config.py`. Después se ordena por código
y `week_start`. Se trabaja directamente con tablas Arrow: no se convierte a
Pandas para escribir, no se recastean columnas ni se imputan valores. Se conservan
las 15 columnas en su orden original, los tipos y nulabilidad de sus campos y
todos los valores, incluidos ceros. No se agregan etiquetas, ventanas, rezagos,
características, canales endémicos ni una selección automática.

**Se incluyen las 992 semanas de cada municipio**, también la última semana
parcial que comienza el 2025-12-29 y termina efectivamente el 2025-12-31. Por tanto,
se esperan 5.952 filas, seis municipios y seis posiciones parciales. Es distinto
de las 5.946 posiciones completas utilizadas en la figura y las tablas anteriores.
No se extrapolan los tres días finales a una semana completa.

Comando desde la raíz, con las dependencias existentes de `requirements/data.txt`:

```powershell
python -m src.pilot_selection.subset
```

Forma explícita equivalente:

```powershell
python -m src.pilot_selection.subset --input data/processed/master_weekly_screening.parquet --output data/processed/pilot_municipalities_weekly.parquet
```

El comando rechaza cualquier salida existente, también si apareció después de la
comprobación inicial. Para repetirlo sin sobrescribir, indicar un nombre nuevo
mediante `--output`, siempre dentro de `data/processed/`. No ejecuta operaciones
Git o DVC, ni accede a remotos. El seguimiento DVC del derivado queda pendiente de
autorización; el Parquet permanece local y excluido de Git por las reglas actuales.

Procedencia almacenada **dentro del Parquet**, sin archivo auxiliar:

- `denguealertas.derivation`: ruta relativa y SHA-256 del maestro, códigos leídos
  de `PILOTS`, autoridad de selección, operación, versión Arrow y perfiles agregados
  de fuente y derivado: filas, claves, municipios, semanas, nulos, tipos, rangos y
  totales. No incorpora rutas privadas absolutas.
- `denguealertas.data_dictionary`: definiciones originales, sin cambios.
- `denguealertas.rule`: campos de la regla de screening independientes del alcance
  territorial. No se trasladan campos agregados o arbitrarios de alcance del maestro.
- `denguealertas.warnings`: advertencias específicas del derivado. Se elimina la
  advertencia antigua de que los municipios finales siguen pendientes. No se copia
  el resto de metadatos globales del maestro, incluido su índice Pandas.

Antes de escribir se comprueban el esquema y el calendario del maestro, la
presencia de todos los pilotos, la ausencia de duplicados y nulos, las 992 fechas
conservadas y una posición parcial por municipio. Se contrasta la igualdad exacta
de campos, tipos y valores con el subconjunto del maestro. El Parquet se codifica
y relee primero en memoria y luego se crea de forma exclusiva, se sincroniza y
se relee desde disco. La igualdad incluye los nuevos metadatos al comprobar la
lectura; al conciliar con el maestro se exceptúan únicamente sus metadatos globales
y el orden de almacenamiento. El SHA-256 del maestro se compara antes de preparar
y al terminar de guardar. La salida de consola informa controles, tamaño y hashes.

No se garantiza publicación transaccional frente a una interrupción abrupta: un
fallo durante la escritura puede dejar un archivo incompleto. Solo una ejecución
terminada con código cero y conciliación aprobada certifica la exportación. No se
eliminan ni sobrescriben automáticamente archivos tras un fallo.

Pruebas sintéticas del procedimiento y regresión opcional contra el maestro real:

```powershell
python -m pytest tests/pilot_selection/test_subset.py -q
python -m pytest tests/pilot_selection/test_subset.py -q --master-parquet data/processed/master_weekly_screening.parquet
```

Cubren igualdad exacta del subconjunto, códigos como texto, tipos, orden, semanas
parciales, procedencia, sustitución de metadatos de alcance, lectura de `PILOTS`,
datos inválidos, fuente intacta, fallos de codificación y rechazo de sobrescritura
incluso ante creación concurrente de la salida. La regresión real no necesita
escribir una segunda copia del maestro.

**Comprobación ejecutada en esta ampliación (2026-10-20):** el derivado tiene
5.952 filas y 15 columnas, exactamente seis códigos, 992 semanas por municipio,
seis posiciones parciales, cero duplicados y cero nulos. Suma 231.406 registros:
231.281 en semanas completas y 125 en la parcial. Ocupa 201.592 bytes. Los valores
y campos Arrow coinciden exactamente con las filas del maestro, salvo los
metadatos globales reemplazados según la política anterior. El maestro conservó
su SHA-256 `ccc0ab5360e29cabb8b1f5d3a0feb7fe0f00dffba195c7f55ffe83c9d0a73c5e`.
La ejecución de `tests/pilot_selection/` con regresión real aprobó **52 pruebas**:
34 anteriores del componente y 18 nuevas del filtrado. No fue necesario modificar
ni volver a ejecutar el pipeline de integración.

El cambio de alcance territorial no resuelve la definición final del desenlace ni
la disponibilidad prospectiva de los registros. Persisten las limitaciones de
ceros, deduplicación y anomalía temporal de 2013 documentadas para el maestro.

## Fuentes documentales

Este componente se apoya en [AGENTS.md](../AGENTS.md),
[contexto_metodologico.md](contexto_metodologico.md),
[data_dictionary.md](data_dictionary.md) y [data_pipeline.md](data_pipeline.md),
además de la decisión territorial explícita del equipo comunicada en esta etapa.
No implica una nueva auditoría de las fuentes originales ni una verificación de
sus referencias científicas externas.
