# Selección territorial y evidencia descriptiva reproducible — DengueAlertas

**Decisión del equipo:** 2026-10-20 (hora de Colombia).  
**Estado:** selección de seis municipios piloto **terminada**; código, evidencia y dataset derivado incorporados a `main` mediante **PR #4**, fusionado el 2026-10-21 01:12 UTC (`5cae9f8a`).  
**Referencia de cierre:** `Informe_final_seleccion_municipios_piloto_DengueAlertas.docx`. Consultar también `contexto_metodologico.md` y `data_dictionary.md`.

## 1. Propósito y decisión territorial

Se redujo el conjunto retrospectivo de **78 municipios candidatos** a seis territorios para un prototipo académico de alerta temprana, con **modelos posteriores independientes por municipio**. La selección es **intencional y explícita del equipo**. No se obtuvo de una función de ranking, un clasificador, un modelo de predicción ni una estratificación oficial de transmisión.

| Municipio | DIVIPOLA | Actividad de semanas completas 2007–2022 (%) |
|---|---|---:|
| Cúcuta | `54001` | 98,08 |
| Barranquilla | `08001` | 97,96 |
| Neiva | `41001` | 97,72 |
| Villavicencio | `50001` | 96,77 |
| Cali | `76001` | 96,65 |
| Soledad | `08758` | 93,41 |

Los cinco primeros pertenecían al grupo de mayor continuidad del periodo 2007–2022; **Soledad** se incorporó tras revisión descriptiva mediante una **decisión expresa del equipo**. La selección favorece series persistentes y distintas trayectorias temporales, no pretende generalización nacional ni demuestra desempeño predictivo.

La lista adoptada está codificada una sola vez como `PILOTS` en `src/pilot_selection/config.py`. No es una regla automática para modificar la selección durante una ejecución. Una revisión posterior de estos pilotos sería una nueva decisión científica.

## 2. Datos de partida y convenciones

El insumo es `data/processed/master_weekly_screening.parquet`, reconstruido previamente desde registros del evento 210 de SIVIGILA y covariables ERA5-Land, CHIRPS y DANE. Contiene **77.376 filas, 15 columnas, 78 municipios y 992 semanas por municipio** entre 2007 y 2025. El conteo `confirmed_records_screening` usa la regla retrospectiva provisional de registros con `confirmados = 1`.

La clave de una posición es `municipality_code` (texto DIVIPOLA de cinco dígitos) + `week_start` (lunes de inicio). La convención semanal es **lunes–domingo (W-SUN)**, sin equivalencia verificada con la numeración epidemiológica oficial. Los ceros indican falta de registros elegibles en el extracto, **no** ausencia certificada de dengue.

**Diferencia de tratamientos:**

- Para el **análisis descriptivo**, se excluyó la semana parcial iniciada el 2025-12-29 (tres días). Se examinaron **991 semanas completas por municipio**, **77.298 posiciones** y **636.865 registros elegibles** en el conjunto de 78.
- En el **dataset derivado de seis municipios**, se conservaron todas las **992 semanas** de cada uno, incluyendo seis posiciones parciales, para preservar un subconjunto fiel del maestro. El tratamiento posterior de esa semana para las etiquetas sigue pendiente.

Los años de los resúmenes corresponden al año del lunes inicial; no se divide una semana en dos años. La sensibilidad histórica filtra semanas iniciadas entre 2007 y 2022, con **835 semanas por municipio**. No es una partición de entrenamiento/prueba.

## 3. Cómo se realizó la selección

La etapa se desarrolló como análisis descriptivo reproducible, sin introducir hipótesis de brotes no verificadas:

1. **Perfilamiento de 78 municipios:** volumen total, semanas activas (conteo > 0), proporción de semanas activas, mediana semanal incluidos ceros, perfiles anuales, rachas máximas sin registros y concentración en el año de mayor volumen bruto.
2. **Contraste de heterogeneidad:** cinco municipios con mayor persistencia, cinco cercanos a la mediana y cinco con menor persistencia; las comparaciones no equivalen a una clasificación epidemiológica.
3. **Lectura de series:** examen de incrementos y trayectorias semanales con fechas reales, sin suavizado, con escalas verticales propias para cada municipio.
4. **Sensibilidad 2007–2022:** recálculo independiente de los indicadores sobre el periodo histórico anterior a 2023 para distinguir continuidad persistente de la influencia de picos recientes.
5. **Adopción del equipo:** selección explícita de seis pilotos y producción de un subconjunto territorial del maestro. No se generaron etiquetas ni modelos.

Reglas relevantes: `active_pct` es la proporción de semanas con registros elegibles; `weekly_median` incluye semanas con cero; `max_zero_run` puede atravesar cambios de año y exige calendario sin huecos; `peak_concentration_pct` mide el porcentaje del volumen total concentrado en el año de mayor conteo bruto. La comparación de percentiles usa interpolación lineal y los desempates son deterministas, con DIVIPOLA como referencia. Ninguno de estos indicadores mide directamente incidencia ajustada por población ni clasifica brotes.

## 4. Resultados descriptivos

**Universo de 78, solo semanas completas:**

| Indicador | Resultado |
|---|---:|
| Posiciones / registros elegibles | 77.298 / 636.865 |
| Mediana municipal de semanas activas | 639 de 991 |
| Rango de actividad semanal entre municipios | 45,11 %–98,39 % |
| Rango de mayor racha sin registros elegibles | 2–52 semanas |
| Municipios con máximo anual bruto en 2023–2025 | 54 de 78 |
| Municipios con máximo anual bruto en 2024 | 34 de 78 |

La Figura 1 del informe final muestra la distribución de años de máximo volumen, concentrada en años recientes, especialmente 2024. Esto motivó la comprobación de sensibilidad histórica. La **Figura 2, página 5**, reúne las curvas semanales de Cúcuta, Barranquilla, Neiva, Villavicencio, Cali y Soledad, conservando escalas Y independientes: sus alturas no deben compararse entre municipios sin observar los ejes.

**Seis pilotos, 991 semanas completas por municipio:**

| Municipio | Registros elegibles | Actividad total (%) | Actividad 2007–2022 (%) | Mediana semanal 2007–2022 | Mayor racha sin registros 2007–2022 |
|---|---:|---:|---:|---:|---:|
| Cúcuta | 30.489 | 98,39 | 98,08 | 19 | 2 |
| Barranquilla | 39.758 | 98,28 | 97,96 | 16 | 3 |
| Neiva | 27.717 | 98,08 | 97,72 | 11 | 2 |
| Villavicencio | 25.684 | 97,28 | 96,77 | 14 | 3 |
| Cali | 93.894 | 97,17 | 96,65 | 39 | 9 |
| Soledad | 13.739 | 94,45 | 93,41 | 5 | 5 |

El periodo histórico anterior a 2023 conserva **más del 93 %** de semanas activas en los seis municipios; los primeros cinco superan el 96 %. Entre los pilotos varían los años máximos y la concentración de volumen, sin que eso constituya evidencia de distinta frecuencia de etiquetas de exceso.

La exploración consideró alternativas como Bucaramanga, Guadalajara de Buga, Villeta y Puerto Boyacá. La elección de Soledad en lugar de otra alternativa fue una **decisión de alcance**, no una afirmación de superioridad de su vigilancia o pronosticabilidad.

## 5. Implementación, evidencia y trazabilidad

**Módulo:** `src/pilot_selection/`, independiente de `src/data/`. Contiene `config.py`, `analysis.py`, `plots.py`, `reporting.py`, `subset.py`, `__main__.py` y utilidades. Pruebas en `tests/pilot_selection/`; dependencias en `requirements/pilot-selection.txt`.

Desde la raíz del repositorio y con las dependencias correspondientes, la exploración puede ejecutarse sin sobrescribir salidas existentes:

```bash
python -m src.pilot_selection --input data/processed/master_weekly_screening.parquet --output-dir reports/pilot_selection/runs/nueva_exploracion --figures
```

La evidencia de cierre se conserva en `reports/pilot_selection/evidence/selection_20261020/`:

| Archivos | Contenido |
|---|---|
| `municipal_indicators.csv`, `annual_profiles.csv`, `distribution_summary.csv` | Indicadores de los 78, perfiles anuales y distribución de estadísticas |
| `comparison_15.csv`, `sensitivity_top10.csv`, `sensitivity_comparison.csv` | Contrastes de persistencia y sensibilidad histórica |
| `peak_year_distribution.csv`, `selected_pilots.csv` | Distribución de años máximos y resultados de los seis pilotos |
| `weekly_comparison_6.png`, `weekly_comparison_6.svg` | Figura comparativa de seis series semanales |
| `run_manifest.json` | Procedencia, reglas, entorno y hashes de entradas/resultados |

Los CSV conservan DIVIPOLA textual al releerse (`dtype={"municipality_code": str}`). El análisis rechaza calendarios y esquemas incompatibles y no modifica las fuentes. La evidencia versionada permite reproducir el cálculo y contrastar los resultados. Las pruebas reportadas en las dos fases incluyen **80 aprobaciones** (integración + selección) y luego **52** en la suite específica de selección y derivado; son ejecuciones con pruebas superpuestas, **no 132 pruebas distintas**.

## 6. Dataset de seis municipios, DVC y regeneración

**Producto:** `data/processed/pilot_municipalities_weekly.parquet`. Descriptor versionado: `data/processed/pilot_municipalities_weekly.parquet.dvc`.

El derivado filtra **solo** por `municipality_code ∈ PILOTS`; conserva los **15 campos originales**, ordenados por municipio y `week_start`, con iguales tipos Arrow y valores del maestro. No reconstruye SIVIGILA, no interpola clima, no imputa ni genera variables objetivo. Los metadatos Parquet documentan la derivación, el hash del maestro, la lista de códigos y las advertencias específicas del alcance.

| Control | Valor documentado |
|---|---:|
| Municipios | 6 |
| Semanas por municipio | 992 |
| Filas / columnas | 5.952 / 15 |
| Confirmados elegibles en total | 231.406 |
| Confirmados en 5.946 semanas municipales completas | 231.281 |
| Confirmados en seis posiciones de la semana parcial | 125 |
| Nulos / claves duplicadas | 0 / 0 |
| Tamaño del Parquet | 201.592 bytes |

El descriptor DVC está en `main` desde el **PR #4**. La transferencia a `vps-storage` está reportada en el informe final; **no consta una comprobación de descarga independiente desde otro computador**. El archivo de datos permanece fuera de Git; se requiere DVC y acceso autorizado al remoto para recuperarlo. No almacenar contraseñas en Git.

Regeneración local (solo cuando la salida no existe):

```bash
python -m src.pilot_selection.subset
```

Comprobaciones sintéticas y de regresión contra el maestro disponible:

```bash
python -m pytest tests/pilot_selection/test_subset.py -q
python -m pytest tests/pilot_selection/test_subset.py -q --master-parquet data/processed/master_weekly_screening.parquet
```

La implementación valida claves, fechas, seis municipios, 992 semanas, tipos, valores, semana parcial, metadatos y conservación del maestro. Rechaza sobrescrituras. No garantiza transacción indivisible ante corte abrupto durante la escritura; requiere ejecución finalizada y verificaciones aprobadas para certificar un nuevo archivo.

## 7. Limitaciones y estado científico posterior

- **Sesgo de selección retrospectiva:** tanto los 78 candidatos como la selección de seis pilotos se hicieron con conocimiento de todo el periodo 2007–2025; la sensibilidad hasta 2022 no constituye selección prospectiva ni una partición de evaluación.
- **Persistencia no es suficiencia de positivos:** las semanas con registros no indican cuántas semanas superarán un umbral de canal endémico que aún no se ha definido.
- **Naturaleza de los datos:** confirmaciones finales, ceros de registro, no deduplicación entre episodios y anomalías de fechas 2013; los aumentos representados no son automáticamente brotes.
- **Tiempo y validez externa:** W-SUN no se acreditó como calendario INS; se desconoce disponibilidad histórica exacta de conteos/clima; la muestra intencional no representa nacionalmente a Colombia.
- **Siguiente desarrollo científico:** definir y justificar una referencia endémica local y la variable objetivo, distinguir estado observado de pronóstico, caracterizar etiquetas por municipio y posteriormente establecer horizontes, rezagos, particiones temporales y modelos independientes. Ninguno de esos parámetros ha sido fijado por esta etapa.

La selección territorial **está cerrada como decisión vigente del equipo** para el prototipo. No debe recalcularse automáticamente por una métrica o por el desempeño de futuros modelos; cualquier cambio requiere justificación explícita.
