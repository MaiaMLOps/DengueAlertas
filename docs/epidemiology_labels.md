# EWT y etiquetas epidemiológicas semanales

Contrato aprobado por el equipo el 2026-10-21, implementado, probado y caracterizado.
El cierre local prepara los artefactos para su posterior versionamiento manual;
no acredita commit, integración a `main`, registro DVC ni transferencia remota.
Estas decisiones posteriores completan las etapas de integración y selección.
El producto asigna estados retrospectivos observados;
no contiene pronósticos, horizontes, rezagos, modelos ni episodios persistentes.

## Cálculo y alineación

Se utiliza `confirmed_records_screening`, por municipio. Cada semana pertenece al
año de su lunes `week_start`. Se enumeran los lunes de ese año desde uno, sin
usar numeración ISO ni dividir semanas entre años. Para la semana ordinal `w` del
año `y`, se buscan exactamente los años `y-5, ..., y-1`, conservando todos sus
conteos, incluidos los años epidémicos. Si `w=53` y el año histórico no tiene esa
semana, se utiliza su ordinal 52; no se sustituye ninguna otra semana.

El EWT es `exp(mean(log(C + 0.001))) - 0.001` sobre esos cinco conteos.
Se evalúa con la identidad `0.001 * expm1(mean(log1p(C / 0.001)))`, que evita
cancelación por sustracción cerca de cero. Las historias constantes devuelven
exactamente su conteo, incluidos cinco ceros. Para historias no constantes se
reconocen también las medias exactamente enteras mediante la identidad entera
`producto(1000*C + 1) = (1000*k + 1)^5`. Solo una igualdad exacta permite devolver
`k`; ningún umbral no entero se redondea. Esto evita falsos excesos por residuos
de logaritmo/exponencial alrededor de una igualdad matemática. No hay tolerancia
en la comparación ni umbral epidemiológico mínimo. Una anomalía numérica detiene el cálculo.

La etiqueta es uno solo cuando `conteo > ewt`; la igualdad es cero. Los años
2007–2011 inicializan la referencia. Desde 2012, solo se etiquetan semanas
completas con las cinco referencias válidas. La semana parcial 2025-12-29 se
conserva sin etiqueta; toda semana no elegible tiene ambos campos nulos.
La falta de historia se registra explícitamente; los huecos estructurales del
panel se rechazan, no se rellenan ni se corrigen.

## Componentes y producto

Producto consolidado: `data/processed/epidemiology/pilot_weekly_labels.parquet`,
**5.952 filas × 18 columnas**, seis municipios y 992 semanas por municipio.
Las 15 columnas originales tienen exactamente los tipos y valores del derivado
territorial, sin nulos. `ewt` y `excess_label` tienen cada uno 1.572 nulos
coincidentes; `label_eligible` no tiene nulos. Son 4.380 elegibles (730 por
municipio), 2.870 positivas y 1.510 negativas; las no elegibles comprenden 1.566
posiciones de 2007–2011 y seis posiciones parciales finales.

Se promovió mediante **copia binaria exacta** del preliminar validado, sin
reexportar ni recalcular el Parquet. SHA-256 de ambos:
`b3b151a54b2dc7e8c18eb78757ad6a40093626bb807757bbccb6045b354f2c7f`.
Se conservan todos los metadatos originales, incluido `preliminary_for_team_review`
de la ejecución de origen. El manifiesto adyacente `run_manifest.json` registra
el estado de consolidación y la copia; no se cambia ese metadato histórico para
preservar la identidad binaria. Diccionario completo: `data_dictionary.md`.

- `config.py`: contrato fijo y parámetros aprobados.
- `validation.py`: esquema de entrada, calendario, integridad y conservación.
- `ewt.py`: ordinal, referencias históricas y media geométrica desplazada.
- `labels.py`: añade `ewt` (float64 nullable), `excess_label` (int8 nullable) y
  `label_eligible` (bool sin nulos), conservando las 15 columnas y su orden de filas.
- `pipeline.py` y `__main__.py`: ejecución local explícita y publicación verificada
  en un directorio nuevo, sin sustituir el insumo ni otros productos.

Los metadatos documentan el método, la alineación, las advertencias y el hash del
insumo al exportar. Se conserva la regla de screening y se extiende el diccionario;
los metadatos que describían al insumo como un producto sin etiquetas no se heredan
como descripción del nuevo artefacto. El manifiesto incluye hashes de entrada,
salida, código y metadatos originales, entorno y controles agregados de elegibilidad
y referencias. No se agregan columnas auxiliares de año, ordinal o referencia.

## Verificación y ejecución

Usar Python 3.11 con `requirements/data.txt` y `requirements/data-test.txt`.
Las figuras y la caracterización requieren además `requirements/pilot-selection.txt`.
Todas estas dependencias ya estaban instaladas; no se instalaron otras. En el equipo inspeccionado, este entorno
corresponde a `python`; `.venv` contiene DVC, no las dependencias de cálculo.

```powershell
python -B -m pytest tests/epidemiology -q -p no:cacheprovider
python -B -m pytest tests/epidemiology/test_local_regression.py -q -p no:cacheprovider --pilot-parquet data/processed/pilot_municipalities_weekly.parquet
python -B -m pytest tests/epidemiology -q -p no:cacheprovider --pilot-parquet data/processed/pilot_municipalities_weekly.parquet --labels-parquet data/processed/epidemiology/pilot_weekly_labels.parquet
python -B -m src.epidemiology --output-dir data/processed/epidemiology/nueva_revision
```

Las regresiones locales son optativas. La del insumo contrasta cada EWT con una fórmula Decimal de
60 dígitos y una enumeración histórica independiente. No modifica el insumo.
La del producto consolidado contrasta su contrato, clases, tipos, hash de promoción
y valores originales; no necesita conservar una copia exploratoria del preliminar.
La exportación exige los seis pilotos y el calendario completo de 5.952 filas.
El Parquet y `run_manifest.json` quedan dentro del directorio nuevo, excluido de
Git por las reglas vigentes. El manifiesto se escribe al final tras releer y
verificar el Parquet y el hash del insumo. Una interrupción puede dejar un
directorio incompleto: no debe considerarse terminado ni reutilizarse; no se
afirma atomicidad frente a fallos abruptos del sistema.

Se mantienen las limitaciones del screening retrospectivo: confirmaciones finales,
ceros de registros que no certifican ausencia de enfermedad, anomalías originales,
falta de deduplicación certificada y selección territorial retrospectiva. La
superación de esta referencia central no equivale a un brote oficial.

## Caracterización descriptiva del producto existente

La caracterización lee el Parquet preliminar ya guardado y su manifiesto. No
ejecuta `label_weeks`, no exporta otro dataset etiquetado y no cambia elegibilidad
ni parámetros. Antes de agregar resultados, verifica los hashes, la conservación
de los campos originales, cada EWT con Decimal de 60 dígitos y cada comparación
binaria con la identidad entera exacta
`(1000*conteo_observado + 1)^5 > producto(1000*conteo_histórico + 1)`.
La tolerancia de verificación del valor flotante no se usa para asignar etiquetas.

```powershell
python -B -m pytest tests/epidemiology -q -p no:cacheprovider
python -B -m src.epidemiology.characterization_report --output-dir reports/epidemiology/runs/nueva_caracterizacion
python -B -m src.epidemiology.reference_diagnostic_report --output-dir reports/epidemiology/runs/nueva_revision_referencias
```

`characterization.py` calcula tablas globales, municipales y anuales, estadísticas
de conteos/EWT y rachas de etiquetas positivas. Los porcentajes usan solo semanas
elegibles. Las estadísticas usan desviación estándar muestral y cuantiles lineales.
Las rachas recorren todas las filas: una semana negativa, no elegible o un salto
de calendario las corta; el cambio de año no. Se conservan sus límites observados
para reconocer truncamiento por falta de observación. No son una nueva etiqueta
ni episodios epidemiológicos validados.

`characterization_plots.py` produce series sin suavizar, con fechas reales y
escalas independientes, porcentajes anuales y distribución de duraciones.
`characterization_report.py` audita el producto, exporta los CSV/PNG/SVG en un
directorio nuevo y escribe el manifiesto al final. Las corridas locales quedan
excluidas de Git mediante `/reports/epidemiology/runs/`; no se sobrescriben.

Los informes leen por defecto el Parquet consolidado y su manifiesto adyacente.
El diagnóstico de referencias contrasta también la evidencia final. Para estudiar
una ejecución alternativa se pueden pasar `--input` y `--characterization-dir`;
las rutas de salida siempre deben ser nuevas. Reconstruir la integración o
selección territorial no forma parte de esta reproducción.

## Evidencia final y resultados de conciliación

La selección final está en `reports/epidemiology/evidence/labels_20261021/`.
Su README orienta la lectura; `run_manifest.json` registra cada archivo, descripción,
origen, hash, entorno, pruebas y conciliaciones. Los CSV son copias exactas de las
ejecuciones `characterization_20261021_02` y `reference_review_20261021_01`.
Se conservaron las figuras informativas y los SVG existentes útiles para publicación;
las corridas anteriores permanecen intactas y no se trasladaron indiscriminadamente.

Las 2.870 semanas positivas forman 271 rachas; 20 duran al menos 52 semanas y
reúnen 1.881 semanas. Hay 569 elegibles con algún cero histórico, de las cuales
530 son positivas: 18,47 % de todas las positivas. En las rachas prolongadas,
288 semanas tienen algún cero; dos rachas no contienen ninguno. Los grupos de
tres/cuatro ceros son muy pequeños y no hay semanas elegibles con cinco ceros.

El diagnóstico usa exactamente los antecedentes aprobados, incluidos años de
inicialización, y verifica su alineación contra una enumeración independiente.
Los rangos de EWT bajo (≤P25 municipal) y conteo elevado (>P75 municipal) son
descriptivos, se obtienen de todas las semanas elegibles y no alteran etiquetas.
Las asociaciones son parcialmente mecánicas por la definición `conteo > EWT`;
no prueban causalidad ni valor predictivo. No se construyeron episodios nuevos.

## Versionamiento pendiente y etapa siguiente

El contenido Parquet permanece excluido de Git y se gestionará posteriormente
mediante DVC. Todavía no se creó su descriptor
`data/processed/epidemiology/pilot_weekly_labels.parquet.dvc` ni se verificó una
transferencia de este producto al remoto. `.gitignore` admite exclusivamente ese
futuro descriptor y el manifiesto adyacente, además de los CSV/PNG/SVG/JSON/README
de la evidencia final. No habilita datos originales ni corridas temporales.

Las reglas de `.gitattributes` se limitan a la evidencia final y al manifiesto
adyacente del dataset: `-text` evita conversiones automáticas LF/CRLF y conserva
los bytes de CSV/JSON/SVG/README que sustentan sus hashes. No se modifica la
política de finales de línea del resto del repositorio. Los hashes de código y
documentación son huellas del entorno de ejecución; una normalización posterior
de esos archivos debe distinguirse de una modificación del Parquet o de la evidencia.

El equipo realizará manualmente el registro Git/DVC después de revisar el cierre.
No agregar el Parquet con `git add -f`, no agregar las figuras exploratorias sueltas
y no compartir configuraciones locales o credenciales. La copia binaria final no
autoriza regenerar/sustituir productos existentes. Si se pierde el preliminar local,
el registro de promoción y las instantáneas de manifiestos conservan su procedencia.

Para ejecutar juntas las suites de integración, selección y epidemiología:

```powershell
python -B -m pytest tests/data tests/pilot_selection tests/epidemiology -q -p no:cacheprovider --import-mode=importlib --master-parquet data/processed/master_weekly_screening.parquet --pilot-parquet data/processed/pilot_municipalities_weekly.parquet --labels-parquet data/processed/epidemiology/pilot_weekly_labels.parquet
```

`--import-mode=importlib` evita la colisión entre los archivos homónimos
`test_local_regression.py` de dos suites, sin renombrarlos. Las pruebas de
publicación de `src/data/` requieren enlaces duros: el aislamiento de Codex en
Windows puede bloquearlos con `WinError 5`; en ese caso requieren ejecución fuera
del aislamiento con permiso. No se cambia la publicación segura para eludir esa
restricción ni se requieren privilegios de administrador para NTFS local normal.

En el cierre del 2026-10-21, esta ejecución conjunta terminó con **174 pruebas
aprobadas**, incluyendo las regresiones locales del maestro, pilotos y producto
consolidado. Los 18 CSV recalculados coincidieron exactamente con las corridas
validadas. La copia final mantiene 4.380 elegibles, 2.870 positivas, 1.510 negativas,
271 rachas, 20 prolongadas, 569 elegibles y 530 positivas con algún cero histórico.

Quedan pendientes fecha de emisión, horizontes, disponibilidad histórica simulada,
características y rezagos, entrenamiento, particiones y evaluación cronológica y
despliegue del predictor. La infraestructura existente no demuestra que estos
estados históricos sean predicciones ni que el predictor esté operativo.
