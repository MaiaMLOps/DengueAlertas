# Integración retrospectiva municipio–semana

Esta implementación reproduce exclusivamente el screening histórico autorizado:
evento 210, `confirmados = 1`, territorio de ocurrencia (`COD_DPTO_O` +
`COD_MUN_O`), fecha de síntomas `INI_SIN` y ventana inclusiva
2007-01-01–2025-12-31. La regla es provisional y no define el desenlace final.
No construye etiquetas, canales endémicos, rezagos, ventanas predictivas,
particiones temporales ni modelos.

## Ejecución local

Desde la raíz del repositorio, en un entorno con las dependencias de
`requirements/data.txt`:

```powershell
python -m src.data --sivigila-dir "../Datos SIVIGILA" --covariates "../Covariables/covariates_weekly_78_2007_2025.xlsx" --output "data/processed/master_weekly_screening.parquet" --report "reports/data_quality/integration_quality.json"
python -m pytest tests/data -q
```

Las versiones utilizadas están fijadas en `requirements/data.txt` y
`requirements/data-test.txt`; el reporte registra las versiones efectivas.
No se descargan dependencias ni se contactan servicios remotos al ejecutar.
Las rutas relativas se resuelven desde la raíz del repositorio, incluso si
la función se invoca desde otro directorio. Las cuatro rutas se pueden pasar
por CLI o mediante `PipelineConfig`.

Las entradas pueden estar en otra unidad de Windows o en un recurso UNC. Cuando
no existe una ruta relativa al repositorio, el reporte guarda un alias
`external:sivigila_dir` o `external:covariates`, junto con nombres y hashes de las
fuentes. No guarda la ruta absoluta privada. Para reproducir esa ejecución se
deben suministrar las rutas locales correspondientes mediante CLI; el alias
descriptivo no es por sí mismo una ruta de lectura.

No se sobrescriben archivos existentes. Para repetir una ejecución conservando
la anterior, elegir nuevos nombres **tanto para el Parquet como para el JSON**,
por ejemplo `data/processed/master_weekly_screening_v2.parquet` y
`reports/data_quality/integration_quality_v2.json`. Las salidas deben permanecer
en `data/processed/` y `reports/data_quality/`, respectivamente. Los insumos son
inmutables; sus hashes se verifican antes de exportar.

## Módulos

| Módulo en `src/data/` | Responsabilidad |
|---|---|
| `config.py` | Rutas, regla histórica fija y cifras esperadas de aceptación |
| `normalization.py` | Códigos territoriales y parser explícito de fechas |
| `calendar.py` | Grilla W-SUN y cobertura efectiva del intervalo |
| `sources.py` | Manifiesto, esquemas, hashes y agregación anual/entre archivos |
| `validation.py` | Perfiles agregados, calendario, covariables, claves y uniones |
| `pipeline.py` | Orquestación, trazabilidad, reporte y escritura segura |
| `artifacts.py` | Preparación temporal, lectura de verificación y publicación sin reemplazo |
| `__main__.py` | Interfaz de ejecución local |

Se valida primero el esquema de los 19 Parquet usando metadatos. Después se
procesa un archivo anual a la vez, conservando únicamente sus agregados semanales
para la unión entre años nominales. No se copian microdatos al repositorio.

Las exclusiones epidemiológicas son secuenciales y mutuamente excluyentes:
no confirmado; confirmado sin fecha; confirmado con fecha no interpretable;
confirmado fuera de ventana; confirmado con fecha válida fuera de los candidatos;
incluido. Su suma debe coincidir con las filas de entrada. Ningún registro se
descarta por anomalías entre notificación y síntomas o por identificador repetido.
Los valores territoriales malformados, categorías inesperadas y defectos
estructurales detienen la ejecución; no se reparan silenciosamente.

Los candidatos se toman del integrado de covariables autorizado. Se exige su
grilla completa y se contrastan los 78 municipios con el criterio de aceptación.
Los nombres descriptivos no participan en la unión. No se imputan covariables ni
se eliminan sus extremos. Sus valores se conservan, salvo conversión de tipos
sin alteración numérica.

La integración rechaza explícitamente conteos nulos, incluso con tipos enteros
nullable. El cero se asigna únicamente a claves de covariables sin agregado
epidemiológico correspondiente; un conteo faltante en una clave coincidente es
un error de integridad, no ausencia de registros.

## Producto y significado de las columnas

El Parquet conserva las nueve columnas del integrado de covariables y añade:

| Columna | Significado |
|---|---|
| `confirmed_records_screening` | Número entero de registros confirmados finales elegibles bajo la regla provisional; no personas/episodios únicos certificados |
| `no_eligible_records_observed` | Verdadero si el conteo anterior es cero; no verifica notificación negativa ni ausencia de infecciones |
| `week_end_nominal` | Domingo final nominal, inclusive |
| `week_end_effective` | Fin nominal limitado por 2025-12-31, inclusive |
| `n_days` | Días del intervalo dentro de la ventana autorizada; no mide completitud diaria de vigilancia o del raster |
| `partial_week` | Verdadero cuando `n_days < 7` |

Las fechas se exportan como Arrow `date32`; los códigos siguen siendo texto de
cinco caracteres. Los conteos son `int64`. La última semana conserva inicio
2025-12-29, fin nominal 2026-01-04, fin efectivo 2025-12-31 y tres días, sin
extrapolación. El orden final es código municipal y fecha.

El diccionario completo, la regla y las advertencias se incluyen en el reporte
y en los metadatos del Parquet. No hay supuestos de disponibilidad prospectiva
de confirmaciones o covariables.

## Controles y reporte

Se exige: 78 municipios; 992 semanas por municipio; 77.376 posiciones;
637.271 registros elegibles; 52.831 posiciones positivas en conteo;
24.545 sin registros elegibles; 78 posiciones parciales de tres días;
claves únicas y uniones uno a uno sin multiplicación ni pérdida de conteos.

Estas cifras son comprobaciones, no instrucciones para alterar las entradas.
Si difieren, se genera un JSON de fallo con las observaciones y la ejecución
termina con código distinto de cero **antes de exportar el panel**. Las fallas
estructurales se documentan hasta la etapa alcanzada. No se serializan valores
individuales ni mensajes de lectores que pudieran contenerlos.

La inicialización de referencias, versiones y trazabilidad también está protegida
por el reporte de errores. Las comprobaciones previas de destinos siguen
rechazando sobrescrituras o ubicaciones no autorizadas sin crear un reporte que
reemplace uno existente. Si tampoco puede guardarse el reporte de fallo, la CLI
termina con error e identifica esa limitación sin divulgar el texto de errores
externos.

### Publicación y recuperación ante fallos

El Parquet y el JSON se preparan en archivos temporales únicos dentro de sus
directorios de destino. Se cierran y sincronizan; el Parquet se relee y se compara
con la tabla validada. El JSON completo se serializa antes de publicar cualquiera
de los dos archivos definitivos. La publicación utiliza enlaces duros en el mismo
sistema de archivos, con creación exclusiva: nunca sustituye un destino existente.
Se requiere un sistema de archivos que admita enlaces duros, como NTFS local;
si no los admite, la ejecución falla sin recurrir a una escritura directa del
nombre final.

Un ejecutor aislado puede prohibir la creación de enlaces duros aunque NTFS la
admita. En esta revisión ese bloqueo se observó en el aislamiento de Codex; las
pruebas de publicación se ejecutaron con permiso explícito fuera del aislamiento.
Ese permiso del ejecutor no equivale a requerir privilegios de administrador para
la ejecución normal en un directorio local con permisos suficientes.

Se publica primero el dataset completo y después el reporte. Un reporte con
`status = passed` y checksum coincidente identifica el par terminado. Si la
publicación del reporte falla durante una ejecución controlada, se retira el
dataset publicado por esa ejecución y se intenta guardar un reporte de fallo.
Se limpian los temporales y se preservan los archivos preexistentes.

Dos rutas no constituyen una transacción indivisible del sistema de archivos.
Una terminación abrupta del proceso o un fallo de alimentación puede dejar
temporales o un dataset completo sin su reporte de éxito. Ese artefacto no debe
considerarse una ejecución terminada: requiere revisión y recuperación manual,
sin sobrescribir ni borrar automáticamente archivos preexistentes. No se afirma
durabilidad transaccional frente a fallos del sistema o del almacenamiento.

### Pruebas de regresión

Además de las pruebas de reglas históricas, calendario y uniones, la suite cubre
rutas entre unidades/UNC y referencias no representables; fallos de inicialización;
conteos nulos y relleno limitado a claves ausentes; interrupciones de escritura
Parquet/JSON; discrepancias en la lectura de verificación; fallos al publicar el
reporte; preservación de archivos existentes; y limpieza de temporales ante
errores controlados. Todos los casos utilizan entradas sintéticas.

El reporte contiene inventario y hashes SHA-256; versiones de dependencias;
commit, rama y estado de trabajo local; hashes del código del pipeline;
configuración sin rutas personales absolutas; perfiles de tipos, claves,
nulos y rangos; exclusiones por archivo; anomalías temporales; aportes entre
archivos; conciliación de uniones; controles de aceptación; resumen por municipio;
checksum y lectura de comprobación del Parquet generado.

Los derivados y reportes JSON quedan excluidos de Git. El código, pruebas,
dependencias y esta documentación son versionables. `.gitignore` es una
protección preventiva; no impide que alguien fuerce manualmente un agregado.

## Límites y decisiones pendientes

Se preservan la anomalía temporal de 2013, los ceros observados, el salto
poblacional de Montelíbano y las áreas MGN 2025 fijas. La unicidad interna de un
identificador no certifica deduplicación entre archivos, personas o episodios;
no se aplica deduplicación. Tampoco se certifica equivalencia del calendario
con semanas oficiales INS ni calidad meteorológica por ausencia de nulos.

Los 78 son candidatos seleccionados retrospectivamente. Siguen pendientes la
definición final de caso, municipios finales, uso analítico de semanas parciales,
disponibilidad histórica de información y todo el diseño supervisado. Esta salida
es un panel retrospectivo con conteos de screening, no una matriz supervisada.
