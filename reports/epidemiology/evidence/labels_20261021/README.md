# Evidencia final de etiquetas epidemiológicas de DengueAlertas

Cierre local del 2026-10-21, preparado en feature/epidemiology-labels para revisión
y posterior versionamiento manual. No implica commit, integración en main,
registro DVC ni transferencia remota del Parquet epidemiológico.

Dataset: data/processed/epidemiology/pilot_weekly_labels.parquet, 5.952 x 18,
copia binaria del preliminar validado. Su run_manifest.json adyacente documenta
la promoción; los metadatos internos originales se conservan sin modificaciones.
SHA-256: b3b151a54b2dc7e8c18eb78757ad6a40093626bb807757bbccb6045b354f2c7f.

## Lectura de los archivos

- labels_*: distribución global, municipal y anual; porcentajes solo sobre elegibles.
- statistics_*: EWT y conteos elegibles; desviación estándar muestral y cuantiles lineales.
- positive_runs.csv, run_summary_* y run_duration_distribution.csv: las 271 rachas.
- zeros_*: 0-5 antecedentes iguales a cero; grupos vacíos no se presentan como EWT cero.
- municipal_quartile_cutoffs.csv, quartile_grid_* y low_reference_high_count_combinations.csv:
  rangos descriptivos de 2012-2025; no son umbrales epidemiológicos ni cortes de entrenamiento.
- long_runs_detail.csv y long_runs_annual_profiles.csv: las 20 rachas >=52 semanas.
- Figuras: casos/EWT, distribución anual, duraciones, ceros y revisión de rachas largas.
  Se conservaron los SVG existentes útiles para publicación. La vista histórica completa
  se conserva en SVG; se omitió su PNG redundante y los README exploratorios.

El manifiesto inventaría cada archivo, su descripción, origen, tamaño y SHA-256.
Incluye instantáneas de los tres manifiestos de referencia; no requiere trasladar
sus directorios temporales para interpretar la procedencia.

## Conciliaciones y límites

4.380 elegibles; 2.870 positivas; 1.510 negativas; 1.572 no elegibles.
271 rachas positivas; 20 de al menos 52 semanas, con 1.881 semanas.
569 elegibles y 530 positivas con algún cero histórico. En las rachas largas,
288 semanas tienen algún cero. Las rachas no son brotes oficiales o episodios validados.
Se preservan ceros de registro, confirmaciones retrospectivas y anomalías originales.
Las asociaciones EWT/conteo/etiqueta son parcialmente mecánicas; no prueban causalidad.

## Reproducción y versionamiento pendiente

Consultar docs/epidemiology_labels.md para cálculo, pruebas y caracterización.
Los informes leen el Parquet consolidado y el manifiesto adyacente; generan una
carpeta nueva en reports/epidemiology/runs/, sin sobrescribir esta evidencia.
La suite conjunta se ejecutó con --import-mode=importlib: 174 pruebas aprobadas.
Las pruebas existentes de publicación requieren enlaces duros; en Codex Windows
se ejecutaron fuera del aislamiento con permiso, tras observar WinError 5 dentro.

Los CSV/PNG/SVG/README/manifiestos son elegibles para Git. El contenido Parquet
permanece ignorado y su descriptor DVC se creará manualmente tras la revisión.
No incluir credenciales, corridas temporales ni figuras exploratorias sueltas.
Las reglas acotadas de .gitattributes conservan los bytes de esta evidencia y del
manifiesto del dataset frente a conversiones LF/CRLF; mantenerlas al versionar.
El siguiente trabajo científico es definir el problema predictivo y la evaluación
cronológica; este cierre no contiene horizontes, rezagos, modelos ni despliegue.
