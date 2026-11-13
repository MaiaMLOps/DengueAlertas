# Guía metodológica del pronóstico

Guía para entender, reproducir y auditar el MVP de pronóstico con solo el repositorio. Resume y enlaza; no sustituye a los documentos vinculantes:

| Documento | Papel |
|---|---|
| [`contrato_predictivo.md`](contrato_predictivo.md) | Qué se construye y cómo se evalúa. Vinculante; sus requisitos están en §10 |
| [`especificacion_implementacion.md`](especificacion_implementacion.md) | Cómo se implementa: módulos, fases, pruebas de aceptación, configuración y trazabilidad |
| [`registro_decisiones.md`](registro_decisiones.md) | Por qué: alternativas y motivo de cada decisión D0–D10 y sus enmiendas |
| [`desviaciones.md`](desviaciones.md) | Lo que cambió durante la implementación (DV-01 a DV-03) |
| [`api_salida_d10.md`](api_salida_d10.md) | Esquema de salida de la API |
| [`verificacion/`](verificacion/) | Cálculos independientes con los que se obtuvieron y contrastaron las cifras |
| [`analisis_resultados.md`](analisis_resultados.md) | Análisis integrado de los resultados de la prueba (contrato §8.8): calidad de la estimación, municipios, eventos, robustez y valor práctico. Lo exploratorio va marcado |
| [`../../reports/forecasting/entregables/`](../../reports/forecasting/entregables/) | Versiones en Word y presentación del análisis, una carpeta por corrida |
| [`../../reports/forecasting/INDICE.md`](../../reports/forecasting/INDICE.md) | Mapa de datos, evidencia, figuras y configuración, con su estado |

Esta guía no reporta resultados. Las cifras están en la evidencia de `reports/forecasting/evidence/` y se localizan con `INDICE.md`; su lectura integrada está en `analisis_resultados.md`.

## 1. Problema y pregunta

El canal endémico es la herramienta habitual de vigilancia del dengue: compara los casos de una semana con una banda calculada con los cinco años anteriores. Describe la semana observada, pero no anticipa. El proyecto pregunta:

> **¿Qué tan probable es que el municipio esté en zona epidémica dentro de 8 semanas, y el sistema está alertando?** (contrato §9.2)

- **Variable objetivo (D3):** y_{t,8} = 1[C_{t+8} > U1e_{t+8}], donde C es el conteo de registros confirmados y U1e el límite superior del canal de epiCo (D1.1). Es una clasificación binaria con salida probabilística.
- **Unidad de análisis (D0):** seis municipios piloto, cada uno con su propio modelo.
- **Dos capas de evaluación (contrato §2.4 y §7.1):**
  - *estimación*: ¿es buena la probabilidad semanal? Se juzga con reglas de puntuación propias (log-loss, Brier) frente a las líneas base B2 y B1;
  - *decisión*: ¿sirven las alertas que produce la regla con τ? Se juzga con inicios de brote, frente a la alerta del propio canal.
- **Principio rector:** todo lo que decide (variable objetivo, información disponible, referencias, métricas y criterios) se fijó antes de ver el desempeño de cualquier modelo. La prueba 2023–2025 se abrió una sola vez, con la receta ya congelada.

## 2. Secuencia de trabajo

Cada fase tiene pruebas de aceptación que se escribieron y congelaron antes de implementarla (§5). Las fases se dividieron en unidades de trabajo (U*x.y*), cada una con uno o dos requisitos del contrato y un commit.

| Fase | Entregable | Decisiones |
|---|---|---|
| F0 | Entorno, dependencias, configuración validada y conexión a MLflow | Especificación §2, §8 y §9 |
| F1 | Canal de epiCo y producto del canal con DVC | D1, D1.1, D2, D3 |
| F2 | Ejemplos supervisados y 15 variables, sin fuga | D4, D6 |
| F3 | Cortes por año, líneas base y barrera de prueba | D5, D7, D7.1 (DV-01) |
| F4 | Módulo de eventos | D8 (§7.4) |
| F5a | Arnés de búsqueda y reglas de comparación | D7 (§6.2–§6.3), D9 |
| ◆ | Pre-registro de la búsqueda | D7 (§6.3), especificación §7 |
| F5b | Búsqueda en desarrollo | D5, D7 |
| F6 | Selección y receta congelada | D7 (§6.2), D8 (§7.5), D9 (§8.3) |
| F6b | Conjunto de variables como paso 0 de la selección | D7.2 (DV-02) |
| F7 | Prueba única, ablación, ficha, historial, modelos finales y diagnóstico | D7 (§6.1), D8, D9 (§8.4), D9.1, D10 (§9.4) |
| U7.5 | Reportes derivables y figuras | D8 (§7.2–§7.4), D9 (§8.4) |
| F8 | Registro de modelos, API, frontend | D10 (pendiente) |

**F0, preparación.** Antes de escribir código metodológico había que fijar lo que lo condiciona: versión de Python y de las bibliotecas, la versión del cliente de MLflow igual a la del servidor (el mecanismo de autenticación usa una API privada de MLflow) y una configuración que se valida al cargarse, para que ningún parámetro del contrato quede escrito en el código.

**F1, canal y producto (D1, D1.1, D2, D3).** El objetivo se define frente al canal de epiCo, la parametrización publicada del canal endémico, y no frente a la etiqueta EWT de la etapa anterior. El canal se construye en un producto nuevo, versionado con DVC, a partir del mismo panel de seis pilotos, para que la etiqueta de legado siga intacta y reproducible. Las pruebas de F1 comparan el producto con el legado en conteos, clima y elegibilidad.

**F2, ejemplos y variables (D4, D6).** t es la última semana completa y h = 8 la distancia hasta la semana objetivo. Las 15 variables describen el estado y la dinámica frente al canal, la semana del año objetivo y anomalías climáticas en bloques mensuales. La prueba de truncamiento calcula las variables con la base completa y con la base recortada en t y exige igualdad exacta: es la que detecta una fuga temporal.

**F3, cortes, líneas base y barrera (D5, D7, D7.1).** Cada año evaluado es un ensayo con reentrenamiento: se entrena solo con filas cuya semana objetivo es anterior a la primera emisión del ensayo (D7.1, motivada en DV-01), es decir, con las etiquetas que ya existían ese día. B2 (probabilidad según la zona actual del canal) y B1 (estacionalidad de la semana objetivo) se estiman igual, para que toda comparación sea pareada. La barrera (`guard.py`) impide puntuar semanas de 2023–2025 sin una receta congelada válida.

**F4, eventos (D8, §7.4).** La utilidad operativa se mide con inicios de brote (Umaña et al., 2026), y modelo y canal pasan por el mismo código: misma regla de alerta, misma fecha de confirmación, misma ventana y misma censura en los bordes del periodo.

**F5a, arnés y reglas de comparación (D7, D9).** Antes de correr la búsqueda real se probó el arnés y las reglas que deciden: conteo de victorias pareadas por casilla municipio-año, con umbrales de una prueba binomial exacta de una cola, y puntajes fuera de año (una configuración nunca se elige con el año en que se puntúa).

**Pre-registro (D7, §6.3).** Candidatos, rejillas, presupuesto y semillas quedaron fijados en `configs/forecasting.yaml` antes de ver un solo puntaje de desarrollo, para que el número de candidatos no se ajustara a los resultados.

**F5b, búsqueda (D5, D7).** Se ajustaron los candidatos pre-registrados de regresión logística y LightGBM en los cuatro ensayos de desarrollo, con las líneas base en las mismas filas. La tabla única de puntajes es la fuente de verdad de la selección; MLflow es el registro navegable.

**F6, selección y receta (D7, D8, D9).** El orden es fijo: configuración común de hiperparámetros, comparación de familias con parsimonia, excepciones por municipio, y τ con el criterio de §7.5 (máxima detección de inicios con falsas alarmas no mayores que las del canal). La receta resultante se congela con su hash y su aprobación.

**F6b, conjunto de variables (D7.2, DV-02).** Al preparar la ablación del clima, el conjunto sin clima resultó mejor en desarrollo, y D6 no permitía considerarlo. Se enmendó el contrato antes de abrir la prueba: el conjunto de variables pasa a ser el paso 0 de la selección, con *sin clima* por defecto. La receta anterior quedó sustituida sin haber puntuado 2023–2025 y se conserva como registro.

**F7, prueba única y productos (D7, D8, D9, D9.1, D10).** Con la receta aprobada, se evaluó la prueba una sola vez y se registró en la bitácora. Del mismo cálculo salen la ablación (la familia final con el otro conjunto de variables), la ficha de desempeño, el historial fuera de muestra para la API, los modelos finales entrenados con todas las filas y los insumos del análisis diagnóstico por régimen (D9.1). Las pruebas `test_f7b_*` sustituyen a las de F7 que citaban la receta anterior (DV-02).

**U7.5, reportes derivables.** El orden de recorte de la especificación (§5) permitía dejar para después todo lo que se calcula con predicciones ya guardadas. Estos reportes (confiabilidad, Murphy, clasificación en τ, eventos, curva tipo AMOC y figuras) no reentrenan, no repiten la prueba y no cambian ninguna decisión. Sus fórmulas y cifras se congelaron en `test_f7c_*`.

## 3. Módulos y flujo de datos

### 3.1 Qué lee y qué escribe cada módulo

Los módulos sin línea de comandos no deciden dónde escribir: reciben tablas o rutas de quien los llama. Los que puntúan filas llaman primero a la barrera de prueba.

| Módulo | Responsabilidad | Lee | Escribe |
|---|---|---|---|
| `src/epidemiology/channel_epico.py` | Canal de epiCo: central, gsd, U1e y zonas; canal de cualquier semana (`channel_for_week`) y semanas de referencia (`reference_weeks`) | Panel en memoria; usa el calendario de `ewt.py` sin modificarlo | — |
| `src/epidemiology/ewt.py`, `labels.py`, `pipeline.py`, `__main__.py` | **Legado:** EWT y `excess_label` de la etapa epidemiológica | `pilot_municipalities_weekly.parquet` | `pilot_weekly_labels.parquet` y `run_manifest.json` |
| `src/epidemiology/characterization*.py`, `reference_diagnostic*.py` | **Legado:** caracterización y diagnósticos de la etiqueta EWT | Producto de legado | `reports/epidemiology/` |
| `src/epidemiology/config.py`, `validation.py` | Constantes de D1 (primer año elegible) y validaciones (`require`) | — | — |
| `src/forecasting/config.py` | Carga y valida la configuración; sobrescribe h con `FORECAST_HORIZON` | `configs/forecasting.yaml` | — |
| `src/forecasting/build_channel.py` | Línea de comandos del producto del canal | Configuración, panel de pilotos | `pilot_weekly_channel.parquet` y su manifiesto |
| `src/forecasting/dataset.py` | Verifica el md5 del canal; ejemplos (t, t + h) y 15 variables; entrada de la GRU | Producto del canal | — |
| `src/forecasting/splits.py` | Ensayos por año con t0 y filas de entrenamiento y evaluación | Ejemplos | — |
| `src/forecasting/guard.py` | Barrera de prueba: valida la receta (estructura, hash, versión del canal) en cada llamada | `configs/receta_congelada.yaml` | — |
| `src/forecasting/baselines.py` | B1 y B2 con suavizado de Laplace, por ensayo | Ejemplos y ensayos | — |
| `src/forecasting/models.py` | Pipelines de regresión logística y LightGBM; ajuste y predicción | Filas del ensayo | — |
| `src/forecasting/metrics.py` | Log-loss, Brier, AP, Murphy, confiabilidad, calibración por año, clasificación en τ | Predicciones | — |
| `src/forecasting/events.py` | Alertas, inicios, entradas, emparejamiento, censura y métricas de eventos | Emisiones y serie observada | — |
| `src/forecasting/comparison.py` | Puntajes fuera de año, victorias, umbrales binomiales, McNemar y niveles de conclusión | Tabla única | — |
| `src/forecasting/tracking.py` | Conexión a MLflow con los encabezados de Cloudflare Access o almacén local | `.env` (sin imprimir valores) | Runs de MLflow |
| `src/forecasting/search.py` | Búsqueda de desarrollo, reanudable por unidad | Configuración, canal | `data/processed/forecasting/` o `sin_clima/`: tabla única, predicciones, líneas base, `busqueda_manifest.json`; runs en `dengue_h8_desarrollo` |
| `src/forecasting/selection.py` | Pasos 0–4 de la selección, τ y borrador de receta | Tabla única y predicciones | `receta_borrador.yaml` (a través de `run_selection`) |
| `src/forecasting/run_selection.py` | Línea de comandos de la selección; ejecuta la selección dos veces y exige el mismo resultado | Salidas de las búsquedas y sus `.dvc` | Evidencia `seleccion_*` (CSV, JSON y borrador) |
| `src/forecasting/evaluate.py` | Prueba única: modelo, ablación, B2 y B1 en las mismas filas; resumen de §7 y §8.4 | Receta, configuración, canal, bitácora | `prueba/`: predicciones, ensayos, casillas, resumen, `diagnostico_prueba/`; entrada en `bitacora_prueba.jsonl`; runs en `dengue_h8_prueba` |
| `src/forecasting/artifacts.py` | Ficha de desempeño, historial fuera de muestra y modelos finales | Salidas de la prueba, receta, búsqueda | `prueba/`: `ficha_desempeno.json`, `historial*.parquet`, `modelos_finales/` |
| `src/forecasting/diagnostics.py` | Régimen de cada casilla y los seis insumos de §8.8 | Receta, canal, búsqueda (pasar la del conjunto de la receta, `--busqueda data/processed/forecasting/sin_clima`; el valor por defecto es la búsqueda completa) | `diag_*.csv` |
| `src/forecasting/reports.py` | Reportes derivables de desarrollo y prueba | Búsquedas, `prueba/`, receta, canal | Tablas CSV y `manifiesto_reportes.json` |
| `src/forecasting/figures_reports.py` | Figuras PNG y SVG de los reportes | Tablas de `reports.py` | `reports/forecasting/figures/<fecha>/` |

### 3.2 Flujo

```
pilot_municipalities_weekly.parquet (DVC)
  └─ build_channel ──► pilot_weekly_channel.parquet (DVC, md5 en configs/forecasting.yaml)
       └─ dataset.build_examples (h = 8, 15 variables) ──► splits.yearly_trials (D7.1)
            ├─ search (desarrollo 2019–2022) ──► tabla única + predicciones (DVC) ──► MLflow dengue_h8_desarrollo
            │    └─ run_selection / selection ──► receta_borrador.yaml ──► aprobación ──► configs/receta_congelada.yaml
            │         └─ diagnostics (desarrollo, «usado para elegir») ──► evidencia diag_*.csv
            └─ guard (receta válida) ──► evaluate (prueba 2023–2025, una vez) ──► prueba/ (DVC) + bitácora + MLflow dengue_h8_prueba
                 ├─ artifacts ──► ficha, historial, modelos_finales/ (DVC) ──► API (F8)
                 └─ reports + figures_reports ──► reports/forecasting/evidence/ y figures/ (git)
```

La dependencia va de `src/forecasting/` a `src/epidemiology/`, nunca al revés. `src/forecasting/` no lee el producto de legado `pilot_weekly_labels.parquet`; solo las pruebas de aceptación de F1 lo leen para compararlo con el producto nuevo.

## 4. Reproducibilidad

- **Datos con DVC.** Cada Parquet y cada carpeta de salida tiene su descriptor `.dvc` en git con el md5 del contenido (`.dir` cuando es una carpeta). `dvc pull` trae exactamente esos bytes. El producto del canal se verifica además al cargarse: su md5 debe coincidir con `datos.version_dvc_canal` de `configs/forecasting.yaml`.
- **Semillas.** Una semilla global, 20261112, la misma en los seis municipios: `semilla_muestreo` para sortear los candidatos de LightGBM y `semilla_modelo` como `random_state` de todos los ajustes. LightGBM corre con un hilo y en modo determinista, así que dos ajustes iguales dan las mismas probabilidades bit a bit.
- **Hash de la receta.** `configs/receta_congelada.yaml` lleva un SHA-256 de su propio contenido (sin el campo del hash), calculado sobre un JSON canónico: reordenar el YAML no lo cambia y editar un valor lo invalida. La receta registra también el commit del código, el commit del contrato y la versión del canal. La bitácora de la prueba, el manifiesto de los modelos finales y las etiquetas de MLflow citan ese hash.
- **Identidad de las búsquedas.** Cada búsqueda tiene un `busqueda_id` derivado de su manifiesto (candidatos, hash de los ejemplos, secciones de la configuración y commit). Una búsqueda no se mezcla con otra en la misma carpeta ni se reanuda después de un commit nuevo.
- **Árbol limpio.** Las líneas de comandos que producen evidencia se niegan a correr con cambios sin commit en `src/` o `configs/`, salvo `diagnostics`, que no lo comprueba, y `build_channel`, que lo registra como `src_dirty` en su manifiesto. Ninguna sobrescribe una salida. Cada commit de evidencia cita el commit del que se generó.
- **Bytes de la evidencia.** `.gitattributes` desactiva la conversión de fin de línea en `reports/**/evidence/**` y en `docs/forecasting/verificacion/*.py`, de modo que los hashes citados (manifiestos y md5 de los scripts en las pruebas congeladas) se reproducen sobre la copia de trabajo en cualquier sistema. Las copias de DVC de `resumen_prueba.json`, `ficha_desempeno.json` y `modelos_finales/manifiesto.json` se escribieron en Windows con CRLF y difieren de las de git solo en el fin de línea.
- **Evidencia del canal (F1).** El comando que documenta `reports/epidemiology/evidence/channel_20261111/README.md` (`scripts/describe_channel_epico.py --out-dir reports/epidemiology/evidence/channel_20261111`) sobrescribe la evidencia versionada, porque el script no comprueba si la carpeta existe. Para reproducirla se escribe en una carpeta nueva (`--out-dir reports/epidemiology/runs/canal_<fecha>`, ignorada por git) y se compara.
- **Verificación independiente.** Los scripts de [`verificacion/`](verificacion/) no importan nada de `src/`: recalculan desde el contrato las cifras que las pruebas de aceptación tienen escritas como literales, y las del análisis integrado.

  | Script | Qué contrasta |
  |---|---|
  | `descriptivos_d3.py` | Cifras de la variable objetivo y del canal (D1.1, D3, D4) |
  | `caso_f2.py` | Las 15 variables de un municipio y una semana (F2): los casos literales de `tests/acceptance/test_f2_dataset.py` |
  | `tamanos_d71.py`, `caso_f3_hueco.py` | Filas de entrenamiento por ensayo y líneas base con el corte de D7.1 (F3); `caso_f3_hueco.py` da el caso literal de `tests/acceptance/test_f3_baselines.py` |
  | `eventos_d8.py` | Inicios de brote evaluables por periodo (F4) |
  | `umbrales_d9.py` | Umbrales de las reglas de conteo de victorias (F5a) |
  | `caso_f6.py`, `caso_f6b.py` | Selección, τ, paso 0 y ablación (F6, F6b) |
  | `caso_f7_diagnostico.py` | Régimen, casillas y resúmenes del diagnóstico (F7). Su uso documentado y su `--tau` por defecto (0,10) corresponden a la receta sustituida; para la receta vigente (literales de `test_f7b_diagnostico_desarrollo.py`): `--busqueda data/processed/forecasting/sin_clima --familia logistica --tau 0.05` |
  | `caso_u75.py` | Reportes derivables de U7.5: clasificación en τ, Murphy, calibración anual, eventos y razón de alertas (contraste independiente previo a congelar `test_f7c_reportes_real.py` y `test_f7c_identidades_real.py`). Frente a `reportes_20261112/` coinciden 1.061 de 1.065 cantidades; las 4 diferencias son la convención U7.5-D4 (razón de alertas del canal en desarrollo, 10/63 frente a 9/63, y en Villavicencio 2/6 frente a 1/6, con sus compuestos). Incluye un adaptador para periodos sin inicios, porque `caso_f7_diagnostico.py`, congelado, falla con una tabla de inicios vacía |
  | `analisis_integrado_exploratorio.py` | Cifras exploratorias [X] de `analisis_resultados.md` y contraste independiente parcial de U7.5 (480 cantidades) |
  | `verificar_cifras_analisis.py` | Cada cifra de `analisis_resultados.md` que procede de la evidencia, con un solo redondeo (245 comprobaciones) |

  Se ejecutan desde la raíz con el Python del entorno y en modo aislado, por ejemplo `.venv/Scripts/python.exe -X utf8 -I docs/forecasting/verificacion/<script>.py …`, con la ruta de los datos; las opciones están al inicio de cada script (algunos usos antiguos muestran solo el nombre del archivo o `python`). Los md5 que citan las pruebas congeladas son los de `caso_f6b.py` y `caso_f7_diagnostico.py`; `caso_f6.py` queda fijado porque `caso_f6b.py` lo importa.

## 5. Gobernanza

- **Receta congelada y bitácora.** La receta la genera la selección como borrador y la aprueba el responsable de la implementación (nombre y fecha en el campo `aprobacion`). La prueba solo corre con una receta válida, y cada ejecución queda en `reports/forecasting/bitacora_prueba.jsonl` en el momento de iniciar (un intento fallido también cuenta).
- **Prueba única.** Ningún sistema (modelo, B1, B2, ablación ni alerta del canal) se puntúa sobre semanas objetivo de 2023–2025 antes de la receta. La prueba se ejecutó una vez; una segunda ejecución exige `--razon` y queda registrada. Ninguna métrica de prueba cambia una decisión.
- **Pruebas de aceptación congeladas.** Cada fase escribe sus pruebas de aceptación (`tests/acceptance/test_f<fase>_*.py`) con cifras literales de los scripts independientes; el responsable de la implementación revisa sus afirmaciones y se congelan en un commit propio antes de implementar. Después se añade una regla `Edit(...)` de denegación en `.claude/settings.json`; la verificación la hace git: estos archivos no deben aparecer en `git diff` ni en `git log` posteriores a su commit de congelamiento. `tests/acceptance/conftest.py` solo puede ganar opciones nuevas.
- **Pruebas sustituidas (DV-02).** Cuando una enmienda vuelve obsoleta una prueba congelada, la prueba no se edita: `tests/conftest.py` la marca como `sustituida_dv02` y la omite con una razón que nombra su reemplazo, de modo que aparece como omitida en todo informe. El reemplazo va en un archivo con nombre propio (`test_f7b_*`) y su propia regla.
- **Legado congelado.** `ewt`, `excess_label`, `pilot_weekly_labels.parquet`, `src/epidemiology/ewt.py`, `src/epidemiology/labels.py` y los descriptores DVC existentes no se modifican: reproducen el informe de avance y su auditoría.
- **Cómo proponer una enmienda.**
  1. Detener la parte afectada y describir el problema con la cláusula del contrato que toca.
  2. Escribir una entrada `DV-NN` en [`desviaciones.md`](desviaciones.md) con el formato del archivo (cláusula, problema, alternativas, decisión, impacto, commit), **antes** de continuar.
  3. Obtener la aprobación del responsable de la implementación. Si cambia una decisión metodológica, se registra además como enmienda (D*x.y*) en [`registro_decisiones.md`](registro_decisiones.md) y en el contrato.
  4. Si la enmienda afecta pruebas congeladas, aplicar el mecanismo de DV-02. Si afecta la receta, se genera una receta nueva y la anterior se conserva como registro.
  5. Nada se decide mirando la prueba: una enmienda posterior a la prueba no puede cambiar umbrales, ventanas, k, W ni τ.

## 6. Cómo leer la evidencia y las figuras

**Rótulos.** Lo que viene de desarrollo (2019–2022) se rotula «usado para elegir»: sirvió para decidir, así que no es una estimación del desempeño. El desempeño se lee solo en prueba (2023–2025). En desarrollo, cada casilla usa la configuración común elegida sin su propio año (fuera de año).

**Evidencia** (`reports/forecasting/evidence/`). Cada carpeta lleva la fecha de generación y su estado en `INDICE.md`. Las tablas de `reportes_20261112/` tienen un manifiesto (`manifiesto_reportes.json`) con el commit, la receta y las entradas; las notas de convención (cómo se cuenta la razón de alertas, por qué los dos agregados de casillas pueden diferir en signo, cuándo aparece NaN) están en `INDICE.md`.

**Figuras** (`reports/forecasting/figures/20261112/`, PNG y SVG). Sistemas: *receta* (modelo vigente, sin clima), *ablación* (misma familia con el conjunto completo), B2, B1 y el *canal*.

| Figura | Qué muestra | Cómo se lee |
|---|---|---|
| `confiabilidad_<periodo>` | Diagrama de confiabilidad de la receta y de la ablación: 10 intervalos de igual frecuencia de p̂; eje x, p̂ media del intervalo; eje y, frecuencia observada de y = 1 | La diagonal es la calibración perfecta. Un punto bajo la diagonal indica que el sistema sobreestima en ese intervalo; sobre ella, que subestima. El texto indica cuántas semanas tiene cada intervalo |
| `amoc_<periodo>` | Curva tipo AMOC: cada punto es un τ de la rejilla. Panel izquierdo, capacidad de detección; derecho, anticipación mediana; ambos frente a falsas alarmas por municipio-año. El canal es un rombo; el τ de la receta, un círculo vacío | Más arriba y más a la izquierda es mejor. En desarrollo hay dos curvas: la que eligió τ (receta en los cuatro años) y la configuración común fuera de año de las tablas. En prueba la curva es descriptiva y no cambia τ (contrato §7.5) |
| `mapa_bss_<periodo>` | Matriz municipio × año con el BSS del modelo frente a B2 en cada casilla | Azul: el modelo mejora a B2; rojo: B2 es mejor; gris: empate. El color se recorta en ±0,5; el número de cada celda es el valor exacto |
| `serie_<código>` | Serie 2019–2025 de un municipio en tres franjas. Arriba: casos semanales C y límite U1e. En medio: alertas del modelo y del canal, desde su confirmación t_c hasta su fin. Abajo: p̂ del modelo y B2, con la línea de τ | p̂ y las alertas se dibujan en la semana de emisión t y se refieren a t + 8: para contrastar una probabilidad con lo observado se mira la franja superior 8 semanas después. Las líneas discontinuas son inicios de brote evaluables; la línea gris vertical separa desarrollo de prueba |
| `resumen_municipios` | Tres paneles por municipio: BSS frente a B2 con las filas del municipio juntas; inicios avisados por el modelo y por el canal; falsas alarmas por municipio-año | Marcador hueco: desarrollo; lleno: prueba. Círculo: modelo; rombo: canal. La etiqueta de cada municipio indica sus inicios evaluables en cada periodo; donde no hay ninguno, no se dibuja punto de detección |

## 7. Glosario

**Identificadores**

| Identificador | Significado |
|---|---|
| D0–D10 | Decisiones del contrato: D0 alcance territorial; D1 conteo, calendario y canal; D2 umbral de exceso (U1e); D3 variable objetivo; D4 emisión, información disponible y horizonte; D5 líneas base y modelos; D6 variables; D7 validación y selección; D8 métricas y regla de decisión; D9 incertidumbre y reglas de comparación; D10 salida para el usuario |
| D*x.y* | Enmienda de una decisión: D1.1 parametrización única del canal; D7.1 entrenamiento cerrado en la primera emisión de cada ensayo; D7.2 conjunto de variables como paso de selección; D9.1 análisis diagnóstico integrado |
| DV-NN | Desviación registrada durante la implementación: DV-01 (corte de entrenamiento, origen de D7.1), DV-02 (receta sustituida por D7.2 antes de la prueba), DV-03 (edición editorial de los documentos) |
| F0–F8 | Fases de la implementación (§2). Sufijos: F5a/F5b (arnés y búsqueda), F6b (D7.2), F7b (pruebas que reemplazan a las de F7 sustituidas por DV-02), F7c (pruebas de U7.5) |
| U*x.y* | Unidad de trabajo *y* de la fase *x*: uno o dos requisitos del contrato y un commit |
| contrato §10.*n* | Requisito *n* de la lista acumulada del contrato |

**Decisiones de plan de las unidades.** No son decisiones del contrato (D0–D10). Se citan así en el código y en las pruebas congeladas de F6b, F7 y F7c, y se reconstruyen aquí desde esos archivos:

| Identificador | Qué fija | Dónde se cita |
|---|---|---|
| F7, decisión 1 | La ablación del clima de D5 (contrato §4.4): la familia final con el otro conjunto de variables | `desviaciones.md` (DV-02, «decisión D1»), `reports/forecasting/INDICE.md`, `tests/forecasting/test_search.py` («F7, D1») |
| U7.1-D2 | El historial de desarrollo usa las predicciones guardadas de la configuración de la receta, con el rótulo «usado para elegir» | `src/forecasting/artifacts.py`, `tests/acceptance/test_f7_artefactos.py`, `test_f7b_prueba_real.py` |
| U7.1-D3 | El diagnóstico de desarrollo usa la configuración común fuera de año, que cambia por año; sus cifras de eventos se reportan como salen, aunque difieran de las de la curva de τ | `tests/acceptance/test_f7_diagnostico_desarrollo.py`, `test_f7b_diagnostico_desarrollo.py`, `src/forecasting/reports.py` («as diagnostics.py, D3») |
| U7.1-D4 | El estado de alerta se calcula por periodo y no cruza de desarrollo a prueba | `src/forecasting/artifacts.py`, `tests/acceptance/test_f7_artefactos.py`, `tests/forecasting/test_artifacts.py` |
| U7.5-D1 | Intervalos de confiabilidad: 10 intervalos de igual frecuencia (`pd.qcut`), recalculados por año | `tests/acceptance/test_f7c_formulas.py` |
| U7.5-D2 | Descomposición de Murphy con esos intervalos; residuo WBV − 2·WBC | `tests/acceptance/test_f7c_formulas.py` |
| U7.5-D3 | Clasificación en τ: emisión positiva si p ≥ τ; NaN cuando el denominador es 0 | `tests/acceptance/test_f7c_formulas.py` |
| U7.5-D4 | Razón de alertas de `events._summarize` (cuenta inicios, así que puede superar 1) y duraciones descriptivas | `tests/acceptance/test_f7c_formulas.py`, `docs/forecasting/verificacion/analisis_integrado_exploratorio.py` |
| U7.5-D5, sentido (a) | Habilidad pareada por casilla municipio-año, con los dos agregados | `tests/acceptance/test_f7c_formulas.py` |
| U7.5-D5, sentido (b) | Insumos de cada periodo: en desarrollo, la receta y la ablación con su configuración común fuera de año y las líneas base guardadas; en prueba, los cuatro sistemas de `predicciones_prueba.parquet` | `tests/acceptance/test_f7c_identidades_real.py`, `src/forecasting/reports.py` («decision D5 of U7.5»). El plan usa «D5» con dos sentidos; ambos se conservan tal como están en los archivos |
| PARADA 2 (plan de F6b) | Reconstruida a partir del docstring de `test_f6breal_seleccion.py`: punto de detención en que el responsable de la implementación revisó la selección real de D7.2 y su verificación independiente (`caso_f6b.py`) antes de congelar esa prueba | `tests/acceptance/test_f6breal_seleccion.py` |

No se encontró en el repositorio ninguna U7.1-D1 ni U7.1-D5 con sentido propio.
| [H], [C], [D], [I], [R], [P], [X] | Marcas de los documentos: hecho verificado, cálculo reproducible con un script, decisión, inferencia, recomendación, pendiente y exploratorio (cálculo no declarado antes de abrir la prueba; describe, no concluye) |

**Tiempo y canal**

- **t (emisión):** la última semana lunes–domingo con dato completo. **h:** semanas entre t y la semana objetivo t + h; h = 8.
- **Canal endémico de epiCo:** con los conteos de la misma semana ordinal en los cinco años anteriores, *central* = media geométrica y **U1e** = límite superior (central por el factor del intervalo t). **Zonas:** seguridad (C ≤ central), alerta (central < C ≤ U1e), epidemia (C > U1e).
- **`target_u1e`:** 1 si la semana está en zona epidémica. **Semana en zona epidémica:** el vocabulario del proyecto; «brote» se reserva para el inicio de brote.
- **Inicio de brote:** primera de ≥ 3 semanas sobre U1e tras ≥ 2 por debajo (Umaña et al., 2026). **Entrada:** semana sobre U1e precedida de una por debajo (evento secundario).
- **Episodio y régimen (D9.1):** episodio = racha de ≥ 3 semanas en zona epidémica. Un año es *epidémico* si tiene semanas de un episodio, *post-epidémico* si no lo es y el anterior sí, *tranquilo* en otro caso. Usa semanas posteriores a la emisión: es solo un estrato de diagnóstico.

**Evaluación**

- **Ensayo:** un municipio y un año evaluado, con su reentrenamiento. **t0:** primera emisión evaluada del ensayo; se entrena solo con filas con t + h ≤ t0 (D7.1). **Origen móvil:** la secuencia de ensayos anuales.
- **Casilla:** un municipio en un año (24 en desarrollo, 18 en prueba). **Victoria:** pérdida estrictamente menor en una casilla; un empate no cuenta.
- **B2:** P(y = 1 | zona actual del canal), la referencia principal. **B1:** P(y = 1 | semana del año objetivo), estacionalidad. Ambas con suavizado de Laplace y las mismas filas de entrenamiento del ensayo.
- **Mejora de log-loss:** 1 − LL/LL_B2, la métrica que decide hiperparámetros y familia. **BSS:** 1 − Brier/Brier_B2, la principal de reporte. **Murphy:** descomposición del Brier en confiabilidad, resolución e incertidumbre.
- **Fuera de año:** la configuración que se puntúa en un año se eligió sin ese año. **Configuración común:** la misma en los seis municipios. **Excepción:** un municipio se aparta de la común solo si gana en los 4 años de desarrollo.
- **Paso 0 (D7.2):** elección del conjunto de variables por familia. **Completo:** 15 variables. **Sin clima:** las 6 que no son anomalías climáticas. **Ablación:** la familia final con el otro conjunto; mide el aporte del clima y no compite con la receta.
- **Emisión positiva:** p̂ ≥ τ (modelo) o zona de alerta (canal). **Alerta:** racha de ≥ k = 2 emisiones positivas, fechada en su confirmación **t_c**. **Alerta de continuidad:** la zona observada en t_c ya es epidemia; se cuenta aparte. **Alerta juzgable:** la que tiene semanas suficientes dentro del periodo para saber si fue falsa. **Falsa alarma:** alerta sin inicio en las W = 12 semanas siguientes.
- **Capacidad de detección:** inicios avisados / inicios evaluables. **Razón de alertas:** inicios avisados por alertas juzgables / alertas juzgables. **Anticipación:** s − t_c. **Δw:** anticipación del modelo menos la del canal en un inicio avisado por ambos. **Censura:** se excluye todo objeto que necesite semanas fuera del periodo.
- **Curva tipo AMOC:** detección y anticipación frente a falsas alarmas sobre la rejilla de τ.
- **Niveles de conclusión (contrato §8.4):** «mejora consistente», «tendencia favorable» o «sin mejora», según la métrica global y el número de casillas (o de inicios discrepantes) ganados. No alcanzar «mejora consistente» no prueba que no haya mejora. **Nivel descriptivo por municipio (§8.8):** favorable, mixto o desfavorable. **H-R, H-E:** hipótesis secundarias de régimen y de eventos, contrastadas solo en prueba de forma descriptiva.

**Gobernanza y artefactos**

- **Pre-registro:** candidatos, rejillas y semillas fijados antes de la búsqueda (sección `busqueda` de `configs/forecasting.yaml`). **Tabla única:** puntajes de todos los candidatos por municipio y año; fuente de verdad de la selección.
- **Receta congelada:** la configuración elegida en desarrollo (familia, conjunto de variables, hiperparámetros por municipio, τ, k, W, ablación), con procedencia, aprobación y hash. **Barrera de prueba:** `guard.py`. **Bitácora:** registro de ejecuciones de la prueba. **Prueba única:** la evaluación 2023–2025, ejecutada una vez.
- **Ficha de desempeño:** resultados de prueba que sirve la API. **Historial fuera de muestra:** predicciones del origen móvil 2019–2025; nunca incluye el modelo final. **Modelo final:** entrenado con todas las filas, solo para emisiones posteriores a la última verdad conocida.
- **«Usado para elegir»:** rótulo de todo resultado de desarrollo. **`sustituida_dv02`:** marca de una prueba congelada que una enmienda volvió obsoleta; se omite y nombra su reemplazo (§5).

**Cómo leer los hashes**

| Forma | Ejemplo | Qué es |
|---|---|---|
| 7 caracteres hexadecimales | `abf08c2` | Commit de git abreviado (`git show abf08c2`) |
| 64 hex, abreviado como `xxxxxxxx…xxxx` | `de74d4ab…fda5` | SHA-256 de la receta (8 primeros y 4 últimos caracteres); el valor completo está en `hash_sha256` de `configs/receta_congelada.yaml` |
| 20 hex | `ae6225180d37d025224b` | `busqueda_id`: los primeros 20 caracteres del SHA-256 del manifiesto de la búsqueda |
| 32 hex, a veces abreviado | `99fd1231…6cfc` | md5 de DVC de un archivo; con sufijo `.dir`, de una carpeta. Se compara con el campo `md5` del `.dvc` |
| `SHA-256 xxxxxxxx…xxxx` de un Parquet | `b3b151a5…2c7f` | Huella del archivo de datos, registrada en los manifiestos |

## 8. Pendientes (F8)

- Registro de los seis modelos finales en el registro de modelos de MLflow como `DengueAlertas_<Municipio>` (nombre del municipio; propuesta de la especificación §9), con las etiquetas de la especificación §9, de forma compatible con el descubrimiento de modelos de `src/api.py`.
- API: `/predict`, `/predict/upload`, `/historial` y `/desempeno` según [`api_salida_d10.md`](api_salida_d10.md), incluido el conflicto del parámetro `threshold` de `/predict` con contrato §9.7 (especificación §10) y la validación del contrato de entrada de datos subidos (contrato §3.3).
- Frontend con las cuatro vistas de contrato §9.4 y la URL de la API como variable.
- Secciones de despliegue del README raíz: enlace a la aplicación, uso de la API, ejemplos de entrada y salida (plantilla CSV y respuesta JSON) y credenciales de ejemplo (contrato §10, punto 21).
- Pruebas de aceptación de F8 (contrato §10, puntos 19 y 20), escritas y congeladas antes de implementar.
- Revisión de la API actual (`src/api.py`, anterior al MVP), según la auditoría de coherencia del 2026-11-13:
  - sirve los modelos `ModeloPrueba` y `DengueAlertas_*` de otra etapa (Cali, Bucaramanga, Cartagena, Ibagué y Villavicencio), con `MODEL_THRESHOLD` = 0,205, sin los seis pilotos ni la receta congelada;
  - **`/predict` rellena con 0,0 las variables que faltan, ignora los errores de conversión de tipos y, si el modelo no da probabilidad, usa una probabilidad sustituta de 0,85 o 0,15 según la clase**: esa `probability` no es una probabilidad del modelo;
  - carga los modelos de forma secuencial (no «en paralelo», como dice su comentario), y cuántos carga depende de lo que encuentra en MLflow;
  - el mapa de municipios usa códigos de departamento tipo ISO 3166-2, no DANE, y no cita la fuente de la densidad;
  - escribe `logs/api_<stage>.jsonl` con las entradas completas (carpeta ignorada por git desde la auditoría);
  - sus docstrings y comentarios están en español sin tildes, aunque la convención del código es el inglés;
  - la ficha versionada trae claves fuera de la convención de `api_salida_d10.md` §2 (`mcnemar.nivel`, `B2`/`B1`, `municipality_code`), que la API normaliza al servirla (`api_salida_d10.md` §7).

El análisis cualitativo de §8.8 y el contraste de H-R y H-E están en [`analisis_resultados.md`](analisis_resultados.md) (03891f3).

**Sensibilidades no ejecutadas en el MVP [P].** Quedan como posible desarrollo posterior, sin compromiso de realizarlas: la GRU opcional (contrato §4.2), los horizontes h ∈ {2, 4, 12} (contrato §3.6), el canal U3a y las ventanas de 3 o 7 años (`registro_decisiones.md`, D2), la validación externa (`registro_decisiones.md`, D0) y el contraste del canal con epiCo en R (contrato §10.1).
