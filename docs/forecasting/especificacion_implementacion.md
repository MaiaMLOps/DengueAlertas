# Especificación de implementación: forecasting MVP (v0.2)

**Estado:** v0.2 del 2026-11-10. Resuelve los [P] de la v0.1 con las decisiones del chat «Desarrollo guiado con Claude Code» y con lo verificado en el repositorio (`main`, commit `77ceaa3`). Lo que sigue marcado **[P]** tiene un momento fijado para resolverse.

**Fuentes de verdad:**
- `contrato_predictivo.md`: qué se construye; los requisitos están en §10.
- `registro_decisiones.md`: por qué se construye así.
- `api_salida_d10.md`: la salida de la API.

Si este documento contradice al contrato, manda el contrato.

**Rama:** `feature/forecasting-mvp`, creada desde `main`.

Convención: **[H]** hecho verificado; **[D]** decisión; **[I]** inferencia; **[R]** recomendación; **[P]** pendiente.

## 0. Cambios respecto a la v0.1 y correspondencia de rutas

- **Cambios:**
  - modo de trabajo definido (§2);
  - separación entre lo nuevo y el legado (§3);
  - barrera técnica sobre el periodo de prueba (§4);
  - F5 dividida en F5a y F5b (§5);
  - pruebas de aceptación literales por fase (§6);
  - configuración y trazabilidad (§8 y §9);
  - variables de entorno alineadas con `src/api.py` (§10).
- **Correspondencia de rutas:** el contrato cita documentos del proyecto de Claude. En el repositorio están en:

  | Proyecto de Claude | Repositorio |
  |---|---|
  | `claude/02_contrato_predictivo.md` | `docs/forecasting/contrato_predictivo.md` (copia literal) |
  | `claude/01_registro_decisiones.md` | `docs/forecasting/registro_decisiones.md` (copia literal) |
  | `claude/specs/implementacion_forecasting.md` | `docs/forecasting/especificacion_implementacion.md` (este archivo) |
  | `claude/specs/api_salida_d10.md` | `docs/forecasting/api_salida_d10.md` |
  | `claude/verificacion/*.py` | `docs/forecasting/verificacion/*.py` |

  Las desviaciones se registran en `docs/forecasting/desviaciones.md` y se consolidan después en `01`. Claude Code no tiene acceso al proyecto de Claude.

## 1. Reglas no negociables

1. **Periodo de prueba.** Las semanas objetivo de 2023 a 2025 no se evalúan hasta que la receta esté congelada en `configs/receta_congelada.yaml` y Jesús la apruebe (contrato §6.1 y §7.7). Lo impone la barrera técnica de §4.
2. **Sin fuga.** Ninguna feature usa semanas posteriores a t. Lo verifica la prueba de truncamiento (contrato §10, punto 7).
3. **Comparaciones pareadas.** Todos los candidatos, las líneas base y la ablación usan las mismas filas y los mismos cortes (contrato §6.1).
4. **Legado intacto.** `ewt` y `excess_label` no se modifican; el canal nuevo vive en módulos y productos nuevos (contrato §10, puntos 2 y 3; §3 de este documento).
5. **Registro completo.** Se registran todos los candidatos, no solo el elegido (contrato §10, punto 9).
6. **Desviaciones.** Cualquier desviación del contrato se registra en `desviaciones.md`, con la aprobación de Jesús, antes de continuar.
7. **Credenciales.** No se leen, imprimen ni versionan `.env` ni las credenciales de DVC.
8. **Control humano.** No hay commits, pushes, `dvc add` ni `dvc push`, instalaciones ni merges sin aprobación explícita (`AGENTS.md` §5).
   - **DVC contra el remoto [D, F1].** `.dvc/config.local` usa `ask_password = true`, y DVC espera la contraseña por teclado; en una sesión no interactiva se queda sin salida. Por eso los comandos contra el remoto (`dvc push`, `dvc pull`, `dvc fetch`, `dvc status -c`) los ejecuta solo Jesús en su terminal. Claude Code no los lanza ni modifica `.dvc/config.local`; sí puede ejecutar comandos locales (`dvc status` sin `-c`, y `dvc add` con aprobación).

## 2. Modo de trabajo [D]

**Entorno.**
- App de escritorio de Claude, pestaña Code, sesión **Local** sobre el repositorio en el PC de Jesús (Windows), sin Worktree.
- Las sesiones en la nube quedan como contingencia, solo para módulos de funciones puras con pruebas sintéticas (F4 o las reglas de D9). Se usan si el jueves hay atraso y Jesús lo decide.

**Reparto.**
- Claude Code escribe las implementaciones que se pactan en el plan de cada unidad.
- Jesús entiende y aprueba el diseño antes de que se escriba. Después revisa los resultados y, si quiere, el código.
- Jesús no escribe código.

**Ciclo por unidad.** Una unidad cubre uno o dos requisitos de contrato §10 y dura de 30 a 90 minutos.
1. **Plan**, en modo Plan: qué se construye y por qué, las funciones con sus entradas y salidas, las pruebas en lenguaje natural con su valor esperado y su origen, y los riesgos.
2. **Aprobación** de Jesús.
3. **Implementación**, en modo Accept edits.
4. **Informe de paso** con el formato fijo de `CLAUDE.md`.
5. **Decisión de Jesús:** commit, recorrido del código o corrección.

Un commit por unidad.

**Modos de permisos.**
- F0 en modo Manual, para que Jesús vea qué hace Claude Code y qué pide aprobar.
- Desde F1, Plan seguido de Accept edits.

**Dos niveles de pruebas.**
- *Aceptación* (`tests/acceptance/test_fN_*.py`): contienen las cifras literales de §6, que vienen de scripts independientes del repositorio. Se escriben al inicio de cada fase, Jesús revisa sus afirmaciones y se congelan en un commit propio antes de implementar.
  - Tras el congelamiento se añade `Edit(/tests/acceptance/test_fN_*.py)` a `.claude/settings.json`.
  - Cada informe de paso verifica con git que esos archivos no cambiaron.
  - [H] La regla de denegación cubre las herramientas de edición de Claude Code y los comandos de shell que reconoce, pero no un programa que abra el archivo por su cuenta. Por eso la regla previene y git verifica.
- *Unitarias* (`tests/forecasting/`): Claude Code las escribe libremente dentro de cada unidad, con datos sintéticos.

**Revisiones.**
- *Obligatorias:*
  - las afirmaciones de las pruebas de aceptación de cada fase;
  - el pre-registro de §7, antes de F5b;
  - la receta congelada, antes de F7;
  - toda desviación.
- *Recomendadas:* recorrido del código del constructor de features (F2) y de la censura de eventos (F4). Ahí una fuga o un error no lo detecta ninguna cifra agregada.
- *Opcionales:* las demás.

**Papel del chat de desarrollo guiado.** Ahí se diseñan los mensajes de cada fase y se discuten las dudas sobre el contrato que surjan en un plan, antes de aprobarlo.

**Commits:** formato `tipo(alcance): descripción` en español. Tipos: `feat`, `fix`, `test`, `docs`, `chore` y `refactor`. El cuerpo cita el requisito, por ejemplo `contrato §10.7`.

## 3. Separación entre lo nuevo y el legado [D]

- **Rama desde `main`.** No se parte de `Oscar-exceso-de-casos` ni de `feat/modelado-n2-oscar`.
- **No se borra el legado.** El EWT, las etiquetas, el código del avance y las 174 pruebas siguen siendo necesarios para reproducir el informe de avance y la auditoría.
- **Procedencia.** El producto del canal se construye a partir de `data/processed/pilot_municipalities_weekly.parquet` (5.952 × 15, sin etiquetas), el mismo insumo que usa `src/epidemiology/` [H, `config.py`]. No se construye desde `pilot_weekly_labels.parquet`. La prueba de aceptación de F1 compara los conteos, el clima y la elegibilidad con el parquet de legado: solo las pruebas leen el legado, nunca `src/forecasting/`.
- **Dos capas, solo una versionada.**
  1. *Producto del canal*, versionado con DVC. Una fila por municipio y semana, con:
     - las columnas de `pilot_municipalities_weekly`;
     - `epico_central`, `epico_upper`, `target_u1e` y `channel_zone`;
     - un indicador de elegibilidad con la regla de D1.

     Ruta: `data/processed/epidemiology/pilot_weekly_channel.parquet`, con un manifiesto adyacente como el del legado. Sin `ewt` ni `excess_label`.
  2. *Ejemplos supervisados con las 15 features.* Dependen de h, así que no se versionan como producto: los genera el código de forma determinista a partir del canal y de h. Puede haber una caché local ignorada por git.
- **Código.**
  - Lo nuevo vive en `src/forecasting/` y en `src/epidemiology/channel_epico.py`.
  - `channel_epico.py` puede importar las utilidades de calendario de `src/epidemiology/ewt.py` (`week_ordinal` y `historical_references`), que implementan las convenciones de D1, incluida la sustitución de la semana 53. No las modifica [R].
  - Una prueba verifica que el producto nuevo no tiene `ewt` ni `excess_label` y que `src/forecasting/` no referencia el parquet de legado.
- **Estilo [H].** El repositorio no tiene `pyproject.toml` ni configuración de lint o formato. Las dependencias están separadas en `requirements/*.txt` y Python es 3.11 (`Dockerfile`, `requirements/data.txt`). F0 añade una configuración mínima de `ruff` solo si Jesús lo aprueba; no se reformatea el código existente.

## 4. Estructura de módulos [D]

| Ruta | Responsabilidad | Requisitos (contrato §10) |
|---|---|---|
| `src/epidemiology/channel_epico.py` | Canal de epiCo: central, gsd, U1e y zonas; `build_channel_table`, transformación pura del panel al producto (sin E/S ni lectura de configuración); `reference_weeks` y `channel_for_week` para cualquier semana (F2) | 1, 2, 3, 4 |
| `src/forecasting/build_channel.py` | Línea de comandos del producto del canal: carga la sección `canal` de la configuración, lee el panel y calcula su hash, llama a `build_channel_table` y escribe el producto y su manifiesto sin sobrescribir | 2 |
| `src/forecasting/config.py` | Carga y validación de `configs/forecasting.yaml` con Pydantic; sobrescritura de h por variable de entorno | 5, 14 |
| `src/forecasting/dataset.py` | Ejemplos supervisados (t, t + h) y las 15 features; entrada de la GRU | 5, 7 |
| `src/forecasting/splits.py` | Cortes por año de la semana objetivo | 11 |
| `src/forecasting/guard.py` | **Barrera de prueba:** toda función que puntúe filas con semana objetivo en los años de prueba exige una receta congelada válida, con hash y aprobación; si no existe, lanza un error | §1, regla 1 |
| `src/forecasting/baselines.py` | B1 y B2 (Laplace, por pliegue). La alerta descriptiva del canal está en `events.py` (F4) | 8 |
| `src/forecasting/models.py` | Pipelines de regresión logística, LightGBM y GRU opcional | 9, 10 |
| `src/forecasting/metrics.py` | Métricas de estimación y de clasificación | 16 |
| `src/forecasting/events.py` | Alertas, alertas de continuidad, inicios, emparejamiento, censura y métricas de eventos; emisiones positivas del modelo (`model_positive`) y del canal (`channel_positive`), que pasan por el mismo código | 8, 13, 14, 15 |
| `src/forecasting/search.py` | Arnés de búsqueda y tabla única de puntajes | 12 |
| `src/forecasting/comparison.py` | Puntajes fuera de año, conteo de victorias, binomial y niveles (D9) | 17 |
| `src/forecasting/selection.py` | Hiperparámetros → familia → τ → receta congelada | §6.2, §8.3, §7.5 |
| `src/forecasting/evaluate.py` | Prueba única, ablación y ficha de desempeño | §8.4 |
| `src/forecasting/artifacts.py` | Historial fuera de muestra, ficha, modelo final y registro | 18 |
| `src/forecasting/diagnostics.py` (F7) | Análisis diagnóstico integrado (D9.1): régimen de cada casilla y los seis insumos de contrato §8.8, para desarrollo y prueba, desde predicciones guardadas | 22 |
| `src/forecasting/tracking.py` | Conexión a MLflow con los encabezados de Cloudflare Access (§9) y respaldo local | 9 |
| `configs/forecasting.yaml` | Parámetros (§8) | 5, 14 |
| `configs/receta_congelada.yaml` | Se genera en F6 y lo aprueba Jesús (§8) | §6.1 |
| `requirements/forecasting.txt` | Dependencias de modelado, fijadas en F0 | — |
| `tests/forecasting/` | Pruebas unitarias | Todos |
| `tests/acceptance/` | Pruebas de aceptación por fase (§6) | Todos |

**Canal: reparto y reutilización [D, F1].**
- La dependencia va de `src/forecasting/` a `src/epidemiology/`, nunca al revés: las funciones del canal reciben `n_years`, `shift` y `alpha` solo por nombre y sin valores por defecto, y `first_year` es `src/epidemiology/config.FIRST_LABEL_YEAR`.
- `build_channel_table` exige el calendario del panel (`validate_source`: desde 2007-01-01 hasta, como máximo, 2025-12-31), así que no sirve para semanas posteriores ni para datos subidos.
- **Resuelto en F2 [H, U2.2 y U2.4].**
  - `reference_weeks(weeks, week_start, *, n_years)` (`channel_epico.py`) es la **única** función que elige las semanas de referencia: mismo ordinal en los cinco años anteriores, en orden de año; la sustitución 53 → 52 solo se aplica cuando el año no tiene ordinal 53 en el calendario; si la semana existe y falta el dato, devuelve `None`. Delega en `ewt.historical_references` sin modificarla. La usan el canal y clim_s (contrato §5.1), así que comparten la alineación por construcción.
  - `channel_for_week(counts, week_start, *, n_years, shift, alpha)` calcula central y U1e de cualquier semana, también posterior a 2025 y sin que sea fila del panel, a partir de los conteos de las semanas **completas** de un municipio (quien la llama excluye la semana parcial). Llama a `epico_channel` con el mismo vector de historia que `build_channel_table`: reproduce `epico_central` y `epico_upper` bit a bit en las 4.380 semanas elegibles y devuelve `None` en las 1.566 de 2007–2011.
  - Reconstruir el producto desde el árbol limpio (commit 9ac13bb) da el mismo md5 `99fd1231…6cfc`.
- **Pendientes para F8.** `/predict/upload` reutiliza `channel_for_week` y `reference_weeks`, pero:
  - no puede pasar por `validate_source`, porque exige que la serie empiece el 2007-01-01 y termine a más tardar el 2025-12-31; F8 necesita una validación propia del contrato de entrada (contrato §3.3);
  - un CSV subido puede traer completa la semana del 2025-12-29, que en la base es parcial (3 días). F8 decide cómo se trata: como semana completa del archivo subido (entonces es referencia válida de las semanas de 2026 con ordinal 52) o con la regla de la base.
- Producto: `data/processed/epidemiology/pilot_weekly_channel.parquet`, con el manifiesto `pilot_weekly_channel_manifest.json` (no `run_manifest.json`, que es del legado). `configs/forecasting.yaml` fija `datos.version_dvc_canal` con el md5 del `.dvc`.
- **Verificación del producto (F2) [H, U2.2].** `load_channel_product(path, *, expected_md5)` (`src/forecasting/dataset.py`) lee el parquet en bloques de 1 MiB, calcula su md5, lo compara con `datos.version_dvc_canal` y lee la tabla de esos mismos bytes; si no coinciden, lanza `ChannelVersionError` con los dos md5. El md5 de DVC 3 (`hash: md5`) es el de los bytes del archivo, así que es comparable. Además, `build_examples` compara `n`, `desplazamiento` y `cuantil_t` de la configuración con los metadatos del canal del producto.

**Barrera de prueba: detalle [R].**
- `guard.py` expone una única función de verificación. La llaman `metrics`, `events`, `baselines` y `evaluate` cuando reciben filas cuyo año de semana objetivo cae en la prueba.
- La receta es válida si contiene el hash de su contenido, el commit del código, el commit del contrato y la aprobación de Jesús (nombre y fecha), y si el hash coincide con el contenido.
- `evaluate.py` registra la ejecución de la prueba (fecha, commit y hash de la receta) en un archivo de bitácora versionado. Una segunda ejecución exige una razón explícita, que se registra.
- Calcular features y etiquetas de esos años no está bloqueado: el modelo final se entrena con todo y los conteos descriptivos del contrato ya son públicos.

**Ubicación de resultados [D].**
- `data/processed/forecasting/`, con DVC: tabla única de puntajes, predicciones fuera de muestra de desarrollo y prueba, historial para la API y modelos serializados de respaldo.
- `reports/forecasting/evidence/<fecha>/`, con git: tablas pequeñas del reporte (casillas, ficha en JSON, resumen de selección) y figuras.

Las excepciones de `.gitignore` se añaden de forma explícita, como en `reports/epidemiology/evidence/` [H: `.gitignore` ignora `*.csv` y `*.parquet`].

## 5. Fases, puntos de control y orden de recorte [D]

| Fase | Entregable | Aceptación (§6) | Control de Jesús | Día objetivo |
|---|---|---|---|---|
| F0 | Preparación: rama, inventario del entorno, `requirements/forecasting.txt`, conectividad DVC y MLflow, documentos en `docs/forecasting/`, `AGENTS.md` y `docs/` actualizados, `.gitignore`, esqueleto de `config.py` y `tracking.py` | §6.F0 | Revisión, en modo Manual | Mié 11 temprano |
| F1 | Canal de epiCo y producto nuevo con DVC | §6.F1 | Revisión de pruebas | Mié 11 |
| F2 | Ejemplos supervisados y features | §6.F2 | Revisión; recorrido de código recomendado | Mié 11 |
| F3 | Cortes, líneas base y barrera de prueba | §6.F3 | Revisión | Mié 11–Jue 12 |
| F4 | Módulo de eventos | §6.F4 | Revisión; recorrido recomendado (censura) | Jue 12 |
| F5a | Arnés de búsqueda y reglas de D9, sin correr la búsqueda real | §6.F5a | Revisión | Jue 12 |
| ◆ | **Pre-registro** (§7) | — | **Aprobación obligatoria** | Jue 12 |
| F5b | Búsqueda en desarrollo | §6.F5b | Revisión | Jue 12 noche–Vie 13 |
| F6 | Hiperparámetros → familia → τ → receta congelada | §6.F6 | **Aprobación obligatoria de la receta** | Vie 13 |
| F7 | Prueba única, ablación, ficha, modelo final e historial | §6.F7 | Revisión de resultados | Vie 13 |
| F8 | Registro en MLflow, API, frontend y README | §6.F8 | Demo | Vie 13–Dom 15 |
| Opcional | GRU y sensibilidad h ∈ {2, 4, 12} | Contrato §4.2 y §3.6 | Decisión según el tiempo | — |

**Dependencias.** F1 → F2 → F3 → F5a → F5b → F6 → F7 → F8.
- F4 solo necesita la serie observada y unas emisiones, así que puede adelantarse o hacerse en la nube.
- F8 avanza en paralelo desde F0 con datos simulados (`api_salida_d10.md` §11).
- La GRU solo compite en la selección si está lista antes de F6 (contrato §4.3).

**Orden de recorte, fijado antes de ver resultados:**
1. Sensibilidad al horizonte (contrato §3.6).
2. GRU (contrato §4.2).
3. **Reportes derivables.** Todo lo que se calcula a partir de predicciones y puntajes ya guardados puede pasar a la semana 7 sin afectar ninguna decisión: la curva tipo AMOC, el desglose de falsas alarmas, la calibración por año y las tablas extensas. F7 debe, como mínimo, ejecutar la prueba una vez, guardar todas las predicciones y puntajes por casilla y generar la ficha que necesita la API.

## 6. Pruebas de aceptación por fase

Cifras de contrato §2.2, §5.4, §6.1, §7.4, §8.2 y `registro_decisiones.md` (D1.1 y D3), calculadas con los scripts de `docs/forecasting/verificacion/`. Se escriben como valores literales, no como llamadas a esos scripts.

**F0**
- `configs/forecasting.yaml` carga y valida: h = 8, k = 2, W = 12, rejilla de τ de 0,05 a 0,95 con paso 0,05 (19 valores), desarrollo 2019–2022 y prueba 2023–2025. Un valor inválido produce un error claro.
- `FORECAST_HORIZON=4` sobrescribe h y un valor no entero falla.
- `tracking.py` falla con un mensaje claro si faltan `CF_ACCESS_CLIENT_ID` o `CF_ACCESS_CLIENT_SECRET`, sin imprimir sus valores.

**F1** (con `--channel-parquet` y `--legacy-labels-parquet`; este nombre evita chocar con la opción `--labels-parquet` del legado en `tests/epidemiology/conftest.py`, U0.5)
- El producto tiene 5.952 filas (6 × 992); la clave (municipio, `week_start`) es única; no tiene `ewt` ni `excess_label`.
- Los conteos, `temperature_c`, `precipitation_mm` y `relative_humidity_pct` son idénticos a los del parquet de legado, y la elegibilidad coincide con `label_eligible`.
- Hay 4.380 semanas elegibles y 945 con `target_u1e = 1`.
- Por municipio:

  | Municipio | Positivas | Rachas | Inicios (Umaña, U1e) | Entradas 0→1 |
  |---|---|---|---|---|
  | Barranquilla `08001` | 171 | 28 | 5 | 28 |
  | Soledad `08758` | 204 | 46 | 7 | 46 |
  | Neiva `41001` | 152 | 26 | 6 | 26 |
  | Villavicencio `50001` | 165 | 32 | 7 | 31 |
  | Cúcuta `54001` | 129 | 30 | 3 | 30 |
  | Cali `76001` | 124 | 13 | 3 | 13 |

  Totales: 175 rachas, 88 de una semana, 31 inicios y 174 entradas.
- `epico_upper ≥ epico_central` en todas las semanas elegibles, y `epico_central > ewt` en las 4.380.
- Zonas: seguridad, alerta y epidemia en 37,9 %, 40,5 % y 21,6 % (a tres decimales: 0,379; 0,405; 0,216). P(C > central) = 0,621.
- Prueba unitaria de la fórmula: reproduce epiCo con redondeo a 5 decimales; |qt(0,025; 4)| = 2,776445 (seis decimales); U1e = 0 si central = 0.

**F2** (h = 8)
- Hay 711 filas por municipio y 4.266 en total; el primer t es el 2012-03-19 y el último, el 2025-10-27.
- La semana objetivo es t + 8 semanas y es elegible.
- Están las 15 columnas de contrato §5.2, con sus nombres exactos y sin faltantes.
- **Truncamiento:** para al menos 50 orígenes t por municipio, distribuidos en todo el rango y siempre con el primero y el último, las 15 features calculadas con la base completa y con la base recortada en t son exactamente iguales.
- U1e_{t+8} no cambia al truncar en t en ninguna fila.

**Lo que se congeló en F2 [H].** Archivos `tests/acceptance/test_f2_dataset.py` y `test_f2_channel_loader.py` (commit ac850b5) y `test_f2gru_input.py` (commit 2e9a0b6), cada uno con su regla `Edit(...)` de denegación. Además de las cifras anteriores:
- **Base recortada en t:** las 15 columnas fuente de un solo municipio con `week_start ≤ t`, sin columnas del canal; las cantidades del canal y clim_s se recalculan desde ella. La muestra son 56 orígenes por municipio: 50 repartidos (con el primero y el último), los tres cuyo objetivo tiene ordinal 53, los tres cuyo t tiene ordinal 53 y el caso A.
- **Sensibilidad:** alterar semanas ≤ t de la base recortada cambia `crecimiento_4s`, `cambio_umbral` y `anom_*_m1`; así las features salen de la base que se pasa, no de un archivo.
- **Consistencia del canal:** `channel_for_week` reproduce `epico_central` y `epico_upper` en las 4.380 semanas elegibles y devuelve `None` en las 1.566 de 2007–2011; el canal de t + 8 calculado con los conteos ≤ t es igual al del producto en las 4.266 filas.
- **Tres casos literales**, calculados por Jesús fuera del repositorio (intermedios del canal y las 15 features):
  - A: Villavicencio `50001`, t = 2018-12-03 (t + 8 = 2019-01-28, en el año siguiente);
  - B: Barranquilla `08001`, t = 2018-11-05 (t + 8 = 2018-12-31, ordinal 53, con las cinco referencias en la semana 52);
  - C: Barranquilla `08001`, t = 2012-12-31 (t con ordinal 53; clim_t usa 2007-12-31 y la semana 52 de 2008–2011).
- **Oráculo independiente** en pandas de las 15 features en los orígenes de la muestra, validado contra los tres casos literales.
- **Cargador:** el producto real pasa la verificación md5 y una copia con un byte alterado se rechaza nombrando los dos md5.
- **Entrada de la GRU** (`test_f2gru_input.py`, contrato §5.3): secuencia de 4.266 × 12 × 8 (log C_s, pos_s, zona_s en one-hot y las tres anomalías; cronológica, de t − 11 a t) y 4.266 × 3 entradas estáticas (`cambio_umbral`, `sem_sin`, `sem_cos`); mismas filas que `build_examples`; igualdad exacta entre la base completa y la recortada; coherencia exacta con `posicion_4s`, `crecimiento_4s`, `zona_t`, las estáticas y `anom_*_m1..m3`.
- **Control negativo** (`tests/forecasting/test_truncation_control.py`): importa la comparación congelada y comprueba que detecta clim_s con todos los años y C_{t+1} en lugar de C_t.

**Convención de tolerancias [D, F2].** Igualdad exacta (`==`) cuando se compara una función consigo misma (truncamiento, `channel_for_week` frente al producto, coherencia GRU–tabular); `rel=1e-9, abs=1e-12` cuando se comparan dos implementaciones (oráculo, casos literales).

**Convención de agregación [D, F2].** Toda suma o media sobre semanas es una suma de izquierda a derecha, de la semana más antigua a t. Por eso la entrada de la GRU reproduce las features tabulares bit a bit.

**Guarda de la semana objetivo [D, F2].** `build_features` y `build_gru_input` fallan si la referencia más reciente de t + h es posterior a t (posible desde h = 53); con h ∈ {2, 4, 8, 12} nunca ocurre.

**La GRU en la búsqueda.** Su entrada está construida y probada, pero su inclusión en la búsqueda se decide en el pre-registro (§7).

**F3**
- **Corte de entrenamiento (DV-01, contrato §6.1 enmendado por D7.1):** en cada ensayo Y y municipio, t0 es la emisión más temprana entre las filas evaluadas, y ninguna fila de entrenamiento tiene t + h > t0. Las filas evaluadas son exactamente las del año Y.
- Desarrollo: 208 filas evaluadas por municipio, 1.248 en total, 319 positivas.
- Prueba: 156 por municipio, 936 en total, 322 positivas.
- En cada ensayo Y, ninguna fila de entrenamiento tiene la semana objetivo en Y ni después, y las filas evaluadas son exactamente las del año Y.
- B1 y B2 se estiman solo con el entrenamiento del ensayo; con datos sintéticos, el suavizado de Laplace da (positivos + 1)/(casos + 2).
- **Barrera:** pedir una métrica sobre filas de 2023–2025 sin una receta válida lanza el error de `guard.py`; con una receta de prueba sintética válida, no.

**Lo que se congeló en F3 [H].** Archivos `tests/acceptance/test_f3_splits.py`, `test_f3_baselines.py` y `test_f3_guard.py` (commit 9a8f607), con su regla `Edit(...)` de denegación (e7cae5c). Procedencia de los literales: `docs/forecasting/verificacion/tamanos_d71.py` y `caso_f3_hueco.py` (31997c5). Además de las cifras anteriores:
- **Ensayos** (`splits.yearly_trials`): uno por municipio y año, con `t0`, `train` y `evaluate` como etiquetas de índice de las filas de `build_examples`. Ejemplo de 2019: t0 = 2018-11-12; la última fila de entrenamiento es la emitida el 2018-09-17 (objetivo 2018-11-12); las emitidas del 2018-09-24 al 2018-11-05 no entran en el entrenamiento de 2019 y sí en el de 2020. t0 y las filas de entrenamiento de cada ensayo y municipio se reportaron sin congelarse (de 340 filas en 2019 a 653 en 2025; 7 menos que con el corte anterior en cada ensayo y municipio).
- **Líneas base:** B2 por (municipio, `zona_t`) y B1 por (municipio, ordinal de t + h), con Laplace y solo con el entrenamiento del ensayo. Un grupo sin casos da 0,5; con los datos reales no ocurre. Caso literal: Villavicencio `50001`, ensayo 2019, t = 2018-12-03 (zona epidemia, objetivo 2019-01-28 con ordinal 4), B2 = 21/43 y B1 = 5/32.
- **Bordes de B1 [D, U3.1]:** el ordinal 53 cuenta como 52, en las filas de entrenamiento y en la evaluada, y la ventana es circular sobre 52 semanas, {((o − 1 + d) mod 52) + 1 : d = −k..k}. La semiventana k = 2 está en `lineas_base.b1_semiventana` (1 ≤ k ≤ 25).
- **Barrera y receta mínima:** `guard.check_test_access(target_weeks, access)` es la única verificación y relee la receta en cada llamada. Una receta es válida si tiene `procedencia` (`commit_codigo` y `commit_contrato` de 40 caracteres hexadecimales, `version_dvc_canal` igual a la configuración), `aprobacion` (nombre no vacío y fecha) y `hash_sha256`. El hash es el SHA-256 de `json.dumps(receta sin hash_sha256, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)` en UTF-8, con fechas ISO: reordenar el YAML o entrecomillar la fecha no lo cambia. Se rechazan las claves YAML duplicadas. Las pruebas usan solo recetas sintéticas en `tmp_path`; `configs/receta_congelada.yaml` no existe hasta F6.
- **Quién llama a la barrera:** `metrics.log_loss`, `brier` y `average_precision`, y `baselines.b1_predict` y `b2_predict`, con las semanas objetivo de las filas que puntúan, antes de validar nada. `splits` no la llama, porque no puntúa. Las filas de entrenamiento no pasan por ella: el ensayo de 2024 entrena con objetivos de 2023.
- **Nota sobre `test_f3_guard.py`:** su función auxiliar `_score` importa `baselines`, `guard` y `metrics` en la misma línea, así que sus casos solo pasan cuando existen los tres módulos.

**Convenciones de F3 [D].**
- **Recorte del log-loss:** ε = 1e-15 (`metricas.recorte_log_loss`), el mismo para todos los sistemas; es una protección numérica. Con B1 y B2 nunca actúa. Un error extremo cuesta como máximo 34,54 en una fila; los dos topes difieren en 8·10⁻⁴ porque 1 − ε se redondea en float64.
- **AP sin positivas:** NaN, porque la métrica no está definida (scikit-learn devolvería 0,0).
- **Mejora frente a una referencia:** `improvement(score, reference)` = 1 − score/reference, para la mejora de log-loss y para el BSS; falla si la referencia es ≤ 0.
- **Comprobaciones de las líneas base:** fallan si un municipio evaluado no tiene entrenamiento o si alguna etiqueta de entrenamiento es posterior a su t0 (D7.1). `yearly_trials` falla si un municipio no tiene filas en el año evaluado.

**F4**
- Pruebas sintéticas de contrato §10, punto 13:
  - una emisión positiva aislada no produce alerta;
  - la alerta se fecha en la segunda semana de la racha;
  - un inicio 12 semanas después de t_c se empareja y uno a 13 semanas, no;
  - una alerta de continuidad no cuenta como detección ni como falsa alarma;
  - la censura excluye todo objeto que use semanas fuera del periodo.
- Con datos reales: 10 inicios evaluables de 11 en desarrollo (se censura Villavicencio 2019-01-28) y 10 de 10 en prueba.
- El módulo de eventos cuenta inicios en prueba, pero no calcula detección ni falsas alarmas de ningún sistema sobre la prueba: eso pasa por la barrera.

**Lo que se congeló en F4 [H].** Archivos `tests/acceptance/test_f4_events.py` (series sintéticas) y `test_f4_periods.py` (datos reales, `--channel-parquet`), commit 332a2f6, con su regla `Edit(...)` de denegación (c770e1a). Se validaron antes del congelamiento con una implementación de referencia desechable (46 de 46) y tres variantes erróneas que fallan: alerta fechada en la primera semana, W = 13 y sin censura por ventana. Implementación: U4.2 en 15583fc y U4.3 en 89ef8f9. Además de las cifras anteriores:
- **Serie elegible completa:** 31 inicios y 174 entradas por municipio (§6.F1), con `outbreak_onsets` y `zone_entries`.
- **Por periodo:** 11 inicios en desarrollo (5 de los 10 evaluables son de 2019) y 10 en prueba, con las fechas de `eventos_d8.py`; 45 y 41 entradas; 24 y 18 municipio-años. Las entradas evaluables (44 y 41) no se congelaron, porque el contrato no las reporta.
- **API:** `alerts(weeks, positive, zones, *, k)`, `model_positive(p, *, tau)`, `channel_positive(zones)`, `outbreak_onsets`, `zone_entries`, `period_onsets(rows, observed, *, window)` (sin barrera), `municipality_years(rows)` y `evaluate_period(rows, positive, observed, *, k, window, access) -> PeriodEvents` (barrera primero). Las emisiones de un periodo son las filas evaluadas de `splits.yearly_trials`.

**Convenciones de F4 [D, aprobadas por Jesús el 2026-11-11].**
- **Pertenencia y censura.** Por municipio, F es la primera emisión y L la última semana objetivo del periodo. Un inicio o una entrada pertenece al periodo si s ∈ [primer objetivo, L]. Los inicios se identifican en la serie elegible completa y después se censuran:
  - por `"ventana"` si s − (W + 1) < F;
  - solo los inicios, por `"confirmacion"` si s + 2 > L.

  Las entradas se conocen en s y solo se censuran por ventana. Una alerta es juzgable si t_c + W + 2 ≤ L.
- **Aviso y razón de alertas.** Una alerta no juzgable puede avisar un inicio evaluable y cuenta en la detección. La razón de alertas solo cuenta en el numerador los inicios avisados por alguna alerta juzgable, para que numerador y denominador sean la misma población. Caso real: Villavicencio 2022-10-31 en desarrollo, avisado por el canal solo con t_c = 2022-10-03.
- **Inicios en semanas de emisión anteriores al primer objetivo** (nov–dic del año previo). No entran en la detección, pero evitan una falsa alarma. Con los datos reales no hay ninguno.
- **Constantes.** 3 y 2 de la definición de inicio son constantes de módulo (`ONSET_MIN_RUN` y `ONSET_PRIOR_BELOW`). W + 1 y W + 2 se derivan de W, y k y W no tienen valor por defecto.
- **Desglose por año.** El inicio cuenta en el año de s y la alerta en el año de la semana objetivo de su fila t_c.
- **Alineación.** `positive` como `pd.Series` debe tener exactamente el índice de `rows`; como array es posicional.
- **Nota para F7.** Una alerta juzgable seguida de un inicio censurado no es falsa alarma: entra en el denominador de la razón de alertas sin sumar en el numerador.

**Canal en desarrollo [C, solo informado; primer cálculo con la regla de contrato §7.4, §10.15].**
- Detección 10/10, con anticipación mediana de 3 semanas (rango 1–7).
- 63 alertas juzgables y 48 falsas alarmas, es decir, 2,0 por municipio-año; 12 de ellas seguidas de alguna semana en zona epidémica.
- Razón de alertas 9/63; 0 alertas de continuidad.
- Detección de entradas 36/44.
- Ninguna cifra se congeló ni se usó para decidir. No se calculó nada del canal sobre la prueba.

**F5a**
- Umbrales: 13 de 18 (azar 0,048), 17 de 24 (azar 0,032), McNemar de una cola {d < 5: imposible; 5→5, 6→6, 7→7, 8→7, 9→8, 10→9} y 4 de 4 = 0,0625.
- Un empate no cuenta como victoria.
- El puntaje fuera de año de una configuración nunca usa el año en que se puntúa.
- La lista de candidatos es idéntica en los seis municipios; si difiere, falla.
- En una corrida de humo con dos candidatos y datos reales de desarrollo, la tabla única tiene 6 × 2 × 4 filas por familia y ninguna fila de 2023–2025.

**Convenciones de F5a [D, aprobadas por Jesús el 2026-11-11].**
- **Desempate.** Si dos candidatos tienen la misma mejora promedio frente a B2, se elige el que aparece primero en la lista pre-registrada. La tabla única guarda esa posición en la columna `orden`.
- **Familia y excepciones de hiperparámetros.** Resuelto en el pre-registro (§7, «Orden de selección para F6»): las familias se comparan con sus configuraciones comunes fuera de año. F5a solo entrega, para cada casilla, la configuración común y la propia, ambas elegidas sin el año puntuado.
- **Predicciones por fila.** `run_search` también guarda en `data/processed/forecasting/` las predicciones por fila (`municipality_code`, `t`, `target_week`, `familia`, `candidato`, `anio`, `y`, `p`), para que F6 elija τ sin reentrenar.
- **Aceptación añadida a la lista anterior.** Hay que reproducir lo siguiente:
  - las predicciones tienen 1.248 filas por (familia, candidato) y ninguna de 2023 o posterior;
  - el log-loss recalculado desde ellas es igual, de forma exacta, al de la tabla única;
  - `logloss_b2` es el mismo para todos los candidatos de una casilla y coincide exactamente con `b2_predict` más `log_loss`;
  - hay un run terminado por (municipio, familia, candidato);
  - se cumplen los niveles de §8.4 y los de eventos.

**Lo que se congeló en F5a [H].**
- **Pruebas.** Archivos `tests/acceptance/test_f5a_comparison.py` (29 pruebas sintéticas) y `test_f5a_search.py` (8 pruebas con `--channel-parquet`), en el commit 6a2dd12, con su regla `Edit(...)` de denegación en 2d390c7.
- **Validación previa.** Antes del congelamiento se validaron con una implementación de referencia desechable (37 de 37) y con cinco variantes erróneas, que fallan donde deben:
  - umbral de dos colas;
  - empate contado como victoria;
  - elección que incluye el año puntuado;
  - sin validación de la lista de candidatos;
  - búsqueda que puntúa 2023.
- **Implementación.**

  | Unidad | Commit | Contenido |
  |---|---|---|
  | U5.2 | 5da5716 | `models.py` |
  | U5.3 | c5128cb | `search.py`, esquema `busqueda` y `tracking.allow_file_store` |
  | U5.4 | 9c2ffc7 | `comparison.py` y `comparacion.alfa` |

  Al cierre de F5a pasa toda la aceptación con datos: 261 de 261.
- **API.**
  - `search`: `Candidate(name, params)`, `candidates_from_config(busqueda)`, `run_search(examples, candidates, *, config, output_dir, tracking_uri)`, `validate_score_table(table, *, municipalities, years)` y la línea de comandos `python -m src.forecasting.search [--config] [--salida] [--permitir-cambios]`.
  - `comparison`: `chance_probability`, `win_threshold`, `count_wins`, `discordant_wins`, `out_of_year_scores`, `weekly_level`, `event_level` y `combined_level`.
  - `models`: `make_model`, `fit_model` y `predict_positive`, con las familias `logistica` y `lightgbm`.
- **Salidas de la búsqueda** en `data/processed/forecasting/`:
  - `tabla_unica_desarrollo.parquet` y `predicciones_desarrollo.parquet`;
  - `lineas_base_desarrollo.parquet` y `predicciones_lineas_base_desarrollo.parquet`;
  - `busqueda_manifest.json`, `busqueda.log` y la carpeta `partes/`.
- **Reanudación.**
  - Cada unidad (municipio, familia, candidato) se escribe de forma atómica y se salta al relanzar.
  - Un run FINISHED de una unidad sin carpeta se reutiliza si sus métricas coinciden (rel = 1e-12).
  - Solo se marcan como FAILED los runs RUNNING con el mismo `busqueda_id`.
  - El `busqueda_id` incluye el commit: una búsqueda no se reanuda después de un commit nuevo.
  - La línea de comandos se niega a correr con cambios sin commit en `src/` o `configs/`, salvo con `--permitir-cambios`.

**Humo de F5a (U5.5) [H, 2026-11-11].**
- **Configuración.** Datos reales de desarrollo con 2 candidatos por familia: regresión logística con C ∈ {0,01; 3} y los dos primeros LightGBM del espacio de §7. Todo se guardó fuera del repositorio, con MLflow local.
- **Corte forzado.** La línea de comandos se mató (TerminateProcess) tras 14 de 36 unidades. Al relanzarla:
  - saltó las 14 unidades hechas;
  - quedaron 36 runs FINISHED únicos y ninguna carpeta temporal;
  - la tabla y las predicciones son idénticas (sin contar `segundos`) a las de una corrida sin corte.
- **Tiempos.**
  - Por ensayo, el ajuste y la predicción tardan 0,06 s en LightGBM y 0,01 s en la regresión logística.
  - Por unidad (4 ensayos, métricas y run local), 0,25–0,37 s y 0,06–0,15 s.
  - Los 25 LightGBM de §7, medidos sobre datos sintéticos del mismo tamaño, tardan 7,0 s por municipio (máximo 0,58 s por candidato).
- **Estimación para F5b** (198 unidades): alrededor de 1 minuto de cálculo, más la latencia de unos 200 runs en el servidor (3 llamadas cada uno), es decir, entre 10 y 25 minutos en total. No hace falta paralelismo.

**F5b**
- La tabla única queda completa: 6 municipios × número de candidatos × 4 años, por familia, sin faltantes. Hay un run de MLflow (o del respaldo local) por municipio, familia y candidato.

**Cierre de F5b [H, 2026-11-11].**
- **Ejecución.** Búsqueda `7f34b69832eea69407c6`, lanzada por Jesús desde Git Bash con `python -B -m src.forecasting.search`:
  - código en el commit cd9d702 (pre-registro de §7);
  - servidor MLflow, experimento `dengue_h8_desarrollo`;
  - de 23:43:40 a 23:47:11 (hora de Colombia), es decir, **3 min 31 s**;
  - 198 de 198 unidades: 6 municipios × (B2, B1, 6 de regresión logística y 25 de LightGBM);
  - salida 0, sin unidades saltadas, reutilizadas ni fallidas.
- **Tiempo por unidad.** Media de 1,04 s en LightGBM, 1,13 s en regresión logística y 0,8 s en las líneas base. Casi todo el tiempo es la latencia del registro en el servidor: en local, una unidad de LightGBM tardaba 0,25–0,37 s.
- **Conteos** (comprobados por Jesús):
  - tabla única con 600 filas de LightGBM (6 × 25 × 4) y 144 de regresión logística (6 × 6 × 4);
  - años 2019–2022;
  - última semana objetivo 2022-12-26;
  - predicciones por fila: 31.200 de LightGBM y 7.488 de regresión logística, es decir, 1.248 por candidato.
- **Runs.** Según el log, se registró un run por unidad: 198 en total (186 de modelos y 12 de líneas base). Claude no consultó el servidor.
- **Versionado con DVC.** Cinco archivos, cada uno con su `.dvc` en git:

  | Archivo | md5 |
  |---|---|
  | `tabla_unica_desarrollo.parquet` | `a38ab5ef…0c` |
  | `predicciones_desarrollo.parquet` | `ac5a28f3…ca` |
  | `lineas_base_desarrollo.parquet` | `a826f366…2a` |
  | `predicciones_lineas_base_desarrollo.parquet` | `7fb8390c…c5` |
  | `busqueda_manifest.json` | `48b930bf…89` |

  `partes/` y `busqueda.log` quedan solo en local, ignorados por las reglas existentes de `.gitignore`.
- **Resultados.** Ningún resultado de la búsqueda (qué candidato o qué familia gana) se miró ni se reportó al cerrar F5b. Las decisiones son de F6, con el orden de §7.

**F6**
- `receta_congelada.yaml` tiene todos los campos de §8.
- Volver a ejecutar la selección a partir de la tabla única guardada produce la misma receta.
- τ pertenece a la rejilla y la restricción de falsas alarmas se cumple, o se reporta que no se cumple (contrato §7.5).

**F7**
- La bitácora registra una ejecución de la prueba.
- **Análisis diagnóstico integrado (contrato §8.8 y §10.22, D9.1).**
  - Un módulo genera los seis insumos de §8.8 a partir de predicciones guardadas, sin reentrenar: tabla de casillas, resúmenes por municipio, año y régimen (con las dos agregaciones), estratos de transición por año, inicios y entradas, alertas juzgables y serie semanal.
  - Desarrollo: se calcula con las predicciones de F5b y las configuraciones fuera de año de la familia elegida; no pasa por la barrera. Prueba: solo después de la receta, con la barrera.
    - La configuración diagnosticada en cada casilla de desarrollo es la común fuera de año (elegida sin y) de la familia elegida, es decir, `candidato_comun` de `comparison.out_of_year_scores`. Si hubiera excepciones, se usa la que §8.3 compara en cada caso.
    - El régimen se clasifica sobre el año calendario de la serie elegible completa, con los episodios identificados en toda la serie, no solo sobre las filas evaluadas. El año de una casilla es el de sus semanas objetivo. Para 2019 y 2023, el año anterior (2018, 2022) se clasifica con la misma regla.
  - Pruebas sintéticas de la clasificación de régimen: episodio de 3 semanas que cruza el cambio de año (ambos años epidémicos), año sin episodio tras uno epidémico (post-epidémico), año con una racha de 2 semanas (no epidémico).
  - Las cifras de desarrollo (victorias y diferencias por casilla, conteos por régimen) se comparan con un cálculo independiente de este chat antes de congelarlas, como en F1–F6.
  - Los insumos son parte del mínimo obligatorio de F7 (§5, orden de recorte): sin ellos el análisis no se puede reconstruir. La redacción cualitativa puede pasar a la semana 7.
- `/historial` solo contiene predicciones del origen móvil y ninguna del modelo final con t ≤ 2025-10-27.
- La ficha contiene todos los campos de `api_salida_d10.md` §7 y la tabla de 18 casillas.

**F8**
- Las pruebas de contrato §10, punto 19.

## 7. Pre-registro [D, aprobado por Jesús el 2026-11-11, antes de F5b]

Contrato §6.3 exige fijarlo antes de correr el desarrollo. Se discutió con el arnés ya probado (F5a) y quedó en la sección `busqueda` de `configs/forecasting.yaml`. `src/forecasting/search.py` lo expande en candidatos con `candidates_from_config`.

| Familia | Pre-registro |
|---|---|
| Regresión logística (L2) | C ∈ {0,01; 0,03; 0,1; 0,3; 1; 3}, en ese orden; `max_iter` = 5000; `class_weight=None`. Si no converge, es un error (`models.py`); en el humo convergieron C = 0,01 y C = 3 |
| LightGBM | 25 candidatos aleatorios. Espacio: `max_depth` entero 2–5; `num_leaves` entero 4–31 con tope 2^`max_depth`; `learning_rate` 0,01–0,1 en escala log; `n_estimators` entero 100–600; `min_child_samples` entero 10–50; `reg_lambda` real 0–10; `subsample` y `colsample_bytree` reales 0,6–1,0. Fijo: `subsample_freq` = 1. Sin pesos de clase ni parada temprana; un hilo y modo determinista |
| GRU | **Fuera de F5b** (orden de recorte de §5). Si hay tiempo, se reporta después de F6 como análisis adicional, sin cambiar la familia elegida (contrato §4.3) |
| Semillas | Semilla global 20261112, la misma en los seis municipios: `semilla_muestreo` (numpy `default_rng`, con los parámetros sorteados en orden alfabético) y `semilla_modelo` (`random_state` de todos los candidatos) |

**Observaciones de LightGBM, resueltas** (documentación de LightGBM 4.6.0, la versión fijada en F0):
- **`num_leaves`.** Con `max_depth` d, un árbol tiene como mucho 2^d hojas, y la documentación recomienda `num_leaves` ≤ 2^`max_depth`. Por eso `num_leaves` se sortea después de `max_depth`, con tope 2^`max_depth` (`tope_por_profundidad`). Consecuencia: con profundidad 2 el único valor posible es 4. Con la semilla 20261112, 8 de los 25 candidatos tienen profundidad 2 y 4 hojas.
- **`subsample`.** Según la documentación, `subsample_freq` ≤ 0 desactiva el submuestreo. Por eso se fija `subsample_freq` = 1.
- **Ajustes categóricos** (`min_data_per_group`, `cat_smooth`, `max_cat_to_onehot`). Quedan con los valores por defecto. Con 3 zonas se usa la partición uno contra el resto, y una zona con 30 filas sí se separa (prueba de U5.2; en los entrenamientos reales, la zona epidemia tiene entre 21 y 95 filas).

**Orden de selección para F6 [D].** Todo con puntajes fuera de año y las reglas de D9 (contrato §8.3, `comparison.py`):
1. **Configuración común de cada familia:** el candidato con la mayor mejora promedio frente a B2, primero sobre los años y después sobre los municipios (contrato §6.2).
2. **Comparación de familias con las configuraciones comunes fuera de año.** LightGBM reemplaza a la regresión logística solo si su mejora promedio fuera de año es mayor y además gana al menos 17 de las 24 casillas.
3. **Excepción de familia por municipio, con configuraciones comunes:** la familia alternativa gana a la común en ese municipio en los 4 años.
4. **Excepciones de hiperparámetros dentro de la familia final de cada municipio:** el mejor candidato propio, elegido sin y, gana a la configuración común, elegida sin y, en los 4 años. Si se cumple, el municipio usa su mejor candidato propio elegido con los 4 años.

**Desempate:** ante una mejora promedio exactamente igual, se elige el candidato de menor `orden`, es decir, el primero en la lista pre-registrada.

**Por qué este orden.** La regla de excepción necesita los 4 años, así que no puede calcularse fuera de año. Si las excepciones de hiperparámetros entraran en la comparación de familias, el año puntuado influiría en su propia elección. Por eso las familias se comparan con configuraciones comunes y las excepciones se aplican al final.

**Seguimiento.** `seguimiento.modo` pasa a `servidor` en el mismo commit del pre-registro (§9). El experimento `dengue_h8_desarrollo` es compartido: el arnés solo marca o reutiliza runs con su propio `busqueda_id`.

## 8. Configuración [D]

**`configs/forecasting.yaml`**, validado con Pydantic:

| Sección | Contenido |
|---|---|
| `datos` | Ruta del producto del panel y del canal; versión DVC esperada |
| `canal` | n = 5; desplazamiento = 1; cuantil t de 0,025 |
| `horizonte` | h = 8; sobrescribible con `FORECAST_HORIZON` |
| `features` | w = 4; bloques m1–m3; ventana de 12 semanas |
| `validacion` | Años de desarrollo [2019, 2022] y de prueba [2023, 2025]; primer año de entrenamiento 2012 |
| `lineas_base` | Semiventana de B1, k = 2 (contrato §4.1; F3) |
| `metricas` | Recorte del log-loss, ε = 1e-15 (contrato §10.16; F3) |
| `comparacion` | α = 0,05, el umbral de azar de las reglas de conteo de D9 (contrato §8.2; F5a) |
| `decision` | k = 2; W = 12; rejilla de τ (0,05 a 0,95, paso 0,05) |
| `busqueda` | Pre-registro (§7, aprobado el 2026-11-11): rejilla de C, espacio de LightGBM, presupuesto de 25 candidatos y semillas |
| `seguimiento` | Nombres de experimentos y modo (servidor o respaldo local) |

**`configs/receta_congelada.yaml`**, generado en F6:
- familia común y excepciones por municipio, con sus diferencias anuales;
- hiperparámetros por municipio;
- τ, k y W;
- hash SHA-256 del contenido sin el propio campo de hash;
- commit del código, commit de `contrato_predictivo.md` y versión DVC del canal;
- aprobación: nombre y fecha de Jesús. **Este campo lo escribe Jesús o se escribe con su instrucción explícita en el chat.**

## 9. Trazabilidad y MLflow [D, salvo lo indicado]

**Servidor del equipo [H, mensaje del compañero que administra la infraestructura y `src/api.py`]:**
- MLflow en `https://platform.arkoiot.com`, protegido por Cloudflare Zero Trust. Cada petición necesita los encabezados `CF-Access-Client-Id` y `CF-Access-Client-Secret`, tomados de `.env` (`.env.example` ya lo documenta; `.env` está en `.gitignore`).
- El mecanismo es un `RequestHeaderProvider` registrado en el registro de MLflow, que es una API privada (`_request_header_provider_registry`).
- Hay RBAC con cuatro roles: Administradores, Analistas, Runners y Solo Lectura.
- La API descubre modelos registrados con nombre `DengueAlertas_*` y la etiqueta `enabled = true`. Los carga por el alias igual a `STAGE` en minúscula (`staging` o `production`) y lee las etiquetas `municipality_code`, `department` y `depto_id` si existen; no son obligatorias. [H, F0 U0.7] Los modelos `DengueAlertas_*` registrados hoy solo tienen la etiqueta `enabled` y el alias `staging`.
- Hay dos ambientes: producción (`demo.arkoiot.com`) y pruebas (`staging.denguealertas.pages.dev`). Según el compañero, la API valida cada 45 segundos y toma el esquema registrado del modelo.

**`src/forecasting/tracking.py` [R]:**
- Toma el mecanismo de `mlflow_auth.py` del compañero y el de `src/api.py`, con dos cambios:
  - la URI de seguimiento sale de `MLFLOW_TRACKING_URI`, con `https://platform.arkoiot.com` por defecto;
  - si faltan las credenciales, falla de inmediato con un mensaje claro, en lugar de enviar encabezados vacíos.
- Como depende de una API privada de MLflow, la versión del cliente se fija en `requirements/forecasting.txt`, igual a la del servidor. [H, F0 U0.7] El servidor corre **MLflow 3.17.0** (endpoint `/version`, leído con `scripts/check_mlflow_readonly.py`); el cliente fijado es `mlflow-skinny==3.17.0`.

**Experimentos:**
- `dengue_h8_desarrollo`: un run por municipio, familia y candidato, con los puntajes de cada año como métricas separadas (`logloss_2019` … `logloss_2022`, `brier_YYYY`, `mejora_b2_YYYY`). Las líneas base van como runs con familia `B1` o `B2`. En la GRU, un run por arquitectura y semilla.
- `dengue_h8_prueba`: la prueba única, la ablación del clima y las líneas base, con la ficha como artefacto.
- `dengue_h8_final`: los modelos finales por municipio.

**Etiquetas de cada run:** `municipio`, `familia`, `candidato`, `fase`, `commit`, `contrato_commit`, `datos_dvc` y `receta_hash` (desde F6).

**La fuente de verdad de las decisiones es la tabla única versionada en `data/processed/forecasting/`.** MLflow es el registro navegable. Si MLflow falla, ninguna decisión se pierde.

**Respaldo local [D, contingencia].** Si en F0 el servidor no responde o Jesús no tiene credenciales o un rol con permiso de escritura, los runs van a un almacén local dentro del repositorio, ignorado por git (`mlruns/` ya está en `.gitignore`), que se respalda aparte. Solo los modelos finales se registran en el servidor en F8. F0 elige el tipo de almacén según la versión de MLflow fijada.

**Cliente y almacén fijados en F0 [D, U0.3].** El cliente es `mlflow-skinny`, primero 2.16.2 y, desde el cierre de F0, **3.17.0**, igual a la versión del servidor. Razón de usar skinny: el paquete `mlflow` completo 2.16.2 exige `pyarrow<18` y choca con `pyarrow==21.0.0` de `requirements/data.txt`; skinny trae el mismo cliente de seguimiento sin esa dependencia. Como skinny no incluye SQLAlchemy, el respaldo local es el almacén de archivos en `mlruns/`. `seguimiento.modo` queda en `local` hasta antes de F5b, porque los runs del servidor no se pueden borrar; el regreso a `servidor` va en un commit explícito. [H] Pasó a `servidor` en el commit del pre-registro (§7, 2026-11-11).

**Almacén de archivos con MLflow 3.17 [H, F5a U5.1; D, aprobado por Jesús el 2026-11-11].** MLflow 3.17 rechaza el almacén de archivos (`mlruns/`) con una excepción, salvo que esté activada `MLFLOW_ALLOW_FILE_STORE=true`. En F0 no se detectó porque las pruebas de `tracking.py` simulan `set_tracking_uri` y nunca escribieron un run. El respaldo local de F0 necesita esa opción y se activa así:
- en el entorno del propio proceso (`os.environ`), justo antes de apuntar MLflow a la URI;
- solo cuando la URI de seguimiento empieza por `file:`;
- nunca como variable global del sistema o del usuario, ni en `.env`, ni al importar un módulo.

En modo servidor no se activa.

**Registro de modelos finales [P: acordar en F8 con quien administra la API].** Propuesta, compatible con `src/api.py`:
- un modelo registrado por municipio, `DengueAlertas_<Municipio>`;
- etiquetas `enabled = true`, `municipality_code`, `department`, `depto_id`, `receta_hash` y `familia`;
- alias `staging` primero; `production` después de validar la demo;
- firma de entrada y salida obligatoria.

[H] El mapa de enrutamiento de `src/api.py` hoy contiene otros municipios (Cartagena, Bucaramanga, Ibagué) y tiene que adaptarse a los seis pilotos.

## 10. Variables de entorno [D; sustituye la tabla de `api_salida_d10.md` §9]

| Variable | Estado [H, `src/api.py` y `Dockerfile`] | Por defecto | Uso |
|---|---|---|---|
| `STAGE` | Existe | `Production` | Ambiente; elige el alias del modelo |
| `MLFLOW_TRACKING_URI` | Existe | `https://platform.arkoiot.com` | Servidor MLflow (API y pipeline) |
| `CF_ACCESS_CLIENT_ID`, `CF_ACCESS_CLIENT_SECRET` | Existen | — | Cloudflare Access; solo en `.env` o en el entorno del servidor |
| `MODEL_THRESHOLD` | Existe (hoy `0.205`) | τ de la receta | τ. Si difiere del congelado: `parametros_congelados = false` y advertencia `parametros_modificados` |
| `MODEL_NAME` | Existe | — | Modelo base de la API actual; se revisa en F8 |
| `LOG_DIR` | Existe | — | Bitácoras de la API |
| `FORECAST_HORIZON` | Nueva | `8` | h; solo valores con modelo entrenado (pipeline y API) |
| `ALERT_K` | Nueva | `2` | k; si difiere, igual que τ |
| `DATA_VERSION` | Nueva | — | Versión DVC del canal que se informa en `versiones.datos` |
| URL de la API en el frontend | [P] | — | `frontend/src` hoy no lee variables `VITE_*` [H]; se define con Santiago |

**[H] Conflicto con el contrato que hay que resolver en F8:** `/predict` hoy acepta un parámetro `threshold` del usuario. Contrato §9.7 excluye un τ ajustable por el usuario. En F8 se elimina, o se conserva y entonces la respuesta marca `parametros_congelados = false`.

## 11. Definición de terminado

- Los 22 requisitos de contrato §10 están cumplidos o tienen su desviación registrada.
- Las pruebas están en verde: las del legado, las de `tests/forecasting/` y todas las de aceptación ejecutadas con datos.
- La ficha de desempeño se generó desde la prueba única registrada en la bitácora.
- La API y el frontend sirven las cuatro vistas.
- El README tiene ejemplos de entrada y salida, variables de entorno y la gestión del modelo en MLflow.

## 12. Preguntas abiertas

**Resueltas:**
- ~~Qué rol tiene Jesús y si puede crear experimentos y registrar modelos.~~ Rol **Runners**: puede crear experimentos y registrar modelos en `platform.arkoiot.com` [respuesta de Jesús, 2026-11-11]. `seguimiento.modo` sigue en `local` hasta antes de F5b (§9).
- ~~Qué versión de MLflow corre en el servidor.~~ **MLflow 3.17.0** [H, F0 U0.7; §9]; cliente `mlflow-skinny==3.17.0`.

**Abiertas:**
- **Para el compañero que administra arkoiot** (antes del jueves 12, porque F5b lo necesita):
  - si las credenciales de DVC y las de MLflow son las mismas personas o cuentas;
  - cómo conviven los modelos `DengueAlertas_*` ya registrados (Bucaramanga, Cali, Cartagena, Ibagué, Villavicencio) con los seis pilotos;
  - si `/predict` y el descubrimiento de modelos soportan seis modelos `DengueAlertas_*` con las etiquetas de §9;
  - qué firma espera la API.
- **Rol de Daniel** y reparto de fases dentro del equipo; en particular, quién toma F8 de la API.
- Nombre de la variable de la URL de la API en el frontend.
