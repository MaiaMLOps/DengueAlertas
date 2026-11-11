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
| `src/epidemiology/channel_epico.py` | Canal de epiCo: central, gsd, U1e y zonas; `build_channel_table`, transformación pura del panel al producto (sin E/S ni lectura de configuración) | 1, 2, 3, 4 |
| `src/forecasting/build_channel.py` | Línea de comandos del producto del canal: carga la sección `canal` de la configuración, lee el panel y calcula su hash, llama a `build_channel_table` y escribe el producto y su manifiesto sin sobrescribir | 2 |
| `src/forecasting/config.py` | Carga y validación de `configs/forecasting.yaml` con Pydantic; sobrescritura de h por variable de entorno | 5, 14 |
| `src/forecasting/dataset.py` | Ejemplos supervisados (t, t + h) y las 15 features; entrada de la GRU | 5, 7 |
| `src/forecasting/splits.py` | Cortes por año de la semana objetivo | 11 |
| `src/forecasting/guard.py` | **Barrera de prueba:** toda función que puntúe filas con semana objetivo en los años de prueba exige una receta congelada válida, con hash y aprobación; si no existe, lanza un error | §1, regla 1 |
| `src/forecasting/baselines.py` | B1 y B2 (Laplace, por pliegue) y alerta descriptiva del canal | 8 |
| `src/forecasting/models.py` | Pipelines de regresión logística, LightGBM y GRU opcional | 9, 10 |
| `src/forecasting/metrics.py` | Métricas de estimación y de clasificación | 16 |
| `src/forecasting/events.py` | Alertas, alertas de continuidad, inicios, emparejamiento, censura y métricas de eventos | 13, 14, 15 |
| `src/forecasting/search.py` | Arnés de búsqueda y tabla única de puntajes | 12 |
| `src/forecasting/comparison.py` | Puntajes fuera de año, conteo de victorias, binomial y niveles (D9) | 17 |
| `src/forecasting/selection.py` | Hiperparámetros → familia → τ → receta congelada | §6.2, §8.3, §7.5 |
| `src/forecasting/evaluate.py` | Prueba única, ablación y ficha de desempeño | §8.4 |
| `src/forecasting/artifacts.py` | Historial fuera de muestra, ficha, modelo final y registro | 18 |
| `src/forecasting/tracking.py` | Conexión a MLflow con los encabezados de Cloudflare Access (§9) y respaldo local | 9 |
| `configs/forecasting.yaml` | Parámetros (§8) | 5, 14 |
| `configs/receta_congelada.yaml` | Se genera en F6 y lo aprueba Jesús (§8) | §6.1 |
| `requirements/forecasting.txt` | Dependencias de modelado, fijadas en F0 | — |
| `tests/forecasting/` | Pruebas unitarias | Todos |
| `tests/acceptance/` | Pruebas de aceptación por fase (§6) | Todos |

**Canal: reparto y reutilización [D, F1].**
- La dependencia va de `src/forecasting/` a `src/epidemiology/`, nunca al revés: las funciones del canal reciben `n_years`, `shift` y `alpha` solo por nombre y sin valores por defecto, y `first_year` es `src/epidemiology/config.FIRST_LABEL_YEAR`.
- `build_channel_table` exige el calendario del panel (`validate_source`: desde 2007-01-01 hasta, como máximo, 2025-12-31), así que no sirve para semanas posteriores ni para datos subidos.
- **Pendiente para F2:** una función del canal que no dependa del calendario del panel y calcule central y U1e para una semana arbitraria a partir de la serie de conteos del municipio (por ejemplo, U1e_{t+8} y las bandas hasta t + 8 de contrato §9.3).
- F8 reutiliza la lógica del canal mediante esa función. En F8 se decide si `/predict/upload` usa esa función o una versión de `build_channel_table` con la validación relajada.
- Producto: `data/processed/epidemiology/pilot_weekly_channel.parquet`, con el manifiesto `pilot_weekly_channel_manifest.json` (no `run_manifest.json`, que es del legado). `configs/forecasting.yaml` fija `datos.version_dvc_canal` con el md5 del `.dvc`.
- **Verificación del producto (F2).** Antes de construir features, el cargador del canal calcula el md5 del parquet en disco y lo compara con `datos.version_dvc_canal`; si no coinciden, falla con un mensaje claro. No basta con el md5 que declara el `.dvc`.

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

**F3**
- Desarrollo: 208 filas evaluadas por municipio, 1.248 en total, 319 positivas.
- Prueba: 156 por municipio, 936 en total, 322 positivas.
- En cada ensayo Y, ninguna fila de entrenamiento tiene la semana objetivo en Y ni después, y las filas evaluadas son exactamente las del año Y.
- B1 y B2 se estiman solo con el entrenamiento del ensayo; con datos sintéticos, el suavizado de Laplace da (positivos + 1)/(casos + 2).
- **Barrera:** pedir una métrica sobre filas de 2023–2025 sin una receta válida lanza el error de `guard.py`; con una receta de prueba sintética válida, no.

**F4**
- Pruebas sintéticas de contrato §10, punto 13:
  - una emisión positiva aislada no produce alerta;
  - la alerta se fecha en la segunda semana de la racha;
  - un inicio 12 semanas después de t_c se empareja y uno a 13 semanas, no;
  - una alerta de continuidad no cuenta como detección ni como falsa alarma;
  - la censura excluye todo objeto que use semanas fuera del periodo.
- Con datos reales: 10 inicios evaluables de 11 en desarrollo (se censura Villavicencio 2019-01-28) y 10 de 10 en prueba.
- El módulo de eventos cuenta inicios en prueba, pero no calcula detección ni falsas alarmas de ningún sistema sobre la prueba: eso pasa por la barrera.

**F5a**
- Umbrales: 13 de 18 (azar 0,048), 17 de 24 (azar 0,032), McNemar de una cola {d < 5: imposible; 5→5, 6→6, 7→7, 8→7, 9→8, 10→9} y 4 de 4 = 0,0625.
- Un empate no cuenta como victoria.
- El puntaje fuera de año de una configuración nunca usa el año en que se puntúa.
- La lista de candidatos es idéntica en los seis municipios; si difiere, falla.
- En una corrida de humo con dos candidatos y datos reales de desarrollo, la tabla única tiene 6 × 2 × 4 filas por familia y ninguna fila de 2023–2025.

**F5b**
- La tabla única queda completa: 6 municipios × número de candidatos × 4 años, por familia, sin faltantes. Hay un run de MLflow (o del respaldo local) por municipio, familia y candidato.

**F6**
- `receta_congelada.yaml` tiene todos los campos de §8.
- Volver a ejecutar la selección a partir de la tabla única guardada produce la misma receta.
- τ pertenece a la rejilla y la restricción de falsas alarmas se cumple, o se reporta que no se cumple (contrato §7.5).

**F7**
- La bitácora registra una ejecución de la prueba.
- `/historial` solo contiene predicciones del origen móvil y ninguna del modelo final con t ≤ 2025-10-27.
- La ficha contiene todos los campos de `api_salida_d10.md` §7 y la tabla de 18 casillas.

**F8**
- Las pruebas de contrato §10, punto 19.

## 7. Pre-registro [P: se aprueba en el punto de control del jueves 12, antes de F5b]

Contrato §6.3 exige fijarlo antes de correr el desarrollo. Esta propuesta inicial se discute con el arnés ya probado (F5a).

| Familia | Propuesta inicial |
|---|---|
| Regresión logística (L2) | C ∈ {0,01; 0,03; 0,1; 0,3; 1; 3}; `class_weight=None`; `max_iter` suficiente para converger |
| LightGBM | 25 candidatos aleatorios con semilla fija. Espacio orientado a regularizar: `num_leaves` 4–31, `max_depth` 2–5, `learning_rate` 0,01–0,1 (escala log), `n_estimators` 100–600, `min_child_samples` 10–50, `reg_lambda` 0–10, `subsample` y `colsample_bytree` 0,6–1,0 |
| GRU (opcional) | 2–3 arquitecturas de una capa (16 o 32 unidades, dropout 0,2), parada temprana con el último año de cada entrenamiento, 5 semillas |
| Semillas | Una semilla global documentada, la misma en los seis municipios |

**Puntos que hay que resolver en el punto de control [I]:**
- Con `max_depth` d, un árbol tiene como mucho 2^d hojas. `num_leaves` > 2^`max_depth` no tiene efecto, así que conviene acotar `num_leaves` en función de `max_depth` o muestrear solo uno de los dos.
- `subsample` < 1 solo tiene efecto en LightGBM si `subsample_freq` > 0 (verificar con la documentación de la versión fijada en F0).

## 8. Configuración [D]

**`configs/forecasting.yaml`**, validado con Pydantic:

| Sección | Contenido |
|---|---|
| `datos` | Ruta del producto del panel y del canal; versión DVC esperada |
| `canal` | n = 5; desplazamiento = 1; cuantil t de 0,025 |
| `horizonte` | h = 8; sobrescribible con `FORECAST_HORIZON` |
| `features` | w = 4; bloques m1–m3; ventana de 12 semanas |
| `validacion` | Años de desarrollo [2019, 2022] y de prueba [2023, 2025]; primer año de entrenamiento 2012 |
| `decision` | k = 2; W = 12; rejilla de τ (0,05 a 0,95, paso 0,05) |
| `busqueda` | Vacía hasta el pre-registro (§7); después, espacios, presupuesto y semillas |
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

**Cliente y almacén fijados en F0 [D, U0.3].** El cliente es `mlflow-skinny`, primero 2.16.2 y, desde el cierre de F0, **3.17.0**, igual a la versión del servidor. Razón de usar skinny: el paquete `mlflow` completo 2.16.2 exige `pyarrow<18` y choca con `pyarrow==21.0.0` de `requirements/data.txt`; skinny trae el mismo cliente de seguimiento sin esa dependencia. Como skinny no incluye SQLAlchemy, el respaldo local es el almacén de archivos en `mlruns/`. `seguimiento.modo` queda en `local` hasta antes de F5b, porque los runs del servidor no se pueden borrar; el regreso a `servidor` va en un commit explícito.

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

- Los 21 requisitos de contrato §10 están cumplidos o tienen su desviación registrada.
- Las pruebas están en verde: las del legado, las de `tests/forecasting/` y todas las de aceptación ejecutadas con datos.
- La ficha de desempeño se generó desde la prueba única registrada en la bitácora.
- La API y el frontend sirven las cuatro vistas.
- El README tiene ejemplos de entrada y salida, variables de entorno y la gestión del modelo en MLflow.

## 12. Preguntas abiertas

- **Para el compañero que administra arkoiot** (antes del jueves 12, porque F5b lo necesita):
  - qué rol tiene Jesús y si puede crear experimentos y registrar modelos;
  - qué versión de MLflow corre en el servidor;
  - si las credenciales de DVC y las de MLflow son las mismas personas o cuentas;
  - si `/predict` y el descubrimiento de modelos soportan seis modelos `DengueAlertas_*` con las etiquetas de §9;
  - qué firma espera la API.
- **Rol de Daniel** y reparto de fases dentro del equipo; en particular, quién toma F8 de la API.
- Nombre de la variable de la URL de la API en el frontend.
