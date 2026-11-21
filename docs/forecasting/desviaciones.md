# Desviaciones del contrato predictivo durante la implementación

Registro de toda desviación de `contrato_predictivo.md` aprobada por el responsable de la implementación durante la implementación en `feature/forecasting-mvp`. Cada entrada se escribe **antes** de continuar con la parte afectada. Al cerrar la semana, las entradas se consolidan en `registro_decisiones.md`.

Formato de cada entrada:

```
## DV-NN — <título> (AAAA-MM-DD)
- Cláusula afectada: contrato §…
- Problema encontrado: …
- Alternativas consideradas: …
- Decisión del responsable de la implementación: …
- Impacto en resultados o en el artículo: …
- Commit: …
```

## DV-01 — Entrenamiento cerrado en la primera emisión de cada ensayo (2026-11-11)
- Cláusula afectada: contrato §6.1 (corte de validación), §6.4 y §10, punto 11. Enmienda D7.1 en `registro_decisiones.md`.
- Problema encontrado: con el corte «entrenamiento con objetivo antes de Y», las primeras h − 1 = 7 emisiones evaluadas de cada ensayo (del t0 al t0 + 6 semanas) se hacían con un modelo entrenado con etiquetas de semanas posteriores a su emisión (por ejemplo, ensayo 2019: emisión del 2018-11-12 con etiquetas hasta el 2018-12-31). Por la autocorrelación del estado, esas etiquetas informan sobre el objetivo. La prueba de truncamiento no lo detecta, porque la información entra por los parámetros y no por las variables. Anotado durante la redacción del contrato y retomado antes de congelar las pruebas de F3.
- Alternativas consideradas: mantener el corte y declararlo como limitación; reentrenar en cada emisión (×52 entrenamientos); ensayos por año de emisión (rompe la barrera de prueba y las cifras verificadas).
- Decisión del responsable de la implementación: en el ensayo Y, t0 es la emisión más temprana entre las filas evaluadas de Y; el entrenamiento usa solo las filas con t + h ≤ t0, desde 2012. Se aplica a modelos, B1, B2, ablación, desarrollo y prueba. Las filas evaluadas no cambian.
- Impacto en resultados o en el artículo: 7 filas menos de entrenamiento por ensayo y municipio; las filas evaluadas no cambian. Se declara en el artículo que cada ensayo simula un reentrenamiento anual el día de su primera emisión.
- Commit: bf477f5 (enmienda); implementada en f0589b3 (`splits.yearly_trials`) y en las líneas base de 0fe3eca; pruebas congeladas en 9a8f607.

## DV-02 — Receta sustituida por la enmienda D7.2 antes de abrir la prueba (2026-11-12)
- Cláusula afectada: contrato §4.2, §4.4, §6.6 (nuevo), §7.1 y §8.4 (enmienda D7.2, commit 847b3ba); especificación §6.F6b y §6.F7. La receta congelada del 2026-11-12 (`configs/receta_congelada.yaml`, hash de contenido `d2135a39…da98`, commit fc31f13: logística completa, `logistica_03`, τ = 0,10) queda sustituida.
- Problema encontrado: al preparar la ablación de F7 (decisión D1), la logística sin el bloque climático (búsqueda `26dc1af622fb5966e56e`, carpeta `data/processed/forecasting/ablacion_clima/`) resultó mejor en desarrollo que la completa (detalle en `registro_decisiones.md`, D7.2). D6 había fijado las variables a priori, así que la selección no podía considerar ese conjunto. El responsable de la implementación enmendó el contrato: el conjunto de variables pasa a ser el paso 0 de la selección, por familia.
- Evidencia de que la prueba sigue cerrada: `reports/forecasting/bitacora_prueba.jsonl` no existe (ninguna ejecución registrada); `src/forecasting/evaluate.py` y `artifacts.py` no existen; ninguna métrica se ha calculado sobre semanas objetivo de 2023–2025.
- Alternativas consideradas: las de `registro_decisiones.md`, D7.2 (mantener la receta completa y reportar la ablación; tomar la logística sin clima directamente; conjunto por municipio).
- Decisión del responsable de la implementación: aplicar D7.2 con una búsqueda sin clima de las dos familias (`data/processed/forecasting/sin_clima/`, `busqueda_id` propio), el paso 0 en `selection.py` y una receta nueva que incluya el conjunto de variables y la definición de la ablación. `ablacion_clima/` se conserva como registro del origen de D7.2.
- Decisiones de implementación tomadas con la opción más conservadora (se presentan al responsable de la implementación en el informe de paso):
  1. **Barrera cerrada desde ya.** La receta sustituida se copió byte a byte en `reports/forecasting/evidence/seleccion_20261112/receta_congelada_sustituida_d2135a39.yaml` y `configs/receta_congelada.yaml` se retiró del árbol. Sin receta, la barrera rechaza cualquier puntaje sobre 2023–2025 hasta que el responsable de la implementación apruebe la nueva. Consecuencia: la línea de comandos del diagnóstico de desarrollo (`python -m src.forecasting.diagnostics`) no corre hasta entonces, porque lee τ y la familia de la receta.
  2. **`ablacion.yaml` deja de usarse.** Nunca llegó a escribirse; la ablación se lee de la receta nueva (`ablacion`: familia, conjunto y configuración por municipio).
  3. **Pruebas congeladas sustituidas: no se editan.** `tests/conftest.py` (archivo nuevo; `tests/acceptance/conftest.py` no cambia) las marca como `sustituida_dv02` y las omite con una razón que nombra su reemplazo, de modo que aparecen como SKIPPED en `-rA`:
     - `test_f7_evaluacion.py`, completo: `run_test` con `ablation` explícito y sistema `sin_clima` → `test_f7b_evaluacion.py` (U7.3).
     - `test_f7_artefactos.py`: las cuatro pruebas de la ficha (sistema `sin_clima`) y la del modelo final (receta sin conjunto) → `test_f7b_artefactos.py` (U7.4).
     - `test_f7_prueba_real.py`: bitácora (hash `d2135a39…`), predicciones (`logistica_03`, `sin_clima`), ficha (τ = 0,10), historial (`logistica_03` de F5b) y modelos finales → `test_f7b_prueba_real.py` (U7.3).
     - `test_f7_diagnostico_desarrollo.py::test_tau_matches_the_frozen_recipe` (lee la receta retirada) → `test_f7b_diagnostico_desarrollo.py`, con contraste independiente antes de congelar.
  4. **Pruebas congeladas que quedan como registro y siguen pasando:** `test_f6_real.py` y `test_f6receta_borrador.py` (selección y borrador con el conjunto completo; la selección con el conjunto completo forzado debe reproducirlas) y el resto de `test_f7_diagnostico_desarrollo.py` (diagnóstico de la logística completa fuera de año, con su contraste de `caso_f7_diagnostico.py`).
  5. **Pruebas congeladas que siguen vigentes sin cambio:** F0–F5a, `test_f6_selection.py` (pasos 1–4 y τ), `test_f7_regimen.py`, `test_f7_diagnostico.py`, las pruebas de historial, estado de alerta y nivel descriptivo de `test_f7_artefactos.py` y las de `test_f7_prueba_real.py` que no dependen de la receta (ensayos, casillas recalculadas, insumos de diagnóstico).
- Impacto en resultados o en el artículo: se declara que el conjunto de variables se eligió en desarrollo tras observar la ablación, antes de abrir la prueba, y que la receta anterior quedó sustituida sin evaluarse (`registro_decisiones.md`, D7.2). Las cifras de F6 con el conjunto completo se conservan como registro.
- Commit: este registro y la retirada de la receta; la búsqueda, el paso 0 y la aceptación de F6b en sus propios commits (ver `reports/forecasting/INDICE.md`).

## DV-03 — Edición editorial posterior a la prueba (2026-11-12)
- Cláusula afectada: ninguna. Alcance: `contrato_predictivo.md`, `registro_decisiones.md`, `especificacion_implementacion.md`, `desviaciones.md`, `api_salida_d10.md` y `reports/forecasting/INDICE.md`.
- Problema encontrado: los seis documentos citaban personas por su nombre, herramientas y conversaciones de trabajo, y rutas o documentos que no existen en el repositorio (`claude/…`, `01`, `02`, `00` §5, códigos de la auditoría del informe de avance), lo que impedía leerlos solo con el repositorio.
- Alternativas consideradas: dejar los documentos como estaban y añadir una tabla de equivalencias; se descartó porque varias referencias no tienen equivalente en el repositorio.
- Decisión del responsable de la implementación: cambio solo de redacción. Las personas se nombran por su rol (experto de dominio; responsable de infraestructura y API; responsable de la implementación; el equipo), las menciones a herramientas y conversaciones pasan a redacción impersonal y las rutas apuntan a `docs/forecasting/`; lo que no existe en el repositorio se resume en el texto o se elimina. En `especificacion_implementacion.md` §3, las ramas de partida descartadas se describen por su función («las ramas de modelado del avance») para no nombrar a un integrante. No cambia ninguna decisión, regla, cifra, número ni título de sección.
- Fuera del alcance: los scripts de `docs/forecasting/verificacion/` no se editan, porque pruebas congeladas citan su md5 (`test_f6breal_seleccion.py`, `test_f7b_diagnostico_desarrollo.py` y `test_f7_diagnostico_desarrollo.py`); sus docstrings conservan la referencia a la sesión de trabajo en que se escribieron.
- Hechos corregidos (únicas excepciones a «solo redacción»):
  1. `registro_decisiones.md`, D7.1, «Cifras»: los scripts `tamanos_d71.py` y `caso_f3_hueco.py` «se copiaron» (31997c5) a `docs/forecasting/verificacion/`, en lugar de «se copian».
  2. `registro_decisiones.md`, D6, «Evidencia»: las filas disponibles y la prueba de truncamiento se remiten a `tests/acceptance/test_f2_dataset.py`, que las comprueba, en lugar de a un cálculo no versionado.
- Versión vinculante: para la receta congelada y la prueba, la del contrato en el commit 847b3ba (`procedencia.commit_contrato` de `configs/receta_congelada.yaml`) y, para los demás documentos, la vigente en cada commit citado. Este cambio no altera esa procedencia; un `git log` del contrato posterior a este commit ya no coincide con `commit_contrato`, y no debe leerse como un cambio del contrato.
- Impacto en resultados o en el artículo: ninguno.
- Evidencia: el diff del commit de esta edición.
- Commit: el de esta edición (ver git log de desviaciones.md).

## Notas posteriores (2026-11-13)

Notas añadidas tras la auditoría de coherencia del 2026-11-13. No son desviaciones: no cambian ninguna cláusula, decisión ni cifra. Las entradas anteriores se conservan sin cambios.

1. **Rama.** La cabecera de este archivo menciona la implementación «en `feature/forecasting-mvp`». Esa rama se fusionó en `main` (PR #8, 03f5970); las entradas posteriores se registran sobre `main`. De las desviaciones, DV-01 y DV-02 están consolidadas en `registro_decisiones.md` como D7.1 y D7.2; DV-03 es editorial y no tiene entrada de decisión.
2. **«Decisión D1» en DV-02.** En «Problema encontrado», «la ablación de F7 (decisión D1)» se refiere a la decisión 1 del plan de F7: la ablación del clima de D5 (contrato §4.4). No es D1 del contrato (conteo, calendario y canal). Las decisiones de plan están en el glosario de `docs/forecasting/README.md`.
3. **Commits de DV-02 y DV-03.** DV-02 se registró en 804504f. DV-03 se registró en 855af78 y se completó en 7d5a117: su punto 3 de «Hechos corregidos» pasó a «Decisión», y se añadió «Fuera del alcance». Esa edición modificó una entrada existente, contra la regla de que este archivo solo crece; se deja constancia aquí y no se revierte.
4. **Orden de `analisis_resultados.md` frente a §8.8.** El contrato §8.8 fija una estructura «de lo particular a lo general»: perfil por municipio, relato por año, síntesis, contraste de H-R y H-E, efecto de las métricas y conclusiones. El análisis integrado conserva los seis contenidos, pero adelanta tres secciones de síntesis sobre la calidad de la estimación, su valor práctico y los eventos (secciones 3 a 5), antes del relato por año (sección 6). Así, el relato por año puede apoyarse en las métricas y los estratos ya definidos. Es un cambio editorial de orden, no de contenido: la Tabla 0 del documento indica dónde está cada contenido de §8.8, y ninguna conclusión de §8.4 cambia.
5. **Erratas de los documentos pre-registrados.** `contrato_predictivo.md`, `registro_decisiones.md` y `especificacion_implementacion.md` reciben una sección final «Erratas (2026-11-13)». Su texto anterior no se reescribe, y la procedencia de la receta (`commit_contrato` 847b3ba) no cambia.

## DV-04 — Rejilla ampliada de C evaluada con el procedimiento preregistrado (2026-11-20)
- Cláusula afectada: ninguna. Contrato §6.2–§6.3 (rejilla de C de la logística) y D9 (regla de reemplazo), solo como referencia.
- Problema encontrado: el contraste de familias de desarrollo (`reports/tesis_contraste_modelos/`) reportó la logística con C = 2, de una rejilla de 19 valores de 0,001 a 100, como la mejor configuración (mejora media 22,45 % frente a B2). Esa cifra se eligió y se reportó sobre las mismas 24 casillas, mientras que la receta se elige fuera de año, así que no indicaba por sí sola si la receta debía cambiar.
- Alternativas consideradas: adoptar C = 2 directamente (se descartó porque elige con las casillas que reporta); no evaluarlo (se descartó porque deja sin respuesta la cifra del contraste).
- Decisión del responsable de la implementación: evaluar la rejilla ampliada con el procedimiento preregistrado de R0, sin cambiar nada más, con un protocolo y una regla de decisión fijados antes del cálculo (`docs/forecasting/analisis_complementario.md`, commit 7d33839). Resultado: C común = 2, mejora media fuera de año 0,2049 (R0: 0,2109), 6 casillas ganadas, 6 perdidas y 12 empates frente a R0. No se cumplen las condiciones 2 y 3: se mantiene la receta congelada, y el resultado queda como análisis de sensibilidad a la regularización. No se generó una receta nueva ni se repitió la evaluación de prueba.
- Impacto en resultados o en el artículo: ninguno en la receta ni en las conclusiones de §8.4. Se puede reportar como sensibilidad a la regularización y como medida del optimismo de la selección dentro de muestra (0,2245 frente a 0,2049).
- Commit: protocolo 7d33839; resultado cc04a14 (`reports/forecasting/evidence/regularizacion_c_ampliada_20261120/`); reproducción previa de R0 en 7311a81.

## DV-05 — Regla de emisión alternativa B01, análisis posterior a la prueba (2026-11-20)
- Cláusula afectada: contrato §7.4 (regla de emisión p ≥ τ), solo como comparación. La regla del contrato no cambia.
- Problema encontrado: en el diagnóstico de desarrollo, dos inicios no se avisaron porque la alerta de la receta ya estaba encendida antes de la ventana [s − W, s − 1], de modo que no se confirmó ninguna alerta nueva dentro de ella.
- Alternativas consideradas: dejar el diagnóstico solo como limitación; cambiar τ, k o W (se descartó: se habrían ajustado con un periodo de prueba ya consultado).
- Decisión del responsable de la implementación: formular y evaluar B01 (p ≥ τ y zona observada en t distinta de `epidemia`) como análisis posterior declarado, con las mismas probabilidades, τ = 0,05, k = 2 y W = 12. Se añadieron `events.model_positive_outside_epidemic`, `events.emission_positive` y el parámetro `decision.regla_emision` en `configs/forecasting.yaml`, con la regla del contrato (`umbral`) por defecto; la selección y la evaluación preregistradas no lo leen. La regla se formuló después de conocer la evaluación preregistrada de prueba y se evaluó una vez en prueba sin cambiar τ, k ni W: **no es un resultado confirmatorio**. Cifras en `analisis_complementario.md` (prueba: 9 de 10 inicios avisados frente a 4 de la regla actual y 8 del canal; 26 falsas alarmas frente a 18 y 30; anticipación mediana 1,0 semanas).
- Impacto en resultados o en el artículo: ninguno en la receta, la API ni las conclusiones de §8.4. La adopción de B01 en el producto es una decisión pendiente; si se adopta, requiere su propia enmienda y una evaluación en un periodo no consultado.
- Commit: 19869a2 (`reports/forecasting/evidence/complementario_20261120/`; regresión en `tests/acceptance/test_b01_regla_emision.py`).

## DV-06 — Análisis de continuidad y referencia M0, descriptivos y posteriores a la prueba (2026-11-20)
- Cláusula afectada: ninguna. Contrato §7.2 y §8.8, solo como referencia.
- Problema encontrado: la evaluación preregistrada no separa el desempeño de la receta en las semanas que ya están en zona epidémica, ni mide cuánto de ese desempeño aporta la posición continua frente al canal por sí sola.
- Alternativas consideradas: incorporar estos análisis a §8.8 (se descartó: §8.8 se fijó antes de la prueba).
- Decisión del responsable de la implementación: (1) análisis de continuidad descriptivo, con protocolo fijado antes del cálculo (commit 8bb73ee), sobre las predicciones guardadas de la receta y de B2, primero desarrollo y luego prueba, sin seleccionar nada; (2) referencia M0, una logística L2 con `posicion_4s` como única variable, con la rejilla, el preprocesamiento, el corte y la elección fuera de año de la receta; sus cifras de prueba usan C = 3 fijado en desarrollo y se reportan tal cual.
- Impacto en resultados o en el artículo: ninguno en la receta ni en las conclusiones de §8.4. Ambos son descriptivos: en zona epidémica, la receta mejora a B2 (BSS 0,252 en desarrollo y 0,136 en prueba); M0 supera a R0 en desarrollo (mejora media 0,2311 frente a 0,2109) y queda por debajo en prueba (log-loss 0,4525 frente a 0,4409).
- Commit: M0 en d299fd4 (`m0_posicion_20261120/`); continuidad en 8bb73ee (protocolo) y ecc35b7 (`continuidad_20261120/`).

## DV-07 — Regla B01 adoptada como regla de alerta del producto (2026-11-20)
- Cláusula afectada: contrato §9.1 («la misma regla de alerta») y §9.3 (estados de alerta de la vista principal); §7.4 recibe la definición de la regla del producto. Enmienda D10.1, con el texto propuesto en `docs/forecasting/enmienda_d10_1_propuesta.md`, pendiente de aplicación en `contrato_predictivo.md` y `registro_decisiones.md`.
- Problema encontrado: con la regla preregistrada (p̂_t ≥ τ), una racha de emisiones positivas que sigue encendida durante y después de un episodio no confirma una alerta nueva dentro de la ventana del inicio siguiente. En el diagnóstico de desarrollo, dos inicios quedaron sin aviso por esa razón. En prueba, la regla preregistrada avisó 4 de 10 inicios y el canal, 8 de 10.
- Alternativas consideradas: mantener la regla preregistrada en el producto; cambiar τ, k o W (descartado: se ajustarían con un periodo de prueba ya consultado); mostrar la zona del canal como alerta.
- Decisión del responsable de la implementación: el producto alerta con B01 (p̂_t ≥ τ y zona observada en t distinta de epidemia), con cuatro estados (sin alerta, señal en observación, alerta de inicio confirmada, continuidad epidémica). Declaraciones:
  1. La regla del producto **se formuló después de la evaluación preregistrada**, a partir de un diagnóstico en desarrollo (DV-05).
  2. **No ajusta ningún parámetro:** τ = 0,05, k = 2, W = 12, la receta congelada y la probabilidad son los mismos.
  3. **Su evaluación en 2023–2025 no es confirmatoria:** ese periodo ya se había consultado al formular la regla.
  4. **Queda pendiente una validación prospectiva** con datos posteriores a 2025.
- Configuración: `configs/forecasting.yaml` separa la regla preregistrada (`decision.regla_emision`, fija en `umbral`, que la selección y la evaluación no pueden cambiar) de la regla del producto (`producto.regla_alerta: umbral_fuera_de_epidemia`, que lee la API).
- Impacto en resultados o en el artículo: ninguno en la receta, la selección ni las conclusiones preregistradas de §8.4, que se siguen reportando con la regla p̂_t ≥ τ. La ficha de desempeño muestra las cifras de B01 en prueba rotuladas «evaluación posterior no confirmatoria», junto al resultado preregistrado y al de continuidad. El artículo declara el origen posterior de la regla.
- Commit: propuesta de enmienda en el commit «docs(d10): propuesta de enmienda D10.1»; justificación, especificación de la API y configuración en los commits siguientes de la rama `integracion/contraste-familias`.

## DV-08 — Registro de los modelos finales desde la interfaz y etiquetas diferidas al despliegue (2026-11-20)
- Cláusula afectada: ninguna del contrato (§9.6 solo como referencia). Especificación §9, «Registro de modelos finales» (etiquetas `enabled = true`, `municipality_code`, `department`, `depto_id`, `receta_hash` y `familia`; alias `staging` primero), y §12 (el rol Runners puede registrar modelos).
- Problema encontrado: (1) el servidor de MLflow rechazó la creación de modelos registrados con el token del equipo (`PERMISSION_DENIED`). Su política reserva el registro oficial a la interfaz, a partir del modelo guardado en un run, o a un pipeline de CI/CD, que el repositorio no tiene. (2) `src/api.py` carga todo modelo `DengueAlertas_*` con `enabled = true`, por alias y, si no lo hay, por su última versión. Con `enabled = true` desde el registro, la API anterior a F8, que no está adaptada a las variables de la receta, los cargaría en producción y en pruebas antes del despliegue.
- Alternativas consideradas: pedir un token con escritura en el registro; registrar con `enabled = true` y alias `staging` como propone §9 (descartado por el punto 2); registrar con `enabled = false` (innecesario, porque sin la etiqueta la API tampoco los descubre).
- Decisión del responsable de la implementación: es una decisión de secuencia de despliegue.
  1. Los runs se crean por script: `scripts/registrar_modelos_finales.py --solo-runs`, en el experimento `dengue_h8_final`, con parámetros, etiquetas, el `.joblib` original y el modelo guardado con firma. El script no llama al registro de modelos.
  2. El responsable de la implementación registró desde la interfaz los seis modelos `DengueAlertas_<Municipio>` en versión 1, **sin etiquetas ni alias**.
  3. `--verificar`, de solo lectura, comprobó en cada uno la versión 1, el run de origen (por id y md5), su modelo guardado y que las probabilidades cargadas desde `models:/` son idénticas a las del `.joblib`.
  4. Las etiquetas del modelo registrado y los alias se difieren al despliegue de F8, a cargo de su responsable, en este orden: etiquetas (con `enabled = true`), alias `staging`, prueba en el ambiente de pruebas, alias `production`. Los valores exactos están en `modelos/README.md`, «Habilitación en F8».
- Impacto en resultados o en el artículo: ninguno. Los modelos registrados son los mismos archivos de `modelos/` y de la salida DVC (mismos md5 y probabilidades idénticas), y la receta, la evaluación y la API no cambian. Hasta F8, la API actual no carga estos modelos. Queda para F8 cómo tratar el último recurso de `src/api.py` (cargar la última versión cuando no existe el alias del ambiente).
- Commit: script de registro en cae96bf y d42cc0e (rama `feat/registro-modelos`); el run de 08001 se creó con cae96bf y los otros cinco con d42cc0e. Documentación en el commit «docs(registro): modelos finales registrados en MLflow desde la interfaz».
