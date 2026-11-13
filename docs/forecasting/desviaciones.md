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
