# Desviaciones del contrato predictivo durante la implementación

Registro de toda desviación de `contrato_predictivo.md` aprobada por Jesús durante la implementación en `feature/forecasting-mvp`. Cada entrada se escribe **antes** de continuar con la parte afectada. Al cerrar la semana, las entradas se consolidan en `registro_decisiones.md` (documento `01` del proyecto de Claude).

Formato de cada entrada:

```
## DV-NN — <título> (AAAA-MM-DD)
- Cláusula afectada: contrato §…
- Problema encontrado: …
- Alternativas consideradas: …
- Decisión de Jesús: …
- Impacto en resultados o en el artículo: …
- Commit: …
```

## DV-01 — Entrenamiento cerrado en la primera emisión de cada ensayo (2026-11-11)
- Cláusula afectada: contrato §6.1 (corte de validación), §6.4 y §10, punto 11. Enmienda D7.1 en `registro_decisiones.md`.
- Problema encontrado: con el corte «entrenamiento con objetivo antes de Y», las primeras h − 1 = 7 emisiones evaluadas de cada ensayo (del t0 al t0 + 6 semanas) se hacían con un modelo entrenado con etiquetas de semanas posteriores a su emisión (por ejemplo, ensayo 2019: emisión del 2018-11-12 con etiquetas hasta el 2018-12-31). Por la autocorrelación del estado, esas etiquetas informan sobre el objetivo. La prueba de truncamiento no lo detecta, porque la información entra por los parámetros y no por las variables. Identificado en el documento de metodología («Observación sobre el corte») y retomado antes de congelar las pruebas de F3.
- Alternativas consideradas: mantener el corte y declararlo como limitación; reentrenar en cada emisión (×52 entrenamientos); ensayos por año de emisión (rompe la barrera de prueba y las cifras verificadas).
- Decisión de Jesús: en el ensayo Y, t0 es la emisión más temprana entre las filas evaluadas de Y; el entrenamiento usa solo las filas con t + h ≤ t0, desde 2012. Se aplica a modelos, B1, B2, ablación, desarrollo y prueba. Las filas evaluadas no cambian.
- Impacto en resultados o en el artículo: 7 filas menos de entrenamiento por ensayo y municipio; las filas evaluadas no cambian. Se declara en el artículo que cada ensayo simula un reentrenamiento anual el día de su primera emisión.
- Commit: (pendiente)
