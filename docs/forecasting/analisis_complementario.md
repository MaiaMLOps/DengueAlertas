# Análisis complementario

Análisis posteriores a la evaluación preregistrada de la receta congelada (`configs/receta_congelada.yaml`). No modifican el contrato, la receta ni la evaluación única de prueba. Cada protocolo se fija en este documento, con su propio commit, antes de calcular su resultado; los resultados se añaden después, en secciones aparte.

## Protocolo declarado: rejilla ampliada de C

Fijado el 2026-11-20, antes de calcular el resultado.

**Procedimiento.** Es el mismo procedimiento de selección preregistrado de la receta vigente (R0): tabla única de desarrollo, configuración común por la mejora media de log-loss frente a B2, puntuación fuera de año, corte de entrenamiento D7.1 y variables `sin_clima`. Solo cambia la rejilla de C, que pasa a ser la de 19 valores de 0,001 a 100 (`configs/contraste_familias.yaml`, corrida `logistica_ampliada`): 0,001; 0,005; 0,01; 0,05; 0,1; 0,25; 0,5; 1; 1,5; 2; 3; 4; 5; 7,5; 10; 15; 25; 50; 100. `max_iter` = 5000.

**Salidas que se reportan.**

- El C común elegido fuera de año para cada año de desarrollo y el C común con los cuatro años.
- La mejora media de log-loss frente a B2, el log-loss y el Brier fuera de año.
- Las casillas ganadas a B2 y las casillas ganadas y perdidas frente a R0.

**Referencia directa.** Se puntúa también C = 2 fijo en los cuatro años, con el mismo esquema, para compararlo con el valor reportado en el contraste de familias (tabla 01, dentro de muestra).

**Regla de decisión, fijada antes del cálculo.** La rejilla ampliada reemplaza a la receta congelada solo si se cumplen las tres condiciones:

1. el C común con los cuatro años es distinto de 3;
2. la mejora media fuera de año supera la de R0 (0,2109);
3. gana al menos 17 de las 24 casillas frente a R0 (regla de reemplazo del contrato, D9).

En cualquier otro caso se mantiene la receta congelada, y el resultado se reporta como análisis de sensibilidad a la regularización.

**Cálculo.** `scripts/analisis_regularizacion.py ampliada`, sobre la reproducción verificada de R0 (`reports/forecasting/evidence/reproduccion_r0_20261120/`). Una casilla se gana con un log-loss estrictamente menor; un empate no es victoria. Solo se usan semanas objetivo de desarrollo (2019–2022). Tras el cálculo, el trabajo se detiene hasta que el responsable de la implementación confirme cómo seguir.
