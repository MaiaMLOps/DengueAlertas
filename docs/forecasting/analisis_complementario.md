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

## Protocolo declarado: continuidad

Fijado el 2026-11-20, antes de calcular el resultado. Análisis descriptivo y posterior a la evaluación única de prueba: no se selecciona nada y se reporta salga como salga.

**Pregunta.** En las semanas en que el municipio ya está en zona epidémica, ¿la probabilidad de la receta vigente de seguir en zona epidémica en t + 8 es mejor que la de B2?

**Datos.** Las predicciones guardadas de la receta congelada y de B2, sin reentrenar: desarrollo fuera de año (`reports/forecasting/evidence/diagnostico_desarrollo_sin_clima_20261112/diag_serie.csv`, 1.248 filas) y prueba (`reports/forecasting/evidence/prueba_20261112/diag_serie.csv`, 936 filas). El objetivo es `y` = `target_u1e` en t + 8.

**Estratos.** `zona_t = epidemia` y `zona_t ≠ epidemia` (zona observada del canal en la semana de emisión t).

**Métricas por estrato**, para la receta y para B2 sobre las mismas filas:

- log-loss (recorte ε de la configuración) y Brier, sobre las filas del estrato juntas;
- BSS frente a B2 = 1 − Brier_receta / Brier_B2, y la mejora relativa de log-loss frente a B2;
- confiabilidad y resolución: descomposición de Murphy con 10 intervalos de igual frecuencia de p (`reports.murphy_decomposition`, la misma de los reportes U7.5), calculada dentro de cada estrato;
- casillas municipio-año (año de la semana objetivo) con al menos 5 semanas en el estrato: número de casillas y casillas en que la receta tiene Brier y log-loss estrictamente menores que B2.

**Orden y reporte.** Primero desarrollo y luego prueba. Se reporta salga como salga; no se elige ningún umbral, estrato ni parámetro con estos resultados.

**Conteo descriptivo adicional.** Cuántas falsas alarmas de B01 (regla de la sección anterior, con τ, k y W de la receta) empiezan en las 13 semanas siguientes al final de un episodio, en prueba y en desarrollo. Episodio: racha de al menos 3 semanas elegibles consecutivas con `target_u1e = 1` en toda la serie elegible (contrato §8.8; `diagnostics.EPISODE_MIN_RUN`); su final es la última semana de la racha, e. Una falsa alarma empieza en la primera semana de su racha de emisiones positivas (`Alert.start`); se cuenta si esa semana está en [e + 1, e + 13] para algún episodio del mismo municipio. Se reporta también el total de falsas alarmas de B01 en cada periodo.

**Cálculo.** `scripts/analisis_continuidad.py`. Las métricas de prueba pasan por la barrera del periodo de prueba (receta congelada aprobada).
