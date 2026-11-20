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

## Resultado: rejilla ampliada de C

Calculado después del protocolo anterior (commit 7d33839), sin cambiar el procedimiento. Evidencia: `reports/forecasting/evidence/regularizacion_c_ampliada_20261120/` (`resumen_regularizacion.json`, `casillas_fuera_de_anio.csv`, `rejilla_dentro_de_muestra.csv`). R0 se reconstruyó antes con el código integrado y coincide con la evidencia versionada (`reproduccion_r0_20261120/`: tabla única idéntica a la almacenada; casillas fuera de año con una diferencia máxima de 1,1e-16 y la misma elección en las 24).

| | Rejilla ampliada (19 C) | R0 (6 C, receta) | C = 2 fijo |
|---|---|---|---|
| C común con los cuatro años | 2 | 3 | 2 |
| C fuera de año (2019 / 2020 / 2021 / 2022) | 1,5 / 15 / 1 / 1 | 1 / 3 / 1 / 1 | 2 en todos |
| Mejora media de log-loss frente a B2 | 0,2049 | 0,2109 | 0,2245 |
| Log-loss (B2: 0,3854) | 0,3164 | 0,3145 | 0,3110 |
| Brier (B2: 0,1243) | 0,1013 | 0,1010 | 0,0998 |
| Casillas ganadas a B2 | 16 de 24 | 16 de 24 | 16 de 24 |
| Frente a R0: ganadas / perdidas / empates | 6 / 6 / 12 | — | 19 / 5 / 0 |

**Regla.** (1) El C común con los cuatro años (2) es distinto de 3: se cumple. (2) La mejora media fuera de año (0,2049) no supera la de R0 (0,2109): no se cumple. (3) La rejilla ampliada gana 6 de 24 casillas frente a R0: no se cumple. **Se mantiene la receta congelada**; el responsable de la implementación confirmó la decisión y F4 (nueva receta y nueva evaluación) no se ejecutó.

**Lectura como análisis de sensibilidad.** Con el procedimiento preregistrado, ampliar la rejilla no mejora la receta: la elección fuera de año es inestable (en 2020 elige C = 15) y empata con R0 en la mitad de las casillas. C = 2 fijo reproduce exactamente la cifra del contraste de familias (22,45 %, log-loss 0,311, Brier 0,0998, 16 casillas), pero ese C se eligió con los cuatro años: su 19–5 frente a R0 es una comparación dentro de muestra y no cuenta para la regla. La diferencia entre la cifra del contraste (0,2245) y la de la rejilla ampliada con el procedimiento preregistrado (0,2049) mide el optimismo de elegir y reportar sobre las mismas casillas.

**C = 2 frente a C = 3, ambos fijos.** La comparación directa usa las mismas 24 casillas y la misma B2, con C fijo en los cuatro años (C = 3 es el C común de la receta con los cuatro años; R0, en cambio, usa el C elegido fuera de año en cada año). C = 2 gana 10 casillas y pierde 14 por log-loss. La diferencia media de la mejora frente a B2 (C = 2 menos C = 3) es:

| Año | Diferencia media de la mejora frente a B2 | Casillas ganadas por C = 2 (de 6) |
|---|---|---|
| 2019 | −0,0008 | 1 |
| 2020 | +0,0143 | 5 |
| 2021 | −0,0053 | 2 |
| 2022 | −0,0044 | 2 |
| Los cuatro años | +0,0009 | 10 |

La ventaja media de C = 2 (+0,0009) proviene de 2020: es el único año en que C = 2 mejora en promedio, y su diferencia (+0,0143) compensa las de los otros tres años, en los que C = 3 es mejor. Con una ventaja media tan pequeña y concentrada en un año, la elección entre C = 2 y C = 3 depende de qué años entran en el promedio. Evidencia: `c2_frente_a_c3.json` y `c2_frente_a_c3_casillas.csv` en `reports/forecasting/evidence/regularizacion_c_ampliada_20261120/` (`scripts/analisis_regularizacion.py c2c3`).

## Referencia M0: modelo mínimo de posición

**Definición.** Regresión logística L2 sin pesos de clase, con `posicion_4s` como única variable, estandarizada con estadísticas de entrenamiento: el mismo preprocesamiento y estimador que la receta, sin `zona_t` ni las demás variables. Misma rejilla de C (6 valores), mismo corte D7.1 y misma configuración común fuera de año. Mide cuánto del desempeño de la receta ya está contenido en la posición continua frente al canal. Código: `src/forecasting/complementary.py` (`position_model`) y `scripts/analisis_m0.py`. Evidencia: `reports/forecasting/evidence/m0_posicion_20261120/`.

**Desarrollo (fuera de año).** C = 3 elegido en los cuatro años y con los cuatro años. Mejora media de log-loss frente a B2 0,2311; log-loss 0,3328; Brier 0,1036; 21 de 24 casillas ganadas a B2; frente a R0, 13 casillas ganadas y 11 perdidas por log-loss.

**Prueba (solo reporte).** M0 con C = 3 fijado en desarrollo, reentrenado una vez por municipio y año de prueba como en la evaluación única (`evaluate.run_test`), sobre las mismas 936 filas:

| Sistema | Log-loss | Brier | AP |
|---|---|---|---|
| M0 | 0,4525 | 0,1494 | 0,751 |
| Receta vigente | 0,4409 | 0,1413 | 0,765 |
| B2 | 0,4708 | 0,1537 | 0,612 |

M0 gana a B2 en 11 de 18 casillas por Brier (10 de 18 por log-loss) y a la receta en 7 de 18 por Brier y por log-loss. Las cifras de la receta y de B2, calculadas sobre las predicciones guardadas, coinciden con `prueba_20261112/resumen_prueba.json`.

**Lectura.** En desarrollo, M0 iguala o supera a la receta en mejora media frente a B2 y en casillas ganadas a B2, aunque con mayor log-loss medio (la mejora media promedia cocientes por casilla); en prueba, la receta es mejor en las tres métricas globales y en 11 de 18 casillas. La mayor parte de la ventaja sobre B2 ya está en la posición continua frente al canal; las otras cinco variables aportan una mejora pequeña y no uniforme.

## Regla de emisión alternativa B01

**Definición.** La emisión de la semana t es positiva si p ≥ τ **y** la zona observada del canal en t no es `epidemia` (`events.model_positive_outside_epidemic`). Mismas probabilidades, τ = 0,05 de la receta, k = 2 y W = 12; `alerts` y `evaluate_period` no cambian. En la configuración, `decision.regla_emision` vale `umbral` (la regla del contrato) por defecto; la selección y la evaluación preregistradas no lo leen.

**Origen.** Análisis posterior declarado: la regla se formuló después de conocer la evaluación preregistrada de prueba, a partir de un diagnóstico en desarrollo. Dos inicios de desarrollo no se avisaron porque la alerta ya estaba encendida antes de la ventana: no se confirmó ninguna alerta nueva en [s − W, s − 1]. Se evaluó una vez en prueba sin cambiar τ, k ni W y **no es un resultado confirmatorio**. Después se adoptó como regla de alerta del producto (enmienda D10.1, DV-07); la justificación está en la sección «Justificación de la regla B01 en el producto».

**Cálculo.** `scripts/analisis_b01.py` sobre las predicciones guardadas de la receta (`prueba_20261112/diag_serie.csv` y `diagnostico_desarrollo_sin_clima_20261112/diag_serie.csv`); la regla actual se verifica contra las emisiones guardadas. Evidencia: `reports/forecasting/evidence/complementario_20261120/` (`b01_resumen.json`, `b01_eventos.csv` por periodo, municipio y año, y `b01_inicios_entradas.csv`). La regresión de las cifras de control está en `tests/acceptance/test_b01_regla_emision.py`.

| Periodo | Sistema | Inicios avisados | Entradas avisadas | Alertas | Falsas alarmas | Semanas bajo alerta | Anticipación mediana (contrato / primera alerta) |
|---|---|---|---|---|---|---|---|
| Prueba | Regla actual | 4 de 10 | 13 de 41 | 26 | 18 | 565 de 936 | 4,5 / 7,5 |
| Prueba | B01 | 9 de 10 | 24 de 41 | 38 | 26 | 192 de 936 | 1,0 / 3,0 |
| Prueba | Canal | 8 de 10 | 19 de 41 | 39 | 30 | 393 de 936 | 1,5 / 2,0 |
| Desarrollo (fuera de año) | Regla actual | 8 de 10 | 21 de 44 | 43 | 24 | 586 de 1.248 | 7,5 / 8,0 |
| Desarrollo (fuera de año) | B01 | 10 de 10 | 38 de 44 | 59 | 37 | 244 de 1.248 | 2,0 / 8,0 |
| Desarrollo (fuera de año) | Canal | 10 de 10 | 36 de 44 | 65 | 48 | 406 de 1.248 | 3,0 / 5,5 |

Semanas bajo alerta: semanas de emisión entre la confirmación de cada alerta y su última semana. Anticipación del contrato: desde la alerta más reciente confirmada en la ventana; «primera alerta»: desde la primera. Alertas: todas, incluidas las de continuidad (4 de la regla actual en prueba y 5 en desarrollo); las falsas alarmas se cuentan sobre las alertas juzgables que no son de continuidad.

**Inicios discrepantes (McNemar, una cola).** En prueba, B01 frente a la regla actual: 5 discrepantes, los 5 a favor de B01 (nivel «mejora consistente»); B01 frente al canal: 1, a favor de B01 («tendencia favorable»). En desarrollo: 2 frente a la regla actual (ambos a favor de B01) y ninguno frente al canal.

**Composición.** En prueba, 268 de las 272 emisiones positivas de B01 caen en zona de alerta (4 en zona de seguridad); el filtro de probabilidad descarta 196 de las 464 semanas en zona de alerta (42,2 %). En desarrollo: 294 de 355 en zona de alerta (61 en seguridad) y 217 de 511 semanas de alerta descartadas (42,5 %).

**Lectura.** B01 avisa casi todos los inicios con muchas menos semanas bajo alerta que la regla actual y que el canal, pero con poca anticipación (mediana de 1 semana en prueba con la medida del contrato) y más falsas alarmas que la regla actual. En la práctica, se comporta como el canal filtrado por la probabilidad del modelo.

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

## Resultado: continuidad

Calculado después del protocolo anterior (commit 8bb73ee), primero desarrollo y luego prueba. Evidencia: `reports/forecasting/evidence/continuidad_20261120/` (`continuidad_resumen.json` y `continuidad_casillas.csv`).

| Periodo | Estrato | Filas (positivas) | Log-loss receta / B2 | Brier receta / B2 | BSS frente a B2 | Confiabilidad receta / B2 | Resolución receta / B2 | Casillas (≥ 5 semanas) ganadas por Brier / log-loss |
|---|---|---|---|---|---|---|---|---|
| Desarrollo | zona_t = epidemia | 310 (231) | 0,5595 / 0,6950 | 0,1873 / 0,2503 | 0,252 | 0,0230 / 0,0919 | 0,0249 / 0,0302 | 8 / 8 de 12 |
| Desarrollo | zona_t ≠ epidemia | 938 (88) | 0,2335 / 0,2831 | 0,0725 / 0,0827 | 0,123 | 0,0026 / 0,0030 | 0,0152 / 0,0064 | 16 / 16 de 23 |
| Prueba | zona_t = epidemia | 340 (243) | 0,5769 / 0,6422 | 0,1935 / 0,2240 | 0,136 | 0,0124 / 0,0469 | 0,0244 / 0,0264 | 10 / 9 de 14 |
| Prueba | zona_t ≠ epidemia | 596 (79) | 0,3633 / 0,3730 | 0,1115 / 0,1136 | 0,018 | 0,0063 / 0,0100 | 0,0131 / 0,0117 | 8 / 9 de 17 |

Confiabilidad y resolución: descomposición de Murphy con 10 intervalos de igual frecuencia dentro de cada estrato (B2 en zona epidémica de desarrollo tiene 7 intervalos por empates de p).

**Lectura descriptiva.** En las semanas ya en zona epidémica, la probabilidad de la receta de seguir sobre U1e en t + 8 es mejor que la de B2 en los dos periodos (BSS 0,252 y 0,136). La mejora viene sobre todo de la confiabilidad (0,023 frente a 0,092 en desarrollo; 0,012 frente a 0,047 en prueba); la resolución es similar o algo menor que la de B2. Fuera de la zona epidémica, la ventaja de la receta en prueba es pequeña (BSS 0,018; 8 de 17 casillas por Brier).

**Conteo adicional.** Falsas alarmas de B01 que empiezan en las 13 semanas siguientes al final de un episodio: 13 de 37 en desarrollo y 11 de 26 en prueba (por municipio en `continuidad_resumen.json`).

## Tabla comparativa

Cifras de desarrollo (2019–2022, 24 casillas, variables `sin_clima`). Las filas del contraste de familias son las versionadas en `reports/tesis_contraste_modelos/tabla_01_ranking_global_sin_clima.csv` y no se recalcularon. Las filas con distinto método de selección no son comparables entre sí sin ajuste.

| Sistema | Método de selección | Mejora media frente a B2 | Log-loss | Brier | Casillas ganadas a B2 |
|---|---|---|---|---|---|
| Receta vigente (R0): logística, 6 valores de C | Fuera de año (preregistrado) | 0,2109 | 0,3145 | 0,1010 | 16 |
| M0: logística con `posicion_4s`, 6 valores de C | Fuera de año | 0,2311 | 0,3328 | 0,1036 | 21 |
| Logística, rejilla ampliada de 19 C | Fuera de año | 0,2049 | 0,3164 | 0,1013 | 16 |
| Logística, C = 2 fijo | C elegido con los cuatro años (dentro de muestra) | 0,2245 | 0,3110 | 0,0998 | 16 |
| Contraste: logística L2 (`logistica_09`, C = 2) | Dentro de muestra, 25 candidatos | 0,2245 | 0,3110 | 0,0998 | 16 |
| Contraste: SVM rbf (`svm_05`) | Dentro de muestra, 40 candidatos | 0,1726 | 0,3284 | 0,1053 | 18 |
| Contraste: MLP tabular (`mlp_36`) | Dentro de muestra, 40 candidatos | 0,1668 | 0,3442 | 0,1106 | 17 |
| Contraste: CatBoost (`catboost_22`) | Dentro de muestra, 65 candidatos | 0,1568 | 0,3496 | 0,1112 | 18 |
| Contraste: GRU (`gru_00`) | Dentro de muestra, 25 candidatos | 0,1568 | 0,3419 | 0,1058 | 17 |
| Contraste: LightGBM (`lightgbm_20`) | Dentro de muestra, 25 candidatos | 0,1207 | 0,3838 | 0,1190 | 14 |
| Contraste: XGBoost (`xgboost_09`) | Dentro de muestra, 65 candidatos | 0,0922 | 0,3722 | 0,1195 | 14 |
| Contraste: CNN-1D (`cnn1d_03`) | Dentro de muestra, 25 candidatos | −0,0377 | 0,4351 | 0,1367 | 11 |
| B2 | Línea base (sin selección) | 0 | 0,3854 | 0,1243 | — |

Con selección dentro de muestra, ninguna familia supera a la logística; la mejor cifra del contraste (0,2245) cae a 0,2049 con el procedimiento preregistrado. M0, con una sola variable y el mismo procedimiento que la receta, queda por encima de R0 en mejora media y en casillas ganadas a B2 en desarrollo, y por debajo en prueba.

## Justificación de la regla B01 en el producto

Sustenta la enmienda D10.1 (`docs/forecasting/enmienda_d10_1_propuesta.md`) y DV-07. No se hizo ninguna evaluación nueva: todas las cifras salen de la evidencia de las secciones anteriores (`complementario_20261120/` y `continuidad_20261120/`) y las comprueba `docs/forecasting/verificacion/verificar_cifras_b01.py` (51 de 51 coinciden). R0 designa la regla preregistrada (p̂_t ≥ τ) aplicada a las predicciones de la receta vigente; los tres sistemas usan las mismas predicciones, τ = 0,05, k = 2 y W = 12.

**(1) Mecanismo sin parámetros ajustados.** Con R0, una racha de emisiones positivas puede empezar antes de un episodio, seguir durante él y prolongarse después. Mientras la racha no se interrumpe, no se confirma ninguna alerta nueva, de modo que una alerta encendida desde antes del episodio no puede contar como aviso del inicio siguiente: o su confirmación queda fuera de la ventana [s − 12, s − 1], o es una alerta de continuidad, que el contrato excluye de la detección (§7.4). B01 hace negativa la emisión de una semana que ya está en zona epidémica, así que corta la racha al entrar en el episodio y permite una confirmación nueva después de él. Aplica a la emisión el mismo principio con que D8 cuenta aparte las alertas de continuidad: lo que ocurre en zona epidémica no es aviso de un inicio. No introduce ni ajusta ningún parámetro: la condición sobre la zona observada en t es información disponible al emitir (§3.2), y τ, k y W son los congelados.

**(2) Criterio del contrato.** El criterio de §7.5 (falsas alarmas por municipio-año no mayores que las del canal y, con esa restricción, la mayor capacidad de detección), aplicado aquí para comparar reglas con τ fijo, lo cumple B01 en los dos periodos:

| Periodo | Sistema | Inicios avisados | Entradas avisadas | Falsas alarmas | Falsas alarmas por municipio-año | Semanas bajo alerta |
|---|---|---|---|---|---|---|
| Desarrollo (24 municipio-años) | R0 | 8 de 10 | 21 de 44 | 24 | 1,00 | 47,0 % |
| Desarrollo | B01 | 10 de 10 | 38 de 44 | 37 | 1,54 | 19,6 % |
| Desarrollo | Canal | 10 de 10 | 36 de 44 | 48 | 2,00 | 32,5 % |
| Prueba (18 municipio-años) | R0 | 4 de 10 | 13 de 41 | 18 | 1,00 | 60,4 % |
| Prueba | B01 | 9 de 10 | 24 de 41 | 26 | 1,44 | 20,5 % |
| Prueba | Canal | 8 de 10 | 19 de 41 | 30 | 1,67 | 42,0 % |

B01 respeta la restricción (1,54 ≤ 2,00 en desarrollo y 1,44 ≤ 1,67 en prueba) y tiene la mayor detección de los tres sistemas: 10 de 10 en desarrollo, empatada con el canal, y 9 de 10 en prueba. R0 también respeta la restricción, pero detecta menos (8 y 4 de 10). Semanas bajo alerta: semanas de emisión entre la confirmación de cada alerta y su última semana, sobre 1.248 (desarrollo) y 936 (prueba).

**(3) Consistencia entre periodos.** B01 avisa más inicios que R0 en los dos periodos, y nunca al revés. En desarrollo hay 2 inicios discrepantes, ambos a favor de B01 (los dos del diagnóstico que originó la regla; McNemar exacto de una cola p = 0,25). En prueba hay 5 inicios discrepantes, todos a favor de B01: McNemar exacto de una cola, p = 0,5⁵ = 0,03125, por debajo del umbral de azar de D9 (5 %). Frente al canal, B01 avisa 1 inicio más en prueba y ninguno menos. La cifra de prueba no es confirmatoria: la regla se formuló después de consultar ese periodo.

**(4) Costo explícito.**
- **Más alertas y falsas alarmas que R0.** En prueba, 38 alertas y 26 falsas alarmas frente a 26 y 18; en desarrollo, 59 y 37 frente a 43 y 24. Sigue con menos falsas alarmas que el canal (30 en prueba y 48 en desarrollo).
- **Menos semanas bajo alerta.** 20,5 % frente a 60,4 % de R0 y 42,0 % del canal, en prueba: las alertas de B01 son más cortas porque terminan al entrar en zona epidémica.
- **Anticipación corta.** Mediana de 1,0 semanas desde la alerta más reciente (medida del contrato) y de 3,0 desde la primera, frente a 4,5 y 7,5 de R0 y 1,5 y 2,0 del canal, en prueba. B01 avisa con poca antelación.
- **Composición.** 268 de las 272 emisiones positivas de B01 en prueba caen en zona de alerta: en la práctica es la zona de alerta del canal filtrada por la probabilidad del modelo, que descarta 196 de las 464 semanas en zona de alerta (42,2 %).
- **Falsas alarmas posteriores a un episodio.** 11 de las 26 falsas alarmas de B01 en prueba, y 13 de 37 en desarrollo, empiezan en las 13 semanas siguientes al final de un episodio: parte del costo se concentra en el descenso de la epidemia.

**Lo que la regla no cambia.** La probabilidad, la receta, las conclusiones preregistradas de §8.4 y la calidad de la estimación en continuidad (BSS frente a B2 de 0,252 en desarrollo y 0,136 en prueba en las semanas en zona epidémica) son las mismas: B01 solo decide cuándo se enciende la alerta.

## Limitaciones

- **Periodo de prueba ya consultado.** M0, B01 y el análisis de continuidad se formularon o se calcularon después de la evaluación única de prueba. Sus cifras de prueba son descriptivas: no confirman ni sustituyen las conclusiones preregistradas de §8.4, y ninguna se usó para elegir τ, k, W, C, variables ni familia.
- **Pocos eventos.** Hay 10 inicios evaluables en prueba y 10 en desarrollo. Una diferencia de pocos inicios cambia las proporciones en decenas de puntos, y las pruebas de McNemar descansan en 1 a 5 inicios discrepantes.
- **Casillas correlacionadas.** Las casillas de un municipio comparten serie, entrenamiento y régimen, y las semanas consecutivas están autocorrelacionadas. Los conteos de casillas y la binomial suponen independencia y sobrestiman la evidencia.
- **Optimismo dentro de muestra.** Las cifras del contraste de familias se eligen y se reportan sobre las mismas casillas (ver `reports/tesis_contraste_modelos/README.md`). Las familias secuenciales tienen además variación entre GPU y CPU, y tres familias no coinciden entre las tablas 01 y 05 en h = 8.
- **B01 en el producto sin validación confirmatoria.** La API (`docs/forecasting/api_salida_d10.md`, esquema 0.2) usa B01 desde D10.1; la receta y la evaluación preregistrada siguen con la regla p̂_t ≥ τ. La evidencia de B01 en 2023–2025 es posterior a la consulta de la prueba; queda pendiente una validación prospectiva con datos posteriores a 2025.
