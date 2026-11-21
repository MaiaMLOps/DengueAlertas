# Metodología punta a punta — EWT final

## 1. Objetivo

El aporte evalúa si la ingeniería de variables basada en el canal EWT mejora modelos locales de alerta temprana de dengue. La unidad de análisis es municipio–semana de emisión `t`; el horizonte operativo es `h=8` semanas.

Se preservan las decisiones del equipo: seis pilotos, un modelo por municipio, canal histórico local, rezago de disponibilidad, validación cronológica y comparación con referencias. No se interpreta ninguna feature climática o demográfica como efecto causal.

## 2. Dos desenlaces

### 2.1 `zone_future` — desenlace principal

`y_zone_h8(t)=1` si `confirmed_records_screening(t+8) > EWT(t+8)` y la semana objetivo es elegible. Responde directamente: **¿estará el municipio en zona epidémica ocho semanas adelante?**

Ventajas: conserva todas las semanas con canal futuro conocido y permite comparar discriminación y calibración. Riesgo: el estado EWT puede ser muy persistente; una referencia de persistencia puede ser difícil de superar y una prevalencia extrema infla AUC-PR.

### 2.2 `pre_onset` — sensibilidad

1. Se marcan excesos semanales `count > EWT`.
2. Excesos contiguos se agrupan; huecos de hasta dos semanas se fusionan.
3. Se conserva un episodio si dura al menos tres semanas.
4. Las ocho semanas anteriores al inicio reciben 1.
5. Brote y ocho semanas refractarias quedan en `NaN`; el resto conocido recibe 0.

Responde: **¿iniciará un episodio en las próximas ocho semanas, dado que hoy no estamos en brote/refractario?** La regla de inicio es sensible al ruido. Rachas largas pueden dejar muy pocas observaciones evaluables; el pipeline no fuerza un modelo cuando faltan ambas clases.

## 3. Canal EWT

Para municipio `i`, año `Y` y ordinal de lunes `w`, la referencia usa exactamente los conteos de `Y-5,…,Y-1` en el mismo ordinal:

`EWT = exp(mean(log(C + 0.001))) - 0.001`.

Implementación estable: `0.001 * expm1(mean(log1p(C/0.001)))`. Historias constantes e identidades enteras exactas evitan residuos que cambiarían una igualdad. Años epidémicos se mantienen. Si `w=53` no existe en un año histórico, se usa únicamente su ordinal 52. La etiqueta es estricta: igualdad con EWT produce 0.

Inicialización 2007–2011 y semana parcial 2025-12-29 no se etiquetan. El límite superior usa la dispersión geométrica y un cuantil t unilateral; se utiliza como feature, no como etiqueta.

## 4. Ingeniería de variables

Se construyen 66 variables:

- 19 `epi_`: nivel, rezagos, medias, volatilidad, crecimiento, variación interanual y actividad;
- 16 `ch_`: distancia al EWT, estandarización geométrica, relación con límite superior, estado de episodio y canal de la semana objetivo;
- 2 `cal_`: seno/coseno de la semana objetivo;
- 27 `clim_`: rezagos, medias y anomalías de temperatura, precipitación y humedad;
- 2 `demo_`: log-población y log-densidad.

No se incluyen variables espaciales. Las ablaciones son:

- `epi`: 19 epidemiológicas + 16 canal + 2 calendario = 37;
- `epi_clima`: las anteriores + 27 clima = 64;
- `epi_clima_demo`: las anteriores + 2 demografía = 66.

Las columnas totalmente vacías o constantes se descartan usando exclusivamente el tramo de entrenamiento. Consulte `FEATURES.md`.

## 5. Disponibilidad y prevención de fuga

- casos: último valor utilizable `t-1`;
- clima: último valor utilizable `t-1`;
- ventanas móviles: se desplazan antes de calcularse;
- canal anual: usa sólo cinco años anteriores y se conoce para el año en curso;
- canal de `t+8`: si cruza de año, se usa el ordinal equivalente del canal conocido del año de emisión, no un canal que incorporaría casos futuros;
- demografía: arrastre sólo dentro del municipio;
- etiqueta: nunca entra como predictor.

La prueba automatizada altera conteos futuros y exige que las 66 features históricas no cambien.

## 6. Diseño de validación

### 6.1 Selección

Folds rolling-origin con validación en 2019, 2020, 2021 y 2022. Para un fold que inicia en `v`, entrenamiento sólo usa emisiones `< v-8 semanas`. Esto impide que el desenlace `t+8` de una fila de entrenamiento caiga en la validación.

Cuando `pre_onset` no tiene ambas clases en ningún año aislado, se permite un único bloque 2019–2022, purgado y documentado. No se relaja la prueba.

### 6.2 Calibración y umbral

Las probabilidades fuera de muestra de selección alimentan una calibración Platt. El umbral minimiza:

`5 × falsos_negativos + 1 × falsos_positivos`.

El umbral se fija antes de abrir la prueba. La elección 5:1 prioriza sensibilidad; si conduce a alertar casi siempre, debe renegociarse con el equipo y complementarse con una restricción de especificidad o carga anual.

### 6.3 Prueba bloqueada

Años 2023–2025. El entrenamiento final termina ocho semanas antes del primer lunes de prueba. La prueba no decide algoritmo, conjunto de variables, calibrador o umbral. Las últimas ocho emisiones sin desenlace futuro conocido y la semana parcial quedan fuera.

## 7. Candidatos

Por municipio y tarea:

- logística regularizada y balanceada;
- LightGBM de baja capacidad;
- XGBoost de baja capacidad;
- tres ablaciones de variables.

La selección prioriza AUC-PR de validación, luego Brier, recall y simplicidad. Si ningún candidato cumple la restricción de rachas de falsas alertas, se conserva el mejor disponible y se documenta; no se consulta la prueba para cambiarlo.

## 8. Referencias

- **Persistencia:** última bandera EWT conocida (`ch_E_lag0`).
- **Climatología:** tasa histórica suavizada por semana objetivo del año.

El aporte sólo es operativo si mejora referencias en discriminación **y** calibración, no únicamente en AUC-PR. Con prevalencia extrema se deben mirar ROC-AUC, Brier, especificidad, matriz de confusión y número de positivos/negativos.

## 9. Métricas

Semanas: AUC-PR, ROC-AUC, Brier, log-loss, precision, recall, especificidad, F1, TN/FP/FN/TP y rachas de alerta. `pre_onset` añade brotes anticipados, sensibilidad por evento y semanas de antelación.

AUC-PR siempre se reporta junto con prevalencia. Un AUC-PR cercano a 1 con una sola semana negativa no prueba robustez. Brier menor es mejor.

## 10. Resultados reproducidos en esta entrega

La corrida incluida utilizó 5.952 filas (seis municipios, 2007-01-01 a 2025-12-29), derivó 58 episodios y generó 66 features. Para `zone_future`, la logística ganó en cuatro municipios; clima se incorporó en dos ganadores y demografía sólo en el ganador de Barranquilla.

El AUC-PR macro fue alto por la prevalencia extrema, pero el modelo sólo mejoró levemente persistencia en AUC-PR y empeoró Brier en cinco de seis municipios. La especificidad fue de 0 a 0,30. Barranquilla no superó la referencia; Villavicencio tuvo una sola semana negativa. Esto es evidencia de que el pipeline funciona, pero **no de que el predictor esté listo para despliegue**.

`pre_onset` sólo fue evaluable en Barranquilla y Soledad, con 17 y 10 semanas de prueba; cuatro municipios no permitieron una comparación válida. Se conserva como sensibilidad, no como resultado principal.

## 11. Trazabilidad y MLflow

Cada candidato puede publicarse como run. Cada ganador puede registrarse como versión pyfunc con modelo, calibrador, umbral, columnas y evidencia. La configuración prohíbe transición de etapa y alias; toda versión queda en `None`. La huella SHA-256 del insumo y de la configuración se conserva en `run_manifest.json`.

## 12. Limitaciones y siguiente decisión

1. Confirmados finales retrospectivos no reproducen revisiones y demoras de vigilancia.
2. Los pilotos fueron seleccionados con conocimiento histórico.
3. EWT `count > reference` no es una declaración oficial de brote.
4. La elevada persistencia del desenlace puede dominar cualquier modelo.
5. Los intervalos son inestables cuando hay pocos negativos o episodios.
6. La contribución climática/demográfica es predictiva, no causal.

Siguiente paso: acordar un criterio mínimo de mejora frente a persistencia (AUC-PR y Brier), especificidad/carga de alertas, y repetir una simulación prospectiva con disponibilidad real. No promover modelos hasta superar esa compuerta.
