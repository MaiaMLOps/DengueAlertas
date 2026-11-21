# Diccionario de las 66 variables EWT

Cada fila del modelo representa un municipio en una semana de emisión `t`. Casos y clima tienen rezago operativo de una semana. El canal del año usa únicamente los cinco años anteriores. Ninguna variable utiliza el desenlace futuro.

## Conjuntos de variables

- `epi`: 37 columnas = 19 `epi_` + 16 `ch_` + 2 `cal_`.
- `epi_clima`: 64 columnas = `epi` + 27 `clim_`.
- `epi_clima_demo`: 66 columnas = `epi_clima` + 2 `demo_`.

`usable_columns` elimina columnas vacías o constantes usando sólo el entrenamiento de cada fold.

## Historia epidemiológica (19)

| # | Variable | Construcción / interpretación | Última información |
|---:|---|---|---|
| 1 | `epi_logc_lag0` | log(1+C) con rezago operativo 0; en emisión t corresponde a t-1 | casos hasta t-1 |
| 2 | `epi_logc_lag1` | log(1+C) con rezago operativo 1; en emisión t corresponde a t-2 | casos hasta t-1 |
| 3 | `epi_logc_lag2` | log(1+C) con rezago operativo 2; en emisión t corresponde a t-3 | casos hasta t-1 |
| 4 | `epi_logc_lag3` | log(1+C) con rezago operativo 3; en emisión t corresponde a t-4 | casos hasta t-1 |
| 5 | `epi_logc_lag4` | log(1+C) con rezago operativo 4; en emisión t corresponde a t-5 | casos hasta t-1 |
| 6 | `epi_logc_lag8` | log(1+C) con rezago operativo 8; en emisión t corresponde a t-9 | casos hasta t-1 |
| 7 | `epi_logc_lag12` | log(1+C) con rezago operativo 12; en emisión t corresponde a t-13 | casos hasta t-1 |
| 8 | `epi_logc_lag26` | log(1+C) con rezago operativo 26; en emisión t corresponde a t-27 | casos hasta t-1 |
| 9 | `epi_logc_lag52` | log(1+C) con rezago operativo 52; en emisión t corresponde a t-53 | casos hasta t-1 |
| 10 | `epi_logc_mean4` | media móvil de log(1+C) en las 4 semanas anteriores | casos hasta t-1 |
| 11 | `epi_logc_mean8` | media móvil de log(1+C) en las 8 semanas anteriores | casos hasta t-1 |
| 12 | `epi_logc_mean12` | media móvil de log(1+C) en las 12 semanas anteriores | casos hasta t-1 |
| 13 | `epi_logc_std4` | desviación móvil de log(1+C) en las 4 semanas anteriores | casos hasta t-1 |
| 14 | `epi_logc_std8` | desviación móvil de log(1+C) en las 8 semanas anteriores | casos hasta t-1 |
| 15 | `epi_logc_std12` | desviación móvil de log(1+C) en las 12 semanas anteriores | casos hasta t-1 |
| 16 | `epi_growth_1w` | cambio logarítmico entre las dos últimas semanas conocidas | t-1 y t-2 |
| 17 | `epi_growth_4w` | media logarítmica últimas 4 semanas menos 4 previas | casos hasta t-1 |
| 18 | `epi_yoy_4w` | media reciente de 4 semanas menos media de la misma época un año atrás | casos hasta t-1 |
| 19 | `epi_active_frac12` | fracción de las últimas 12 semanas con conteo mayor que cero | casos hasta t-1 |

## Canal EWT y detector (16)

| # | Variable | Construcción / interpretación | Última información |
|---:|---|---|---|
| 20 | `ch_logratio_ewt_lag0` | log((C+1)/(EWT+1)) de la última semana conocida | t-1 |
| 21 | `ch_logratio_ewt_lag1` | misma razón una semana más atrás | t-2 |
| 22 | `ch_logratio_ewt_lag2` | misma razón dos semanas más atrás | t-3 |
| 23 | `ch_E_lag0` | bandera count>EWT de la última semana conocida | t-1 |
| 24 | `ch_E_lag1` | bandera count>EWT con dos semanas de rezago | t-2 |
| 25 | `ch_E_lag2` | bandera count>EWT con tres semanas de rezago | t-3 |
| 26 | `ch_z_ewt_lag0` | distancia geométrica estandarizada al EWT | t-1 |
| 27 | `ch_z_ewt_lag1` | distancia geométrica estandarizada al EWT | t-2 |
| 28 | `ch_z_ewt_max4` | máximo z geométrico de las últimas 4 semanas conocidas | hasta t-1 |
| 29 | `ch_ratio_ub_lag0` | conteo/límite superior EWT | t-1 |
| 30 | `ch_ratio_ub_lag1` | conteo/límite superior EWT | t-2 |
| 31 | `ch_episode_active` | estado de episodio derivado, rezagado | t-1 |
| 32 | `ch_log_ewt_target` | log(1+EWT) para el ordinal de la semana objetivo t+h | canal anual conocido en t |
| 33 | `ch_log_ub_target` | log(1+límite superior) para la semana objetivo | canal anual conocido en t |
| 34 | `ch_gap_to_target_ub` | epi_logc_lag0 menos ch_log_ub_target | casos t-1 + canal objetivo |
| 35 | `ch_log_gsd` | log(1+desviación geométrica) de la banda anual | cinco años previos |

## Calendario objetivo (2)

| # | Variable | Construcción / interpretación | Última información |
|---:|---|---|---|
| 36 | `cal_sin_woy` | seno de 2π×semana objetivo/52.18 | calendario conocido |
| 37 | `cal_cos_woy` | coseno de 2π×semana objetivo/52.18 | calendario conocido |

## Clima (27)

| # | Variable | Construcción / interpretación | Última información |
|---:|---|---|---|
| 38 | `clim_temperature_c_lag0` | temperatura con rezago operativo 0; último dato climático t-1 | t-1 |
| 39 | `clim_temperature_c_lag2` | temperatura con rezago operativo 2; último dato climático t-1 | t-3 |
| 40 | `clim_temperature_c_lag4` | temperatura con rezago operativo 4; último dato climático t-1 | t-5 |
| 41 | `clim_temperature_c_lag6` | temperatura con rezago operativo 6; último dato climático t-1 | t-7 |
| 42 | `clim_temperature_c_lag8` | temperatura con rezago operativo 8; último dato climático t-1 | t-9 |
| 43 | `clim_temperature_c_lag12` | temperatura con rezago operativo 12; último dato climático t-1 | t-13 |
| 44 | `clim_temperature_c_mean4` | media móvil de temperatura en 4 semanas | hasta t-1 |
| 45 | `clim_temperature_c_mean8` | media móvil de temperatura en 8 semanas | hasta t-1 |
| 46 | `clim_temperature_c_anom4` | media 4 semanas de temperatura menos media 52 semanas | hasta t-1 |
| 47 | `clim_precipitation_mm_lag0` | precipitación con rezago operativo 0; último dato climático t-1 | t-1 |
| 48 | `clim_precipitation_mm_lag2` | precipitación con rezago operativo 2; último dato climático t-1 | t-3 |
| 49 | `clim_precipitation_mm_lag4` | precipitación con rezago operativo 4; último dato climático t-1 | t-5 |
| 50 | `clim_precipitation_mm_lag6` | precipitación con rezago operativo 6; último dato climático t-1 | t-7 |
| 51 | `clim_precipitation_mm_lag8` | precipitación con rezago operativo 8; último dato climático t-1 | t-9 |
| 52 | `clim_precipitation_mm_lag12` | precipitación con rezago operativo 12; último dato climático t-1 | t-13 |
| 53 | `clim_precipitation_mm_mean4` | media móvil de precipitación en 4 semanas | hasta t-1 |
| 54 | `clim_precipitation_mm_mean8` | media móvil de precipitación en 8 semanas | hasta t-1 |
| 55 | `clim_precipitation_mm_anom4` | media 4 semanas de precipitación menos media 52 semanas | hasta t-1 |
| 56 | `clim_relative_humidity_pct_lag0` | humedad relativa con rezago operativo 0; último dato climático t-1 | t-1 |
| 57 | `clim_relative_humidity_pct_lag2` | humedad relativa con rezago operativo 2; último dato climático t-1 | t-3 |
| 58 | `clim_relative_humidity_pct_lag4` | humedad relativa con rezago operativo 4; último dato climático t-1 | t-5 |
| 59 | `clim_relative_humidity_pct_lag6` | humedad relativa con rezago operativo 6; último dato climático t-1 | t-7 |
| 60 | `clim_relative_humidity_pct_lag8` | humedad relativa con rezago operativo 8; último dato climático t-1 | t-9 |
| 61 | `clim_relative_humidity_pct_lag12` | humedad relativa con rezago operativo 12; último dato climático t-1 | t-13 |
| 62 | `clim_relative_humidity_pct_mean4` | media móvil de humedad relativa en 4 semanas | hasta t-1 |
| 63 | `clim_relative_humidity_pct_mean8` | media móvil de humedad relativa en 8 semanas | hasta t-1 |
| 64 | `clim_relative_humidity_pct_anom4` | media 4 semanas de humedad relativa menos media 52 semanas | hasta t-1 |

## Demografía (2)

| # | Variable | Construcción / interpretación | Última información |
|---:|---|---|---|
| 65 | `demo_log_population` | log(población vigente, mínimo 1) | valor anual disponible |
| 66 | `demo_log_density` | log(1+densidad poblacional) | valor anual disponible |

## Reglas contra fuga

1. Todas las transformaciones de conteos aplican `shift(1)` antes de rezagos o ventanas.
2. Clima aplica `shift(1)`.
3. El canal de `t+h` se consulta en el canal anual conocido; si cruza de año no se construye un canal futuro.
4. Población/densidad se arrastran sólo dentro del municipio.
5. Variables vacías/constantes se detectan sólo en entrenamiento.
6. Las columnas `y_*`, `excess_flag_ewt`, conteos sin rezago, fechas e identificadores nunca entran en `FEATURE_SETS`.

## Interpretación

- Coeficientes logísticos se reportan en valor absoluto para ranking y con signo por separado.
- Árboles reportan `feature_importances_`; no se comparan magnitudes absolutas entre algoritmos.
- La frecuencia de una familia en el top 15 es descriptiva, no evidencia causal.
- Una variable climática puede ayudar por estacionalidad compartida; debe contrastarse con `cal_` y validación prospectiva.

