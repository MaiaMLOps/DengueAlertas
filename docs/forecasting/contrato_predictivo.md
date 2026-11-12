# 02 — Contrato predictivo
 
**Creado:** 2026-11-09, chat de metodología. **Última actualización:** 2026-11-12 (enmienda D7.2: el conjunto de variables pasa a ser un paso de selección, §6.6; y enmienda D9.1: análisis diagnóstico integrado, §8.8 y §10.22; solo reporte). Versiones anteriores: 2026-11-11 (enmienda D7.1: el entrenamiento de cada ensayo se cierra en su primera emisión, §6.1) y 2026-11-10 (D1.1 y D3–D10 cerrados: contrato completo).
**Propósito:** especificación vinculante de qué se predice, con qué información, contra qué referencias y cómo se evalúa. Es la fuente de verdad para la implementación en la rama `feature/forecasting-mvp`. Las alternativas descartadas y su justificación detallada están en `01_registro_decisiones.md`.
 
Convención: **[H]** hecho verificado; **[C]** cálculo reproducible con el script indicado; **[D]** decisión; **[I]** inferencia; **[R]** recomendación; **[P]** pendiente.
 
**Principio rector [D]:** target, información disponible, referencias, métricas y criterios se fijan **antes** de observar el desempeño de cualquier modelo. Ninguna cláusula se modifica por cómo harían lucir los resultados. Un cambio posterior se registra en `01` con su motivo.
 
## 0. Estado de las decisiones
 
| # | Tema | Estado | Dónde |
|---|---|---|---|
| D0 | Alcance territorial | [D] Seis pilotos, un modelo independiente por municipio | `01` |
| D1 | Conteo, calendario e historia del canal | [D] Contrato del 2026-10-21, enmendado por D1.1 | `01` |
| D1.1 | Parametrización del canal | [D] epiCo con desplazamiento 1 para la línea central y el límite superior | §1 y `01` |
| D2 | Umbral de exceso | [D] U1e | `01` |
| D3 | Variable objetivo | [D] y_{t,h} = 1[C_{t+h} > U1e_{t+h}]; solo clasificación binaria | §2 |
| D4 | Emisión, información disponible y horizontes | [D] t = última semana completa; información de la base hasta t; horizonte principal h = 8; sensibilidad h ∈ {2, 4, 12} | §3 |
| D5 | Líneas base y modelos | [D] B2 estado del canal (principal), B1 climatología estacional (secundaria); regresión logística, LightGBM y GRU opcional con todas las features; familia común; una ablación del clima | §4 |
| D6 | Features | [D] 15 variables (estado y dinámica frente al canal, semana objetivo, anomalías climáticas en bloques mensuales); ventana de 12 semanas; sin variables demográficas | §5 |
| D7 | Validación y selección | [D] Origen móvil con reentrenamiento anual y corte por fecha de la semana objetivo, con el entrenamiento cerrado en la primera emisión de cada ensayo (D7.1); desarrollo 2019–2022, prueba 2023–2025; hiperparámetros con esquema híbrido (configuración común + excepción por municipio) | §6 |
| D7.2 | Conjunto de variables | [D] Paso 0 de la selección: sin clima por defecto; completo solo con mayor mejora media fuera de año y ≥ 17 de 24 casillas; la ablación es la familia final con el otro conjunto | §6.6 |
| D8 | Métricas y regla de decisión | [D] Estimación: la mejora de log-loss frente a B2 decide hiperparámetros y familia; BSS frente a B2 es la principal de reporte. Decisión: regla de alerta con k = 2, fechada en la confirmación y ventana de 12 semanas; τ común que maximiza la detección de inicios con falsas alarmas ≤ las del canal, en desarrollo | §7 |
| D9 | Incertidumbre y reglas de comparación | [D] Conteo de victorias pareadas con prueba binomial exacta de una cola (5 %). Desarrollo, con puntajes fuera de año: excepción por municipio si gana 4 de 4 años; familia más compleja si gana ≥ 17 de 24 casillas. Prueba: tres niveles de conclusión (≥ 13 de 18 casillas) y McNemar exacto para los inicios | §8 |
| D9.1 | Análisis diagnóstico integrado | [D] Solo reporte: régimen de cada casilla (epidémica, post-epidémica, tranquila), seis insumos para desarrollo y prueba, hipótesis secundarias H-R y H-E contrastadas solo en prueba, limitaciones de las métricas | §8.8 |
| D10 | Salida para el usuario final | [D] Vista principal con probabilidad, referencias B1 y B2, estado de alerta (§7.4), canal hasta t + 8 y ficha de prueba; cuatro vistas (pronóstico vigente, historial fuera de muestra, carga de datos, desempeño); API FastAPI ampliada | §9 |
 
## 1. Canal endémico de referencia (D1.1)
 
Todas las cantidades del canal que usan el modelado y el producto siguen la implementación de epiCo (`epiverse-trace/epiCo`, commit `39a45ef`, `endemic_channel`, `method = "geometric"`, `shift = 1`), con n = 5 años previos:
 
    central  = round(exp(mean(log(x + 1))) − 1, 5)
    gsd      = round(sd(log(x + 1)), 5)                     (ddof = 1)
    interval = |qt(0,025; n − 1)| · gsd / √n
    U1e      = central · e^interval                         (0 si central = 0)
 
donde x son los conteos de la misma semana ordinal en los cinco años anteriores (la semana 53 usa la 52 si falta, como en el repositorio). La elegibilidad sigue D1: semanas 2012–2025, sin la semana parcial 2025-12-29.
 
**Zonas del canal:** seguridad si C ≤ central; alerta si central < C ≤ U1e; epidemia si C > U1e. Se cumple U1e ≥ central en todas las semanas [C].
 
**Legado:** las columnas `ewt` y `excess_label` (ε = 0,001) del contrato del 2026-10-21 se conservan sin cambios para reproducir el informe de avance y la auditoría. No se usan en el modelado ni en el producto.
 
## 2. D3 — Variable objetivo (2026-11-09)
 
### 2.1 Definición [D]
 
    y_{t,h} = 1[C_{t+h} > U1e_{t+h}]
 
- C es `confirmed_records_screening` (D1) y U1e es el límite superior de §1. La comparación es estricta.
- Hay un target binario por municipio y por horizonte. La convención de t y el horizonte se fijan en D4 (§3): el horizonte principal es h = 8.
- Solo cuentan semanas objetivo elegibles según D1.
- **Vocabulario:** «semana en zona epidémica». No se llama «brote» a una semana individual; «inicio de brote» se reserva para el evento de §2.4.
### 2.2 Propiedades y cifras
 
**El umbral de la semana objetivo se conoce al emitir [H, por construcción].** U1e_{t+h} usa la misma semana ordinal de los cinco años anteriores al año de t+h. La más reciente de esas semanas ocurrió unas 52 semanas antes de t+h, así que para horizontes de pocas semanas ya se conoce en t. Lo único incierto es C_{t+h}, y U1e_{t+h} puede entrar como feature (D6).
 
**Cifras [C]** (2012–2025, 4.380 semanas elegibles):
 
| Municipio | Positivas | Prevalencia | Rachas (de 1 semana) | Entradas 0→1 | Inicios de brote (Umaña, U1e) |
|---|---|---|---|---|---|
| Barranquilla | 171 | 23,4 % | 28 (13) | 28 | 5 |
| Soledad | 204 | 27,9 % | 46 (21) | 46 | 7 |
| Neiva | 152 | 20,8 % | 26 (10) | 26 | 6 |
| Villavicencio | 165 | 22,6 % | 32 (14) | 31 | 7 |
| Cúcuta | 129 | 17,7 % | 30 (21) | 30 | 3 |
| Cali | 124 | 17,0 % | 13 (9) | 13 | 3 |
| **Total** | **945** | **21,6 %** | **175 (88)** | **174** | **31** |
 
- Las 88 rachas de una semana son el 50,3 % de las rachas, pero solo el 9,3 % de las semanas positivas. El 85 % de las positivas (805) está en rachas de tres semanas o más.
- **Autocorrelación del estado:** entre una semana observada s y s + 5, el estado cambia en el 11,7 % de los pares. Desde fuera de la zona epidémica se entra en el 7,4 % de los casos (253 entradas: 24 en Cali, 63 en Soledad). Desde dentro se sale en el 27,1 % (255 salidas).
**Implicaciones [I]:**
- La persistencia es una referencia fuerte. Eso no se corrige con el target: se gestiona en D5 y D8 exigiendo superarla y midiendo en las transiciones.
- Hay ruido irreducible: U1e es el intervalo de confianza de la media y se supera en ~16 % de las semanas de un proceso estable. Parte de las positivas aisladas son esa fluctuación, lo que pone un techo a la habilidad alcanzable.
### 2.3 Alcance de modelado [D]
 
- **Solo clasificación binaria** con salida probabilística p̂_{t,h} = P̂(y_{t,h} = 1 | información disponible en t).
- Fuera del MVP: modelos de conteo o regresión, y clasificación multiclase.
- **Zonas:** no se predicen. La zona observada entra como información de entrada (D6) y como visualización del canal (D10). Extensión posible después del MVP, sin modificar este contrato: un segundo clasificador binario para P(C > central).
### 2.4 Eventos y transiciones: objetos de evaluación, no target [D en principio; detalle en D8 y D10]
 
**Dos capas.**
- *Estimación:* el modelo produce p̂_{t,h}. Se juzga con reglas de puntuación propias (Brier, log-loss), calibración y métricas de clasificación frente a las líneas base.
- *Decisión:* una regla con parámetros (umbral τ y k = 2 emisiones consecutivas; §7.4) convierte p̂ en alertas. Se juzga con eventos.
**Tres niveles de evaluación**, todos reportados:
 
| Nivel | Pregunta | Tamaño |
|---|---|---|
| Semana | ¿Es buena la probabilidad semanal? | 4.380 semanas |
| Transición | ¿Acierta cuando el estado cambia? Se estratifica por el estado observado al emitir, que forma parte de la información disponible | 291 entradas y 299 salidas entre t y t + 8 |
| Evento | ¿Anticipa episodios? | 31 inicios de brote; 174 entradas a zona epidémica |
 
**Objetos de la evaluación por eventos:**
- *Evento:* se define solo con la serie observada. Inicio de brote según Umaña et al. (2026) con U1e: la primera de ≥ 3 semanas con C > U1e tras ≥ 2 semanas con C ≤ U1e. Se reportan también las entradas a la zona epidémica.
- *Alerta:* se define solo con las predicciones, mediante la regla de decisión.
- *Emparejamiento:* regla propia, declarada en D8 (§7.4). La regla exacta de Umaña et al. no está verificada (Tabla S1 del apéndice).
**Qué se optimiza con eventos:** solo la capa de decisión, en validación y nunca en prueba, con un criterio declarado antes (§7.5). El entrenamiento usa una regla de puntuación propia, y la selección de modelos e hiperparámetros, log-loss (§7.1).
 
**Línea base a nivel de evento:** la alerta descriptiva del propio canal (≥ 2 semanas consecutivas en zona de alerta). La persistencia binaria casi no puede anticipar un inicio, por construcción, y no se usa como referencia (D5, §4.1).
 
**Prohibido:**
- Eliminar filas de entrenamiento o de prueba según información posterior a la emisión, como hizo el masking del avance.
- Usar eventos como target.
### 2.5 Limitaciones declaradas
 
- Se usa la serie final consolidada: el estudio es retrospectivo. Ver §3.4.
- Ruido irreducible de la etiqueta (§2.2).
- Pocos eventos: 31 inicios en total, entre 3 y 7 por municipio. La evaluación por eventos será sobre todo descriptiva.
### 2.6 Evidencia
 
`claude/verificacion/descriptivos_d3.py` (entrada `pilot_weekly_labels.parquet`, SHA-256 `b3b151a5…2c7f`). Reproduce el conteo del estudio de umbrales (945 positivas, 175 rachas) y todas las cifras de §1, §2 y de D1.1 y D3 en `01`. Ejecución: `python -I descriptivos_d3.py <parquet>`.
 
## 3. D4 — Emisión, información disponible y horizontes (2026-11-09)
 
### 3.1 Convención temporal [D]
 
- **t:** la última semana W-SUN con dato completo en la base. Es la «semana actual» del sistema.
- **h:** la distancia, en semanas, entre t y la semana objetivo t + h. El target es y_{t,h} (§2.1).
- **Equivalencia con el avance:** su «h = 4» (último dato en t − 1, etiqueta en t + 4) corresponde a h = 5 en esta convención.
### 3.2 Información disponible en t [D]
 
- **Hasta t, inclusive:** todo lo que contiene la base: conteos, covariables climáticas, cantidades del canal y estado y zona observados.
- **Conocido de antemano:** la central y U1e de la semana objetivo, que se calculan con años anteriores al de t + h y por tanto con semanas anteriores a t, y el calendario de t + h (semana del año).
- **Prohibido:** cualquier dato de semanas posteriores a t.
  - Las cantidades de confirmación diferida solo se usan ya confirmadas: un inicio de brote en la semana s se conoce en s + 2, y la alerta descriptiva del canal en s, en s + 1.
  - Todo preprocesamiento que se ajusta con datos (escalado, imputación, codificaciones) se ajusta solo con el entrenamiento de cada pliegue (D7).
### 3.3 Modo de operación del MVP [D]
 
- **Sin tiempo real.** El MVP no se conecta a fuentes externas: usa la base del proyecto o datos que el usuario sube en la interfaz. En cada t se conoce todo lo que la base contiene hasta t, y no se modelan retrasos de notificación ni latencias de publicación.
- **Contrato de entrada para datos subidos**, validado por la aplicación:
  - semanas W-SUN completas;
  - un municipio entre los seis pilotos (D0);
  - historia suficiente para el canal y las anomalías climáticas: los cinco años completos anteriores al año de la semana t − 11, más todas las semanas hasta t (§5.4);
  - sin valores faltantes en conteos ni en clima;
  - la última semana subida se toma como t.
- **Pronósticos genuinos en el demo.** Con t = 2025-12-22, la última semana completa de la base, el objetivo a h = 8 es la semana del 2026-02-16, cuyo U1e se calcula con 2021–2025. Son pronósticos sin verdad conocida, distintos de la evaluación retrospectiva.
### 3.4 Limitaciones declaradas
 
- **Series consolidadas.** Los modelos se entrenan y evalúan con series consolidadas: conteos con clasificación final, y ERA5-Land y CHIRPS en sus versiones finales. En una operación en tiempo real [H, documentación de los productos]:
  - los conteos recientes por fecha de inicio de síntomas estarían incompletos;
  - el clima vendría de productos preliminares: ERA5-Land-T sale con ~5 días de latencia (la versión final, a los 2–3 meses), y CHIRPS preliminar ~2 días después de cada pentada (la final, en la tercera semana del mes siguiente).
  Si los conteos llegaran con d semanas de retraso, la anticipación operativa sería h − d.
- **Datos subidos incompletos.** Si el usuario sube semanas recientes todavía incompletas, las probabilidades tenderían a subestimar el riesgo [I]. La interfaz debe advertirlo (D10).
### 3.5 Horizonte principal [D]
 
**h = 8, como horizonte único del sistema.** Con él se seleccionan modelos, se fija la regla de alerta, se evalúan los eventos y se construye el producto.
 
**Criterio:** relevancia, dentro de lo viable.
- **Relevancia.**
  - [H] En Singapur, el tiempo que necesitaron las autoridades para controlar brotes fue de 1 a 3 meses, con mediana de 2 (Hii et al., 2012). Su transferencia a Colombia es una inferencia.
  - [C] La alerta descriptiva del canal precede los inicios de brote con una mediana de 3 semanas (alerta más reciente, según el estudio de umbrales, con el EWT como línea de alerta). h = 8 añade unas 5 semanas a esa anticipación; h = 4, apenas una.
- **Viabilidad [C].** El estado observado en t sigue siendo informativo sobre t + h en todo el rango de 4 a 12 semanas:
  | h | 2 | 4 | 6 | 8 | 10 | 12 | 26 | 52 |
  |---|---|---|---|---|---|---|---|---|
  | Kappa (estado en t frente a t + h) | 0,73 | 0,68 | 0,65 | 0,60 | 0,55 | 0,49 | 0,22 | ≈ 0 |
  | P(entrar en zona epidémica \| fuera en t) | 5,8 % | 6,9 % | 7,6 % | 8,6 % | 9,6 % | 10,8 % | 16,6 % | 22,3 % |
  | Entradas | 198 | 237 | 259 | 291 | 323 | 364 | 546 | 708 |
  A h = 8, la kappa por municipio va de 0,51 (Soledad) a 0,65 (Cali). A horizontes cortos el presente lo dice casi todo y el pronóstico aporta poco sobre el canal. Pasados unos 6 meses, el presente no informa y el problema es estacional.
- **Clima [H].** Los rezagos clima–dengue reportados son del orden de semanas a meses: lluvia con asociación máxima a 6–10 semanas y temperatura desde 4 semanas (Sri Lanka); temperatura a 1–2 meses en México y Brasil. A h = 8, el clima disponible en t cae dentro de esa ventana. Su transferencia a Colombia es una inferencia.
- **Principio rector.** Preferir un horizonte por su mayor probabilidad de buen desempeño violaría el principio rector. A h = 8 el modelo puede quedar cerca de la climatología estacional; si ocurre, se reporta así. Por eso D5 debe incluir una climatología por semana del año.
**Desviación declarada:** la propuesta comprometía h = 1, 2 y 4; el avance usó h = 4, que en esta convención equivale a 5.
 
### 3.6 Sensibilidad al horizonte [D, prioridad baja]
 
- **Valores:** h ∈ {2, 4, 12}.
- **Cómo se calcula:** con el mismo pipeline (h es un parámetro), el mismo protocolo de validación (D7) y métricas semanales frente a B2 y B1 (§4.1; D8).
- **Lo que no toca:** no entra en el producto, ni en la regla de alerta, ni en la evaluación por eventos.
- **Prioridad:** es lo primero que se recorta si falta tiempo. Se decidió así antes de ver resultados.
- **Si se ejecuta**, debe ser completa y responder estas preguntas, declaradas ahora:
  1. ¿Hasta qué horizonte supera el modelo al estado del canal (B2) y a la climatología estacional (B1)? Se responde con una curva de habilidad frente a h, con su incertidumbre (D9).
  2. ¿Crece con h el aporte del bloque climático, medido por la ablación de D5 y D6, como sugiere la literatura de rezagos?
  3. ¿Se mantiene el patrón de 1 y 2 en los seis municipios, o es heterogéneo?
### 3.7 Emisión y filas
 
- **Emisión:** una por semana, con origen móvil.
- **Filas:** cada fila es un par (municipio, t) con la semana objetivo t + h elegible según D1. El primer t depende de los rezagos de D6. Para h = 8, el último t con verdad conocida es el 2025-10-27, cuyo objetivo es la semana del 2025-12-22.
- **Varios horizontes:** si se producen (sensibilidad), la forma de hacerlo, con un modelo por horizonte o un modelo multisalida, se decide en D5. La evaluación es siempre por horizonte.
### 3.8 Evidencia
 
- `claude/verificacion/descriptivos_d3.py`, sección 6.
- Hii, Y. L., et al. (2012). Optimal lead time for dengue forecast. *PLoS Neglected Tropical Diseases, 6*(10), e1848.
- El estudio de Sri Lanka (IJERPH, 2016) y la revisión de rezagos de temperatura citados en el chat de metodología. **Verificar sus datos bibliográficos completos antes de citarlos en el artículo.**
## 4. D5 — Líneas base y modelos (2026-11-09)
 
### 4.1 Líneas base [D]
 
Todas se estiman por municipio y solo con los años de entrenamiento de cada pliegue (D7). Las frecuencias empíricas usan suavizado de Laplace, (positivos + 1)/(casos + 2), para evitar probabilidades de 0 o 1, que tienen log-loss infinito cuando fallan.
 
| Línea base | Predicción | Uso |
|---|---|---|
| **B2. Estado del canal (principal)** | P(y_{t,h} = 1 \| zona observada en t): una probabilidad por zona (seguridad, alerta, epidemia) | Referencia de los skill scores principales, en las métricas semanales y de transición |
| **B1. Climatología estacional (secundaria)** | P(y_{t,h} = 1 \| semana del año de t + h), con ventana circular de ±2 semanas | Separar la anticipación de la estacionalidad |
| **Alerta descriptiva del canal** | Primera de ≥ 2 semanas consecutivas en zona de alerta (central de epiCo < C ≤ U1e), confirmada en s + 1 | Referencia a nivel de eventos (D8) |
 
**Cómo se leen los resultados:**
- Si el modelo supera a B2 y a B1, anticipa más allá de lo que dicen el canal y la estación.
- Si supera a B2 pero no a B1, lo que aprendió es sobre todo la estacionalidad.
- Si no supera a B2, no aporta sobre lo que el epidemiólogo ya ve en el canal.
### 4.2 Modelos candidatos [D]
 
Hay tres candidatos:
1. Regresión logística regularizada.
2. LightGBM.
3. GRU: opcional, pero con intención de ejecutarse, después de los dos anteriores.
**Comunes a los tres:**
- Usan el conjunto de variables elegido en desarrollo (§6.6, D7.2): las 15 de D6 o las 6 sin el bloque climático.
- Un modelo por municipio (D0), con estrategia directa a h = 8.
- Las mismas filas y los mismos pliegues (D7).
- Salida probabilística, entrenada con log-loss y sin pesos de clase. La sensibilidad operativa se ajusta con τ en la capa de decisión (§2.4).
- Hiperparámetros con rejillas pequeñas, fijadas antes de evaluar en el periodo de prueba y seleccionadas por log-loss en validación interna (D7).
**Protocolo de la GRU:**
- Misma información que los otros candidatos: la secuencia de las mismas variables base, en la misma ventana de D6.
- Arquitectura pequeña (una capa), con dropout y parada temprana sobre validación interna temporal.
- Al menos 5 semillas, reportando la media y la dispersión.
- Una sola salida (h = 8). La multisalida solo se usa si se ejecuta la sensibilidad de §3.6.
### 4.3 Selección de la familia [D]
 
- **Una familia común para los seis municipios.** Cada municipio tiene su propio ajuste. La familia se elige en desarrollo, nunca en prueba, con la mejora de log-loss frente a B2 en cada municipio (§6.2). Cada familia entra con sus hiperparámetros ya elegidos.
- **Parsimonia** (precisada en D9, §8.3). Se parte de la familia más simple, en este orden: regresión logística, LightGBM, GRU. Una familia más compleja reemplaza a la vigente solo si su mejora promedio es mayor y además gana en al menos 17 de las 24 casillas municipio-año de desarrollo, con puntajes fuera de año.
- **Excepción por municipio** (precisada en D9, §8.3) solo si la familia alternativa gana a la común en ese municipio en los 4 años de desarrollo, con puntajes fuera de año.
- **Momento de la GRU.** La GRU compite solo si está lista antes de la selección. Si se ejecuta después, se reporta como análisis adicional sin cambiar la familia elegida.
- **Reporte.** El artículo reporta todas las familias en cada municipio; el producto usa la familia elegida. «Justificar con los resultados» significa reportar esa comparación completa, no modificar la regla después de verlos.
**Motivo [H, resultado matemático].** Elegir la mejor entre k candidatas con igual desempeño verdadero sesga su puntaje de validación hacia arriba. El sesgo esperado es E[máx de k normales estándar] · σ: 0,56σ con k = 2, 0,85σ con k = 3 y 1,16σ con k = 5. Repetir esa elección en cada municipio, con validaciones cortas y autocorrelacionadas, multiplica las elecciones ruidosas (sesgo del ganador; auditoría, I9).
 
### 4.4 Ablación del clima [D]
 
- **Diseño:** la familia elegida se reentrena con el otro conjunto de variables (§6.6, D7.2): sin el bloque climático si la receta usa el completo, o con él si la receta usa el sin clima. Todo lo demás es idéntico, incluido el protocolo de selección de hiperparámetros.
- **Medida:** diferencia de log-loss y de Brier en prueba, y BSS frente a B2, por municipio y global, con su incertidumbre (D9).
- **Interpretación:** es el aporte predictivo incremental del clima dado el resto de la información. No es un efecto causal.
- **Objetivo:** responde el objetivo específico 3 de la propuesta. Si se ejecuta la sensibilidad de §3.6, la ablación se repite en cada horizonte (pregunta 2 de §3.6).
### 4.5 Fuera de alcance [D]
 
- **De la propuesta:**
  - DLNM+INLA, porque es regresión de conteos (§2.3);
  - ST-GNN, porque combina municipios (D0).
- **Otras familias:**
  - bosque aleatorio, XGBoost y CatBoost, por redundantes con LightGBM;
  - GAM, que queda como trabajo futuro;
  - arquitecturas mayores, por desproporcionadas frente a unas 730 semanas por municipio.
- **Aporte demográfico.** La propuesta prometía medirlo junto al del clima. Resuelto en D6: las covariables demográficas se excluyen y la desviación se declara (§5.6).
## 5. D6 — Features (2026-11-09)
 
### 5.1 Notación
 
Para cada municipio y semana s:
 
- C_s es el conteo; U1e_s y central_s, las cantidades del canal (§1); zona_s ∈ {seguridad, alerta, epidemia}.
- pos_s = log(C_s + 1) − log(U1e_s + 1) es la posición frente al umbral epidémico. Es positiva si la semana está en zona epidémica.
- La anomalía climática semanal es a_s(X) = X_s − clim_s(X), con X ∈ {`temperature_c`, `precipitation_mm`, `relative_humidity_pct`}. Aquí clim_s(X) es el promedio de X en la misma semana ordinal de los cinco años anteriores al año de s (la semana 53 usa la 52), con la misma alineación que el canal. Solo usa el pasado, así que no depende del pliegue.
### 5.2 Variables de la regresión logística y LightGBM [D]
 
**Quince variables.** Todas usan semanas ≤ t, salvo las cantidades de la semana objetivo, que se conocen de antemano (§3.2). Los resúmenes usan una sola ventana, w = 4 semanas, fijada a priori. Los bloques mensuales son m1 = semanas t−3 a t, m2 = t−7 a t−4 y m3 = t−11 a t−8.
 
| # | Variable | Definición | Qué captura |
|---|---|---|---|
| 1 | `zona_t` | Zona del canal en t (categórica) | Estado actual; contiene la información de B2 |
| 2 | `posicion_4s` | Promedio de pos_s, s = t−3 … t | Nivel reciente frente al umbral epidémico |
| 3 | `crecimiento_4s` | log(C_t + 1) − log(C_{t−4} + 1) | Tendencia del último mes |
| 4 | `cambio_umbral` | log(U1e_{t+h} + 1) − log(U1e_t + 1) | Cuánto se mueve el umbral entre hoy y la semana objetivo: estacionalidad expresada por el canal |
| 5–6 | `sem_sin`, `sem_cos` | Seno y coseno de 2π·(w_{t+h} − 1)/52,18, donde w_{t+h} es la semana ordinal de t + h | Época del año de la semana objetivo; contiene la información de B1 |
| 7–9 | `anom_temp_m1..m3` | Promedio de a_s(temperatura) en cada bloque | Temperatura inusual hace 8–11, 12–15 y 16–19 semanas respecto a la semana objetivo |
| 10–12 | `anom_prec_m1..m3` | Suma de a_s(precipitación) en cada bloque | Lluvia acumulada inusual en cada bloque |
| 13–15 | `anom_hum_m1..m3` | Promedio de a_s(humedad relativa) en cada bloque | Humedad inusual en cada bloque |
 
**Bloque climático:** las variables 7 a 15. Es el que se ablaciona en D5 (§4.4).
 
### 5.3 Entrada de la GRU [D]
 
- **Secuencia** de las semanas s = t−11 … t con log(C_s + 1), pos_s, zona_s (codificada one-hot), a_s(temperatura), a_s(precipitación) y a_s(humedad).
- **Entradas estáticas:** `cambio_umbral`, `sem_sin` y `sem_cos`.
Es la misma información de la que se derivan las 15 variables: la GRU aprende su propia representación de la ventana en lugar de usar resúmenes fijos.
 
### 5.4 Disponibilidad y filas [C]
 
- **Primer t:** 2012-03-19. Son 12 semanas con cantidades del canal y anomalías climáticas, ambas disponibles desde 2012-01-02.
- **Último t con verdad conocida (h = 8):** 2025-10-27.
- **Filas:** 711 por municipio, 4.266 en total. Para otros horizontes de la sensibilidad el rango cambia.
- **Datos faltantes:** el panel no tiene valores faltantes en conteos ni en clima [H].
### 5.5 Preprocesamiento [D]
 
- **Regresión logística:** estandarización de las variables numéricas con la media y la desviación de las filas de entrenamiento de cada pliegue. `zona_t` se codifica con dos indicadores, con seguridad como referencia.
- **LightGBM:** sin escalado; `zona_t` entra como categórica.
- **GRU:** estandarización ajustada solo con el entrenamiento.
- **Imputación:** no se imputa nada. Los datos subidos con faltantes se rechazan por el contrato de entrada (§3.3).
### 5.6 Variables excluidas [D]
 
- **Población y densidad.** Dentro de cada municipio solo cambian de un año a otro, así que funcionan como reloj: su correlación con el año es de 0,957 a 0,995 (auditoría, I11). En una validación cronológica, sus valores de prueba quedan fuera del rango de entrenamiento.
- **Área.** Es constante por municipio.
- **Índice de tiempo o año.**
- **`ewt` y `excess_label`:** legado (D1.1).
**Desviación declarada.** Con modelos por municipio, el efecto demográfico no se puede separar de la tendencia temporal. Para estimarlo habría que comparar municipios, y D0 lo excluye. Por eso no se mide el aporte demográfico que prometía el objetivo específico 3; solo se mide el aporte climático.
 
### 5.7 Limitaciones
 
- **Ventanas fijas.** La ventana de 4 semanas, los tres bloques mensuales y la ventana de 12 semanas son elecciones a priori. No se ajustan con datos para no añadir grados de libertad al investigador. Otras combinaciones podrían dar mejores variables.
- **Lectura de la GRU [I].** Si la GRU, con la misma ventana, supera claramente a los modelos con resúmenes, eso indicaría que los resúmenes fijos pierden información.
- **Trabajo futuro.** Optimizar las ventanas dentro de la validación, nunca con el periodo de prueba.
## 6. D7 — Validación y selección (2026-11-10)
 
### 6.1 Esquema: origen móvil con reentrenamiento anual [D]
 
Se evalúa **la receta**, no un modelo ajustado una sola vez. La receta son las features, la familia, los hiperparámetros, el preprocesamiento y la regla de decisión. Lo único que cambia entre ensayos son los datos que ve.
 
- **Corte por la fecha de la semana objetivo, con el entrenamiento cerrado en la primera emisión (enmendado por D7.1, 2026-11-11).** Para evaluar el año Y en cada municipio:
  - la evaluación usa las filas cuya semana objetivo t + h cae en Y (año calendario del lunes de t + h);
  - t0 es la emisión más temprana entre esas filas: la primera emisión del ensayo;
  - el entrenamiento usa las filas, desde 2012, cuya semana objetivo es anterior o igual a t0 (t + h ≤ t0). Son las únicas cuya etiqueta ya existía en t0.

  El modelo de cada ensayo es el que se habría entrenado el día de su primera emisión, y se usa sin cambios en todas las emisiones del ensayo. Entre la última etiqueta de entrenamiento (t0) y la primera evaluada (t0 + h) quedan h semanas.

  Ejemplo (h = 8): la primera semana objetivo de 2019 es la del 2019-01-07, así que t0 = 2018-11-12. El entrenamiento llega hasta la fila emitida el 2018-09-17, cuyo objetivo es el 2018-11-12. Las filas emitidas del 2018-09-24 al 2018-11-05 (objetivos del 2018-11-19 al 2018-12-31) no entran en el entrenamiento de 2019, porque sus etiquetas no existían en t0; sí entran en el de 2020. El pronóstico emitido el 2018-11-26 apunta a la semana del 2019-01-21 y pertenece a la evaluación de 2019.

  **Motivo (D7.1).** La versión del 2026-11-10 cortaba el entrenamiento al final de Y − 1. Garantizaba que ninguna semana objetivo evaluada entrara al entrenamiento, pero no que el entrenamiento usara solo etiquetas existentes al emitir: las primeras h − 1 emisiones evaluadas de cada año (del t0 al t0 + h − 2) se hacían con un modelo que conocía las etiquetas de las semanas entre su emisión y su objetivo. Por la autocorrelación del estado (§2.2), esas etiquetas informan sobre el objetivo. La información entra por los parámetros del modelo, no por las variables, así que la prueba de truncamiento (§10, punto 7) no la detecta. Costo [C]: h − 1 = 7 filas de entrenamiento por ensayo y municipio (83 positivas de 294 filas en los 42 ensayos municipio-año). Las filas evaluadas y los tamaños de la tabla siguiente no cambian.
- **Desarrollo, para decidir:** cuatro ensayos, Y ∈ {2019, 2020, 2021, 2022}. Ahí se eligen los hiperparámetros, la familia y la regla de decisión (D8).
- **Prueba, se mira una sola vez:** Y ∈ {2023, 2024, 2025}.
  - La receta queda congelada; cada año se reentrena con todo lo conocido en su t0.
  - Las métricas se calculan sobre las predicciones de los tres años juntas, y también por año.
- **Modelo del producto:** se entrena al final con todas las filas (objetivos de 2012 a 2025) y la receta congelada. Su emisión es la última semana completa de la base (2025-12-22) y todas sus etiquetas existen en esa fecha (t ≤ 2025-10-27), así que cumple la misma regla.
- **Mismas condiciones para todos.** Todos los candidatos, la ablación y las líneas base usan exactamente las mismas filas de entrenamiento y evaluación en cada ensayo. Las líneas base se estiman con el entrenamiento de cada ensayo, así que las comparaciones son pareadas.
**Tamaños [C]:**
 
| Periodo | Semanas evaluadas | Positivas | Prevalencia | Inicios | Entradas |
|---|---|---|---|---|---|
| Desarrollo 2019–2022 | 1.248 (208 por municipio) | 319 | 25,6 % | 11 | 45 |
| Prueba 2023–2025 | 936 (156 por municipio) | 322 | 34,4 % | 10 | 41 |
 
En cada ensayo, el año evaluado equivale a entre el 10 % y el 14 % del entrenamiento (de 7 a 10 años). Cada semana evaluada en desarrollo entra después en el entrenamiento de los ensayos siguientes, de la prueba y del modelo final: ningún año reciente se pierde para entrenar.
 
### 6.2 Hiperparámetros: esquema híbrido [D]
 
1. **Mismos candidatos para todos.** Por familia se fija una sola lista de candidatos, con semilla, idéntica en los seis municipios.
2. **Búsqueda por municipio.** Cada municipio corre la búsqueda con sus propios datos y con los cortes de §6.1. Se guardan los puntajes de cada candidato en cada año de desarrollo.
3. **Tabla única** con municipio, familia, candidato, año y log-loss. En cada municipio y año, el log-loss se expresa como mejora frente a B2: 1 − log-loss del modelo / log-loss de B2. Así se normaliza la dificultad de cada municipio.
4. **Configuración común por familia:** el candidato con la mayor mejora promedio. Se promedia primero sobre los años y después sobre los municipios.
5. **Excepción por municipio.** El mejor candidato propio de un municipio reemplaza a la configuración común solo si la supera de forma consistente, año a año: gana en los 4 años de desarrollo con puntajes fuera de año (D9, §8.3).
**Justificación:**
- *Ruido de selección.* Elegir el mejor de k candidatos con puntajes ruidosos sesga a favor al ganador en E[máx de k normales]·σ: 1,97σ con k = 25 [C]. Si el puntaje que decide es el promedio de seis municipios, su ruido baja a σ/√6 y ese sesgo, a unos 0,80σ, bajo el supuesto de independencia entre municipios. Como los municipios comparten los mismos años, la reducción real es algo menor.
- *Evidencia por municipio escasa y desigual.* En desarrollo, Neiva tiene positivas solo en 2019; Cali tiene 20 positivas en total. Barranquilla y Soledad tienen más de 75.
- *Comparabilidad.* Sin la misma lista de candidatos no se puede calcular una configuración común ni comprobar que un municipio necesita otra.
- *Agregación parcial.* El esquema es una forma discreta de lo que en estadística se llama *partial pooling*: se parte de la estimación conjunta y solo se aparta de ella cuando la evidencia local lo justifica.
### 6.3 Estrategia de búsqueda [D]
 
| Familia | Estrategia |
|---|---|
| Regresión logística (L2) | Rejilla de C, unos 6 valores en escala logarítmica |
| LightGBM | Búsqueda aleatoria con presupuesto fijo, unos 20–30 candidatos, en un espacio compacto orientado a regularizar. El número de árboles entra en la búsqueda, sin parada temprana, y la semilla del modelo es fija |
| GRU | 2 a 4 arquitecturas pequeñas y fijas, comunes a todos los municipios, sin excepciones por municipio por costo. Parada temprana con el último año de cada entrenamiento; ≥ 5 semillas |
 
- **Declaración previa.** El espacio, el presupuesto y las semillas se fijan en la especificación de implementación, antes de correr el desarrollo.
- **Herramientas.** `GridSearchCV` y `RandomizedSearchCV` son válidos con estas condiciones:
  - `cv` como lista explícita de pares (entrenamiento, validación) por año, según §6.1;
  - preprocesamiento dentro de un `Pipeline`;
  - `scoring="neg_log_loss"` y `refit=False`;
  - el mismo `random_state` en los seis municipios.
  No se usan `KFold` ni `TimeSeriesSplit`, porque no cortan por año calendario ni por la fecha de la semana objetivo.
- **Sin búsqueda bayesiana ni adaptativa.** Propondría candidatos distintos en cada municipio, lo que rompe la tabla única, y con validación ruidosa sobreajusta la selección (Cawley y Talbot, 2010, *Journal of Machine Learning Research*; verificar la cita).
### 6.4 Diferencias con el informe de avance [H]
 
| | Avance | Este contrato |
|---|---|---|
| Partición | `TimeSeriesSplit(n_splits=3)` por filas, sin hueco; el «OOT» es el último pliegue de la misma validación | Corte por año de la semana objetivo, con el entrenamiento cerrado en la primera emisión de cada ensayo (hueco de h semanas, D7.1); prueba separada que no se usa para decidir |
| Hiperparámetros | Fijos y sin validar, iguales para todos (por ejemplo, 200 árboles, tasa 0,03, profundidad 5) | Elegidos en desarrollo con una regla de puntuación propia; común por defecto, excepción por municipio con evidencia |
| Elección del modelo | El «mejor ensamble» por municipio se eligió mirando el mismo pliegue que se reporta | Familia común elegida en desarrollo; el desempeño se reporta en una prueba no usada para elegir |
| Métrica de selección | AUC-PR y F1 | Log-loss expresado como mejora frente a B2 |
 
### 6.5 Limitaciones
 
- **Pocos eventos.** 11 inicios en desarrollo y 10 en prueba. Cali no tiene inicios en la prueba (el último fue en 2019), así que allí la evaluación por eventos solo mide falsas alarmas.
- **Prevalencia distinta entre periodos:** 25,6 % en desarrollo frente a 34,4 % en prueba, por el ciclo epidémico de 2023–2024. La calibración se revisa por año (D8 y D9).
- **Cambio de régimen.** Entre 2015 y 2018 hubo un periodo tranquilo en casi todos los municipios; desde 2019 el régimen es más activo. El primer ensayo de desarrollo entrena con 2012–2018.
- **Pocos años para medir la variación entre ellos.** Con cuatro años de desarrollo, la variación de un año a otro se estima con pocos puntos. Por eso la regla de excepción de D9 exige ganar los 4 años (§8.3).
### 6.6 Conjunto de variables (D7.2, 2026-11-12) [D]

**Dos conjuntos.** *Completo:* las 15 variables de §5.2. *Sin clima:* las 6 que no pertenecen al bloque climático (`zona_t`, `posicion_4s`, `crecimiento_4s`, `cambio_umbral`, `sem_sin`, `sem_cos`).

**Búsqueda.** Para cada familia y cada conjunto se corre la misma búsqueda: candidatos y semillas pre-registrados, cortes de §6.1 (D7.1) y tabla única con la misma estructura. La selección de hiperparámetros de §6.2 se aplica igual a cada combinación de familia y conjunto.

**Paso 0 de la selección, por familia.** El conjunto por defecto es *sin clima*, el más simple. *Completo* lo reemplaza solo si, con las configuraciones comunes fuera de año de cada conjunto (§8.3):
1. su mejora media fuera de año frente a B2 es mayor, y
2. gana al menos 17 de las 24 casillas de desarrollo por log-loss (un empate no cuenta).

No hay excepciones por municipio para el conjunto de variables.

**Después, en el orden de siempre y con el conjunto elegido de cada familia:** comparación de familias (§4.3 y §8.3); excepción de familia por municipio; excepción de hiperparámetros por municipio dentro de la familia y el conjunto finales; τ (§7.5) con las predicciones de desarrollo de la configuración elegida.

**Ablación (§4.4).** En cada municipio, la familia final con el otro conjunto de variables, con su configuración elegida por el mismo protocolo (configuración común y excepción de hiperparámetros). No compite con la receta: mide el aporte del clima.

**Motivo y origen:** `01`, D7.2. Se adoptó antes de abrir la prueba; la receta congelada anterior (hash `d2135a39…da98`, conjunto completo) queda sustituida sin haber puntuado 2023–2025.

## 7. D8 — Métricas y regla de decisión (2026-11-10)
 
### 7.1 Qué se mide: estimación y decisión [D]
 
El sistema tiene dos capas (§2.4) y cada una se juzga con métricas distintas:
- **Estimación:** ¿es buena la probabilidad p̂_{t,8}? Se juzga con reglas de puntuación propias, calibración y discriminación, en los niveles de semana y de transición.
- **Decisión:** ¿es útil la alerta que produce la regla con τ? Se juzga con eventos, frente a la alerta descriptiva del canal.
Las métricas se dividen en dos grupos. **Las que deciden** son pocas, se fijan aquí y se usan solo en desarrollo. **Las que solo se reportan** son todas las demás y se calculan en desarrollo y en prueba. Ninguna métrica de prueba cambia una decisión.
 
**Métrica que decide cada elección:**
 
| Decisión | Capa | Métrica | Datos | Regla |
|---|---|---|---|---|
| Hiperparámetros (§6.2) | Estimación | Mejora de log-loss frente a B2: 1 − LL/LL_B2 | Desarrollo, por municipio y año | Promedio sobre años y luego sobre municipios; excepción por municipio según D9 |
| Familia (§4.3) | Estimación | La misma | Desarrollo | Mayor mejora promedio; parsimonia y excepción según D9 |
| τ (§7.5) | Decisión | Capacidad de detección de inicios de brote, con falsas alarmas por municipio-año ≤ las del canal | Desarrollo, predicciones fuera de muestra concatenadas | §7.5 |
 
**Orden:** hiperparámetros → conjunto de variables (§6.6) → familia → τ → receta congelada (configuración y registro de experimentos) → prueba, una sola vez → modelo final.
 
**Por qué decide el log-loss y no otras métricas:**
- [H] El log-loss y el Brier son reglas de puntuación estrictamente propias: su valor esperado solo es óptimo si la probabilidad emitida es la verdadera (Gneiting y Raftery, 2007). Premian a la vez la discriminación y la calibración.
- [D] Decide el log-loss porque es la pérdida de entrenamiento (se ajusta y se selecciona con el mismo criterio) y porque penaliza más los errores hechos con confianza. El Brier se reporta porque es acotado, interpretable y descomponible.
- [H] AUC-PR y AP solo dependen del orden de las predicciones, no de su calibración. F1, sensibilidad, especificidad y VPP dependen de un umbral, así que elegir con ellas mezclaría las dos capas. Se reportan, no deciden.
### 7.2 Métricas de estimación (reporte) [D]
 
**Nivel semana** (todas las filas del periodo):
 
| Métrica | Referencia | Lectura |
|---|---|---|
| **BSS frente a B2 (principal de reporte)** | B2 | 1 − Brier/Brier_B2. Si es > 0, el modelo mejora lo que el canal ya dice |
| BSS frente a B1 | B1 | Mejora sobre la estacionalidad |
| Mejora de log-loss frente a B2 y frente a B1 | B2, B1 | La métrica de selección, reportada también en prueba |
| Brier y su descomposición de Murphy (confiabilidad, resolución, incertidumbre) | — | De dónde viene la mejora o el error. Se aproxima con los intervalos del diagrama de confiabilidad (Murphy, 1973) |
| AP (precisión promedio) | π del periodo | Discriminación. Un clasificador sin información tiene AP ≈ π |
| Diagrama de confiabilidad | Diagonal | 10 intervalos de igual frecuencia de p̂, con el número de semanas por intervalo |
| Calibración en grande por año | 0 | Media de p̂ menos prevalencia observada, por año. Necesaria porque la prevalencia pasa de 25,6 % en desarrollo a 34,4 % en prueba (§6.5) |
 
**Nivel transición.** Las mismas métricas propias (BSS y mejora de log-loss frente a B2) y la AP se calculan por separado en dos estratos, según el estado observado en t:
- **Fuera de la zona epidémica en t:** las positivas son entradas.
- **Dentro de la zona epidémica en t:** las negativas son salidas.
Estratificar es legítimo porque el estado en t es información disponible al emitir. Dentro de cada estrato B2 sigue siendo informativa, porque distingue seguridad de alerta.
 
### 7.3 Métricas de clasificación en τ (reporte) [D]
 
- **Qué se evalúa:** la emisión semanal ŷ_t = 1[p̂_t ≥ τ], con el τ elegido en §7.5, frente a y_{t,8}. No es la regla de alerta (k = 2), que se evalúa con eventos (§7.4).
- **Métricas:** sensibilidad, especificidad, VPP, F1 y MCC. Se reportan para el modelo y para B2 con el mismo τ.
- **Referencia de F1:** se acompaña de la F1 de un clasificador que marca todas las semanas como positivas, 2π/(1 + π). Vale 0,41 en desarrollo (π = 0,256) y 0,51 en prueba (π = 0,344). Una F1 por debajo de esa cifra no supera a marcarlo todo.
### 7.4 Regla de alerta y evaluación por eventos [D]
 
**Origen [H].**
- Las definiciones de alerta, inicio de brote, ventana de oportunidad, capacidad de detección y razón de alertas vienen de Umaña et al. (2026). Allí describen el comportamiento del canal; los autores aclaran que no lo evaluaron como herramienta de pronóstico.
- La propuesta del grupo trasladó la alerta al modelo, con el EWT como umbral, y añadió la comparación Δw frente al canal.
- El informe de avance no implementó la regla (auditoría, I12 y M15).
Este contrato adopta esas definiciones con cuatro precisiones, necesarias para comparar un predictor con el canal: la fecha de confirmación, la ventana explícita, las alertas de continuidad y las falsas alarmas como tasa.
 
**Definiciones:**
 
| Componente | Umaña et al. / propuesta | Este contrato | Soporte |
|---|---|---|---|
| Emisión positiva | Canal: semana en zona de alerta. Propuesta: el modelo proyecta casos sobre el EWT | **Modelo:** p̂_t ≥ τ, con p̂_t = P(zona epidémica en t + 8). **Canal:** zona observada en t = alerta (central < C_t ≤ U1e_t) | D3; EWARS-csd |
| Alerta | Primera de ≥ 2 semanas consecutivas | Racha maximal de ≥ 2 emisiones positivas consecutivas (k = 2), que cuenta como una sola alerta. Igual para modelo y canal | Umaña et al.; estudio multicéntrico (2 señales = 1 periodo) |
| Fecha de la alerta | Primera semana de la racha | Semana de confirmación t_c (la segunda de la racha), que es cuando se conoce. Para comparar con las ventanas de Umaña se suma una semana | Información disponible en t_c (§3.2) |
| Alerta de continuidad | No existe en el canal | Alerta del modelo cuya zona observada en t_c es epidemia. Se cuenta aparte y no entra en detección ni en falsas alarmas | Propia |
| Evento principal | ≥ 3 semanas sobre el límite superior tras ≥ 2 por debajo | Igual, con U1e (§2.4). **Secundario:** entrada a la zona epidémica | Umaña et al.; D3 |
| Emparejamiento | Tabla S1, no verificada | Un inicio en s está avisado si hay una alerta, no de continuidad, con t_c ∈ [s − 12, s − 1] (W = 12) | Estudio multicéntrico (rezago de 1 a 12 semanas) |
| Anticipación | Ventana de oportunidad | s − t_c, medida desde la alerta más reciente de la ventana | Umaña et al.; propuesta |
| Falsa alarma | Implícita en la razón de alertas | Alerta, no de continuidad, sin inicio de brote en [t_c + 1, t_c + 12] | Curvas AMOC |
 
**Consecuencias de la regla:**
- *Precisión del momento.* El nivel de eventos pregunta si hubo un aviso utilizable; la precisión del momento la juzga el nivel semanal.
- *Alerta prolongada.* Una alerta encendida más de 12 semanas antes del inicio no lo avisa y cuenta como falsa alarma. La regla penaliza al sistema que se queda encendido.
- *Justificación de las alertas de continuidad.* Sin separarlas, el modelo sumaría falsas alarmas por acertar durante una epidemia: ningún inicio puede seguir en 12 semanas, porque el inicio exige 2 semanas por debajo. El canal no puede alertar en zona epidémica, por construcción. La separación es legítima porque condiciona en el estado observado en t_c; el masking del avance condicionaba en el futuro.
- *Falsas alarmas.* La definición es estricta, coherente con Umaña et al. Solo como información, las falsas alarmas se desglosan en las seguidas de alguna semana en zona epidémica (exceso corto) y las seguidas de ninguna.
**Censura en los bordes [D].** Ningún objeto puede usar semanas fuera del periodo evaluado (desarrollo o prueba). Las emisiones de un periodo son las de sus filas (§6.1). Se excluyen:
- los inicios cuya ventana [s − 12, s − 1] y la confirmación de una alerta en s − 12 (emisión en s − 13) no estén cubiertas por emisiones del periodo;
- los inicios que se confirman (s + 2) después de la última semana objetivo del periodo;
- las alertas con t_c + 14 posterior a la última semana objetivo del periodo, porque no pueden juzgarse.
Los objetos censurados se reportan.
 
**Simetría [D].** El modelo y el canal se evalúan sobre las mismas semanas de emisión, con el mismo k, la misma fecha, la misma ventana y la misma censura.
 
**Inicios evaluables [C]:**
- **Desarrollo:** 10 de 11. Se censura Villavicencio 2019-01-28, cuya ventana empieza antes de la primera emisión del periodo. De los 10, 5 son de 2019.
- **Prueba:** 10 de 10.
- **Municipio-años:** 24 en desarrollo y 18 en prueba.
**Métricas de eventos:**
 
| Métrica | Definición | Papel |
|---|---|---|
| **Capacidad de detección** | Inicios avisados / inicios evaluables (Umaña et al.) | Decide τ (objetivo) |
| **Falsas alarmas por municipio-año** | Falsas alarmas / municipio-años del periodo | Decide τ (restricción) |
| Razón de alertas | Inicios avisados / alertas juzgables (Umaña et al.) | Reporte |
| Anticipación | Mediana y rango de s − t_c en los inicios avisados | Reporte |
| Δw | Anticipación del modelo menos la del canal, en los inicios avisados por ambos (propuesta) | Reporte inicio por inicio, sin prueba (§8.4) |
| Detección de entradas | Como la capacidad de detección, con entradas a la zona epidémica | Reporte secundario |
| Alertas de continuidad | Número por municipio-año | Reporte |
| Curva tipo AMOC | Capacidad de detección y anticipación mediana frente a falsas alarmas por municipio-año, sobre la rejilla de τ, con el canal como un punto | Reporte. En prueba es descriptiva y no cambia τ |
| Desempeño compuesto | 0,5 · detección + 0,5 · razón de alertas (Umaña et al.) | Reporte, solo por comparabilidad con el artículo |
 
### 7.5 Criterio para elegir τ [D]
 
- **Rejilla:** τ ∈ {0,05; 0,10; …; 0,95}.
- **Datos:** las predicciones fuera de muestra de los cuatro ensayos de desarrollo, concatenadas por municipio, con los seis municipios juntos (τ común). Se usan la familia y los hiperparámetros ya elegidos.
- **Restricción:** las falsas alarmas por municipio-año del modelo no superan las del canal, calculadas sobre las mismas semanas.
- **Objetivo:** la máxima capacidad de detección sobre los 10 inicios evaluables.
- **Empates:** se elige el τ mayor, que produce menos alertas.
- **Si ningún τ cumple la restricción:** se elige el τ con menos falsas alarmas (empate al mayor) y se reporta que el modelo no alcanza el costo del canal.
- **Fijo, sin ajuste:** k = 2, W = 12, la fecha en t_c y la rejilla. No hay τ por municipio.
**Motivo.** La restricción responde una pregunta operativa sin pesos arbitrarios: con un costo en falsas alarmas no mayor que el de la práctica actual, ¿el modelo detecta más inicios, y con más anticipación? El compuesto de Umaña et al. fija el peso α = 0,5, un juicio de valor; se reporta, pero no decide.
 
### 7.6 Agregación [D]
 
- **Global:** las métricas semanales y de transición se calculan sobre las filas de los seis municipios juntas; los eventos, con los conteos sumados.
- **Por municipio y por año:** todas las métricas semanales. Los eventos por municipio se reportan como conteos, porque son pocos.
- **Prueba:** los tres años juntos y cada año por separado.
- **Selección:** la métrica que decide hiperparámetros y familia agrega como fija §6.2 (promedio de las mejoras por municipio), no sobre las filas juntas.
### 7.7 Lo que no se hace [D]
 
- Decidir algo en prueba: la receta (hiperparámetros, familia y τ) se congela antes.
- Elegir con AUC-PR, AP, F1, sensibilidad, VPP o el desempeño compuesto.
- Ajustar k o W con datos, o fijar un τ por municipio.
- Eliminar filas según información posterior a la emisión (§2.4).
### 7.8 Limitaciones
 
- **Pocos eventos.** Hay 10 inicios evaluables en cada periodo, así que la capacidad de detección se mueve en saltos de 10 puntos. La mitad de los de desarrollo son de 2019, de modo que τ queda muy influido por un solo año. Cali no tiene inicios en prueba: allí solo se miden falsas alarmas. La comparación con el canal sigue la regla de §8.4.
- **Cambio de prevalencia.** τ se elige con π = 25,6 % y se aplica con π = 34,4 %. La calibración en grande por año lo hace visible.
- **Definición estricta de falsa alarma.** Cuenta como falsa una alerta seguida de un exceso corto; el desglose de §7.4 lo muestra.
- **Alertas repetidas.** Dos alertas separadas por una semana sin emisión positiva cuentan como dos, aunque un funcionario probablemente ignoraría la segunda (Jiang, Cooper y Neill, 2009). Se aplica igual a ambos sistemas y no se corrige para no añadir parámetros.
- **No es una réplica de Umaña et al.** La regla exacta de su Tabla S1 no está verificada (pregunta 3 para Carlos, `01` D2). La ventana W = 12 y las alertas de continuidad son convenciones de este contrato, con el precedente indicado.
- **Cifras del canal vistas antes.** El estudio de umbrales y D3 reportaron cifras descriptivas de la alerta del canal con otras reglas: precede un inicio el 12 % de las veces, con una anticipación mediana de 3 semanas. k y W se tomaron de la literatura, no de esas cifras. El desempeño del canal bajo esta regla se calcula por primera vez en la implementación, después de congelar esta sección.
### 7.9 Evidencia y referencias
 
- `claude/verificacion/eventos_d8.py` (inicios evaluables por periodo; depende de `descriptivos_d3.py`). Ejecución: `python -I eventos_d8.py <parquet>`.
- Umaña et al. (2026), *JMIR Public Health and Surveillance*, 12, e79914: definiciones, Figura 3 y métricas.
- Propuesta del proyecto de grado, grupo 5, §10: ventana de oportunidad, Δw, McNemar y bootstrap.
- Cardenas, R., et al. (2022). EWARS-TDR para dengue, chikungunya y Zika. *BMC Infectious Diseases, 22*. doi:10.1186/s12879-022-07197-6. En Cúcuta, dengue: sensibilidad 92 %, VPP 68 %, distancia de predicción de 3 a 5 semanas.
- Schlesinger et al. (2024). EWARS-csd en Colombia. *Frontiers in Public Health, 12*, 1323618: probabilidades fuera de muestra de superar el umbral de brote, calibradas por municipio.
- *Alarm variables for dengue outbreaks: a multi-centre study in Asia and Latin America* (repositorio Spiral, Imperial College): 2 señales = 1 periodo; rezago de 1 a 12 semanas. **Verificar autores y revista antes de citarlo.**
- Jiang, X., Cooper, G. F., y Neill, D. B. (2009). Generalized AMOC curves for evaluation and improvement of event surveillance. PMC2815453 (verificar la revista antes de citarlo).
- Gneiting, T., y Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association, 102*(477). Murphy, A. H. (1973). A new vector partition of the probability score. *Journal of Applied Meteorology, 12*. **Verificar los datos bibliográficos de ambas antes de citarlas.**
## 8. D9 — Incertidumbre y reglas de comparación (2026-11-10)
 
### 8.1 Propósito [D]
 
Con 4 años de desarrollo, 3 de prueba y 10 inicios evaluables por periodo, una diferencia pequeña entre dos opciones puede deberse a qué años tocaron. D9 fija, antes de ver resultados, cuándo una diferencia se trata como real:
- **en desarrollo,** para tomar las decisiones que D5 y D7 dejaron abiertas: la parsimonia entre familias y las excepciones por municipio;
- **en prueba,** para redactar las conclusiones del artículo.
Trata la incertidumbre **de la evaluación**. La incertidumbre de cada pronóstico la expresa la propia probabilidad p̂, si está calibrada; cómo mostrarla al usuario es D10.
 
### 8.2 Principio: contar victorias pareadas [D]
 
- **Unidades de comparación:**
  - la **casilla municipio-año**: las predicciones de un municipio en un año, hechas por su propio modelo en orden cronológico. Hay 6 × 4 = 24 en desarrollo y 6 × 3 = 18 en prueba;
  - el **año**, para las excepciones de un solo municipio;
  - el **inicio de brote**, para los eventos.
- **Victoria:** en una unidad, la opción evaluada tiene menor pérdida que la referencia sobre las mismas filas. Un empate no cuenta como victoria.
- **Referencia de azar:** si ambas opciones fueran equivalentes, ganar cada unidad sería como lanzar una moneda (p = 0,5). El umbral es el menor número de victorias cuya probabilidad por azar es menor que 5 % (prueba binomial exacta).
- **Una cola.** Todas las preguntas tienen una sola dirección y se declaran ahora: ¿la opción más compleja supera a la más simple?, ¿el modelo supera a la referencia? Nada se decide a partir de que una opción sea significativamente peor; si lo es, se reporta como descripción.
- **Métrica de la victoria:**
  - *en desarrollo,* log-loss, la métrica que decide en D8 (§7.1);
  - *en prueba,* Brier, porque el BSS frente a B2 es la métrica principal de reporte (§7.2). El conteo con log-loss se reporta junto.
- **Sin simulación.** No hay remuestreo ni intervalos; la tabla de casillas se publica completa, así que cualquiera puede verificar el conteo.
**Umbrales [C]:**
 
| Unidades | Umbral | Probabilidad por azar | Potencia si la opción gana de verdad el 75 % de las unidades |
|---|---|---|---|
| 18 casillas (prueba) | 13 de 18 | 4,8 % | 72 % |
| 24 casillas (desarrollo) | 17 de 24 | 3,2 % | 77 % |
| 4 años (excepción por municipio) | 4 de 4 | 6,25 % (lo más estricto posible con 4 años) | — |
| d inicios discrepantes (McNemar) | d < 5: imposible concluir; 5 de 5, 6 de 6, 7 de 7, 7 de 8, 8 de 9, 9 de 10 | < 5 % | — |
 
### 8.3 Reglas en desarrollo [D]
 
**Puntajes fuera de año.** Si una opción se elige y se juzga con los mismos años, casi siempre parecerá mejor (sesgo del ganador, §4.3). Por eso, para cada municipio y año de desarrollo y:
- cada configuración, la propia y la común, se elige con los otros tres años de desarrollo;
- se puntúa en y.
La configuración común sin y se elige con los seis municipios, como en §6.2. Son consultas a la tabla única de §6.2, sin entrenamientos adicionales. La GRU se puntúa en cada casilla con el promedio de la pérdida de sus semillas.
 
**Excepción de hiperparámetros por municipio (§6.2).** El mejor candidato propio, elegido sin y, gana a la configuración común, elegida sin y, en los 4 años de desarrollo. Si se cumple, el municipio usa su mejor candidato propio elegido con los 4 años. No aplica a la GRU (§6.3).
 
**Familia común (§4.3).** Comparación secuencial: LightGBM frente a la regresión logística, y después la GRU, si está lista, frente a la vigente. La familia más compleja reemplaza a la vigente solo si se cumplen dos condiciones:
1. su mejora promedio fuera de año frente a B2 es mayor;
2. gana en al menos 17 de las 24 casillas de desarrollo, con puntajes fuera de año.
**Excepción de familia por municipio (§4.3).** La familia alternativa gana a la común en ese municipio en los 4 años de desarrollo, con puntajes fuera de año.
 
**Todas las excepciones se reportan**, con sus cuatro diferencias anuales.
 
### 8.4 Conclusiones en prueba [D]
 
**Comparaciones semanales:**
- el modelo frente a B2 (principal);
- el modelo frente a B1;
- la ablación: la receta frente a la misma familia con el otro conjunto de variables (§6.6); se reporta como el aporte del bloque climático.
En cada casilla gana la opción con menor Brier. La conclusión tiene tres niveles:
 
| Nivel | Condición |
|---|---|
| **Mejora consistente** | Métrica global favorable (BSS global > 0, o Brier global menor en la ablación) y al menos 13 de 18 casillas ganadas |
| **Tendencia favorable** | Métrica global favorable y 10 a 12 de 18 casillas ganadas |
| **Sin mejora** | Cualquier otro caso |
 
- **Con log-loss:** se aplican los mismos niveles. Si los niveles con Brier y con log-loss difieren, se reportan ambos y la conclusión adopta el menor.
- **Por municipio:** no hay conclusión estadística, porque son 3 casillas. Se reportan el BSS del municipio y sus tres valores anuales.
**Eventos (frente al canal):**
- *Detección:* inicios avisados de los 10 evaluables, para el modelo y para el canal.
- *McNemar exacto de una cola,* sobre los inicios en que discrepan (uno avisa y el otro no). Hay **mejora consistente** si las victorias del modelo alcanzan el umbral de §8.2, y **tendencia favorable** si el modelo gana la mayoría de los inicios discrepantes sin alcanzarlo. En cualquier otro caso, **sin mejora**.
- *Falsas alarmas por municipio-año* del modelo y del canal: descriptivas.
- *Δw:* inicio por inicio, en los inicios avisados por ambos, sin prueba.
- *Desarrollo:* las comparaciones de eventos son solo descriptivas, porque τ se eligió con esos mismos datos.
**Redacción.** No alcanzar «mejora consistente» no prueba que no haya mejora: indica que con 3 años la evidencia no basta. El artículo lo redacta así.
 
### 8.5 Lo que no se hace [D]
 
- Bootstrap e intervalos de confianza.
- Intervalos o pruebas por municipio.
- McNemar sobre semanas: la autocorrelación viola el supuesto de independencia.
- Prueba estadística de Δw.
- Correcciones por comparaciones múltiples.
### 8.6 Limitaciones
 
- **Correlación entre casillas del mismo año.** 2019 y 2023–2024 fueron epidémicos en varios municipios a la vez, así que el conteo trata como independientes casillas que no lo son del todo y puede sobrestimar la evidencia. Los resultados por año (§7.6) lo hacen visible.
- **Potencia baja.** Un modelo que gana de verdad el 75 % de las casillas alcanza «mejora consistente» el 72 % de las veces en prueba.
- **Varias afirmaciones sin corrección.** En prueba se hacen cuatro afirmaciones (frente a B2, frente a B1, el clima y el canal), cada una con 5 %. Si todas las diferencias fueran nulas e independientes, la probabilidad de que al menos una salga favorable por azar sería ≈ 19 %.
- **Excepciones espurias.** Cada regla de 4 de 4 tiene 6,25 % por azar. Con entre 18 y 24 comparaciones posibles (hiperparámetros de dos familias y familia, en seis municipios), cabe esperar entre una y dos excepciones espurias. Su costo es bajo: si ambas opciones son equivalentes, elegir cualquiera rinde casi igual.
- **El 5 % es una convención,** fijada antes de ver resultados.
### 8.7 Evidencia y referencias
 
- `claude/verificacion/umbrales_d9.py`: umbrales, probabilidades por azar y potencia. Ejecución: `python -I umbrales_d9.py`.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika, 12*(2), 153–157. **Verificar los datos bibliográficos antes de citarla.**
- La auditoría del avance (I9e) y la propuesta (§10) recomendaban bootstrap y McNemar; la desviación se justifica en `01`, D9.
### 8.8 Análisis diagnóstico integrado (D9.1, 2026-11-12) [D]

**Propósito.** Además de las conclusiones de §8.4, el reporte incluye un análisis descriptivo, año por año y municipio por municipio: dónde aporta el modelo frente a B2, B1 y el canal, dónde no, qué patrones de los datos acompañan cada caso y cómo influyen en la lectura las propiedades de las métricas. No cambia ninguna decisión ni conclusión: la receta, los criterios de §8.3 y §8.4 y la regla de alerta quedan iguales. Se fija antes de abrir la prueba (motivo en `01`, D9.1).

**Régimen de cada casilla municipio-año.** Se clasifica con la serie observada y la noción de racha de §2.4, sin umbrales nuevos. Un *episodio* es una racha de ≥ 3 semanas consecutivas en zona epidémica (C > U1e) en la serie elegible completa. La clasificación se hace sobre el año calendario de esa serie, con los episodios identificados en toda ella, no solo sobre las filas evaluadas. El año de una casilla es el de sus semanas objetivo. Para 2019 y 2023, el año anterior (2018, 2022) se clasifica con la misma regla.
- *Epidémica:* al menos una semana del año calendario pertenece a un episodio.
- *Post-epidémica:* no es epidémica y el año anterior sí lo es, con la misma regla.
- *Tranquila:* el resto.

La clasificación usa semanas posteriores a la emisión, así que es solo un estrato de diagnóstico: no es variable, no decide nada y no se presenta como información disponible al emitir.

**Insumos obligatorios.** Se generan con el mismo código para desarrollo y para prueba. En desarrollo se usa, en cada casilla, la configuración común fuera de año (elegida sin y, §8.3) de la familia elegida; si hubiera excepciones, la que §8.3 compara en cada caso. Se rotula «usado para elegir»; en prueba, la receta congelada.
1. *Tabla de casillas:* municipio, año, régimen, filas, positivas, prevalencia; p̂ medio del modelo, de B2 y de B1, y su calibración en grande (p̂ medio − prevalencia); log-loss y Brier de los tres; diferencias absolutas y relativas frente a B2 y frente a B1; victoria por log-loss y por Brier.
2. *Resúmenes* de esa tabla por municipio, por año y por régimen, con cada métrica agregada de dos formas: promedio de las mejoras relativas por casilla y métrica calculada sobre las filas juntas.
3. *Estratos de transición* (§7.2), también por año.
4. *Inicios y entradas evaluables:* municipio, semana, régimen, aviso del modelo y del canal, anticipación de cada uno y Δw.
5. *Alertas juzgables* del modelo y del canal: t_c, continuidad, falsa alarma, si fue seguida de alguna semana en zona epidémica y régimen.
6. *Serie semanal* por municipio, para figuras: semana de emisión y objetivo, C, central y U1e de la semana objetivo, zona en t, y, p̂, B1, B2 y alertas del modelo y del canal.

**Hipótesis secundarias.** Se formularon con los resultados de desarrollo y se contrastan solo en prueba, de forma descriptiva y sin pruebas estadísticas adicionales (§8.5). La comparación frente a B1 ya es una conclusión principal de §8.4.
- *H-R, régimen:* (a) en las casillas epidémicas, el modelo gana a B2 por log-loss en más de la mitad; (b) en las post-epidémicas, pierde en más de la mitad; (c) en las post-epidémicas, la calibración en grande del modelo (p̂ medio − prevalencia, con signo) es mayor que la de B2 en más de la mitad, es decir, el modelo sobreestima más que B2.
- *H-E, eventos frente al canal:* (a) el modelo avisa menos inicios; (b) tiene menos falsas alarmas por municipio-año; (c) en los inicios avisados por ambos, Δw > 0 en más de la mitad.

Cada parte se declara sostenida, no sostenida o no evaluable (estrato con menos de 2 casillas o de 2 inicios). El análisis reporta también lo que contradiga estas hipótesis.

**Nivel descriptivo por municipio.** Cada municipio tiene su propio modelo, así que el análisis empieza por él. En prueba, cada municipio recibe un nivel descriptivo frente a B2, con la misma lógica de §8.4 aplicada a sus 3 casillas:
- *favorable:* BSS del municipio > 0 y gana en al menos 2 de sus 3 años;
- *mixto:* solo una de las dos condiciones;
- *desfavorable:* ninguna.

Se calcula con Brier y con log-loss y se adopta el menor, como en §8.4. Se reporta igual frente a B1 y para los eventos (inicios avisados y falsas alarmas frente al canal). No es una conclusión estadística: con 3 casillas, un modelo equivalente a B2 gana al menos 2 de 3 años la mitad de las veces. Por eso se acompaña de las casillas de desarrollo fuera de año del mismo municipio, como contexto de consistencia.

**Análisis cualitativo.** Estructura fija, de lo particular a lo general:
1. Perfil por municipio, uno por uno: nivel descriptivo, casillas de desarrollo y de prueba, episodios disponibles en su entrenamiento, dónde aporta y dónde no.
2. Relato por año (2019–2025): qué ocurrió en la serie observada de cada municipio (régimen, episodios, inicios) y cómo se comportaron el modelo, B2, B1 y el canal.
3. Síntesis de los seis municipios: conclusiones globales de §8.4 y qué patrones son comunes y cuáles son propios de cada municipio.
4. Contraste de H-R y H-E.
5. Efecto de las métricas en la lectura (lista siguiente).
6. Conclusiones integradas, separando hechos, inferencias e hipótesis.

**Comunicación por municipio.** Los municipios con nivel favorable se destacan como tales en el artículo y en la ficha de desempeño (§9.4, vista 4), porque es donde el sistema puede aportar más; los desfavorables se reportan con la misma visibilidad. Ningún municipio se omite ni se presenta solo en el agregado.

**Qué puede afirmar.** Puede asociar un resultado con patrones verificables en los datos: calibración en grande, régimen, número de episodios en el entrenamiento, desplazamiento del canal tras una epidemia y tamaño de la casilla. No atribuye resultados a variables concretas ni a causas epidemiológicas (inmunidad, serotipos, notificación); si se mencionan, es como hipótesis con referencia verificada. No modifica las conclusiones de §8.4.

**Limitaciones de las métricas que el análisis discute:**
- el promedio de mejoras relativas es inestable cuando el log-loss de B2 es pequeño (años sin positivas) y puede tener otro signo que la diferencia absoluta;
- la AP no está definida en casillas sin positivas;
- la detección se mueve en saltos de un inicio (10 puntos con 10 evaluables);
- las casillas de un mismo año están correlacionadas (§8.6);
- la restricción de falsas alarmas de τ puede no estar activa;
- la etiqueta U1e se mueve con su línea base: tras una epidemia, los años epidémicos entran en la referencia del canal.

## 9. D10 — Salida para el usuario final (2026-11-10)
 
### 9.1 Alcance y principio [D]
 
- **Qué fija:** qué dice el sistema, a quién, con qué datos y con qué lenguaje, y el contrato de la API que lo entrega.
- **Qué no fija:** el diseño visual. Lo define el equipo de frontend sobre la infraestructura existente (API FastAPI y frontend React en Cloudflare Pages); la maqueta del chat de metodología es orientativa.
- **Primera versión.** El equipo puede añadir elementos que aporten valor, sean ilustrativos o muestren el trabajo realizado, siempre que usen las fuentes de datos de §9.4 y las reglas de §9.5.
- **Principio:** el usuario ve exactamente lo que se evaluó: la misma probabilidad, las mismas referencias, la misma regla de alerta (§7.4) y el desempeño medido en prueba.
### 9.2 Usuario y pregunta [D]
 
- **Usuario de referencia:** un epidemiólogo de una secretaría de salud. En la evaluación del despliegue, también el jurado (usabilidad, 35 puntos).
- **Pregunta que responde:** ¿qué tan probable es que el municipio esté en zona epidémica dentro de 8 semanas, y el sistema está alertando?
### 9.3 Contenido de la vista principal [D]
 
| Elemento | Contenido | Origen |
|---|---|---|
| Probabilidad | p̂ de zona epidémica en t + 8, en porcentaje entero, con la semana de emisión y la semana objetivo explícitas | §2.1 |
| Referencias | B2 (solo con la zona actual del canal) y B1 (estacionalidad de la semana objetivo), en la misma escala | §4.1 |
| Estado de alerta | Sin alerta; señal en observación (una emisión ≥ τ); alerta confirmada (dos emisiones consecutivas ≥ τ); continuidad epidémica (alerta con la zona observada en t en epidemia). Con τ y k congelados | §7.4 |
| Canal | Casos observados hasta t, con la central y U1e de epiCo y sus zonas, y las bandas ya conocidas hasta t + 8, con la semana objetivo marcada | §1, §3.2 |
| Trayectoria | Probabilidades de las últimas 12 emisiones, con la línea de τ | §7.4 |
| Ficha de desempeño | Resultados de prueba (2023–2025) del municipio y globales (§9.4, vista 4) | §7, §8 |
| Advertencias | Texto fijo de §9.5 | §3.4 |
 
### 9.4 Vistas y origen de los datos [D]
 
1. **Pronóstico vigente.** Emisiones posteriores al 2025-10-27 (último t con verdad conocida) y hasta el 2025-12-22, con el modelo final (§6.1). Son pronósticos genuinos: sus semanas objetivo no tienen verdad conocida. Las referencias B1 y B2 se estiman con las mismas filas de entrenamiento del modelo final. La trayectoria de 12 semanas combina predicciones de prueba (t ≤ 2025-10-27) y del modelo final; todas son fuera de muestra respecto a su semana objetivo, y la interfaz lo indica.
2. **Historial fuera de muestra (2019–2025).** Usa solo las predicciones guardadas del origen móvil (§6.1), con las referencias B1 y B2 de cada ensayo. Marca los inicios avisados y no avisados, las falsas alarmas y las alertas de continuidad (§7.4), y distingue desarrollo (2019–2022, usado para elegir) de prueba (2023–2025). **Prohibido** mostrar predicciones del modelo final para semanas pasadas: ese modelo se entrenó con sus etiquetas y aparentaría un desempeño que no tiene.
3. **Cargar datos** (misma prioridad que las demás vistas). El usuario sube un CSV con el formato del contrato de entrada (§3.3) para uno de los seis pilotos. La aplicación valida el archivo, rechaza con mensajes claros lo que no cumple, calcula el canal y las variables con los datos subidos y emite con el modelo final del municipio. Muestra la misma vista principal y la advertencia de semanas incompletas (§3.4). Se entrega una plantilla CSV de ejemplo. [R] Los datos subidos se usan solo para la consulta y no se almacenan.
4. **Desempeño del modelo.** Solo resultados de prueba, global y por municipio:
   - BSS y mejora de log-loss frente a B2 y B1, con el nivel de conclusión de §8.4;
   - la tabla de casillas municipio-año (§8.2);
   - inicios avisados (k de 10) frente al canal, con el nivel de McNemar (§8.4); falsas alarmas por municipio-año; anticipación mediana;
   - el diagrama de confiabilidad y la curva tipo AMOC con el canal como punto (§7.4).
   Se reporta aunque el resultado no favorezca al modelo. En Cali se indica que no hubo inicios en prueba. Los resultados de desarrollo no se muestran como desempeño, porque se usaron para elegir.
### 9.5 Reglas de comunicación [D]
 
- **Vocabulario de D3:** «semana en zona epidémica»; «brote» solo para los inicios de brote (§2.4).
- **Observación, estimación y predicción** se distinguen siempre: casos observados, canal estimado, probabilidad predicha (propuesta, sección de transparencia).
- **Sin lenguaje causal.** Sin atribuciones por variable en el MVP. Si se muestran variables de entrada (por ejemplo, anomalías de lluvia), se presentan como contexto, no como causa.
- **Probabilidad en porcentaje entero**, sin categorías de riesgo adicionales a los estados de §9.3.
- **Advertencias permanentes:**
  1. Es un pronóstico retrospectivo con datos consolidados (§3.4).
  2. No reemplaza la vigilancia oficial ni tiene aval del INS.
  3. Es una probabilidad, no una certeza; el modelo no identifica causas.
  4. Si las semanas recientes están incompletas, el riesgo se subestima (§3.4).
### 9.6 Contrato de la API [D]
 
Se conserva la API FastAPI existente (`/municipios`, `/schema`, `/health`) y se amplía:
 
| Endpoint | Devuelve | Cálculo |
|---|---|---|
| `/predict` | La salida de §9.3 para un municipio y una semana de emisión de la base | Inferencia con el modelo final |
| `/predict/upload` | La misma salida, a partir de un CSV subido | Validación de §3.3 e inferencia |
| `/historial` | Por municipio, 2019–2025: semana de emisión y objetivo, p̂, B1, B2, estado observado, zona, estado de alerta y eventos | Tabla precalculada |
| `/desempeno` | La ficha de §9.4, vista 4 | Tabla precalculada |
 
**Campos de `/predict`:** municipio (código y nombre), semana de emisión, semana objetivo, horizonte, probabilidad, referencia B2, referencia B1, τ, estado de alerta, zona actual, central y U1e de la semana objetivo, serie del canal para el gráfico (hasta t + 8), trayectoria de las últimas 12 emisiones, versión de la receta, versión de los datos y advertencias.
 
**Parametrización** (variables de entorno documentadas en el README): horizonte (por defecto 8; solo valores con modelo entrenado), τ y k (por defecto, los congelados en la receta), URI o etapa del modelo en MLflow, versión de los datos y URL de la API en el frontend. Si τ o k difieren de los congelados, la API lo indica y la ficha de desempeño deja de aplicar.
 
**Despliegue.** Los modelos por municipio son pequeños y se cargan al iniciar; el historial y la ficha se precalculan, así que esas vistas no hacen inferencia. Si la familia elegida fuera la GRU, aplica §10, punto 10.
 
### 9.7 Fuera del MVP [D]
 
- τ ajustable por el usuario: el desempeño medido solo vale para el τ congelado.
- Mapas o los 78 municipios (D0).
- Conteos pronosticados y varios horizontes (D3, D4).
- Atribuciones por variable (por ejemplo, SHAP).
- Datos en tiempo real (§3.3).
- Cuentas de usuario.
- Cambiar a Streamlit o Power BI: la infraestructura React y FastAPI ya existe.
### 9.8 Limitaciones
 
- La usabilidad no se valida con usuarios reales; queda como trabajo futuro con el experto y con secretarías de salud.
- En el pronóstico vigente no hay verdad para comparar: el respaldo es la ficha de prueba.
- Los datos subidos solo se validan en formato y completitud, no en calidad.
## 10. Requisitos acumulados para la implementación
 
1. **Módulo del canal.** Un módulo en `src/epidemiology/` que reproduzca epiCo (central, gsd y límite superior, redondeo incluido), con pruebas. Opcional: contraste ejecutando epiCo en R.
2. **Producto nuevo.** Un producto versionado con DVC (nombre a fijar en la implementación, por ejemplo `data/processed/epidemiology/pilot_weekly_channel.parquet`) con columnas que declaren la parametrización: `epico_central`, `epico_upper`, `target_u1e` (1[C > U1e] de la semana) y `channel_zone`. No contiene `ewt` ni `excess_label`.
3. **Legado.** `pilot_weekly_labels.parquet` (SHA-256 `b3b151a5…2c7f`) queda congelado. `ewt` y `excess_label` se marcan como legado en `docs/` y `AGENTS.md`.
4. **Pruebas de aceptación:** 4.380 semanas elegibles, 945 positivas, 175 rachas y 31 inicios de brote; central > EWT y U1e ≥ central en todas las semanas.
5. **Horizonte como parámetro.** El pipeline recibe h por configuración o variable de entorno (valor por defecto 8). Esto también puntúa en la rúbrica del despliegue: «parametrización», 15 puntos.
6. **Validación del contrato de entrada** para datos subidos (§3.3), con mensajes claros al usuario.
7. **Prueba de truncamiento (sin fuga).** Para una muestra de orígenes t, todas las features (incluidas U1e_{t+h}, la central de t + h y las anomalías climáticas) se calculan con la base completa y con la base recortada en t, y deben coincidir exactamente. Es la prueba que falla si alguna feature usa una semana posterior a t. Comprobación previa [C]: U1e_{t+8} no cambia al truncar en t en ninguno de los 4.380 pares; la semana más reciente de su historia está 44–45 semanas antes de t.
8. **Líneas base.** B1, B2 y la alerta descriptiva del canal se implementan en el mismo módulo de evaluación que los modelos. Se estiman por pliegue, con suavizado de Laplace, y se reportan en todas las tablas.
9. **Registro de experimentos.** Se registran todos los candidatos, la ablación y las semillas de la GRU, no solo el modelo elegido.
10. **Despliegue.** Si la familia elegida fuera la GRU, la API necesitaría PyTorch para inferencia; hay que evaluar el tamaño y el tiempo de respuesta (rúbrica del despliegue).
11. **Constructor de cortes por año (§6.1).** Pruebas mínimas:
    - ninguna fila de entrenamiento tiene la semana objetivo en el año evaluado ni después;
    - ninguna fila de entrenamiento tiene la semana objetivo posterior a t0, la primera emisión evaluada del ensayo (D7.1);
    - las filas evaluadas son exactamente las del año;
    - el número de filas coincide con 208 por municipio en desarrollo y 156 en prueba.
12. **Arnés de búsqueda (§6.2–6.3).** Debe:
    - guardar los puntajes por candidato, año y municipio;
    - verificar que la lista de candidatos sea idéntica en los seis municipios;
    - calcular la mejora frente a B2, la configuración común y las excepciones según la regla de D9 (§8.3).
    La prueba (2023–2025) solo se ejecuta después de congelar la receta.
13. **Módulo de eventos (§7.4).** Funciones puras que, a partir de las emisiones y la serie observada, producen alertas, alertas de continuidad, inicios, emparejamiento, anticipación, falsas alarmas y censura. El modelo y el canal pasan por el mismo código. Pruebas mínimas con series sintéticas:
    - una emisión positiva aislada no produce alerta;
    - la alerta se fecha en la segunda semana de la racha;
    - un inicio 12 semanas después de t_c se empareja; uno a 13 semanas, no;
    - una alerta de continuidad no cuenta como detección ni como falsa alarma;
    - la censura excluye todo objeto que use semanas fuera del periodo.
    Con los datos reales debe reproducir 10 inicios evaluables en desarrollo y 10 en prueba (`eventos_d8.py`).
14. **Parámetros de la regla en configuración:** k = 2, W = 12 y la rejilla de τ. El τ elegido se guarda con la receta congelada y en el registro de experimentos.
15. **Orden de ejecución.** La alerta del canal bajo la regla de §7.4 se calcula después de congelar §7 (registrar el commit del contrato). Es el primer cálculo de eventos del proyecto con esta regla.
16. **Cálculo de métricas:**
    - log-loss con logaritmo natural, promediado por fila, con el mismo recorte numérico para todos los sistemas;
    - AP con `sklearn.metrics.average_precision_score`;
    - los intervalos del diagrama de confiabilidad, por cuantiles de p̂ dentro de cada periodo;
    - todas las tablas incluyen B2, B1 y, a nivel de eventos, el canal.
17. **Reglas de comparación (§8).** Funciones que:
    - obtienen los puntajes fuera de año a partir de la tabla única, sin reentrenar;
    - cuentan las victorias por casilla, por año o por inicio discrepante;
    - aplican la prueba binomial exacta de una cola;
    - asignan el nivel de conclusión.
    Pruebas mínimas:
    - los umbrales reproducen 13 de 18, 17 de 24 y la tabla de McNemar de §8.2 (`umbrales_d9.py`);
    - un empate no cuenta como victoria;
    - el puntaje fuera de año de una configuración nunca usa el año en que se puntúa.
    La tabla completa de casillas (municipio × año, con la diferencia de Brier y de log-loss) es un producto obligatorio del reporte.
18. **Artefactos precalculados (§9.4).** El pipeline de evaluación genera y versiona, junto con la receta congelada:
    - la tabla del historial fuera de muestra (2019–2025), con p̂, B1, B2, estado observado, zona, estado de alerta y eventos;
    - la ficha de desempeño de prueba, global y por municipio.
19. **API (§9.6).** Pruebas mínimas:
    - la respuesta de `/predict` cumple el esquema;
    - `/historial` coincide fila a fila con las predicciones guardadas del origen móvil y nunca contiene predicciones del modelo final para t ≤ 2025-10-27;
    - el estado de alerta coincide con el del módulo de eventos (punto 13) sobre la misma serie;
    - `/predict/upload` rechaza con mensajes claros los archivos que violan §3.3 y, con datos idénticos a la base, reproduce la salida de `/predict`.
20. **Frontend (§9.3–9.5):** las cuatro vistas, las referencias B1 y B2, los estados de alerta y las advertencias permanentes. El esquema JSON de `/predict` se fija antes de entrenar, para que el frontend avance con valores simulados.
21. **README (rúbrica: 15 puntos):** ejemplos de entrada y salida (plantilla CSV y respuesta JSON), variables de entorno y gestión del modelo en MLflow.
22. **Análisis diagnóstico integrado (§8.8).** Los seis insumos de §8.8 se generan para desarrollo y para prueba con el mismo código y se versionan con la evidencia. La clasificación de régimen tiene pruebas con series sintéticas (episodio de 3 semanas que cruza el cambio de año, año post-epidémico, año con racha de 2 semanas). En desarrollo, las cifras se comparan con un cálculo independiente antes de congelar sus literales.
