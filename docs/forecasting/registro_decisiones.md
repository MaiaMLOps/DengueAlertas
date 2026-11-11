# 01 — Registro de decisiones
 
Cada entrada registra la fecha, la decisión, las alternativas consideradas, la justificación y la evidencia. Las decisiones del contrato predictivo (D0–D10) se describen en `02_contrato_predictivo.md`. Una decisión registrada solo se reabre con una nueva entrada que explique el motivo.
 
---
 
## D1 — Conteo, calendario y EWT base (2026-10-21, contrato epidemiológico del equipo)
 
**Decisión:**
- Conteo: `confirmed_records_screening` (evento 210, `confirmados = 1`, municipio de ocurrencia, `INI_SIN`).
- Calendario: semanas W-SUN.
- EWT = `exp(mean(log(C + 0.001))) − 0.001`, con 5 años previos y años epidémicos incluidos.
- `excess_label = 1[C > EWT]`.
- 2007–2011 se usa solo para inicialización.
**Estado:** se conserva sin cambios en el repositorio. A partir de D2 deja de ser la variable objetivo. **Enmendado por D1.1 (2026-11-09):** la línea central del canal pasa a ser la central de epiCo; `ewt` y `excess_label` quedan como legado, sin uso en el modelado ni en el producto.
**Evidencia:** `docs/epidemiology_labels.md` y `reports/epidemiology/evidence/labels_20261021/` en el repositorio.
 
## D0 — Alcance territorial (2026-11-08)
 
**Decisión:** se mantienen los seis pilotos (Cúcuta 54001, Barranquilla 08001, Neiva 41001, Villavicencio 50001, Cali 76001, Soledad 08758), con modelos independientes por municipio.
**Alternativas descartadas:**
- Reducir a dos o tres pilotos.
- Reemplazarlos por municipios con series «más variables».
**Justificación:**
- Carlos pidió pocos municipios, no un número exacto.
- El criterio de selección (continuidad de la serie) es coherente con el dominio de aplicación del canal endémico: el protocolo INS lo recomienda para transmisión media, alta y muy alta, y Umaña et al. (2026) reportan mejor desempeño del canal a mayor transmisión.
- La heterogeneidad entre los seis es un resultado en sí mismo.
- Elegir municipios según cómo se comportan sus series introduciría un sesgo de selección.
- Las rachas largas del avance se debían al umbral, no a los municipios: con U1e la duración media de las rachas es 5,4 semanas.
**Limitación a declarar:** los pilotos se seleccionaron con conocimiento de 2007–2025; los resultados aplican a municipios con vigilancia continua.
**Opcional (semana 7):** validación externa en 2–3 municipios de los 78, elegidos por una regla fijada de antemano.
 
## D2 — Umbral de exceso (2026-11-08)
 
**Decisión:** el umbral de exceso es **U1e**, el límite superior del canal endémico tal como lo implementa epiCo (`epiverse-trace/epiCo`, commit `39a45ef`, `endemic_channel`, `method = "geometric"`, `shift = 1`), con n = 5 años previos:
 
    central  = round(exp(mean(log(x + 1))) − 1, 5)
    gsd      = round(sd(log(x + 1)), 5)            (ddof = 1)
    interval = |qt(0,025; n − 1)| · gsd / √n
    U1e      = exp(log(central) + interval) = central · e^interval     (0 si central = 0)
 
Una semana está en **zona epidémica** si C > U1e (comparación estricta). La historia y la elegibilidad siguen el contrato D1.
 
**Significado:** U1e es el límite entre las zonas de alerta y de epidemia del canal (Bortman, 1999; Umaña et al., 2026). Es el extremo superior del intervalo de confianza del 95 % de la media geométrica, no un intervalo de predicción: en un proceso estable sin brotes se supera ~16 % de las semanas.
 
**Evidencia:** `claude/estudio_umbrales/04_estudio_umbrales.md` (ejecutado por Fable según `claude/specs/estudio_umbrales.md` v1.0, con criterios fijados antes de ver los resultados). Cifras clave verificadas de forma independiente en el chat de metodología:
- prevalencia 21,6 % (17,0–27,9 % por municipio);
- 175 rachas, de las cuales el 50,3 % dura una semana;
- 38 años casi monoclase de 84;
- al menos 8 entradas por municipio en 2017–2025;
- tasa de cambio de estado P(A_{t+4} ≠ A_{t−1}) = 0,117.
**Desviación declarada de la regla preestablecida:** la regla lexicográfica de la especificación (§5) ordenaba U3a (1.º), U1e (2.º) y U1 (3.º). Se adopta U1e por tres motivos:
 
1. **Validez.** El estudio reveló un artefacto que los criterios no contemplaban. Con ε = 0,001, un cero en la historia entra como ln(0,001) = −6,9 e infla la desviación logarítmica: la mediana de s pasa de 1,12 a 4,25. Eso vuelve inalcanzables a U1 y U3a en el 13 % de las semanas elegibles (29 % en Soledad): sus prevalencias en esas semanas caen a 0,5 % y 2,8 %, frente a 23,2 % de U1e. El artefacto también contamina el conteo de años monoclase que decidió el ranking.
2. **Respaldo.** U1e es la implementación verificada del paquete del grupo de los autores de Umaña et al. (BIOMAC). U3a es una construcción declarada por el equipo: la familia «media + k·DE» se documenta en escala original, no logarítmica.
3. **Costo pequeño.** U1e pasa los tres criterios excluyentes y queda a 3 años monoclase de U3a (3,6 puntos porcentuales).
**Alternativas descartadas:**
- U0, por prevalencia de 65,5 %.
- U2 y U3b, por prevalencias de 0,8 % y 5,3 % y porque Cúcuta queda sin positivos.
- U1, por el artefacto de ε y por no ser la fórmula de epiCo.
- U3a: queda como **sensibilidad** en los resultados.
**Implicaciones y pendientes:**
- U1e se implementa como **extensión** de `src/epidemiology`, con columnas nuevas, sin modificar `ewt` ni `excess_label`. Sus pruebas deben reproducir epiCo, incluido el redondeo; idealmente también un contraste ejecutando epiCo en R.
- Conviven dos desplazamientos: ε = 0,001 en el EWT de D1 y 1 en U1e. La etiqueta no se ve afectada, pero queda pendiente decidir qué línea central muestra el producto: el EWT de D1 o el `central` de epiCo. Fable verificó que U1e nunca queda por debajo del EWT. **Resuelto en D1.1 (2026-11-09):** se usa la central de epiCo.
- Sensibilidad a la ventana (U1e: 11 %, 20 % y 27 % con n = 3, 5 y 7) y U3a se reportan como análisis de sensibilidad.
- El respaldo de Umaña et al. a ε = 0,001 se refiere a la tendencia central y a una métrica interna del canal (detección y razón de alertas, sin verdad externa). Además, ese estudio no evaluó la combinación ventana de 5 años + ε. Hay que citarlo con esa precisión.
**Alternativas adicionales consideradas y descartadas (2026-11-09):**
- **Umbral intermedio entre el EWT y U1e** (por ejemplo, central · e^(I/2)), para tener más positivos. Se descarta por cuatro razones:
  - Los positivos no escasean: 945 semanas, unas 157 por municipio.
  - Lo escaso son los eventos, y eso se aborda con la evaluación, no con el target.
  - Una línea intermedia no tiene respaldo institucional ni en epiCo y debilita la interpretación de «zona epidémica».
  - Bajar el umbral para mejorar el aprendizaje sería ajustar el target a conveniencia.
  U1e ya es moderado: en un proceso estable se supera ~16 % de las semanas, frente a 50 % del EWT y 2,5 % de un límite de predicción. La sensibilidad operativa se ajusta con el umbral de decisión τ sobre la probabilidad (D8/D10), no cambiando la definición de exceso.
- **Cambiar la ventana retrospectiva** (n = 3 o 7). Con U1e, en los años comunes 2014–2025, la prevalencia es 11,4 %, 19,8 % y 27,0 %, y los años monoclase son 43, 35 y 28. La razón es que k = t*/√n vale 2,48, 1,24 y 0,92. Se mantiene n = 5 porque:
  - es el contrato aprobado y la mejor ventana entre las evaluadas por Umaña et al. (5–9 años);
  - sigue mejor la dinámica reciente;
  - conserva los años etiquetados 2012–2013;
  - elegir n después de ver la sensibilidad sería post hoc.
  **[I]** Parte del aumento de prevalencia con n = 7 refleja una referencia más rezagada frente a la tendencia de los registros. Las ventanas de 3 y 7 años se reportan como sensibilidad. Si el equipo quisiera más positivos, n = 7 (línea base institucional) es la única alternativa defendible y debe adoptarse antes de modelar, no después.
**Preguntas para Carlos:**
1. ¿Le parece adecuado el límite superior de epiCo (con desplazamiento 1) como umbral de exceso?
2. Su artículo (Umaña et al., 2026) recomienda desplazar 0,001 en lugar de una unidad. ¿Aplicaron ese desplazamiento también a la desviación geométrica del límite superior? ¿Observaron el inflado de la desviación con ceros en la historia?
3. ¿Tiene disponible la regla de emparejamiento entre alertas y brotes (Tabla S1 del apéndice)?
4. (Añadida el 2026-11-09, ver D1.1.) ¿Le parece aceptable usar como línea central la de epiCo (desplazamiento 1) en lugar de la recomendación de 0,001, por coherencia con el límite superior?
5. (Añadida el 2026-11-10, ver D8.) ¿Es razonable una ventana de 12 semanas entre la alerta y el inicio del brote para que una secretaría de salud pueda actuar? ¿Una alerta seguida solo de un exceso corto (1 o 2 semanas en zona epidémica) debería contarse como falsa alarma?
## D1.1 — Parametrización única del canal (2026-11-09; enmienda a D1)
 
**Decisión:** todas las cantidades del canal que usan el modelado y el producto (línea central, límite superior y zonas) se calculan con la parametrización de epiCo: desplazamiento 1 y n = 5. La línea central del canal es `central = round(exp(mean(log(x + 1))) − 1, 5)`, no el EWT de D1. Fórmulas y zonas en `02` §1.
 
**Motivo: coherencia, no superioridad del desplazamiento 1.**
- U1e = central · e^interval se construye sobre la central de epiCo. Usar el EWT como línea central junto con U1e mezcla dos desplazamientos y rompe la geometría del canal: la zona entre ambas líneas deja de ser la zona de alerta de epiCo.
- Solo hay dos canales coherentes: todo con ε = 0,001 (su límite superior es U1, descartado en D2 por el artefacto de ε en la dispersión) o todo con desplazamiento 1 (epiCo). Solo el segundo sobrevive.
**Lo que dice la literatura [H].** Umaña et al. (2026) concluyen que desplazar 0,001 en lugar de una unidad mejora el desempeño del canal en todos los niveles de transmisión, en promedio un 23,07 % (P < 0,01). Matices:
- su métrica es interna al canal (detección y razón de alertas, con brotes definidos por el propio canal);
- parte de su argumento es evitar umbrales nulos en municipios de baja transmisión, lo que en nuestros pilotos es marginal (0,2 % de las positivas con C ≤ 2, según el estudio de umbrales);
- no está verificado si aplicaron 0,001 también a la dispersión del límite superior (pregunta 2 para Carlos).
Por eso, en la línea central nos apartamos de su recomendación, y hay que declararlo.
 
**Magnitud del cambio [C].**
- **Monotonía:** G(c) = exp(mean(log(x + c))) − c no decrece con c, porque G′(c) = GM(x + c)/HM(x + c) − 1 ≥ 0 (la media armónica no supera a la geométrica). Por tanto, la central de epiCo nunca queda por debajo del EWT. En los datos queda por encima en las 4.380 semanas elegibles.
- **Diferencia:** la razón mediana central/EWT es 1,036 en semanas con historia sin ceros (3.811) y 3,834 con algún cero (569).
- **Etiquetas:** P(C > central) = 62,1 % frente a P(C > EWT) = 65,5 %; 1[C > línea] cambia según la línea en el 3,4 % de las semanas.
**Implementación.** No se sobrescriben `ewt` ni `excess_label`:
- la auditoría, el estudio de umbrales y las cifras de D2 se verificaron contra el archivo actual (SHA-256 `b3b151a5…2c7f`);
- las 174 pruebas definen `ewt` con ε;
- cambiar el contenido de una columna sin cambiarle el nombre altera su significado en silencio.
Se crean un módulo y un producto nuevos con columnas de nombre explícito (`02` §10). Las columnas con ε quedan congeladas como legado, marcadas como obsoletas y ausentes del producto de modelado.
 
**Declarar en el artículo:** la desviación respecto de la recomendación de Umaña et al. para la tendencia central, con la magnitud anterior.
**Evidencia:** `claude/verificacion/descriptivos_d3.py`, sección 4.
 
## D3 — Variable objetivo (2026-11-09)
 
**Decisión:** y_{t,h} = 1[C_{t+h} > U1e_{t+h}], semana individual en zona epidémica, con clasificación binaria y salida probabilística. Definición, propiedades y cifras en `02` §2.
 
**Restricción de alcance (decisión del equipo):**
- Solo clasificación binaria en el MVP, sin modelos de conteo, de regresión ni multiclase.
- Carlos pidió un target simple y no contemplar regresión. La multiclase no estaba prohibida; se descarta por conveniencia (ver abajo).
**Principio de evaluación acordado:**
- Eventos y transiciones son objetos de evaluación, no target.
- Se separan dos capas: estimación (reglas de puntuación propias y métricas de clasificación) y decisión (eventos).
- Hay tres niveles de evaluación: semana, transición y evento.
- Los eventos solo ajustan la capa de decisión, en validación.
Detalle en `02` §2.4; se concreta en D8 y D10.
 
**Alternativas descartadas [C]:**
- **Ventana** («al menos una semana en zona epidémica en las h siguientes»). Prevalencia ~30 % (21–42 % por municipio). Pierde el «cuándo», que es lo que se quiere medir, y acumula las fluctuaciones aisladas de varias semanas.
- **Inicio de brote como target.** 31 inicios en 14 años, entre 3 y 7 por municipio. Cali tiene 3 (2013, 2015 y 2019, ninguno desde entonces) y Cúcuta 3 (2021, 2024 y 2025). Además, la etiqueta exige dos semanas posteriores. Con modelos por municipio es inviable como target; pasa a ser el evento de evaluación.
- **Semana en episodio sostenido** (racha de ≥ 3 semanas). 805 positivas (18,4 %). Quita 140 positivas aisladas o cortas a costa de:
  - mirar hasta dos semanas después de la semana objetivo, lo que exige un hueco de validación de h + 2;
  - usar una definición que no es estándar en el canal;
  - acercarse a la lógica del masking.
  El ruido de semanas aisladas se trata en la regla de alerta, no en el target.
- **Multiclase por zona** (seguridad, alerta, epidemia; 37,9 %, 40,5 % y 21,6 % con la central de epiCo).
  - *La frontera seguridad/alerta reproduce el target del avance:* prevalencia de 62,1 %, cambio de estado a 5 semanas de 14,2 % y 33 municipio-años casi monoclase de 84, frente a 65,5 %, 15,0 % y 32 de U0. Copiar la última zona acierta el 74,9 %.
  - *La decisión de alertar solo depende de P(epidemia).* «Alerta» ocuparía el 40,5 % de las semanas, y la alerta descriptiva del canal precede un inicio solo el 12 % de las veces.
  - *Costo:* métricas ordinales, líneas base multiclase y seis tipos de transición, con cuatro días de implementación y un artículo de 12 páginas.
  - *Extensión posible:* un segundo clasificador binario P(C > central), sin cambiar el contrato.
  - *Uso de las zonas:* la zona observada pasa a D6 (feature) y a D10 (visualización). Nota de fuga: la matriz de transición (236 de 253 entradas a la zona epidémica vienen de la zona de alerta) se calculó con todos los años. El feature debe justificarse por el diseño del canal, no por esa cifra.
- **Modelos de conteo o regresión** (por ejemplo, derivar P(C_{t+h} > U1e_{t+h}) de una distribución predictiva). Excluidos por la restricción de alcance.
- **Entrenar o evaluar solo filas con estado observado 0, para modelar entradas.** No es un target distinto: es una estratificación legítima de la evaluación (D8), porque el estado al emitir forma parte de la información disponible. Se diferencia del masking del avance, que condicionaba en información futura.
**Evidencia:** `claude/verificacion/descriptivos_d3.py`, secciones 1, 2, 3 y 5.
 
## D4 — Emisión, información disponible y horizontes (2026-11-09)
 
**Decisión:** detalle en `02` §3.
- **Convención:** t es la última semana con dato completo y h la distancia desde ella.
- **Información:** en t se conoce todo lo que la base contiene hasta t, más la central y U1e de la semana objetivo y su calendario.
- **Operación:** el MVP no opera en tiempo real; hay un contrato de entrada para los datos que suba el usuario.
- **Horizonte:** h = 8 es el horizonte principal y único del sistema.
- **Sensibilidad:** h ∈ {2, 4, 12}, con prioridad baja. Si se ejecuta, debe responder tres preguntas declaradas: horizonte de habilidad, aporte del clima frente a h y heterogeneidad entre municipios.
**Justificación del horizonte.** Se eligió por relevancia, dentro de lo viable.
- **Relevancia:** en Singapur, el tiempo de control fue de 1 a 3 meses, con mediana de 2 (Hii et al., 2012). La alerta del canal ya anticipa unas 3 semanas en mediana.
- **Viabilidad [C]:** a h = 8, la kappa entre el estado en t y en t + 8 es 0,60 (0,51–0,65 por municipio) y hay 291 entradas.
- **Clima:** los rezagos clima–dengue reportados, de semanas a meses, siguen siendo aprovechables a esa distancia.
**Alternativas descartadas:**
- **h = 4 como horizonte principal** (propuesta y avance). Su única justificación propia era superar en una semana la anticipación mediana de la alerta del canal. Preferirlo por su mayor probabilidad de buen desempeño violaría el principio rector. Queda como sensibilidad.
- **h = 6 o h = 10.** No tienen un anclaje propio que los distinga.
- **h = 12 como principal.** Llega al tiempo máximo de control reportado, pero el presente informa bastante menos (kappa 0,49). Queda como sensibilidad.
- **Varios horizontes operativos** (por ejemplo, 1 a 8). Multiplican la selección de modelos, la regla de alerta, la evaluación por eventos y la interfaz, lo que no cabe en el MVP. La evaluación por eventos no exige horizontes contiguos, porque una alerta a un solo h puede emparejarse con un inicio mediante una ventana; la contigüidad solo aporta resolución.
- **Solo h = 8, sin sensibilidad.** Se perdería la evidencia empírica sobre relevancia frente a viabilidad. Se conserva la sensibilidad con prioridad baja.
- **Diseño con retraso real de los conteos o cortes distintos por fuente.** El MVP no opera en tiempo real. La latencia queda como limitación declarada, con anticipación operativa h − d.
**Desviación declarada:** respecto de la propuesta (h = 1, 2 y 4) y del avance (h = 4, que en la convención nueva equivale a 5).
**Evidencia:** `claude/verificacion/descriptivos_d3.py`, sección 6; referencias en `02` §3.8.
 
## D5 — Líneas base y modelos (2026-11-09)
 
**Decisión:** detalle en `02` §4.
- **Líneas base:**
  - B2, estado del canal: P(y | zona observada en t). Es la principal.
  - B1, climatología estacional: secundaria.
  - La alerta descriptiva del canal, a nivel de eventos.
- **Candidatos:** regresión logística y LightGBM, más una GRU opcional con intención de ejecutarse. Todos usan todas las features de D6.
- **Selección:** una familia común elegida por log-loss de validación. Una excepción por municipio solo con evidencia clara.
- **Ablación:** una sola, del bloque climático, con la familia elegida.
**Justificación:**
- *B2 como principal:* es la referencia más transparente, porque mide al modelo contra lo que el epidemiólogo ya ve en el canal.
- *B1:* a h = 8 es la única forma de distinguir «anticipa» de «aprendió la estación».
- *La ablación:* responde el objetivo específico 3 de la propuesta, que prometía cuantificar el aporte incremental de las covariables climáticas.
- *La familia común:* evita el sesgo del ganador al repetir elecciones ruidosas por municipio.
**Alternativas descartadas:**
- **Persistencia binaria (B3).** B2 la contiene. Una predicción 0/1 no es probabilística: su log-loss es infinito cuando falla. A nivel de eventos casi no puede anticipar inicios, por construcción.
- **Solo B2.** No separa anticipación de estacionalidad.
- **Climatología global constante como línea base.** Solo da contexto (la prevalencia π).
- **Diseño factorial 2 × 2** (regresión logística y LightGBM, con y sin clima), propuesto en el chat de metodología. Se reemplaza por candidatos con todas las features y una sola ablación: responde la misma pregunta con menos modelos y métricas.
- **Ablación en todas las familias.** Multiplica las métricas sin cambiar la respuesta al objetivo 3.
- **Elegir la familia por municipio.** Con k candidatas de igual desempeño, la ganadora queda sesgada a favor en E[máx de k normales]·σ: 0,56σ con dos, 0,85σ con tres y 1,16σ con cinco. Por municipio, σ es grande y se harían seis elecciones ruidosas.
- **Familias adicionales.**
  - Bosque aleatorio, XGBoost y CatBoost son redundantes con LightGBM.
  - Un GAM queda como trabajo futuro, por si la brecha entre la regresión logística y LightGBM resulta grande y conviene interpretar la no linealidad.
  - Las arquitecturas mayores son desproporcionadas para unas 730 semanas por municipio.
  Cada candidata adicional aumenta la multiplicidad de la selección y la carga de evaluación.
- **De la propuesta:** DLNM+INLA (regresión de conteos, excluida en D3) y ST-GNN (combina municipios, contra D0). Desviación declarada.
## D6 — Features (2026-11-09)
 
**Decisión:** detalle en `02` §5. Son 15 variables:
- `zona_t`, `posicion_4s` y `crecimiento_4s`: estado y dinámica frente al canal;
- `cambio_umbral`, `sem_sin` y `sem_cos`: lo que se conoce de antemano sobre la semana objetivo;
- nueve anomalías climáticas: temperatura, precipitación y humedad en tres bloques mensuales que terminan en t.
La ventana es de 12 semanas y todos los resúmenes usan w = 4, fijado a priori. La GRU recibe la secuencia de las mismas fuentes. No hay variables demográficas.
 
**Justificación:**
- **Resúmenes frente a rezagos.** Doce rezagos de cuatro series son unas 48 columnas casi colineales para unas 700 filas: colinealidad en la regresión logística y cortes espurios en LightGBM (auditoría, C4). Los resúmenes codifican nivel, tendencia y posición frente al umbral. La vista secuencial la aporta la GRU.
- **Clima como anomalía.** El clima crudo repite la estacionalidad que ya aportan `sem_sin` y `sem_cos`. El target también es una anomalía frente a los cinco años anteriores, así que la escala es coherente. Al calcularse solo con el pasado, no puede filtrar información.
- **Bloques mensuales.** El efecto del clima es acumulativo. Tres bloques cubren rezagos de 8 a 19 semanas respecto a la semana objetivo, el rango que reporta la literatura revisada en D4. Son 9 columnas en lugar de 36.
- **Covariables futuras conocidas.** `cambio_umbral` usa U1e_{t+h}, que se calcula con años anteriores al de t + h. Comprobación [C]: en los 4.380 pares, la semana más reciente de su historia está 44–45 semanas antes de t, y truncar la base en t no cambia su valor.
**Alternativas descartadas:**
- **Rezagos explícitos semanales** (unas 48 columnas): colinealidad, cortes espurios e importancias arbitrarias entre rezagos vecinos.
- **Clima crudo:** duplica la estacionalidad. **Doce rezagos climáticos semanales:** ruido y 36 columnas.
- **Semanas en zona epidémica en las últimas 8** y **distancia al umbral objetivo:** redundantes con `posicion_4s`, `zona_t` y `cambio_umbral`. Además, la primera introducía un segundo parámetro de ventana arbitrario.
- **Cambio de la central entre t y t + h:** duplica `cambio_umbral`.
- **Ventanas ajustadas con datos:** añaden grados de libertad al investigador. Quedan como trabajo futuro, y solo dentro de la validación.
- **Población, densidad y área.** Con modelos por municipio, población y densidad solo varían por año y funcionan como reloj (correlación con el año de 0,957 a 0,995; auditoría, I11), y el área es constante. El efecto demográfico no se puede separar del tiempo sin comparar municipios, lo que D0 excluye. **Desviación declarada** respecto al objetivo específico 3: solo se mide el aporte climático.
- **Población como denominador de tasas:** el target usa conteos (D1).
**Limitación declarada:** las ventanas son elecciones a priori y otras combinaciones podrían rendir más.
**Evidencia:** las filas disponibles (711 por municipio desde t = 2012-03-19) y la prueba de truncamiento se calcularon en el chat de metodología. La prueba de truncamiento pasa a ser un requisito de implementación (`02` §10, punto 7).
 
## D7 — Validación y selección (2026-11-10)
 
**Decisión:** detalle en `02` §6.
- **Origen móvil con reentrenamiento anual,** con corte por el año de la semana objetivo.
- **Desarrollo:** cuatro ensayos (2019, 2020, 2021 y 2022), para elegir hiperparámetros, familia y regla de decisión.
- **Prueba:** 2023 a 2025, con la receta congelada, mirada una sola vez.
- **Modelo final:** entrenado con 2012–2025.
- **Hiperparámetros:** esquema híbrido. Mismos candidatos en todos los municipios, configuración común por mejora promedio frente a B2, y excepción por municipio solo si gana de forma consistente año a año (regla en D9).
- **Búsqueda:** rejilla para la regresión logística, búsqueda aleatoria con presupuesto fijo para LightGBM y arquitecturas fijas para la GRU.
**Justificación:**
- *Coherencia con la propuesta.* La propuesta comprometía una evaluación «rolling-origin, con reentrenamiento».
- *Realismo.* El sistema operaría reentrenándose cada año con todo lo conocido.
- *Hueco automático.* El corte por la fecha de la semana objetivo elimina la fuga del avance (I9).
- *Esquema híbrido.* El sesgo del ganador al elegir entre k candidatos ruidosos es E[máx]·σ: 1,97σ con k = 25. Si se promedia sobre seis municipios, baja a unos 0,80σ bajo independencia. La evidencia por municipio es desigual: en desarrollo, Neiva solo tiene positivas en 2019 y Cali 20 en total. Con la misma lista de candidatos, la excepción por municipio se puede comprobar en lugar de afirmarse.
**Alternativas descartadas [C]:**
- **Una sola partición fija** (tipo 70/15/15). Deja un solo año para decidir, y los modelos de los años tardíos no ven la historia reciente.
- **`KFold` y `TimeSeriesSplit`.** El primero mezcla el futuro con el pasado. El segundo corta por número de filas, no por año ni por la fecha de la semana objetivo.
- **Desarrollo 2018–2022.** El ensayo de 2018 entrena con el régimen tranquilo de 2012–2017 (Cúcuta solo tiene 19 positivas ahí), y el papel de «año tranquilo» ya lo cumple 2020 (prevalencia de 7,7 %). Aportaba 14 entradas y un quinto año para medir la variación.
- **Desarrollo 2020–2022** (planteado por Jesús, para no restar historia reciente). Con origen móvil, 2019 entra igualmente en el entrenamiento de todos los ensayos posteriores. Quitarlo de la validación no suma entrenamiento, pero deja a Neiva sin ninguna positiva en desarrollo y a Villavicencio con 10. En total quedarían 200 positivas, 5 inicios y 30 entradas, frente a 319, 11 y 45.
- **Desarrollo 2018–2021 con prueba 2022–2025.** Desarrollo más débil: 247 positivas y 9 inicios.
- **Prueba solo en 2024–2025.** Solo 6 inicios, y dos municipios sin ninguno.
- **Prueba sin reentrenamiento** (origen fijo). Evaluaría un sistema desactualizado que nadie operaría así.
- **Hiperparámetros elegidos de forma independiente en cada municipio.** Seis elecciones ruidosas, sin forma de comprobar si un municipio necesita realmente otra configuración.
- **Configuración común sin excepciones.** Ignora a los municipios que sí tienen evidencia para personalizar.
- **Búsqueda bayesiana o adaptativa.** Genera candidatos distintos por municipio y sobreajusta la selección cuando la validación es ruidosa.
- **Lo hecho en el avance:**
  - hiperparámetros fijos sin validar;
  - el «mejor ensamble» elegido por municipio mirando el mismo pliegue que se reporta;
  - métricas de selección no propias (AUC-PR y F1).
**Limitaciones declaradas:**
- 10 inicios en la prueba; Cali sin inicios.
- Prevalencia de 25,6 % en desarrollo frente a 34,4 % en prueba.
- Cambio de régimen después de 2018.
- Pocos años para medir la variación entre ellos.
**Evidencia:** los conteos de eventos por año y municipio de cada partición candidata, y los valores de E[máx de k normales], se calcularon en el chat de metodología. Son descriptivos de la etiqueta: no se entrenó ningún modelo.
 
## D8 — Métricas y regla de decisión (2026-11-10)
 
**Decisión:** detalle en `02` §7.
- **Dos capas con métricas separadas.**
  - *Estimación:* la mejora de log-loss frente a B2 decide hiperparámetros y familia. El BSS frente a B2 es la métrica principal de reporte. También se reportan el BSS frente a B1, el Brier con su descomposición, la AP con π como referencia, la calibración (diagrama y calibración en grande por año) y las mismas métricas en las transiciones.
  - *Decisión:* una regla de alerta fija y un τ común elegido con eventos en desarrollo.
- **Clasificación en τ, solo reporte:** sensibilidad, especificidad, VPP, F1 (con su referencia trivial 2π/(1 + π)) y MCC.
- **Regla de alerta.** Adopta las definiciones de Umaña et al. (2026) con cuatro precisiones:
  - la alerta se fecha en la semana de confirmación;
  - la ventana de emparejamiento es de 12 semanas;
  - las alertas de continuidad se cuentan aparte;
  - las falsas alarmas se miden como tasa por municipio-año.
  El canal se evalúa con la misma regla.
- **τ:** máxima capacidad de detección de inicios con falsas alarmas por municipio-año ≤ las del canal; rejilla de 0,05; empate al τ mayor.
**Origen de la regla [H].**
- Umaña et al. definen alerta, inicio de brote, ventana de oportunidad, capacidad de detección y razón de alertas para describir el canal. Aclaran que no lo evaluaron como herramienta de pronóstico.
- La propuesta del grupo trasladó la alerta al modelo, con el EWT, y añadió Δw.
- El avance no implementó nada de esto (auditoría, I12 y M15).
**Justificación:**
- *Log-loss para decidir.* Es una regla de puntuación estrictamente propia y la misma pérdida del entrenamiento.
- *Soporte de la regla en la literatura.*
  - k = 2: Umaña et al., y el estudio multicéntrico de Asia y Latinoamérica (2 señales = 1 periodo).
  - W = 12: el rezago de 1 a 12 semanas del mismo estudio.
  - La estructura probabilidad → umbral → alarma evaluada contra brotes: EWARS y EWARS-csd en Colombia (Cardenas et al., 2022; Schlesinger et al., 2024).
  - Las falsas alarmas por unidad de tiempo frente a la oportunidad: curvas AMOC (Jiang, Cooper y Neill, 2009).
- *Fecha en la confirmación.* Una racha de dos semanas solo se conoce al final de la segunda. Fechar en la primera regala una semana de anticipación que no existió.
- *Alertas de continuidad aparte.* Sin separarlas, el modelo acumula falsas alarmas por acertar durante una epidemia, algo que al canal no le puede pasar. Se condiciona en el estado observado en t_c, que es información disponible; el masking del avance condicionaba en el futuro.
- *Restricción en lugar del compuesto.* Responde «con un costo no mayor que la práctica actual, ¿detecta más y antes?» sin fijar un peso α.
**Alternativas descartadas:**
- **Elegir con AUC-PR o F1** (como en el avance). No son reglas de puntuación propias; F1 depende de un umbral y mezcla las dos capas. Se reportan.
- **Brier como métrica de selección.** También es propia; se prefirió el log-loss por coherencia con el entrenamiento. El Brier queda como reporte principal (BSS).
- **Fechar la alerta en la primera semana de la racha** (convención de Umaña et al.). Supone una semana de información no disponible.
- **Emparejar sin ventana, o con ventana abierta.** «Inicio avisado» queda indefinido y premia alertas permanentes.
- **Ajustar k y W en validación.** Con 10 inicios evaluables, cada parámetro ajustado es otra oportunidad de ajustar ruido.
- **Desempeño compuesto (α = 0,5) como criterio.** El peso es un juicio de valor; se reporta por comparabilidad.
- **τ por municipio.** Pocos eventos: Cali no tiene inicios en prueba y Neiva solo tiene positivas de desarrollo en 2019.
- **τ por una métrica semanal** (máxima F1 o índice de Youden). Optimiza la emisión aislada, no la pregunta operativa por eventos.
- **Contar las alertas de continuidad como falsas alarmas.** Sesga la comparación contra el modelo.
- **Falsa alarma laxa** (sin ninguna semana epidémica en la ventana) como definición principal. Queda como desglose informativo; la estricta es coherente con Umaña et al.
- **Usar semanas de otro periodo en la evaluación por eventos.** En desarrollo implicaría mirar etiquetas de prueba; se censura en los bordes.
**Limitaciones declaradas:**
- 10 inicios evaluables por periodo, 5 de los de desarrollo en 2019.
- Cambio de prevalencia entre periodos.
- Definición estricta de falsa alarma.
- Alertas repetidas contadas por separado.
- La regla exacta de Umaña et al. no está verificada (pregunta 3 para Carlos).
- Algunas cifras descriptivas de la alerta del canal ya se vieron con otras reglas (estudio de umbrales y D3). k y W se tomaron de la literatura, no de esas cifras.
**Evidencia:** `claude/verificacion/eventos_d8.py` (inicios evaluables: 10 de 11 en desarrollo y 10 de 10 en prueba); referencias en `02` §7.9. Pregunta 5 para Carlos añadida en D2.
 
## D9 — Incertidumbre y reglas de comparación (2026-11-10)
 
**Decisión:** detalle en `02` §8. Una sola idea: contar victorias pareadas y compararlas con el azar (moneda con p = 0,5), con una prueba binomial exacta de una cola al 5 %.
- **Desarrollo,** con puntajes fuera de año (cada configuración se elige sin el año en que se puntúa):
  - excepción de hiperparámetros o de familia por municipio si gana en 4 de 4 años;
  - una familia más compleja reemplaza a la más simple si su mejora promedio es mayor y gana en al menos 17 de 24 casillas municipio-año.
  La victoria se mide con log-loss.
- **Prueba:** tres niveles de conclusión frente a B2, frente a B1 y para la ablación del clima. La victoria se mide con Brier, con el conteo con log-loss como contraste.
  - *Mejora consistente:* métrica global favorable y al menos 13 de 18 casillas.
  - *Tendencia favorable:* métrica global favorable y 10 a 12 casillas.
  - *Sin mejora:* cualquier otro caso.
- **Eventos:** McNemar exacto de una cola sobre los inicios discrepantes entre el modelo y el canal, con los mismos tres niveles. Δw se reporta inicio por inicio, sin prueba.
**Origen [H]:** la auditoría del avance (I9e) recomendaba bootstrap por bloques anuales, McNemar en decisiones pareadas y resultados por año. La propuesta prometía McNemar para detecciones y falsas alarmas, y bootstrap estratificado por municipio. D9 mantiene el propósito y simplifica el método.
 
**Justificación:**
- *Necesidad.* Sin estas reglas, D5 («la más simple si la diferencia está dentro de la incertidumbre») y D7 («excepción si gana de forma consistente») no son ejecutables. Además, el avance reportó diferencias sin ninguna medida de incertidumbre (I9e).
- *Conteo en lugar de bootstrap.* Con 18 casillas en prueba, el bootstrap supone lo mismo que el conteo (casillas independientes), es más difícil de explicar y no aporta mucho más que mirar la tabla. El conteo no simula datos, no reordena nada y se verifica con la tabla publicada.
- *Puntajes fuera de año.* Elegir y juzgar con los mismos años favorece siempre al elegido (sesgo del ganador). Además, sin esta corrección LightGBM, con ~25 candidatos, quedaría favorecido frente a la regresión logística, con ~6. Se obtienen de la tabla única sin entrenar nada adicional.
- *Una cola.* Todas las preguntas son direccionales y se declaran antes. Ninguna decisión depende de que una opción sea significativamente peor.
- *Tres niveles en prueba.* Evitan la trampa binaria: un buen modelo que no alcanza el umbral con 3 años queda como tendencia favorable, no como fracaso. La métrica global aporta la magnitud y el conteo, la consistencia.
**Alternativas descartadas:**
- **Bootstrap por bloques municipio-año,** con intervalos de confianza. Ver la justificación anterior.
- **Bootstrap por años calendario con los seis municipios juntos** (la recomendación literal de la auditoría). Hay 3 bloques en prueba, que solo admiten 10 remuestreos distintos.
- **Dos colas** (14 de 18, 18 de 24, McNemar 6 de 6). Más conservador sin necesidad, dado que las preguntas son direccionales. Con un modelo que gana de verdad el 75 % de las casillas, la potencia en prueba bajaba de 72 % a 52 %.
- **Umbrales más laxos** (12 de 18, con 12 % por azar; 3 de 4 años para las excepciones, con 31 %). Las afirmaciones pierden valor y aparecerían excepciones por ruido.
- **Conclusión binaria** (supera o no supera).
- **Intervalo de la diferencia pareada para la excepción de familia por municipio** (redacción previa de §4.3). Con 4 años por municipio, un intervalo no es creíble.
- **McNemar sobre semanas.** La autocorrelación viola la independencia.
- **Prueba de Δw.** Hay muy pocos inicios avisados por ambos sistemas.
- **Intervalo exacto para la capacidad de detección.** Se reportan los conteos (k de 10), que comunican lo mismo.
- **Correcciones por comparaciones múltiples y pruebas por municipio.**
**Limitaciones declaradas:**
- Casillas del mismo año correlacionadas.
- Potencia baja: no alcanzar «mejora consistente» no prueba que no haya mejora.
- Cuatro afirmaciones en prueba sin corrección (≈ 19 % de al menos un falso positivo bajo nulidad total).
- Entre una y dos excepciones espurias esperables, de bajo costo.
- El 5 % es una convención.
**Evidencia:** `claude/verificacion/umbrales_d9.py` (umbrales, probabilidades por azar y potencia).
 
## D10 — Salida para el usuario final (2026-11-10)
 
**Decisión:** detalle en `02` §9. Es una primera versión; el equipo de frontend puede complementarla dentro de las reglas de §9.4 y §9.5.
- **Principio:** el usuario ve exactamente lo que se evaluó: la misma probabilidad, las mismas referencias, la misma regla de alerta y el desempeño de prueba.
- **Vista principal:**
  - la probabilidad de zona epidémica en t + 8, con las fechas explícitas;
  - las referencias B2 y B1;
  - el estado de alerta de §7.4, con cuatro estados;
  - el canal de epiCo con las bandas conocidas hasta t + 8;
  - las probabilidades de las últimas 12 emisiones;
  - la ficha de prueba;
  - advertencias permanentes.
- **Cuatro vistas, todas con la misma prioridad:**
  - pronóstico vigente, con el modelo final;
  - historial fuera de muestra 2019–2025, solo con las predicciones guardadas del origen móvil;
  - carga de datos, según el contrato de entrada de §3.3;
  - desempeño del modelo, solo con resultados de prueba.
- **API:** se amplía la FastAPI existente con `/predict`, `/predict/upload`, `/historial` y `/desempeno`. La parametrización va por variables de entorno.
**Justificación:**
- *Coherencia.* Si la interfaz usa la misma regla y las mismas referencias que la evaluación, el desempeño reportado aplica a lo que el usuario ve.
- *Referencias B1 y B2 visibles* (pedido de Jesús). Muestran cuánto agrega el modelo sobre el canal y sobre la estación, ya están calculadas para la evaluación y evidencian el trabajo metodológico.
- *Historial solo fuera de muestra.* El modelo final se entrenó con las etiquetas de 2012–2025; mostrarlo sobre semanas pasadas sería la misma fuga criticada en el avance, trasladada a la interfaz.
- *Carga de datos con prioridad completa* (pedido de Jesús: el equipo quiere verla). Ya estaba en el contrato de entrada de §3.3 y demuestra inferencia sobre datos nuevos (rúbrica de funcionamiento).
- *Rúbrica del despliegue.* Usabilidad (35 puntos): instrucciones, etiquetas, explicaciones y visualizaciones. Parametrización (15 puntos): variables de entorno. README (15 puntos): plantilla CSV y respuesta JSON como ejemplos.
- *Infraestructura existente.* La API FastAPI, el frontend React en Cloudflare Pages y el registro MLflow ya existen; fijar el esquema JSON permite que el frontend avance en paralelo con valores simulados.
**Alternativas descartadas:**
- **Mostrar solo la probabilidad, sin referencias.** El usuario no podría juzgar qué agrega el modelo.
- **Historial con el modelo final.** Predicciones dentro de muestra que aparentarían un desempeño inexistente.
- **τ ajustable por el usuario.** El desempeño medido solo vale para el τ congelado.
- **Categorías de riesgo adicionales** (bajo, medio, alto). No se evaluaron; los estados de alerta de §7.4 sí.
- **Atribuciones por variable** (por ejemplo, SHAP). Añaden complejidad y favorecen lecturas causales.
- **Carga de datos como prioridad secundaria** (propuesta inicial en el chat de metodología). Descartada por el equipo.
- **Ficha con resultados de desarrollo.** Se usaron para elegir; solo la prueba es una medición limpia.
- **Mapas, los 78 municipios, conteos o varios horizontes.** Fuera de alcance por D0, D3 y D4.
- **Cambiar a Streamlit o Power BI** (mencionados en la propuesta). La infraestructura React y FastAPI ya está desplegada.
**Limitaciones declaradas:**
- La usabilidad no se valida con usuarios reales.
- El pronóstico vigente no tiene verdad para comparar.
- Los datos subidos solo se validan en formato y completitud.
**Evidencia:** criterios de evaluación del despliegue (`Entregable y criterios de evaluación.docx`); estado del repositorio en `00` §5; propuesta del grupo, sección de transparencia.
 