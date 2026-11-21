# Especificación de la salida de la API — DengueAlertas (v0.2)
 
**Estado:** v0.2 del 2026-11-20 (enmienda D10.1: regla de alerta del producto B01, cuatro estados y probabilidad etiquetada según el estado; texto propuesto en `enmienda_d10_1_propuesta.md`, DV-07). Versión anterior: v0.1 del 2026-11-10, derivada de `contrato_predictivo.md` §9 (D10). La receta está congelada (`configs/receta_congelada.yaml`: regresión logística sin clima, τ = 0,05) y la ficha de desempeño ya existe (`data/processed/forecasting/prueba/ficha_desempeno.json`; copia en `reports/forecasting/evidence/prueba_20261112/`). La implementación de la API es parte de F8.
**Para:** el equipo de frontend, el responsable de infraestructura y API, y la implementación del pipeline.
**Regla de cambios:** ver §10. Los nombres de campo, los tipos y los valores permitidos son estables; los valores numéricos, τ, la familia del modelo y las versiones se toman de la receta congelada y de la ficha.
 
## 1. Para qué sirve
 
La API entrega lo que muestra la interfaz (`contrato_predictivo.md` §9.3) y nada más. El frontend no calcula probabilidades, estados de alerta ni zonas: los recibe ya calculados, para que lo que ve el usuario sea exactamente lo que se evaluó.
 
## 2. Convenciones
 
- **Fechas:** texto `YYYY-MM-DD` con el lunes que inicia la semana (lunes a domingo, W-SUN), igual que `week_start` en los datos.
- **Municipios:** código DIVIPOLA como texto de 5 caracteres, con el cero inicial (`"08001"`). Pilotos: `08001` Barranquilla, `08758` Soledad, `41001` Neiva, `50001` Villavicencio, `54001` San José de Cúcuta, `76001` Cali.
- **Probabilidades:** número entre 0 y 1, con 4 decimales. El frontend las muestra como porcentaje entero (`contrato_predictivo.md` §9.5).
- **Casos:** entero (`confirmed_records_screening`).
- **Central y U1e del canal:** número con hasta 5 decimales (redondeo de epiCo). El frontend redondea al mostrar.
- **Ausencias:** `null`; nunca se omite un campo.
- **Nombres:** español, `snake_case`, sin tildes.

**Valores permitidos (enumeraciones):**
 
| Campo | Valores | Significado |
|---|---|---|
| `zona` | `seguridad`, `alerta`, `epidemia` | Zona del canal (`contrato_predictivo.md` §1) |
| `estado_alerta` | `sin_alerta`, `senal_en_observacion`, `alerta_inicio_confirmada`, `continuidad_epidemica` | Regla de alerta del producto (D10.1; `contrato_predictivo.md` §7.4 y §9.3) |
| `etiqueta_probabilidad` | `entrar_en_zona_epidemica`, `seguir_en_zona_epidemica` | Lectura de la probabilidad según el estado (D10.1) |
| `regla_alerta` | `umbral_fuera_de_epidemia`, `umbral` | Regla del producto (B01) y regla preregistrada (p̂ ≥ τ), que solo aparece en la ficha |
| `origen` | `modelo_final`, `prueba_origen_movil`, `desarrollo_origen_movil`, `carga_usuario` | De dónde sale cada predicción (`contrato_predictivo.md` §9.4) |
| `periodo` | `desarrollo`, `prueba` | Partición de `contrato_predictivo.md` §6.1 |
| `nivel_conclusion` | `mejora_consistente`, `tendencia_favorable`, `sin_mejora` | `contrato_predictivo.md` §8.4 |
| `nivel_descriptivo` | `favorable`, `mixto`, `desfavorable` | Nivel por municipio en prueba (`contrato_predictivo.md` §8.8, D9.1) |
| `tipo_evento` | `inicio_avisado`, `inicio_no_avisado`, `falsa_alarma`, `alerta_continuidad`, `inicio_censurado`, `alerta_censurada` | `contrato_predictivo.md` §7.4 |
| `sistema` | `modelo`, `canal` | Quién emite la alerta |
 
**Estado de alerta (lo calcula la API con la regla del producto, B01; D10.1).** Una emisión es positiva si p̂ ≥ τ **y** la zona observada en su semana de emisión no es `epidemia`. Con la zona observada en t y las emisiones hasta t:
- `continuidad_epidemica`: la zona observada en t es `epidemia`. La emisión es negativa por la regla, así que no hay alerta del modelo.
- `sin_alerta`: la zona observada en t no es `epidemia` y p̂_t < τ.
- `senal_en_observacion`: la zona observada en t no es `epidemia`, p̂_t ≥ τ y la emisión anterior no fue positiva (racha de 1 < k emisiones positivas).
- `alerta_inicio_confirmada`: las k = 2 últimas emisiones son positivas (≥ τ y fuera de zona epidémica). El estado se mantiene mientras siga la racha.

**Probabilidad según el estado.** El número es siempre p̂ = P(zona epidémica en la semana objetivo t + 8). Se etiqueta `seguir_en_zona_epidemica` en `continuidad_epidemica` y `entrar_en_zona_epidemica` en los otros tres estados, siempre con la semana objetivo. No hay bandas de riesgo ni aviso de salida de la zona epidémica: no se evaluaron (`contrato_predictivo.md` §9.5).

## 3. Endpoints
 
| Método y ruta | Uso | Cálculo |
|---|---|---|
| `GET /health` | Estado del servicio (existente) | — |
| `GET /municipios` | Lista de pilotos y ventana de emisión vigente | — |
| `GET /schema` | Esquema de entrada para la carga (existente; actualizar) | — |
| `GET /predict` | Vista principal para una semana de la base | Modelo final |
| `POST /predict/upload` | Vista principal a partir de un CSV | Validación e inferencia con el modelo final |
| `GET /historial` | Predicciones fuera de muestra 2019–2025 y eventos | Tabla precalculada |
| `GET /desempeno` | Ficha de desempeño en prueba | Tabla precalculada |
 
FastAPI genera la documentación interactiva en `/docs` a partir de modelos Pydantic que reproduzcan este documento.
 
## 4. `GET /predict`
 
**Parámetros:**
- `municipio`: obligatorio.
- `semana_emision`: opcional; por defecto, la última semana completa de la base (2025-12-22).

Solo admite semanas de la ventana vigente: posteriores al 2025-10-27 y hasta el 2025-12-22. Para otra fecha responde el error `fuera_de_ventana_vigente` e indica usar `/historial`. Nunca devuelve predicciones del modelo final para semanas con verdad conocida (`contrato_predictivo.md` §9.4).
 
**Campos de la respuesta:**
 
| Campo | Tipo | Contenido |
|---|---|---|
| `version_esquema` | texto | `"0.2"` |
| `municipio` | objeto | `codigo`, `nombre` |
| `semana_emision` | fecha | t |
| `semana_objetivo` | fecha | t + h |
| `horizonte_semanas` | entero | h (8) |
| `origen` | enumeración | `modelo_final` |
| `probabilidad` | número | p̂ de zona epidémica en la semana objetivo |
| `etiqueta_probabilidad` | enumeración | `entrar_en_zona_epidemica` o `seguir_en_zona_epidemica`, según `decision.estado_alerta` |
| `referencias` | objeto | `b2_estado_canal` y `b1_climatologia_estacional` (probabilidades) |
| `decision` | objeto | `regla_alerta` (`umbral_fuera_de_epidemia`), `tau`, `k`, `estado_alerta`, `emisiones_positivas_consecutivas` (racha de emisiones ≥ τ fuera de zona epidémica que termina en t; 0 en continuidad epidémica) y `parametros_congelados` (booleano; `false` si τ o k difieren de los de la receta) |
| `continuidad` | objeto o `null` | Solo en `continuidad_epidemica` (si no, `null`): `probabilidad_seguir` (= `probabilidad`), `b2_estado_epidemia` (referencia B2 del estado epidémico; = `referencias.b2_estado_canal`), `semana_inicio_episodio` (primera semana de la racha actual de semanas observadas en zona epidémica hasta t) y `trayectoria_episodio` (una entrada por emisión desde `semana_inicio_episodio` hasta t: `semana_emision`, `semana_objetivo`, `probabilidad`, `b2_estado_canal`) |
| `canal_actual` | objeto | `casos`, `central`, `u1e` y `zona` de la semana de emisión |
| `canal_objetivo` | objeto | `central` y `u1e` de la semana objetivo (se conocen de antemano) |
| `serie_canal` | lista | 52 semanas observadas hasta t y 8 futuras. Cada elemento: `semana`, `casos`, `central`, `u1e`, `zona` y `tipo` (`observada` o `futura`; en las futuras, `casos` y `zona` son `null`) |
| `trayectoria` | lista | Las últimas 12 emisiones hasta t. Cada elemento: `semana_emision`, `semana_objetivo`, `probabilidad`, `sobre_tau`, `zona` (observada en la semana de emisión), `emision_positiva` (regla del producto) y `origen` (`prueba_origen_movil` o `modelo_final`) |
| `versiones` | objeto | `receta`, `familia`, `modelo` (URI de MLflow) y `datos` (versión DVC) |
| `advertencias` | lista | Objetos con `codigo` y `texto` (§4.1) |
 
**Ejemplo** (valores ilustrativos; las listas largas aparecen recortadas):
 
```json
{
  "version_esquema": "0.2",
  "municipio": {"codigo": "41001", "nombre": "Neiva"},
  "semana_emision": "2025-12-22",
  "semana_objetivo": "2026-02-16",
  "horizonte_semanas": 8,
  "origen": "modelo_final",
  "probabilidad": 0.4123,
  "etiqueta_probabilidad": "entrar_en_zona_epidemica",
  "referencias": {"b2_estado_canal": 0.1832, "b1_climatologia_estacional": 0.2410},
  "decision": {
    "regla_alerta": "umbral_fuera_de_epidemia", "tau": 0.05, "k": 2, "estado_alerta": "alerta_inicio_confirmada",
    "emisiones_positivas_consecutivas": 2, "parametros_congelados": true
  },
  "continuidad": null,
  "canal_actual": {"casos": 38, "central": 22.14, "u1e": 41.73, "zona": "alerta"},
  "canal_objetivo": {"central": 30.21, "u1e": 52.88},
  "serie_canal": [
    {"semana": "2024-12-30", "casos": 12, "central": 15.02, "u1e": 28.41, "zona": "seguridad", "tipo": "observada"},
    {"semana": "2026-02-16", "casos": null, "central": 30.21, "u1e": 52.88, "zona": null, "tipo": "futura"}
  ],
  "trayectoria": [
    {"semana_emision": "2025-10-06", "semana_objetivo": "2025-12-01", "probabilidad": 0.1210, "sobre_tau": true, "zona": "epidemia", "emision_positiva": false, "origen": "prueba_origen_movil"},
    {"semana_emision": "2025-12-22", "semana_objetivo": "2026-02-16", "probabilidad": 0.4123, "sobre_tau": true, "zona": "alerta", "emision_positiva": true, "origen": "modelo_final"}
  ],
  "versiones": {"receta": "de74d4ab…fda5", "familia": "logistica", "modelo": "models:/DengueAlertas_Neiva/1", "datos": "dvc:xxxxxxxx"},
  "advertencias": [
    {"codigo": "retrospectivo", "texto": "Pronóstico retrospectivo con datos consolidados."},
    {"codigo": "sin_aval_ins", "texto": "No reemplaza la vigilancia oficial ni tiene aval del INS."},
    {"codigo": "probabilidad_no_certeza", "texto": "Es una probabilidad, no una certeza; el modelo no identifica causas."},
    {"codigo": "semanas_incompletas", "texto": "Si las semanas recientes están incompletas, el riesgo se subestima."}
  ]
}
```
 
**En continuidad epidémica** (fragmento ilustrativo): `"etiqueta_probabilidad": "seguir_en_zona_epidemica"`, `"decision": {"estado_alerta": "continuidad_epidemica", "emisiones_positivas_consecutivas": 0, …}` y

```json
"continuidad": {
  "probabilidad_seguir": 0.6812,
  "b2_estado_epidemia": 0.5410,
  "semana_inicio_episodio": "2025-11-17",
  "trayectoria_episodio": [
    {"semana_emision": "2025-11-17", "semana_objetivo": "2026-01-12", "probabilidad": 0.7420, "b2_estado_canal": 0.5410},
    {"semana_emision": "2025-12-22", "semana_objetivo": "2026-02-16", "probabilidad": 0.6812, "b2_estado_canal": 0.5410}
  ]
}
```

### 4.1 Advertencias
 
- **Permanentes** (siempre presentes; `contrato_predictivo.md` §9.5): `retrospectivo`, `sin_aval_ins`, `probabilidad_no_certeza`, `semanas_incompletas`.
- **Condicionales:** `parametros_modificados` (τ o k distintos de la receta: la ficha de desempeño no aplica).

## 5. `POST /predict/upload`
 
**Entrada:** formulario `multipart/form-data` con el campo `archivo` (CSV, UTF-8, separador coma, decimal punto).
 
**Plantilla CSV:** usa los mismos nombres de columna que la base, para que un extracto de la base sea un archivo válido.
 
| Columna | Tipo | Regla |
|---|---|---|
| `municipality_code` | texto | Un solo piloto por archivo |
| `week_start` | fecha | Lunes; semanas consecutivas, sin huecos ni duplicados |
| `confirmed_records_screening` | entero | ≥ 0 |
| `temperature_c` | número | Sin faltantes |
| `precipitation_mm` | número | ≥ 0, sin faltantes |
| `relative_humidity_pct` | número | Entre 0 y 100, sin faltantes |
 
**Historia mínima (`contrato_predictivo.md` §3.3):** los cinco años completos anteriores al año de la semana t − 11, más todas las semanas hasta t, donde t es la última semana del archivo. Ejemplo: con t = 2025-12-22, t − 11 cae en 2025, así que el archivo debe empezar en la primera semana de 2020.
 
**Salida:** la misma de `/predict`, con `origen` igual a `carga_usuario` y un bloque adicional `validacion` (`filas`, `semana_inicial`, `semana_final`).
 
**Errores de validación (código 422):**
 
| Código | Causa |
|---|---|
| `columnas_faltantes` | Falta alguna columna de la plantilla |
| `municipio_no_piloto` | El código no es uno de los seis pilotos |
| `varios_municipios` | El archivo trae más de un municipio |
| `semana_no_lunes` | Alguna `week_start` no es lunes |
| `semanas_faltantes` | Hay huecos en la secuencia semanal |
| `semanas_duplicadas` | Hay semanas repetidas |
| `historia_insuficiente` | No alcanza la historia mínima |
| `valores_faltantes` | Hay celdas vacías |
| `valores_invalidos` | Valores fuera de rango o no numéricos |
 
**[R]** Los datos subidos se usan solo para la consulta y no se almacenan (`contrato_predictivo.md` §9.4).
 
## 6. `GET /historial`
 
**Parámetros:** `municipio` (obligatorio), `desde` y `hasta` (opcionales; por defecto todo 2019–2025).
 
**Respuesta:**
- `municipio`, `tau`, `regla_alerta` (`umbral_fuera_de_epidemia`) y `version_esquema`.
- `filas`: una por semana de emisión. Cada fila: `semana_emision`, `semana_objetivo`, `periodo`, `origen` (`desarrollo_origen_movil` o `prueba_origen_movil`), `probabilidad`, `etiqueta_probabilidad`, `b2_estado_canal`, `b1_climatologia_estacional`, `zona_emision`, `estado_alerta` (regla del producto), `casos_objetivo`, `central_objetivo`, `u1e_objetivo`, `zona_objetivo` y `observado_objetivo` (0 o 1).
- `eventos`: lista de objetos con `tipo_evento`, `sistema`, `semana` (la del evento), `semana_alerta` (t_c, si aplica) y `anticipacion_semanas` (si aplica, desde la alerta más reciente de la ventana). Para `sistema = modelo`, los eventos son los de la regla del producto: con B01 no hay `alerta_continuidad`, porque la emisión en zona epidémica es negativa.
```json
{
  "version_esquema": "0.2",
  "municipio": {"codigo": "54001", "nombre": "San José de Cúcuta"},
  "tau": 0.05,
  "regla_alerta": "umbral_fuera_de_epidemia",
  "filas": [
    {"semana_emision": "2023-11-20", "semana_objetivo": "2024-01-15", "periodo": "prueba", "origen": "prueba_origen_movil",
     "probabilidad": 0.5210, "etiqueta_probabilidad": "entrar_en_zona_epidemica", "b2_estado_canal": 0.1832,
     "b1_climatologia_estacional": 0.3105, "zona_emision": "alerta", "estado_alerta": "alerta_inicio_confirmada",
     "casos_objetivo": 41, "central_objetivo": 18.40, "u1e_objetivo": 33.12, "zona_objetivo": "epidemia", "observado_objetivo": 1}
  ],
  "eventos": [
    {"tipo_evento": "inicio_avisado", "sistema": "modelo", "semana": "2024-01-15", "semana_alerta": "2023-11-20", "anticipacion_semanas": 8}
  ]
}
```
 
(Valores ilustrativos. El inicio de Cúcuta del 2024-01-15 es real; las probabilidades y la alerta no.)

**Historial precalculado.** El historial generado en F7 (`artifacts.history_table`, en `data/processed/forecasting/prueba/`) tiene los estados de la versión 0.1, con la regla preregistrada, y sus pruebas congeladas lo fijan así. En F8, el historial de la API se genera en una carpeta nueva con los estados 0.2, calculados con `events.product_alert_states` y `producto.regla_alerta`; el historial de F7 se conserva como registro.
 
## 7. `GET /desempeno`
 
**Parámetro:** `municipio` (opcional; sin él devuelve la ficha global).
 
**Contenido** (prueba 2023–2025; `contrato_predictivo.md` §9.4, vista 4, con D10.1):
- `estimacion`: `bss_vs_b2`, `bss_vs_b1`, `mejora_logloss_vs_b2`, `mejora_logloss_vs_b1` y, para cada comparación (`vs_b2`, `vs_b1`, `ablacion_clima`), un objeto `conclusion` con `nivel_conclusion`, `casillas_ganadas` y `casillas_total`. No depende de la regla de alerta.
- `eventos_regla_producto` (B01; se muestra con la etiqueta «Evaluación posterior no confirmatoria»):
  - `regla_alerta`: `umbral_fuera_de_epidemia`; `etiqueta`: `evaluacion_posterior_no_confirmatoria`;
  - `inicios_evaluables`, `inicios_avisados`, `entradas_evaluables`, `entradas_avisadas`;
  - `falsas_alarmas` y `falsas_alarmas_por_municipio_anio`;
  - `fraccion_semanas_bajo_alerta` (semanas de emisión entre la confirmación de cada alerta y su última semana);
  - `anticipacion_mediana_semanas`: `desde_alerta_mas_reciente` (medida del contrato) y `desde_primera_alerta`;
  - `mcnemar_vs_regla_preregistrada`: `discrepantes`, `a_favor_regla_producto`.
- `eventos_regla_preregistrada` (p̂ ≥ τ; resultado preregistrado de §8.4, junto al anterior):
  - `regla_alerta`: `umbral`;
  - `inicios_evaluables`, `avisados_modelo`, `avisados_canal`;
  - `mcnemar`: `discrepantes`, `a_favor_modelo`, `nivel_conclusion`;
  - `falsas_alarmas_por_municipio_anio`: `modelo`, `canal`;
  - `anticipacion_mediana_semanas`: `modelo`, `canal`.
- `continuidad` (descriptivo, junto a los anteriores; semanas de prueba con la zona observada en t en epidemia): `bss_vs_b2_zona_epidemia_prueba`, `filas`, `casillas_ganadas_brier` y `casillas` (municipio-años con al menos 5 semanas en el estrato).

**Valores globales de la ficha 0.2** (de `reports/forecasting/evidence/complementario_20261120/`, `prueba_20261112/` y `continuidad_20261120/`; verificados con `docs/forecasting/verificacion/verificar_cifras_b01.py`):

| Bloque | Cifra | Valor |
|---|---|---|
| Regla del producto (B01), evaluación posterior no confirmatoria | Inicios avisados | 9 de 10 |
| | Entradas avisadas | 24 de 41 |
| | Falsas alarmas (por municipio-año) | 26 (1,44) |
| | Semanas bajo alerta | 20,5 % (192 de 936) |
| | Anticipación mediana (desde la alerta más reciente / desde la primera) | 1,0 / 3,0 semanas |
| | Inicios discrepantes frente a la regla preregistrada | 5, todos a favor de B01 |
| Regla preregistrada (p̂ ≥ τ), resultado de §8.4 | Inicios avisados (modelo / canal) | 4 / 8 de 10; McNemar «sin mejora» (1 de 6 discrepantes a favor del modelo) |
| | Falsas alarmas por municipio-año (modelo / canal) | 1,00 / 1,67 |
| | Anticipación mediana (modelo / canal) | 4,5 / 1,5 semanas |
| Continuidad (zona epidémica), prueba | BSS frente a B2; casillas ganadas por Brier | 0,136; 10 de 14 |
- `calibracion`: lista de intervalos con `probabilidad_media`, `frecuencia_observada` y `n`.
- `curva_amoc`: lista con `tau`, `deteccion`, `falsas_alarmas_por_municipio_anio` y `anticipacion_mediana_semanas`, con la regla preregistrada (no se recalcula para B01, cuyo τ no se eligió con ella).
- `casillas`: lista con `municipio`, `anio`, `brier_modelo`, `brier_b2`, `brier_b1`, `logloss_modelo`, `logloss_b2` y `gana_vs_b2`.
- `notas`: textos fijos. Por ejemplo, para Cali: «No hubo inicios de brote en 2023–2025: solo se miden falsas alarmas».

**Ficha por municipio.** Los bloques de eventos se dan también por municipio, como conteos (`contrato_predictivo.md` §7.6); los de B01 salen de `complementario_20261120/b01_eventos.csv` (nivel `municipio`). En la ficha generada (`ficha_desempeno.json`), por municipio `conclusion.nivel_conclusion` es `null` y se informa `nivel_descriptivo` (con `nivel_descriptivo_brier` y `nivel_descriptivo_logloss`; `contrato_predictivo.md` §8.8); `mcnemar.nivel_conclusion` es `null`. La ficha también trae campos adicionales (compatibles según §10) y algunas claves fuera de la convención de §2: `mcnemar.nivel` con el valor «sin mejora» (con espacio), las claves `B2` y `B1` en mayúscula y `municipality_code`. La ficha versionada no se regenera: en F8 la API normaliza estos campos al servirlos.
 
## 8. Errores
 
Formato único:
 
```json
{"error": {"codigo": "fuera_de_ventana_vigente", "mensaje": "La semana 2024-03-04 tiene verdad conocida. Consulte /historial.", "detalles": []}}
```
 
Códigos HTTP: 404 (municipio desconocido), 422 (parámetros o archivo inválidos) y 500 (error interno, con un mensaje genérico).
 
## 9. Configuración (variables de entorno)

**Sustituida por la especificación §10** (`FORECAST_HORIZON`, `MODEL_THRESHOLD`, `ALERT_K`, `DATA_VERSION`, `MODEL_NAME`, `STAGE`; la URL de la API en el frontend se define en F8). La tabla siguiente se conserva como registro.

| Variable | Por defecto | Uso |
|---|---|---|
| `DENGUE_HORIZONTE` | 8 | Solo valores con modelo entrenado |
| `DENGUE_TAU` | El de la receta congelada | Si difiere, `parametros_congelados = false` |
| `DENGUE_K` | 2 | Ídem |
| `DENGUE_MODELO_URI` | — | Modelo o etapa en el registro MLflow |
| `DENGUE_DATOS_VERSION` | — | Versión DVC de la base |
| `VITE_API_URL` | — | URL de la API en el frontend |
 
Los nombres definitivos deben armonizarse con los de `src/api.py` y documentarse en el README.

**Regla de alerta.** No es una variable de entorno: la API la lee de `producto.regla_alerta` en `configs/forecasting.yaml` (`umbral_fuera_de_epidemia`, D10.1). La selección y la evaluación preregistradas usan `decision.regla_emision`, fija en `umbral`. Si `producto.regla_alerta` se cambiara, el bloque `eventos_regla_producto` de la ficha dejaría de aplicar.
 
## 10. Qué puede cambiar y cómo
 
- **Estable:** nombres de campo, tipos y enumeraciones. Un cambio aquí sube `version_esquema` y se avisa al frontend.
- **Cambios de 0.1 a 0.2 (D10.1):** `estado_alerta` cambia `alerta_confirmada` por `alerta_inicio_confirmada` y redefine los cuatro estados con la regla B01; `decision.emisiones_consecutivas_sobre_tau` pasa a `decision.emisiones_positivas_consecutivas`; se añaden `etiqueta_probabilidad`, `decision.regla_alerta`, `continuidad`, `trayectoria[].zona`, `trayectoria[].emision_positiva`, `regla_alerta` en `/historial`, `filas[].etiqueta_probabilidad` y `filas[].zona_emision`; la ficha separa `eventos_regla_producto`, `eventos_regla_preregistrada` y `continuidad`.
- **Tomado de la receta congelada:** τ = 0,05, la familia (`logistica`) y las versiones. Los datos simulados usan τ = 0,05.
- **Compatible:** agregar campos nuevos sin quitar ni renombrar los existentes.

## 11. Datos simulados para el frontend
 
Para que los datos simulados se parezcan a los reales, deben respetar estas invariantes, que también sirven como pruebas de la API:
- `zona` es coherente con `casos`, `central` y `u1e` (`contrato_predictivo.md` §1).
- `estado_alerta` es coherente con la trayectoria, la zona de emisión y τ (§2): `continuidad_epidemica` si y solo si `canal_actual.zona` es `epidemia`; `alerta_inicio_confirmada` si y solo si las dos últimas entradas de la trayectoria tienen `emision_positiva = true`.
- `etiqueta_probabilidad` es `seguir_en_zona_epidemica` si y solo si `estado_alerta` es `continuidad_epidemica`, y `continuidad` es `null` en los otros estados.
- `semana_objetivo` = `semana_emision` + 8 semanas.
- `serie_canal` tiene 60 elementos: los 52 primeros, `observada`, y los 8 últimos, `futura`.
- `trayectoria` tiene 12 elementos y el último corresponde a `semana_emision`.
- Todas las probabilidades están entre 0 y 1.