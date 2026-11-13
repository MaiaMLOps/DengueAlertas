# Especificación de la salida de la API — DengueAlertas (v0.1)
 
**Estado:** borrador del 2026-11-10, derivado de `contrato_predictivo.md` §9 (D10). Sirve para que el frontend avance con datos simulados mientras se entrenan los modelos.
**Para:** el equipo de frontend, el responsable de infraestructura y API, y la implementación del pipeline.
**Regla de cambios:** ver §10. Los nombres de campo, los tipos y los valores permitidos son estables; los valores numéricos, τ, la familia del modelo y las versiones se llenan cuando la receta se congele.
 
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
| `estado_alerta` | `sin_alerta`, `senal_en_observacion`, `alerta_confirmada`, `continuidad_epidemica` | Regla de alerta (`contrato_predictivo.md` §7.4 y §9.3) |
| `origen` | `modelo_final`, `prueba_origen_movil`, `desarrollo_origen_movil`, `carga_usuario` | De dónde sale cada predicción (`contrato_predictivo.md` §9.4) |
| `periodo` | `desarrollo`, `prueba` | Partición de `contrato_predictivo.md` §6.1 |
| `nivel_conclusion` | `mejora_consistente`, `tendencia_favorable`, `sin_mejora` | `contrato_predictivo.md` §8.4 |
| `tipo_evento` | `inicio_avisado`, `inicio_no_avisado`, `falsa_alarma`, `alerta_continuidad`, `inicio_censurado`, `alerta_censurada` | `contrato_predictivo.md` §7.4 |
| `sistema` | `modelo`, `canal` | Quién emite la alerta |
 
**Estado de alerta (lo calcula la API):**
- `sin_alerta`: la emisión actual está bajo τ.
- `senal_en_observacion`: la emisión actual es ≥ τ, pero la anterior no.
- `alerta_confirmada`: las dos últimas emisiones son ≥ τ (k = 2).
- `continuidad_epidemica`: hay alerta y la zona observada en la semana de emisión ya es `epidemia`.
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
| `version_esquema` | texto | `"0.1"` |
| `municipio` | objeto | `codigo`, `nombre` |
| `semana_emision` | fecha | t |
| `semana_objetivo` | fecha | t + h |
| `horizonte_semanas` | entero | h (8) |
| `origen` | enumeración | `modelo_final` |
| `probabilidad` | número | p̂ de zona epidémica en la semana objetivo |
| `referencias` | objeto | `b2_estado_canal` y `b1_climatologia_estacional` (probabilidades) |
| `decision` | objeto | `tau`, `k`, `estado_alerta`, `emisiones_consecutivas_sobre_tau` y `parametros_congelados` (booleano; `false` si τ o k difieren de los de la receta) |
| `canal_actual` | objeto | `casos`, `central`, `u1e` y `zona` de la semana de emisión |
| `canal_objetivo` | objeto | `central` y `u1e` de la semana objetivo (se conocen de antemano) |
| `serie_canal` | lista | 52 semanas observadas hasta t y 8 futuras. Cada elemento: `semana`, `casos`, `central`, `u1e`, `zona` y `tipo` (`observada` o `futura`; en las futuras, `casos` y `zona` son `null`) |
| `trayectoria` | lista | Las últimas 12 emisiones hasta t. Cada elemento: `semana_emision`, `semana_objetivo`, `probabilidad`, `sobre_tau` y `origen` (`prueba_origen_movil` o `modelo_final`) |
| `versiones` | objeto | `receta`, `familia`, `modelo` (URI de MLflow) y `datos` (versión DVC) |
| `advertencias` | lista | Objetos con `codigo` y `texto` (§4.1) |
 
**Ejemplo** (valores ilustrativos; las listas largas aparecen recortadas):
 
```json
{
  "version_esquema": "0.1",
  "municipio": {"codigo": "41001", "nombre": "Neiva"},
  "semana_emision": "2025-12-22",
  "semana_objetivo": "2026-02-16",
  "horizonte_semanas": 8,
  "origen": "modelo_final",
  "probabilidad": 0.4123,
  "referencias": {"b2_estado_canal": 0.1832, "b1_climatologia_estacional": 0.2410},
  "decision": {
    "tau": 0.35, "k": 2, "estado_alerta": "alerta_confirmada",
    "emisiones_consecutivas_sobre_tau": 2, "parametros_congelados": true
  },
  "canal_actual": {"casos": 38, "central": 22.14, "u1e": 41.73, "zona": "alerta"},
  "canal_objetivo": {"central": 30.21, "u1e": 52.88},
  "serie_canal": [
    {"semana": "2024-12-30", "casos": 12, "central": 15.02, "u1e": 28.41, "zona": "seguridad", "tipo": "observada"},
    {"semana": "2026-02-16", "casos": null, "central": 30.21, "u1e": 52.88, "zona": null, "tipo": "futura"}
  ],
  "trayectoria": [
    {"semana_emision": "2025-10-06", "semana_objetivo": "2025-12-01", "probabilidad": 0.1210, "sobre_tau": false, "origen": "prueba_origen_movil"},
    {"semana_emision": "2025-12-22", "semana_objetivo": "2026-02-16", "probabilidad": 0.4123, "sobre_tau": true, "origen": "modelo_final"}
  ],
  "versiones": {"receta": "receta-2026-11-xx", "familia": "regresion_logistica", "modelo": "models:/dengue_h8_41001/1", "datos": "dvc:xxxxxxxx"},
  "advertencias": [
    {"codigo": "retrospectivo", "texto": "Pronóstico retrospectivo con datos consolidados."},
    {"codigo": "sin_aval_ins", "texto": "No reemplaza la vigilancia oficial ni tiene aval del INS."},
    {"codigo": "probabilidad_no_certeza", "texto": "Es una probabilidad, no una certeza; el modelo no identifica causas."},
    {"codigo": "semanas_incompletas", "texto": "Si las semanas recientes están incompletas, el riesgo se subestima."}
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
- `municipio`, `tau` y `version_esquema`.
- `filas`: una por semana de emisión. Cada fila: `semana_emision`, `semana_objetivo`, `periodo`, `origen` (`desarrollo_origen_movil` o `prueba_origen_movil`), `probabilidad`, `b2_estado_canal`, `b1_climatologia_estacional`, `estado_alerta`, `casos_objetivo`, `central_objetivo`, `u1e_objetivo`, `zona_objetivo` y `observado_objetivo` (0 o 1).
- `eventos`: lista de objetos con `tipo_evento`, `sistema`, `semana` (la del evento), `semana_alerta` (t_c, si aplica) y `anticipacion_semanas` (si aplica).
```json
{
  "version_esquema": "0.1",
  "municipio": {"codigo": "54001", "nombre": "San José de Cúcuta"},
  "tau": 0.35,
  "filas": [
    {"semana_emision": "2023-11-20", "semana_objetivo": "2024-01-15", "periodo": "prueba", "origen": "prueba_origen_movil",
     "probabilidad": 0.5210, "b2_estado_canal": 0.1832, "b1_climatologia_estacional": 0.3105, "estado_alerta": "alerta_confirmada",
     "casos_objetivo": 41, "central_objetivo": 18.40, "u1e_objetivo": 33.12, "zona_objetivo": "epidemia", "observado_objetivo": 1}
  ],
  "eventos": [
    {"tipo_evento": "inicio_avisado", "sistema": "modelo", "semana": "2024-01-15", "semana_alerta": "2023-11-20", "anticipacion_semanas": 8}
  ]
}
```
 
(Valores ilustrativos. El inicio de Cúcuta del 2024-01-15 es real; las probabilidades y la alerta no.)
 
## 7. `GET /desempeno`
 
**Parámetro:** `municipio` (opcional; sin él devuelve la ficha global).
 
**Contenido** (solo prueba 2023–2025; `contrato_predictivo.md` §9.4, vista 4):
- `estimacion`: `bss_vs_b2`, `bss_vs_b1`, `mejora_logloss_vs_b2`, `mejora_logloss_vs_b1` y, para cada comparación (`vs_b2`, `vs_b1`, `ablacion_clima`), un objeto `conclusion` con `nivel_conclusion`, `casillas_ganadas` y `casillas_total`.
- `eventos`:
  - `inicios_evaluables`, `avisados_modelo`, `avisados_canal`;
  - `mcnemar`: `discrepantes`, `a_favor_modelo`, `nivel_conclusion`;
  - `falsas_alarmas_por_municipio_anio`: `modelo`, `canal`;
  - `anticipacion_mediana_semanas`: `modelo`, `canal`.
- `calibracion`: lista de intervalos con `probabilidad_media`, `frecuencia_observada` y `n`.
- `curva_amoc`: lista con `tau`, `deteccion`, `falsas_alarmas_por_municipio_anio` y `anticipacion_mediana_semanas`.
- `casillas`: lista con `municipio`, `anio`, `brier_modelo`, `brier_b2`, `brier_b1`, `logloss_modelo`, `logloss_b2` y `gana_vs_b2`.
- `notas`: textos fijos. Por ejemplo, para Cali: «No hubo inicios de brote en 2023–2025: solo se miden falsas alarmas».
Hasta que se ejecute la prueba, los valores numéricos son `null` y el frontend muestra «pendiente».
 
## 8. Errores
 
Formato único:
 
```json
{"error": {"codigo": "fuera_de_ventana_vigente", "mensaje": "La semana 2024-03-04 tiene verdad conocida. Consulte /historial.", "detalles": []}}
```
 
Códigos HTTP: 404 (municipio desconocido), 422 (parámetros o archivo inválidos) y 500 (error interno, con un mensaje genérico).
 
## 9. Configuración (variables de entorno)
 
| Variable | Por defecto | Uso |
|---|---|---|
| `DENGUE_HORIZONTE` | 8 | Solo valores con modelo entrenado |
| `DENGUE_TAU` | El de la receta congelada | Si difiere, `parametros_congelados = false` |
| `DENGUE_K` | 2 | Ídem |
| `DENGUE_MODELO_URI` | — | Modelo o etapa en el registro MLflow |
| `DENGUE_DATOS_VERSION` | — | Versión DVC de la base |
| `VITE_API_URL` | — | URL de la API en el frontend |
 
Los nombres definitivos deben armonizarse con los de `src/api.py` y documentarse en el README.
 
## 10. Qué puede cambiar y cómo
 
- **Estable:** nombres de campo, tipos y enumeraciones. Un cambio aquí sube `version_esquema` y se avisa al frontend.
- **Se llena al congelar la receta:** τ, la familia y las versiones. En los datos simulados, usar τ = 0,35 como valor provisional.
- **Compatible:** agregar campos nuevos sin quitar ni renombrar los existentes.
- **Si la familia elegida es la GRU,** el esquema no cambia.
## 11. Datos simulados para el frontend
 
Para que los datos simulados se parezcan a los reales, deben respetar estas invariantes, que también sirven como pruebas de la API:
- `zona` es coherente con `casos`, `central` y `u1e` (`contrato_predictivo.md` §1).
- `estado_alerta` es coherente con la trayectoria y τ (§2).
- `semana_objetivo` = `semana_emision` + 8 semanas.
- `serie_canal` tiene 60 elementos: los 52 primeros, `observada`, y los 8 últimos, `futura`.
- `trayectoria` tiene 12 elementos y el último corresponde a `semana_emision`.
- Todas las probabilidades están entre 0 y 1.