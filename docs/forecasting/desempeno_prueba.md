# Vista «Desempeño en prueba»: esquema de datos

Contrato entre los datos y la pestaña del frontend. Las cifras del ejemplo son reales: salen de la evidencia versionada y coinciden con las Tablas 4 a 6 del informe final.

## 1. Archivo

- **Ruta:** `frontend/public/desempeno_prueba.json`. Se sirve como `/desempeno_prueba.json`, así que la pestaña lo lee con un `fetch` relativo, sin API, sin MLflow y sin variables de entorno.
- **Generación:** lo produce un script versionado, `scripts/construir_desempeno_prueba.py`, a partir de la evidencia de `reports/forecasting/evidence/`. No se edita a mano.
- **Tamaño:** unos 316 KB. Las probabilidades llevan 4 decimales.
- **Periodo:** 936 emisiones, 156 por municipio, del 2022-11-07 al 2025-10-27, con semanas objetivo del 2023-01-02 al 2025-12-22.
- **Qué no contiene:**
  - predicciones del modelo final;
  - cifras de desarrollo, porque la ficha solo muestra la prueba (contrato §9.4);
  - el contraste de familias.

## 2. Estructura

```
{
  "version_esquema": "1.0",
  "fuentes": [ { "ruta": "...", "sha256": "..." } ],
  "receta": { "hash_sha256": "de74d4ab…", "familia_comun": "logistica", "tau": 0.05, "k": 2, "W": 12 },
  "notas": [ "...", "...", "..." ],
  "global": { "probabilidad": {...}, "eventos": {...} },
  "municipios": { "<DIVIPOLA>": { "nombre", "ficha", "eventos", "inicios", "serie" } }
}
```

### 2.1 `notas`

Son tres textos fijos que la pestaña debe mostrar siempre, por ejemplo al pie:

1. «Resultados retrospectivos de la prueba 2023–2025, evaluada una sola vez con la receta congelada.»
2. «Las probabilidades son las de los modelos de origen móvil de la prueba, no las del modelo desplegado.»
3. «La regla del producto se definió después de la evaluación: sus cifras son descriptivas, no confirmatorias.»

### 2.2 `global.probabilidad`

| Campo | Tipo | Ejemplo | Lectura |
|---|---|---|---|
| `brier` | objeto | `{"modelo": 0.1413, "b2": 0.1537, "b1": 0.2658}` | Error cuadrático medio; menor es mejor |
| `bss_vs_b2`, `bss_vs_b1` | número | `0.0805`, `0.4684` | Mejora relativa del Brier frente a cada referencia; positivo es mejor |
| `ap` | objeto | `{"modelo": 0.7651, "b2": 0.6115}` | Precisión promedio; mayor es mejor |
| `prevalencia` | número | `0.3440` | Fracción de semanas objetivo en zona epidémica |
| `pares_ganados_vs_b2`, `pares_total` | entero | `12`, `18` | Pares municipio-año con menor Brier que B2 |
| `nivel_vs_b2`, `nivel_vs_b1` | texto | `"tendencia_favorable"`, `"mejora_consistente"` | Nivel del contrato §8.4 |

### 2.3 Eventos: `global.eventos` y `municipios.<código>.eventos`

Tienen tres claves, `umbral`, `producto` y `canal`, con los mismos campos:

| Campo | Tipo | Global: umbral / producto / canal |
|---|---|---|
| `inicios_avisados`, `inicios_evaluables` | entero | 4 / 9 / 8 de 10 |
| `entradas_avisadas`, `entradas_evaluables` | entero | 13 / 24 / 19 de 41 |
| `falsas_alarmas` | entero | 18 / 26 / 30 |
| `semanas_bajo_alerta`, `semanas` | entero | 565 / 192 / 393 de 936 |
| `anticipacion_mediana` (desde la alerta más reciente) | número o `null` | 4.5 / 1.0 / 1.5 |
| `anticipacion_mediana_primera` (desde la primera alerta) | número o `null` | 7.5 / 3.0 / 2.0 |
| `posterior` | booleano | `false` / `true` / `false` |

`posterior: true` marca la regla del producto. La pestaña debe mostrar esa marca (por ejemplo, †) con la nota 3.

### 2.4 `municipios.<código>.ficha`

| Campo | Tipo | Neiva (`41001`) |
|---|---|---|
| `nivel` | texto: `favorable`, `mixto` o `desfavorable` | `"favorable"` |
| `bss_vs_b2` | número | `0.1974` |
| `pares_ganados`, `pares_total` | entero | `3`, `3` |
| `ap` | objeto | `{"modelo": 0.7804, "b2": 0.6215}` |
| `prevalencia` | número | `0.3526` |

El nivel se toma de la evidencia y se calcula con el Brier y con el log-loss, adoptando el menor (Tabla 6). Por contrato, los seis municipios se muestran con la misma visibilidad, también el desfavorable (Barranquilla).

### 2.5 `municipios.<código>.inicios`

Hay una entrada por cada inicio de brote evaluable de la prueba, para dibujar los triángulos de la Fig. 3:

```
[
  { "semana": "2023-07-17", "avisado": { "umbral": true,  "producto": true, "canal": true } },
  { "semana": "2023-08-28", "avisado": { "umbral": true,  "producto": true, "canal": true } },
  { "semana": "2024-06-03", "avisado": { "umbral": false, "producto": true, "canal": true } }
]
```

Cali no tuvo inicios en la prueba, así que su lista viene vacía y la pestaña debe decir «Sin inicios de brote en la prueba».

### 2.6 `municipios.<código>.serie`

Hay 156 entradas por municipio, en orden cronológico. Alimentan la gráfica y las franjas.

| Campo | Tipo | Ejemplo (Neiva, 2023-05-22) | Uso |
|---|---|---|---|
| `semana_emision` | fecha ISO | `"2023-05-22"` | Eje de las franjas |
| `semana_objetivo` | fecha ISO | `"2023-07-17"` | Eje de la gráfica de probabilidad (t + 8) |
| `probabilidad` | número | `0.1275` | Probabilidad del modelo (origen móvil) |
| `b2_estado_canal` | número | `0.1552` | Referencia por zona actual del canal |
| `b1_climatologia_estacional` | número | `0.1579` | Referencia por época del año |
| `zona_t` | texto | `"alerta"` | Zona observada en la semana de emisión |
| `zona_epidemica_observada` | booleano | `true` | Si la semana objetivo terminó en zona epidémica (y) |
| `casos_objetivo`, `central_objetivo`, `u1e_objetivo` | número | `53`, `9.8039`, `46.3722` | Casos y canal de la semana objetivo, por si se dibuja la serie del canal |
| `bajo_alerta` | objeto | `{"umbral": true, "producto": true, "canal": true}` | Si la semana está bajo alerta con cada regla, desde la confirmación hasta el fin de la alerta |

## 3. Qué muestra la pestaña con estos datos

1. **Encabezado global.** BSS frente a B2 y frente a B1 con su nivel, y la tabla de eventos de las tres reglas (Tablas 4 y 5 del informe).
2. **Selector de municipio y ficha.** Nivel, BSS, pares, AP y la tabla de eventos del municipio (fila de la Tabla 6).
3. **Gráfica de probabilidad 2023–2025.** El modelo frente a B2 en el eje de la semana objetivo, con las semanas que terminaron en zona epidémica sombreadas.
4. **Franjas de alerta.** Tres franjas (umbral, producto† y canal) en el eje de la semana de emisión, con los inicios marcados y si cada regla los avisó (Fig. 3).
5. **Las tres notas.**

La pestaña no calcula ninguna métrica: solo presenta las del archivo. Si hace falta una cifra nueva, se agrega al script y a su prueba.

## 4. Garantías del archivo

La prueba automática del script (`tests/forecasting/test_desempeno_prueba.py`) comprueba:

- **Cifras globales:** coinciden con las del informe y de la evidencia (BSS 0,0805; 4, 9 y 8 de 10 inicios; 18, 26 y 30 falsas alarmas; 20,5 % y 42,0 % de semanas bajo alerta).
- **Cifras por municipio:** cada fila coincide con la Tabla 6.
- **Totales:** la suma de los municipios da el global.
- **Semanas bajo alerta:** los totales coinciden con la evidencia; el de la regla del producto suma 192.
- **Sin modelo final:** no hay ninguna emisión posterior al 2025-10-27.
- **Reproducibilidad:** al regenerarlo, el archivo sale idéntico byte a byte.

## 5. Regeneración

Desde la raíz del repositorio, con el entorno virtual:

- **Comprobar** que el archivo versionado coincide byte a byte con la evidencia (termina con código distinto de 0 si difiere):

  ```bash
  .venv/Scripts/python.exe -B scripts/construir_desempeno_prueba.py --verificar
  ```

- **Generar** el archivo en su ruta por defecto, o en otra con `--salida <ruta>`. El script no sobrescribe una salida existente y se niega a escribir con cambios sin commit en `src/` o `configs/`; una regeneración aprobada empieza por eliminar el archivo anterior:

  ```bash
  .venv/Scripts/python.exe -B scripts/construir_desempeno_prueba.py
  ```

- **Prueba automática** (§4); se omite con un mensaje explícito si la evidencia no está disponible:

  ```bash
  .venv/Scripts/python.exe -B -m pytest tests/forecasting/test_desempeno_prueba.py -q -p no:cacheprovider
  ```

El SHA-256 de cada fuente se calcula sobre el contenido versionado en git: los bytes exactos de la evidencia, marcada `-text` en `.gitattributes`, y el contenido con saltos de línea LF de los archivos de texto, como `configs/receta_congelada.yaml`. Así el archivo no depende de los saltos de línea de la copia local. El propio JSON está marcado `-text` para que `--verificar` compare los mismos bytes en cualquier sistema.
