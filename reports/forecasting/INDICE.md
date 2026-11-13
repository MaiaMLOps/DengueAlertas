# Índice de resultados del pronóstico

Mapa de las carpetas de resultados del MVP de pronóstico (`feature/forecasting-mvp`): qué contiene cada una, con qué conjunto de variables, de qué búsqueda o commit sale, qué decisión del contrato la originó y si está vigente. Se actualiza en cada unidad. Ningún archivo versionado se mueve: las carpetas sustituidas se conservan como registro.

**Estados:** *vigente* (insumo de la receta en curso), *registro* (se conserva como evidencia; no alimenta la receta en curso), *sustituida* (reemplazada por una decisión posterior), *pendiente* (todavía no generada), *exploratorio* (cálculo descriptivo del análisis integrado que no se declaró antes de la prueba; no alimenta ninguna decisión ni modifica las conclusiones de §8.4).

**Conjuntos de variables (contrato §6.6, D7.2):** *completo*, las 15 variables de §5.2; *sin clima*, las 6 que no son anomalías climáticas (`zona_t`, `posicion_4s`, `crecimiento_4s`, `cambio_umbral`, `sem_sin`, `sem_cos`).

**Prueba 2023–2025:** ejecutada una vez (`reports/forecasting/bitacora_prueba.jsonl`, ejecución 1, 2026-11-12, código abf08c2, receta `de74d4ab…fda5`), verificada de forma independiente por el responsable de la implementación. Cualquier otra ejecución exige una razón registrada.

## Datos (`data/processed/forecasting/`, DVC)

| Carpeta | Contenido | Conjunto | Búsqueda / commit | Origen | Estado |
|---|---|---|---|---|---|
| raíz `data/processed/forecasting/` | Búsqueda de F5b: tabla única, predicciones, líneas base y manifiesto (con `.dvc`); `partes/` y `busqueda.log` solo locales | **Completo**, logística (6) y LightGBM (25) | `7f34b69832eea69407c6`; código cd9d702; `.dvc` en 1a1ad2d | Pre-registro (especificación §7), D5–D7 | **Vigente** como conjunto completo del paso 0 (D7.2) |
| `ablacion_clima/` | Búsqueda de la logística sin clima (6 candidatos); tabla, predicciones, líneas base y manifiesto con `.dvc` (`partes/` y `busqueda.log` solo locales) | Sin clima, solo logística | `26dc1af622fb5966e56e`; código a3b274b | Ablación de F7, decisión D1 (contrato §4.4) | **Registro**: origen de D7.2; no alimenta la selección |
| `sin_clima/` | Búsqueda sin clima de las dos familias (`--variante sin_clima`), 198 de 198 unidades; tabla, predicciones, líneas base y manifiesto con `.dvc` (`partes/` y `busqueda.log` solo locales) | Sin clima, logística (6) y LightGBM (25), mismos candidatos y semillas que F5b | `ae6225180d37d025224b`; código 1f45514 | D7.2 (contrato §6.6) | **Vigente** como conjunto sin clima del paso 0; su logística es idéntica a `ablacion_clima/` |
| `prueba/` | Prueba única de F7: predicciones de modelo, ablación, B2 y B1, ensayos, casillas, resumen de §7 y §8.4 e insumos de §8.8 (`evaluate`, abf08c2); ficha, historial (2.184 filas, 172 eventos) y 6 modelos finales (`artifacts`, bd165ca); todo con `.dvc` (carpetas `diagnostico_prueba/` y `modelos_finales/` con un `.dvc` cada una) | Receta de D7.2 | Ejecución 1 de la bitácora (67bdcb1), receta `de74d4ab…` | Contrato §6.1, §7, §8.4, §8.8 y §9.4 | **Vigente** |

## Evidencia (`reports/forecasting/evidence/`, git)

| Carpeta | Contenido | Conjunto | Commit | Origen | Estado |
|---|---|---|---|---|---|
| `seleccion_20261112/` | Selección de F6 (casillas fuera de año, comparaciones, curva de τ, márgenes), borrador de receta y copia de la receta congelada sustituida (`receta_congelada_sustituida_d2135a39.yaml`) | Completo | f1c0876 (evidencia y borrador); receta fc31f13; copia con DV-02 | F6, contrato §6.2, §7.5 y §8.3 | **Sustituida** por D7.2 (DV-02); se conserva como registro |
| `seleccion_conjuntos_20261112/` | Selección de D7.2: paso 0 por familia (`comparacion_conjuntos.csv`), pasos 1–4, τ, ablación (`ablacion.csv`, `comparaciones_ablacion.csv`) y borrador de receta con `conjunto_variables` y `ablacion` | Sin clima en las dos familias; ablación con el completo | ced43ff (generada desde 1f45514 con el árbol limpio); regresión congelada en 4e01794 | D7.2, contrato §6.6 | **Vigente**: su borrador es la receta congelada |
| `prueba_20261112/` | Copia en git de las tablas pequeñas de `prueba/`: `resumen_prueba.json`, `casillas_prueba.csv`, `ficha_desempeno.json`, los seis `diag_*.csv` de prueba y `manifiesto_modelos_finales.json` (idénticos byte a byte) | Receta de D7.2 | Ver la fila de `prueba/` | Contrato §8.4, §8.8 y §9.4 | **Vigente** |
| `diagnostico_desarrollo_20261112/` | Seis insumos de §8.8 en desarrollo: logística completa con la configuración común fuera de año, τ = 0,10 | Completo | a3b274b (código caae344); regresión en `test_f7_diagnostico_desarrollo.py` | D9.1, contrato §8.8 | **Registro**: corresponde a la receta sustituida (DV-02) |
| `diagnostico_desarrollo_sin_clima_20261112/` | Seis insumos de §8.8 en desarrollo para la receta vigente: logística sin clima con la configuración común fuera de año (2019, 2021 y 2022 `logistica_04`; 2020 `logistica_05`), τ = 0,05, rótulo «usado para elegir»; reproducido por `caso_f7_diagnostico.py` | Sin clima | a1d1cc0 (generado desde 2a1ab0b); regresión congelada en 043c8ad (`test_f7b_diagnostico_desarrollo.py`) | D9.1 y D7.2, contrato §8.8 y §6.6 | **Vigente** |
| `reportes_20261112/` | Reportes derivables de U7.5, desarrollo («usado para elegir») y prueba, todos desde predicciones guardadas: `clasificacion_tau.csv` (§7.3, modelo y B2 en τ = 0,05), `murphy.csv` y `confiabilidad.csv` (§7.2, receta y ablación, 10 intervalos de igual frecuencia, global y por año), `calibracion_anual.csv`, `eventos_razon.csv` (§7.4, modelo y canal, global y por municipio), `duracion_alertas.csv` (descriptivo, fuera del contrato), `habilidad.csv`, `casillas.csv`, `desarrollo_vs_prueba.csv`, `curva_amoc.csv` y `manifiesto_reportes.json` | Receta sin clima; ablación completa | 7c2267c (generado desde 176f89d, árbol limpio); fórmulas e identidades congeladas en 37406f7, cifras en fbb51d8 | Contrato §7.2–§7.4 y §8.4 | **Vigente** |
| `exploratorio_20261112/` | Tablas del análisis integrado (`docs/forecasting/analisis_resultados.md`): clasificación de la etiqueta en τ = 0,05, en el τ de F1 máxima de desarrollo (0,30) y en 0,50, y sobre la rejilla de τ, con la persistencia como referencia (`clasificacion_umbrales.csv`, `clasificacion_curva_tau.csv`); desempeño por zona del canal en t (`zona_canal*.csv`); Murphy de B2 con dos agrupamientos y AP con empates (`murphy_b2.csv`, `ap_empates.csv`); fases, semanas desde el último episodio, estratos y episodios por municipio, rachas (`fases*.csv`, `semanas_desde_episodio.csv`, `estratos_municipio.csv`, `episodios_entrenamiento.csv`, `rachas_positivas.csv`); detalle de inicios y tasa base de alertas (`inicios_detalle.csv`, `tasa_base_alerta.csv`); contraste independiente de U7.5 (`contraste_u75.csv`, 480 de 480 cifras coinciden) y `manifiesto_exploratorio.json` (md5 de los insumos) | Receta sin clima: desarrollo con la configuración común fuera de año, prueba con `predicciones_prueba.parquet` | Generado con `docs/forecasting/verificacion/analisis_integrado_exploratorio.py` sobre 03f5970; commit pendiente | Contrato §8.8 (D9.1) | **Exploratorio** |

**Notas de `reportes_20261112/` (convenciones, sin interpretación):**
- **Insumos.** En desarrollo, la receta usa la configuración común fuera de año de la búsqueda sin clima y la ablación la de la búsqueda completa (2019, 2021 y 2022 `logistica_03`; 2020 `logistica_04`). B2 y B1 son las líneas base guardadas. En prueba se usan los cuatro sistemas de `predicciones_prueba.parquet`.
- **Curva AMOC de desarrollo.** Tiene dos configuraciones: `eleccion_tau_receta`, con `logistica_05` en los cuatro años (la curva que eligió τ: 22 falsas alarmas en 0,05), y `comun_fuera_de_anio`, que es la de las demás tablas (24). La de prueba es descriptiva y no cambia τ (§7.5).
- **Razón de alertas.** Es la de `events._summarize`: inicios evaluables avisados por alertas juzgables, divididos por las alertas juzgables. Cuenta inicios, así que puede superar 1. `alertas_que_avisan` y `proporcion_falsas_alarmas` son informativas.
- **Los dos agregados de casillas pueden diferir en signo.** En desarrollo, el BSS de la ablación frente a B2 es −0,084 como promedio de casillas y +0,070 con las filas juntas.
- **Clasificación en τ.** Sensibilidad, especificidad, VPP y MCC son NaN cuando su denominador es 0.
- **Residuo de Murphy.** Es WBV − 2·WBC (Stephenson, Coelho y Jolliffe, 2008).

## Figuras (`reports/forecasting/figures/`, git)

| Carpeta | Contenido | Commit | Estado |
|---|---|---|---|
| `20261112/` | PNG y SVG de U7.5: `confiabilidad_*`, `amoc_*` y `mapa_bss_*` de desarrollo («usado para elegir») y de prueba; `serie_<código>` 2019–2025 por municipio (p̂ y alertas en la semana de emisión t, referidos a t + 8, con las alertas en una franja propia); `resumen_municipios` (sin punto de detección donde no hay inicios evaluables) | Regeneradas desde 300cb08 (primera versión en 7c2267c, desde 176f89d); las tablas son idénticas byte a byte y el manifiesto registra 300cb08 | **Vigente** |
| `exploratorio_20261112/` | PNG y SVG del análisis integrado: `clasificacion_pr`, `clasificacion_umbral`, `zona_canal`, `fases_aporte`, `fases_municipio`, `semanas_desde_episodio`, `inicios_prueba` y `regimen_casillas`; rótulo [X] en el título | Mismo script y commit que `evidence/exploratorio_20261112/` | **Exploratorio** |

## Entregables (`reports/forecasting/entregables/`, git)

| Carpeta | Contenido | Fuente | Estado |
|---|---|---|---|
| `analisis_20261112/` | Análisis integrado en Word (`analisis_resultados.docx`) y presentación de referencia para el equipo (`presentacion_resultados.pdf`), con su `README.md` | `docs/forecasting/analisis_resultados.md`; evidencia `prueba_20261112/`, `reportes_20261112/` y `exploratorio_20261112/` | **Vigente** (versión de lectura; la fuente es el Markdown) |

## Configuración

| Archivo | Contenido | Estado |
|---|---|---|
| `configs/receta_congelada.yaml` | Receta de D7.2 aprobada por el responsable de la implementación el 2026-11-12: bytes del borrador de `seleccion_conjuntos_20261112/` más aprobación y hash `de74d4abd33537433ffc9d1572c6cdd8620a2608fcc2b67ee4a1752aa865fda5`. Logística sin clima `logistica_05` (C = 3) en los seis municipios, τ = 0,05, k = 2, W = 12; ablación: logística completa `logistica_03` (C = 0,3) | **Vigente** |
| `seleccion_20261112/receta_congelada_sustituida_d2135a39.yaml` | Receta anterior (logística completa, `logistica_03`, τ = 0,10), retirada antes de abrir la prueba (DV-02) | **Sustituida**; registro |
