# Índice de resultados del pronóstico

Mapa de las carpetas de resultados del MVP de pronóstico (`feature/forecasting-mvp`): qué contiene cada una, con qué conjunto de variables, de qué búsqueda o commit sale, qué decisión del contrato la originó y si está vigente. Se actualiza en cada unidad. Ningún archivo versionado se mueve: las carpetas sustituidas se conservan como registro.

**Estados:** *vigente* (insumo de la receta en curso), *registro* (se conserva como evidencia; no alimenta la receta en curso), *sustituida* (reemplazada por una decisión posterior), *pendiente* (todavía no generada).

**Conjuntos de variables (contrato §6.6, D7.2):** *completo*, las 15 variables de §5.2; *sin clima*, las 6 que no son anomalías climáticas (`zona_t`, `posicion_4s`, `crecimiento_4s`, `cambio_umbral`, `sem_sin`, `sem_cos`).

**Prueba 2023–2025:** sin ejecutar. `reports/forecasting/bitacora_prueba.jsonl` no existe. La receta vigente es la de D7.2 (`configs/receta_congelada.yaml`, hash `de74d4ab…fda5`), así que la barrera ya permite puntuar 2023–2025; la ejecución única se revisa con Jesús antes de lanzarla.

## Datos (`data/processed/forecasting/`, DVC)

| Carpeta | Contenido | Conjunto | Búsqueda / commit | Origen | Estado |
|---|---|---|---|---|---|
| raíz `data/processed/forecasting/` | Búsqueda de F5b: tabla única, predicciones, líneas base y manifiesto (con `.dvc`); `partes/` y `busqueda.log` solo locales | **Completo**, logística (6) y LightGBM (25) | `7f34b69832eea69407c6`; código cd9d702; `.dvc` en 1a1ad2d | Pre-registro (especificación §7), D5–D7 | **Vigente** como conjunto completo del paso 0 (D7.2) |
| `ablacion_clima/` | Búsqueda de la logística sin clima (6 candidatos); tabla, predicciones, líneas base y manifiesto con `.dvc` (`partes/` y `busqueda.log` solo locales) | Sin clima, solo logística | `26dc1af622fb5966e56e`; código a3b274b | Ablación de F7, decisión D1 (contrato §4.4) | **Registro**: origen de D7.2; no alimenta la selección |
| `sin_clima/` | Búsqueda sin clima de las dos familias (`--variante sin_clima`), 198 de 198 unidades; tabla, predicciones, líneas base y manifiesto con `.dvc` (`partes/` y `busqueda.log` solo locales) | Sin clima, logística (6) y LightGBM (25), mismos candidatos y semillas que F5b | `ae6225180d37d025224b`; código 1f45514 | D7.2 (contrato §6.6) | **Vigente** como conjunto sin clima del paso 0; su logística es idéntica a `ablacion_clima/` |

## Evidencia (`reports/forecasting/evidence/`, git)

| Carpeta | Contenido | Conjunto | Commit | Origen | Estado |
|---|---|---|---|---|---|
| `seleccion_20261112/` | Selección de F6 (casillas fuera de año, comparaciones, curva de τ, márgenes), borrador de receta y copia de la receta congelada sustituida (`receta_congelada_sustituida_d2135a39.yaml`) | Completo | f1c0876 (evidencia y borrador); receta fc31f13; copia con DV-02 | F6, contrato §6.2, §7.5 y §8.3 | **Sustituida** por D7.2 (DV-02); se conserva como registro |
| `seleccion_conjuntos_20261112/` | Selección de D7.2: paso 0 por familia (`comparacion_conjuntos.csv`), pasos 1–4, τ, ablación (`ablacion.csv`, `comparaciones_ablacion.csv`) y borrador de receta con `conjunto_variables` y `ablacion` | Sin clima en las dos familias; ablación con el completo | ced43ff (generada desde 1f45514 con el árbol limpio); regresión congelada en 4e01794 | D7.2, contrato §6.6 | **Vigente**: su borrador es la receta congelada |
| `diagnostico_desarrollo_20261112/` | Seis insumos de §8.8 en desarrollo: logística completa con la configuración común fuera de año, τ = 0,10 | Completo | a3b274b (código caae344) | D9.1, contrato §8.8 | **Registro**: corresponde a la receta sustituida; el de la receta nueva está pendiente |

## Configuración

| Archivo | Contenido | Estado |
|---|---|---|
| `configs/receta_congelada.yaml` | Receta de D7.2 aprobada por Jesús Sierra el 2026-11-12: bytes del borrador de `seleccion_conjuntos_20261112/` más aprobación y hash `de74d4abd33537433ffc9d1572c6cdd8620a2608fcc2b67ee4a1752aa865fda5`. Logística sin clima `logistica_05` (C = 3) en los seis municipios, τ = 0,05, k = 2, W = 12; ablación: logística completa `logistica_03` (C = 0,3) | **Vigente** |
| `seleccion_20261112/receta_congelada_sustituida_d2135a39.yaml` | Receta anterior (logística completa, `logistica_03`, τ = 0,10), retirada antes de abrir la prueba (DV-02) | **Sustituida**; registro |
