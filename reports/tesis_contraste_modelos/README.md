# Contraste de familias de modelos (desarrollo 2019–2022)

## Estado

Exploración complementaria en el periodo de desarrollo. **No reemplaza la receta vigente** (`configs/receta_congelada.yaml`), no participa en la selección preregistrada (F6) y no usa el periodo de prueba (2023–2025). Las cifras de esta carpeta no son comparables sin más con las de la receta: el método de selección es distinto (ver «Método de selección»). El análisis que las sitúa frente a la receta vigente está en `docs/forecasting/analisis_complementario.md`.

Configuración: `configs/contraste_familias.yaml` (validada por `load_contrast_config` en `src/forecasting/config.py`). Todas las corridas usan el conjunto de variables `sin_clima` (contrato §6.6, D7.2), salvo la ablación climática de las tablas 02 y 06.

## Contenido

| Archivo | Contenido | Script |
|---|---|---|
| `tabla_01_ranking_global_sin_clima.{csv,md}` y `fig_01_ranking_familias_b2` | Mejor candidato de cada una de las ocho familias: mejora media de log-loss frente a B2 (%) y su desviación estándar entre casillas, log-loss y Brier medios, casillas ganadas a B2 (de 24), corrida de origen y parámetros | `build_thesis_report.py` |
| `fig_02_calibracion_logloss_brier` | Log-loss medio frente a Brier medio de los ocho ganadores | `build_thesis_report.py` |
| `tabla_03_desglose_municipal.md` y `fig_03_heatmap_municipal` | Mejora media frente a B2 (%) por municipio y familia; la figura añade los años ganados (de 4) | `build_thesis_report.py` |
| `tabla_04_detalle_anual_top4.md` | Detalle anual de las cuatro primeras familias. **Vacía en la versión versionada** (ver «Diferencias conocidas») | `build_thesis_report.py` |
| `tabla_02_ablacion_clima_dl.md` y `fig_04_ablacion_clima_dl` | GRU y CNN-1D con 5 canales (sin clima) y con 8 canales (con las tres series climáticas); mejor candidato de cada variante | `build_thesis_report.py` |
| `tabla_05_sensibilidad_horizonte.{csv,md}` y `fig_05_sensibilidad_horizonte` | Los ocho ganadores de la tabla 01, con sus hiperparámetros fijos, re-puntuados en h = 2, 4, 8 y 12 con B2 recalculada para cada h | `run_horizon_climate_mlflow.py` |
| `tabla_06_clima_por_horizonte.{csv,md}` y `fig_06_clima_por_horizonte` | Seis familias tabulares con `sin_clima` y con el conjunto completo, por horizonte; diferencia en puntos porcentuales | `run_horizon_climate_mlflow.py` |

Las figuras están en PNG y PDF. Todas las cifras son de desarrollo: 6 municipios × 4 años (2019–2022) = 24 casillas, con el mismo esquema anual que la búsqueda preregistrada (`splits.yearly_trials`, corte de entrenamiento D7.1) y la misma B2 por casilla.

## Corridas

Cada corrida es una carpeta con una tabla única de desarrollo (`tabla_unica_desarrollo.parquet`, fuera de git); su nombre es el de la carpeta y aparece en la columna `corrida` de la tabla 01.

| Corrida | Familias | Origen |
|---|---|---|
| `sin_clima` | Logística (6 valores de C) y LightGBM (25 candidatos) | Búsqueda preregistrada de F7 (`data/processed/forecasting/sin_clima/`, DVC) |
| `logistica_ampliada` | Logística con 19 valores de C, de 0,001 a 100 | `run_contrast_search.py` |
| `cb_xgb` | CatBoost y XGBoost, 25 candidatos | `run_contrast_search.py` |
| `svm_ampliado` | SVM (núcleo rbf), CatBoost y XGBoost, 40 candidatos | `run_contrast_search.py` |
| `mlp_sin_clima` | MLP tabular, 40 candidatos | `run_contrast_search.py` |
| `dl_secuencial` | GRU y CNN-1D, 25 candidatos por familia y variante | `run_seq_dl.py` |

Las corridas del contraste se registraron en MLflow en el experimento con ID `11` (`experimento_registrado_id`). Las corridas nuevas usan el experimento `dengue_h8_contraste_familias`. No se borra ni se renombra ningún experimento o corrida existente.

## Método de selección

Determinado leyendo el código que generó las tablas (`build_thesis_report.py`, `run_seq_dl.py` y `run_horizon_climate_mlflow.py`); no se re-ejecutó nada.

- **Dentro de muestra.** Para cada familia y variante, el ganador es el candidato con la mayor mejora media de log-loss frente a B2 sobre las 24 casillas de desarrollo (media simple de `mejora_b2 = 1 − logloss / logloss_B2` por casilla; con cuatro años por municipio equivale a promediar primero por municipio). El empate se rompe con el menor log-loss medio. Solo compiten candidatos con las 24 casillas completas. Las cifras de la tabla 01 (mejora, desviación estándar, log-loss, Brier y casillas ganadas a B2) se calculan sobre **las mismas 24 casillas** con que se eligió el candidato. Son, por tanto, estimaciones optimistas del desempeño fuera de muestra, y el sesgo crece con el número de candidatos de la familia (logística: 25 valores de C entre sus dos corridas; CatBoost y XGBoost: 65 cada una; SVM y MLP: 40; LightGBM, GRU y CNN-1D: 25).
- **Candidatos de varias corridas.** Una familia presente en varias corridas compite con todos sus candidatos juntos (logística: `sin_clima` y `logistica_ampliada`; CatBoost y XGBoost: `cb_xgb` y `svm_ampliado`).
- **Horizonte y clima.** Las tablas 05 y 06 no vuelven a elegir: re-puntúan en cada horizonte los ganadores elegidos en h = 8 con las mismas 24 casillas. La variante con clima de la tabla 06 usa los hiperparámetros elegidos sin clima.
- **Diferencia con la receta vigente (R0).** La receta elige la configuración común **fuera de año**: para cada año y de desarrollo, la configuración se elige con los otros tres años (media por municipio y luego entre municipios; empate a la menor posición de la rejilla) y se puntúa en y; ninguna casilla interviene en su propia elección (`comparison.out_of_year_scores`, contrato §6.2). Las cifras de control de R0 en desarrollo (mejora media 0,2109; log-loss 0,3145 frente a 0,3854 de B2; 16 de 24 casillas ganadas a B2; C = 3 con los cuatro años) son fuera de año y se reproducen exactamente con el código integrado (`reports/forecasting/evidence/reproduccion_r0_20261120/`). Las de la tabla 01 son dentro de muestra y no se comparan directamente con ellas.

## Cómo reproducir

Desde la raíz del repositorio, con el `.venv` y las dependencias opcionales instaladas:

```bash
.venv/Scripts/python.exe -m pip install -r requirements/contraste.txt
.venv/Scripts/python.exe -B scripts/run_contrast_search.py --corrida logistica_ampliada
.venv/Scripts/python.exe -B scripts/run_contrast_search.py --corrida cb_xgb
.venv/Scripts/python.exe -B scripts/run_contrast_search.py --corrida svm_ampliado
.venv/Scripts/python.exe -B scripts/run_contrast_search.py --corrida mlp_sin_clima
.venv/Scripts/python.exe -B scripts/run_seq_dl.py
.venv/Scripts/python.exe -B scripts/log_seq_dl_mlflow.py
.venv/Scripts/python.exe -B scripts/build_thesis_report.py
.venv/Scripts/python.exe -B scripts/run_horizon_climate_mlflow.py
```

Todos los scripts leen `--config` (por defecto `configs/contraste_familias.yaml`) y la configuración base que esta nombra (`configs/forecasting.yaml`). `run_contrast_search.py` reutiliza `search.run_search` sin cambios (mismas filas, casillas, líneas base, manifiesto y registro en MLflow), con la sección de búsqueda de la corrida. `run_horizon_sensitivity.py` es la primera versión del análisis de horizonte (cinco familias); las tablas versionadas vienen de `run_horizon_climate_mlflow.py`. Las salidas no se sobrescriben: para regenerar, usar carpetas nuevas.

Dependencias opcionales (`requirements/contraste.txt`): `catboost` y `xgboost` (familias tabulares) y `torch` (GRU y CNN-1D). El pipeline principal, la API y la suite de pruebas no las requieren: `src/forecasting/models.py` las importa solo al construir esas familias. Las versiones con que se generaron las tablas no quedaron registradas.

Los parámetros de los cinco ganadores tabulares se regeneran exactamente desde `configs/contraste_familias.yaml` (prueba `tests/forecasting/test_contraste_familias.py`). En la corrida `cb_xgb`, los hiperparámetros que la lista vigente de `models.py` exige y que esa corrida no muestreó figuran en `fijos` con el valor por defecto de la biblioteca (XGBoost: `gamma = 0`, `reg_alpha = 0`; CatBoost: `min_data_in_leaf = 1`, `random_strength = 1`, `rsm = 1`). Con el núcleo rbf, `degree` y `coef0` de SVM se muestrean pero no afectan al modelo.

## No determinismo de las familias secuenciales

GRU y CNN-1D fijan la semilla de PyTorch (inicialización y orden de los lotes), pero los resultados en GPU (CUDA) y en CPU no son idénticos bit a bit, y algunos núcleos de CUDA no son deterministas entre ejecuciones. Una regeneración puede dar cifras distintas de las versionadas para estas dos familias. Las familias tabulares fijan `random_state` y un hilo, y son reproducibles en el mismo entorno.

## Normalización de nombres de corrida

La columna `corrida` de la tabla 01 (CSV y Markdown) contenía, en seis filas, un prefijo con un nombre propio delante del nombre de la carpeta. Se eliminó ese prefijo; ningún otro valor cambió. Las rutas por defecto de los scripts se cambiaron en el mismo sentido: las carpetas de las corridas del contraste están ahora bajo `data/processed/forecasting/contraste/` y se declaran en `configs/contraste_familias.yaml`.

## Diferencias conocidas

- **Tabla 04 vacía.** La versión de `build_thesis_report.py` que la generó agrupaba por la columna `year`, que solo tienen las tablas secuenciales; las tabulares la llaman `anio`. Como ninguna familia secuencial está entre las cuatro primeras, la tabla quedó sin filas. El script corregido renombra `anio` a `year` al cargar; la tabla no se regeneró.
- **Tabla 05 frente a tabla 01 en h = 8.** Cinco familias coinciden exactamente (logística, SVM, MLP, CatBoost y LightGBM). Tres no: GRU (19,46 % frente a 15,68 %, 17 casillas en ambas), XGBoost (10,59 % y 13 casillas frente a 9,22 % y 14) y CNN-1D (−4,62 % frente a −3,77 %, 11 casillas en ambas). En las secuenciales es compatible con el no determinismo descrito arriba; para XGBoost no se determinó la causa. No se re-ejecutó nada para diagnosticarlo.
- **Lectura de tablas en la versión anterior.** `build_thesis_report.py` recorría todas las tablas únicas bajo `data/processed/forecasting/` y etiquetaba como `sin_clima` las que no tenían columna de variante, entre ellas la de F5b (conjunto completo). Las tablas de las búsquedas preregistradas no declaradas no aportan un ganador: en F5b, la mejor mejora media es 2,21 % (logística) y −1,94 % (LightGBM); en `ablacion_clima`, 22,35 % (logística), por debajo del 22,45 % de `logistica_ampliada`. Restringir la lectura a las corridas declaradas no cambia la tabla 01.
