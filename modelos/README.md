# Modelos finales

Artefactos del modelo final del pronóstico: un modelo por municipio piloto que estima la probabilidad de que el municipio esté en **zona epidémica** del canal de epiCo 8 semanas después de la semana de emisión t (`target_u1e` en t + 8).

Esta carpeta es una **copia de entrega** de la salida DVC `data/processed/forecasting/prueba/modelos_finales/`, que sigue siendo la fuente de verdad (ver «Relación con DVC y MLflow»).

## Qué son

- **Familia.** Regresión logística con penalización L2 (`sklearn.linear_model.LogisticRegression`, C = 3, `max_iter` = 5000, sin pesos de clase), una por municipio.
- **Receta.** La receta congelada [`configs/receta_congelada.yaml`](../configs/receta_congelada.yaml), aprobada el 2026-11-12 antes de la prueba única, con hash SHA-256 `de74d4abd33537433ffc9d1572c6cdd8620a2608fcc2b67ee4a1752aa865fda5`. El mismo hash figura en `manifiesto.json` (`receta_hash`).
- **Variables.** Conjunto `sin_clima` (contrato §5.2 y §6.6): `zona_t`, `posicion_4s`, `crecimiento_4s`, `cambio_umbral`, `sem_sin` y `sem_cos`.
- **Entrenamiento.** Cada modelo se ajustó con todas las filas de su municipio desde 2012 (contrato §6.1): 711 filas, con última emisión 2025-10-27 y última semana objetivo 2025-12-22. Lo produjo `python -m src.forecasting.artifacts` (`train_final_models` y `write_final_models`) después de la prueba única. Las cifras de desempeño de la receta no corresponden a estos modelos, sino a las predicciones de origen móvil de la prueba (evidencia en `reports/forecasting/evidence/prueba_20261112/`).
- **Objeto guardado.** Un `sklearn.pipeline.Pipeline` serializado con `joblib`:
  1. `preparar`: `ColumnTransformer` que estandariza las cinco variables numéricas (`StandardScaler`, ajustado con las filas de entrenamiento) y codifica `zona_t` con dos indicadores (`alerta`, `epidemia`; categoría de referencia `seguridad`). Descarta cualquier otra columna.
  2. `modelo`: `LogisticRegression`, con clases `[0, 1]`.

## Archivos

| Archivo | Municipio | Tamaño (bytes) | md5 |
|---|---|---|---|
| `08001.joblib` | Barranquilla | 4.321 | `400b5088ec5ced8ff7926499e2ea693b` |
| `08758.joblib` | Soledad | 4.321 | `3661d54e3894e1da024085866424da0b` |
| `41001.joblib` | Neiva | 4.321 | `2635b8756a79238e740fa8f6d48ab86e` |
| `50001.joblib` | Villavicencio | 4.321 | `b6a4fb3d4359f9b82b64a94afa9347a9` |
| `54001.joblib` | San José de Cúcuta | 4.321 | `923c437fa5be3af827dfec08d59efbd6` |
| `76001.joblib` | Cali | 4.321 | `0f6d341f588b50ba5e42f36e1277b00d` |
| `manifiesto.json` | Los seis | 2.002 | `5733be0fbf7f064bbccbe80c5061c7ce` |

Los md5 coinciden con los del directorio `modelos_finales` en la caché DVC (descriptor `data/processed/forecasting/prueba/modelos_finales.dvc`, md5 `fca43e9e6b335ef5ea69535d113ba03f.dir`). `manifiesto.json` registra, por municipio, la familia, el candidato (`logistica_05`), el conjunto de variables, los hiperparámetros, las filas y las semanas de la última emisión y del último objetivo. `.gitattributes` marca la carpeta como `-text` para que git no altere los bytes.

## Entrada que esperan

Una tabla `pandas.DataFrame` con una fila por emisión (municipio, t) y las variables del pipeline, calculadas solo con semanas ≤ t:

| Variable | Definición (contrato §5.2) |
|---|---|
| `zona_t` | Zona del canal en t: `seguridad`, `alerta` o `epidemia` |
| `posicion_4s` | Promedio de log(C_s + 1) − log(U1e_s + 1), s = t − 3 … t |
| `crecimiento_4s` | log(C_t + 1) − log(C_{t−4} + 1) |
| `cambio_umbral` | log(U1e_{t+8} + 1) − log(U1e_t + 1) |
| `sem_sin`, `sem_cos` | Seno y coseno de 2π·(w_{t+8} − 1)/52,18, con w_{t+8} la semana ordinal de la semana objetivo |

Las variables no se calculan a mano: las construye `src.forecasting.dataset.build_features` a partir de las quince columnas del panel (conteos completos y clima), recalculando el canal con las semanas de referencia de cada t. Esa función devuelve las quince variables del contrato; el pipeline selecciona por nombre las seis que usa, de modo que el orden y las columnas adicionales no importan. `src.forecasting.models.predict_positive` exige las quince columnas, comprueba que sean finitas y que `zona_t` sea una zona válida, y devuelve p̂ = P(`target_u1e` = 1 en t + 8).

Las semanas de emisión que corresponden al modelo final son las posteriores a su última emisión de entrenamiento: del 2025-11-03 al 2025-12-22, la última semana completa de la base (`docs/forecasting/api_salida_d10.md` §4). Para semanas anteriores, las probabilidades fuera de muestra son las del historial precalculado.

## Ejemplo mínimo

Desde la raíz del repositorio, con el producto del canal descargado (`dvc pull`):

```python
from datetime import date

import joblib

from src.epidemiology.validation import SOURCE_SCHEMA
from src.forecasting.config import load_config
from src.forecasting.dataset import FEATURES, build_features, load_channel_product
from src.forecasting.models import predict_positive

config = load_config()   # configs/forecasting.yaml
product = load_channel_product(config.datos.canal, expected_md5=config.datos.version_dvc_canal)
rows = build_features(product.select(SOURCE_SCHEMA.names), [("41001", date(2025, 12, 22))],
                      h=config.horizonte.h, canal=config.canal, features=config.features)

model = joblib.load("modelos/41001.joblib")
p = predict_positive(model, rows[list(FEATURES)])[0]
print(f"P(zona epidémica en la semana objetivo) = {p:.4f}")
```

El estado de alerta que ve el usuario no es la probabilidad sola: lo calcula `src.forecasting.events.product_alert_states` con las emisiones hasta t, la zona observada, τ y k de la receta y la regla `producto.regla_alerta` de `configs/forecasting.yaml`. El recorrido completo está en [`notebooks/03_uso_del_modelo.ipynb`](../notebooks/03_uso_del_modelo.ipynb).

## Requerimientos y tiempos de inferencia

Solo CPU; no requieren GPU. Dependencias: las de `requirements/forecasting.txt` (scikit-learn 1.7.2, la versión con la que se serializaron; otra versión de scikit-learn puede no cargar los archivos o emitir advertencias). Los tiempos medidos están en el [README raíz](../README.md), §7.

## Relación con DVC y MLflow

- **DVC (fuente de verdad).** Los modelos son una salida del pipeline y se versionan con DVC junto a la prueba única de la que dependen (especificación §4): `data/processed/forecasting/prueba/modelos_finales/`, con su descriptor `.dvc` en git. Si esta copia y la salida DVC difieren, vale la salida DVC. La prueba `tests/forecasting/test_modelos_entrega.py` compara las dos copias byte a byte cuando la salida DVC está disponible localmente y se omite en caso contrario; también comprueba, siempre, que el manifiesto corresponde a la receta congelada.
- **Registro de modelos de MLflow.** La API no lee esta carpeta: cargará cada modelo desde el registro de modelos de MLflow, con el nombre `DengueAlertas_<Municipio>` (por ejemplo, `DengueAlertas_Neiva`; especificación §9). El registro de estos seis modelos es parte de F8 y todavía no se ha hecho.
- **Regeneración.** Los archivos no se reescriben a mano ni se reentrenan aquí. Para reproducirlos se ejecuta `src.forecasting.artifacts` sobre una copia de las salidas de la prueba en una carpeta nueva y se comparan los md5 (README raíz, §6).
