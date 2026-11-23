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
- **Registro de modelos de MLflow.** La API no lee esta carpeta: cargará cada modelo desde el registro de modelos de MLflow (sección siguiente).
- **Regeneración.** Los archivos no se reescriben a mano ni se reentrenan aquí. Para reproducirlos se ejecuta `src.forecasting.artifacts` sobre una copia de las salidas de la prueba en una carpeta nueva y se comparan los md5 (README raíz, §6).

## Registro en MLflow

Los seis modelos están registrados en el servidor del equipo desde el 2026-11-20, cada uno en versión 1:

| Código | Modelo registrado | Versión | Run en `dengue_h8_final` | Modelo guardado del run |
|---|---|---|---|---|
| 08001 | `DengueAlertas_Barranquilla` | 1 | `3f7704358d7b48e192074876fca5158d` | `m-8de1cdfebabf46a68906b24b5762b216` |
| 08758 | `DengueAlertas_Soledad` | 1 | `9fd51150bbbb46979ddc51b752891c77` | `m-dbb6b128eca8409e82c747ea0673f8a9` |
| 41001 | `DengueAlertas_Neiva` | 1 | `00160760df3b450ead918199b9f9b27e` | `m-6c0f07408b1245158d473aec8a0bc3ad` |
| 50001 | `DengueAlertas_Villavicencio` | 1 | `923678add194465b9f39c4faf3b72fe8` | `m-c0222e6a8e9440c9ba85c5603f2c8721` |
| 54001 | `DengueAlertas_Cucuta` | 1 | `f22036f1a0d147c694f3f60b34b4bd4d` | `m-4b0aa2fd53ff47fe9294d27d95cf3e5d` |
| 76001 | `DengueAlertas_Cali` | 1 | `4dbbc1b4b4724c11b7352b9ddc4e13ce` | `m-cb9a1cbe6bae474a81f94b5763a458b7` |

Los nombres usan el nombre del municipio en ASCII, sin tildes (`DengueAlertas_<Municipio>`, especificación §9).

**Actualización 2026-11-23.** Los seis modelos tienen además una **versión 2** creada como copia exacta de la v1: mismo `source` (`models:/m-…`), mismo `run_id` y mismas probabilidades, sin reentrenamiento. Cada modelo quedó con `staging` → v1 (etapa `Staging`) y `production` → v2 (etapa `Production`). La promoción se hizo con el cliente de MLflow sobre el SQLite del servidor y luego `POST /sync` a la API de Producción; no se usó el webhook para no archivar la v1. La sección «Habilitación en F8» de abajo describe el estado original del registro y ya no refleja los alias actuales.

**Ruta de registro.** El servidor no admite escrituras en el registro de modelos con el token del equipo (`PERMISSION_DENIED`): el registro oficial se hace desde la interfaz de MLflow, a partir del modelo guardado en cada run. Por eso el registro tiene tres pasos (DV-08 en `docs/forecasting/desviaciones.md`):

1. **Runs por script.** `scripts/registrar_modelos_finales.py --solo-runs` crea un run por municipio en el experimento `dengue_h8_final`. El script primero verifica los md5 de esta carpeta y la coherencia del manifiesto con la receta, y no llama al registro de modelos. Cada run lleva:
   - parámetros: familia, candidato, conjunto de variables, C, `max_iter`, filas y fechas de entrenamiento;
   - etiquetas: las de la especificación §9, más el código DANE y el md5 del archivo;
   - el `.joblib` original, en `joblib_original/`;
   - el modelo guardado con `mlflow.sklearn.log_model` (serialización cloudpickle), con firma de entrada de las seis variables `sin_clima` y como ejemplo de entrada la emisión del 2025-12-22.

   El script es idempotente: reutiliza el run cuya etiqueta `md5` coincide con el archivo. El run de 08001 se creó en un intento anterior de registro directo y registra el commit `cae96bf`; los otros cinco, `d42cc0e`. El contenido del modelo es el mismo.
2. **Registro desde la interfaz.** Cada modelo guardado se registró manualmente con su nombre, sin etiquetas ni alias.
3. **Verificación.** `scripts/registrar_modelos_finales.py --verificar` es de solo lectura. Comprobó, para cada modelo:
   - que existe la versión 1;
   - que proviene del run con el md5 del archivo local y de su modelo guardado;
   - que, cargada desde `models:/<nombre>/1`, da probabilidades idénticas bit a bit a las del `.joblib` en las ocho emisiones del 2025-11-03 al 2025-12-22. Lo comprobó por la ruta de `mlflow.sklearn` y por la ruta `pyfunc` que usa `src/api.py`.

**Estado: registrados, sin habilitar.** Sin la etiqueta `enabled` y sin alias, los seis modelos son invisibles para la API actual: `src/api.py` solo descubre los modelos `DengueAlertas_*` con `enabled = true`. Se dejan así porque la API todavía no está adaptada a estas variables, así que podría cargarlos y fallar antes de F8. La habilitación y los alias forman parte del despliegue de F8.

### Habilitación en F8

El responsable del despliegue pone estas etiquetas en cada **modelo registrado** (no en la versión). `src/api.py` lee `enabled`, `municipality_code`, `department` y `depto_id`; `depto_id` es «CO» más el código ISO 3166-2 del departamento, como en el mapa del frontend.

| Modelo registrado | `enabled` | `municipality_code` | `department` | `depto_id` |
|---|---|---|---|---|
| `DengueAlertas_Barranquilla` | `true` | `08001` | `Atlántico` | `COATL` |
| `DengueAlertas_Soledad` | `true` | `08758` | `Atlántico` | `COATL` |
| `DengueAlertas_Neiva` | `true` | `41001` | `Huila` | `COHUI` |
| `DengueAlertas_Villavicencio` | `true` | `50001` | `Meta` | `COMET` |
| `DengueAlertas_Cucuta` | `true` | `54001` | `Norte de Santander` | `CONSA` |
| `DengueAlertas_Cali` | `true` | `76001` | `Valle del Cauca` | `COVAC` |

La especificación §9 propone además `receta_hash` (`de74d4abd33537433ffc9d1572c6cdd8620a2608fcc2b67ee4a1752aa865fda5`) y `familia` (`logistica`); la API no los lee.

Orden:

1. **Alias `staging`** en la versión 1. Según `src/webhook_api.py`, crear el alias pasa la versión a la etapa Staging (y archiva la versión que estuviera en esa etapa en el mismo modelo) y sincroniza la API del ambiente de pruebas.
2. **Prueba** de las vistas en el ambiente de pruebas.
3. **Alias `production`**, que la pasa a la etapa Production y sincroniza la API de producción.

**Precaución.** `src/api.py` carga cada modelo habilitado por el alias igual a `STAGE`. Si ese alias no existe, recurre a la etapa clásica y después a la **última versión**. Por eso, con `enabled = true`, la API de producción cargaría la versión 1 aunque todavía no tenga el alias `production`. Antes de habilitar, F8 decide si elimina ese último recurso o acepta que ambos ambientes carguen los modelos al habilitarlos.

**Dependencias de la API (F8).** `requirements/serve.txt` debe fijar `scikit-learn==1.7.2`, la versión con la que se serializaron los modelos, y `mlflow==3.17.0`, la versión del servidor. Hoy fija `scikit-learn>=1.3.0` y `mlflow>=2.0.0`. El cambio lo hace el responsable del despliegue, junto con el `Dockerfile`.
