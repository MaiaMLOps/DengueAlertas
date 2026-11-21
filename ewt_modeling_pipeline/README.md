# DengueAlertas — aporte final de modelamiento EWT

Pipeline ejecutable de punta a punta para el panel semanal de seis municipios: validación del dato, canal EWT, episodios, 66 variables, entrenamiento local, validación rolling-origin purgada, prueba 2023–2025, reporte y publicación opcional en MLflow.

## Decisión metodológica central

Se modelan dos preguntas, siempre a **8 semanas**:

1. **Principal — `zone_future`:** ¿estará el municipio en zona epidémica en `t+8`? La etiqueta es la bandera EWT de la semana objetivo.
2. **Sensibilidad — `pre_onset`:** ¿iniciará un episodio en las siguientes 8 semanas, estando fuera de brote/refractario? Esta etiqueta se conserva porque forma parte de la metodología previa, pero puede quedar sin ejemplos evaluables cuando las rachas de exceso son largas.

El modelo es **local por municipio**. No se mezclan observaciones entre ciudades. La ingeniería de variables compara tres ablaciones: `epi` (historia + canal + calendario), `epi_clima` y `epi_clima_demo`.

## Inicio rápido

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python run_pipeline.py \
  --input data/processed/pilot_municipalities_weekly.parquet \
  --config configs/pipeline.yaml \
  --output outputs/final_run
```

Sin publicación en MLflow:

```bash
python run_pipeline.py --input <panel.parquet> --output outputs/final_run --no-mlflow
```

También puede instalarse como paquete:

```bash
python -m pip install -e .
ewt-final --input <panel.parquet> --output outputs/final_run
```

## Contrato mínimo de entrada

Obligatorio:

- `municipality_code`
- `week_start` (lunes, grilla semanal sin huecos)
- `confirmed_records_screening` (entero no negativo)

Opcional pero necesario para evaluar las familias completas:

- `temperature_c`, `precipitation_mm`, `relative_humidity_pct`
- `population`, `population_density_km2`

El panel esperado contiene los DIVIPOLA 54001, 08001, 41001, 50001, 76001 y 08758. La ejecución falla ante claves duplicadas, días distintos de lunes, huecos estructurales o conteos inválidos.

## Separación temporal

- **Selección:** folds anuales 2019, 2020, 2021 y 2022.
- **Purga:** 8 semanas antes de cada validación y antes de la prueba.
- **Prueba bloqueada:** 2023–2025; no se usa para elegir algoritmo, variables, calibración ni umbral.
- **Calibración:** Platt sobre predicciones fuera de muestra de la selección.
- **Umbral:** costo falso negativo:falso positivo = 5:1, fijado antes de abrir la prueba.

Si `pre_onset` no tiene ambas clases por año, el pipeline intenta un único bloque 2019–2022, todavía purgado. Si selección o prueba siguen sin ambas clases, registra la combinación en `skipped_models.csv` y continúa.

## Modelos y métricas

Candidatos por municipio/tarea:

- regresión logística regularizada (`C=0.3`, ponderación balanceada);
- LightGBM pequeño y regularizado;
- XGBoost pequeño y regularizado;
- conjuntos `epi`, `epi_clima`, `epi_clima_demo`.

Métricas: AUC-PR, ROC-AUC, Brier, log-loss, precision, recall, especificidad, F1, matriz de confusión y rachas de alerta. Para `pre_onset` se añaden sensibilidad por episodio y antelación media. Se contrastan persistencia y climatología.

## Salidas

- `REPORTE_FINAL.md`: reporte compartible con resultados y lectura crítica.
- `results.csv`: ganadores y prueba bloqueada.
- `candidates_validation.csv`: todos los candidatos y métricas de selección.
- `baselines_test.csv`: referencias de persistencia y climatología.
- `skipped_models.csv`: combinaciones no evaluables.
- `prepared_supervised_panel.csv.gz`: panel trazable con canal, etiquetas y features.
- `episodes.csv`: episodios EWT.
- `models/<tarea>/<DIVIPOLA>/model.joblib`: bundle ejecutable.
- En cada modelo: predicciones de validación/prueba, importancias, plan de folds y métricas.
- `run_manifest.json`: SHA-256 del insumo y configuración, versiones y perfil del panel.

## MLflow: registro sin etapa

La configuración apunta al experimento `dengue-ews-ewt-h8-final`. Antes de ejecutar:

```bash
export MLFLOW_TRACKING_URI=https://platform.arkoiot.com
export CF_ACCESS_CLIENT_ID='...'
export CF_ACCESS_CLIENT_SECRET='...'
```

El flujo:

- crea un run para cada candidato evaluado;
- crea un run ganador y una versión registrada por municipio/tarea;
- **no ejecuta transición de etapa** y **no asigna alias**;
- la versión queda con etapa `None`.

`configs/pipeline.yaml` valida que `transition_stage` y `set_alias` sean `null`. Los secretos nunca se imprimen ni se guardan en artefactos. Consulte `docs/MLFLOW.md`.

## Verificación

```bash
pytest -q
```

Las pruebas cubren EWT exacto, episodios, contrato de 66 variables, causalidad temporal, métricas, umbral y prohibición de etapas/alias.

## Documentación

- `docs/METODOLOGIA.md`: formulación, etiquetas, particiones, selección y limitaciones.
- `docs/FEATURES.md`: diccionario de las 66 variables y reglas contra fuga.
- `docs/MLFLOW.md`: autenticación, estructura de runs y comprobación de etapa `None`.
- `outputs/final_run/REPORTE_FINAL.md`: hallazgos de la corrida real incluida.
