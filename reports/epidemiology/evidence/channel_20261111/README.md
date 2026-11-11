# Caracterización descriptiva del canal de epiCo

Unidad intermedia antes de F2: no es una fase del contrato. Solo describe el producto del canal;
no calcula ninguna métrica de la alerta frente a los inicios (eso es F4, contrato §10.15) y no lee el parquet de legado.

## Procedencia

- Producto: `data/processed/epidemiology/pilot_weekly_channel.parquet`, md5 `99fd12311a261264019645d906631cfc` (igual a `datos.version_dvc_canal`; el script falla si no coincide).
- Commit base del código: `79f12c76a448f75cabca6ab705f88d3324db1de2` (el script tenía cambios sin commit al generar).
- sha256 del script `scripts/describe_channel_epico.py`: `bf3df5dfde4472ef6cd7db334107d333136f890a66c2981bc700bdb6fa902f72`.
- Regenerar (desde la raíz del repositorio):

```bash
.venv/Scripts/python.exe scripts/describe_channel_epico.py --out-dir reports/epidemiology/evidence/channel_20261111
```

## Archivos

| Archivo | Contenido |
|---|---|
| `channel_main_six_municipalities.png` / `.svg` | Figura principal: seis paneles, semanas elegibles 2012–2025, eje x compartido, eje y independiente, lineal y sin recorte. Zonas, central, U1e, casos, inicios de brote (31) y franja de desarrollo/prueba solo como referencia. PNG a 300 dpi. |
| `channel_detail_<código>_<municipio>.png` / `.svg` | Un detalle por municipio: escala lineal y log(C + 1). |
| `channel_by_municipality_year.csv` | Por municipio y año (y fila `total`): semanas elegibles, positivas, prevalencia, proporción en cada zona, rachas (por año de inicio) e inicios de brote. Los totales reproducen §6.F1: 4.380 semanas, 945 positivas, 175 rachas, 31 inicios; zonas 0,379 / 0,405 / 0,216. |
| `channel_width_by_municipality.csv` | Anchura del canal: mediana y cuantiles 10 y 90 de U1e/central, semanas con central = 0 y con central < 1. |

## Definiciones

- Racha: bloque máximo de semanas elegibles consecutivas con `target_u1e = 1`.
- Inicio de brote (Umaña et al., U1e): primera semana de una racha de ≥ 3 semanas precedida de ≥ 2 semanas con 0. Misma definición que la prueba de aceptación de F1.
- Bandas: seguridad de 0 a la central, alerta de la central a U1e, epidemia por encima de U1e. Cada semana se dibuja con un valor (escalón), sin interpolar.
- Franja superior de cada panel: desarrollo 2019–2022 y prueba 2023–2025, solo como referencia visual.

## Lecturas del canal

- Central nunca es 0 en las semanas elegibles (mínimo 0.1487); ver `weeks_central_below_1` en la tabla de anchura.
- El canal se calcula por ordinal de semana con los cinco años anteriores, por lo que U1e es irregular de una semana a otra. Mediana de |Δ log(U1e + 1)| entre semanas consecutivas:

| Municipio | Con cambio de año | Dentro del año |
|---|---|---|
| Cúcuta | 0.145 | 0.142 |
| Barranquilla | 0.184 | 0.215 |
| Neiva | 0.22 | 0.154 |
| Villavicencio | 0.365 | 0.164 |
| Cali | 0.276 | 0.121 |
| Soledad | 0.376 | 0.227 |
