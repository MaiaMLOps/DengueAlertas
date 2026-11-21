# Entrega

## Componentes

- Código ejecutable en `src/ewt_final/` y `run_pipeline.py`.
- Configuración reproducible en `configs/pipeline.yaml`.
- Pruebas en `tests/`.
- Documentación metodológica, features y MLflow en `docs/`.
- Corrida real verificada en `outputs/final_run/`.
- Reporte compartible en `outputs/final_run/REPORTE_FINAL.md`.

## Estado

- Pruebas: 5 aprobadas.
- Bundles cargados y verificados: 8.
- Tarea principal `zone_future`: 6/6 municipios evaluados.
- Sensibilidad `pre_onset`: 2/6 evaluados; 4/6 no evaluables por falta de ambas clases.
- MLflow remoto: no publicado en esta sesión por ausencia de la dependencia y credenciales en el entorno. El publicador está implementado para reintento sin reentrenar y no asigna etapa ni alias.
