@AGENTS.md

# Trabajo en el MVP de pronóstico

Este archivo añade a `AGENTS.md` las reglas del pronóstico (`src/forecasting/`). Si `AGENTS.md` contradice este archivo o el contrato predictivo, prevalecen el contrato y este archivo.

## Fuentes de verdad

| Documento | Uso |
|---|---|
| `docs/forecasting/README.md` | Guía: fases, módulos, reproducibilidad, gobernanza y glosario |
| `docs/forecasting/contrato_predictivo.md` | Qué se construye. Vinculante; los requisitos están en §10 |
| `docs/forecasting/especificacion_implementacion.md` | Cómo se construye: módulos, fases, pruebas de aceptación y configuración |
| `docs/forecasting/api_salida_d10.md` | Esquema de salida de la API |
| `docs/forecasting/registro_decisiones.md` y `desviaciones.md` | Por qué se decidió así y qué cambió durante la implementación |
| `docs/forecasting/verificacion/*.py` | Cálculos independientes de las cifras de aceptación |
| `reports/forecasting/INDICE.md` | Estado de cada carpeta de datos, evidencia y figuras |

Si la especificación contradice al contrato, manda el contrato. La metodología (D0–D10 y sus enmiendas) está cerrada: si algo resulta inviable, se detiene la parte afectada, se explica el problema y se espera la decisión del responsable de la implementación. Toda desviación aprobada se registra en `docs/forecasting/desviaciones.md` antes de continuar.

## Reglas no negociables

1. **Periodo de prueba.** Ninguna métrica de ningún sistema (modelo, B1, B2, ablación ni alerta del canal) se calcula sobre semanas objetivo de 2023–2025 sin `configs/receta_congelada.yaml` aprobado. La barrera `src/forecasting/guard.py` lo hace cumplir y no se desactiva. La prueba única ya se ejecutó: `python -m src.forecasting.evaluate` no se relanza sin una decisión del equipo (exige `--razon` y queda en `reports/forecasting/bitacora_prueba.jsonl`).
2. **Sin fuga temporal.** Ninguna variable usa semanas posteriores a t; la prueba de truncamiento lo verifica. Ante una posible fuga, detenerse e informarlo.
3. **Comparaciones pareadas.** Candidatos, líneas base y ablación usan las mismas filas y los mismos cortes.
4. **Legado congelado.** No se modifican `ewt`, `excess_label`, `pilot_weekly_labels.parquet`, `src/epidemiology/ewt.py`, `src/epidemiology/labels.py` ni los descriptores DVC existentes, porque reproducen el informe de avance y su auditoría. `src/forecasting/` no lee el Parquet de legado.
5. **Receta congelada.** `configs/receta_congelada.yaml` no se edita: su hash SHA-256 cubre todo el contenido y la barrera rechaza una receta alterada.
6. **Registro completo.** Se registran todos los candidatos, las líneas base, la ablación y las semillas, no solo el elegido.
7. **Nada se decide mirando la prueba.** No se ajustan umbrales, ventanas, k, W ni τ con datos fuera de lo que fija el contrato.

## Entorno

- **Python 3.11** del entorno virtual del repositorio, siempre por su ruta: `.venv/Scripts/python.exe` en Windows (`.venv/bin/python` en Linux o macOS). No usar el Python global ni suponer el entorno activado.
- **Dependencias:** `requirements/forecasting.txt` (directas) y `requirements/forecasting.lock.txt` (entorno verificado, con DVC). No instalar nada sin aprobación.
- **DVC:** `.venv/Scripts/dvc.exe`. `status` sin `-c` es local. Los comandos contra el remoto (`pull`, `fetch`, `push`, `status -c`) piden la contraseña por teclado y se ejecutan en una terminal interactiva. `add` y `push` requieren aprobación.
- **Credenciales:** `.env` contiene las credenciales de Cloudflare Access para MLflow (plantilla: `.env.example`). **No se lee, no se imprime y su contenido no se copia** en mensajes, código, logs ni commits; los scripts lo cargan con `python-dotenv`. Las credenciales de DVC están solo en `.dvc/config.local`.
- **MLflow:** solo a través de `src/forecasting/tracking.py`, que añade los encabezados de Cloudflare Access. Si `.env` falta o el servidor no responde, se informa y se usa el modo `local` de `configs/forecasting.yaml` (especificación §9). No se inventan credenciales ni se desactiva la verificación TLS.

## Pruebas

`pytest.ini` ya aplica `--import-mode=importlib`.

- Suite completa:

  ```bash
  .venv/Scripts/python.exe -B -m pytest tests/ -q -p no:cacheprovider
  ```

- Aceptación con datos reales (requiere `dvc pull`). Las opciones de datos solo se reconocen si se pasa `tests/acceptance` explícitamente:

  ```bash
  .venv/Scripts/python.exe -B -m pytest tests/acceptance -q -rA -p no:cacheprovider --panel-parquet data/processed/pilot_municipalities_weekly.parquet --channel-parquet data/processed/epidemiology/pilot_weekly_channel.parquet --legacy-labels-parquet data/processed/epidemiology/pilot_weekly_labels.parquet --search-dir data/processed/forecasting --sin-clima-dir data/processed/forecasting/sin_clima --prueba-dir data/processed/forecasting/prueba
  ```

  Sin la opción, una prueba con datos se omite; con la opción y sin el archivo, falla. Una prueba de aceptación omitida no cuenta como aprobada; las marcadas `sustituida_dv02` se omiten a propósito (DV-02).
- **Pruebas de aceptación** (`tests/acceptance/test_f*_*.py`): se escriben al inicio de cada fase con las cifras literales de la especificación §6, se revisan, se congelan en un commit propio y reciben una regla `Edit(...)` de denegación en `.claude/settings.json`. **Nunca se modifican para que pasen.** Si una parece errónea, se detiene el trabajo y se explica por qué. Una prueba obsoleta por una enmienda se marca como sustituida en `tests/conftest.py` y su reemplazo va en un archivo nuevo con nombre propio que no coincida con los patrones ya denegados. `tests/acceptance/conftest.py` solo puede ganar opciones nuevas (con su fixture de ruta), nunca alterar las existentes.
- **Comprobar que las pruebas congeladas no cambiaron** (ambas salidas deben estar vacías):

  ```bash
  git diff --stat HEAD -- "tests/acceptance/test_f*_*.py"
  ```

  ```bash
  git log --oneline <commit de congelamiento>..HEAD -- "tests/acceptance/test_f*_*.py"
  ```

- **Pruebas unitarias:** en `tests/forecasting/`, con datos sintéticos.
- **Datos locales:** convención de `tests/epidemiology/conftest.py`: las pruebas con datos reales se activan con una opción explícita de pytest.

## Convenciones

- **Ubicación:** código nuevo en `src/forecasting/` y `src/epidemiology/channel_epico.py`; pruebas en `tests/forecasting/` y `tests/acceptance/`. Fuera de esas rutas (documentación, `.gitignore`, `requirements/`, `configs/`, API, frontend) solo si la tarea aprobada lo incluye.
- **Nombres:** los que fija el contrato se usan tal cual (variables como `zona_t` o `posicion_4s`; columnas `epico_central`, `epico_upper`, `target_u1e`, `channel_zone`; campos de la API). Lo demás sigue el estilo existente: identificadores y docstrings en inglés, documentación en español.
- **Configuración:** los parámetros van en `configs/forecasting.yaml` y se validan con Pydantic al cargarse. Ningún parámetro del contrato se escribe en el código.
- **Datos y resultados:** los Parquet y artefactos van con DVC en `data/processed/`; la evidencia pequeña, en `reports/forecasting/evidence/<fecha>/` con excepciones explícitas en `.gitignore`. Las líneas de comandos nunca sobrescriben una salida y se niegan a correr con cambios sin commit en `src/` o `configs/`.
- **Errores:** validar las entradas y fallar con mensajes claros. No capturar excepciones para ocultarlas.
- **Commits:** formato `tipo(alcance): descripción` en español, con tipos `feat`, `fix`, `test`, `docs`, `chore` y `refactor` (por ejemplo `feat(canal): …`, `test(aceptacion): …`). Un commit por unidad de trabajo, con el requisito citado en el cuerpo (`contrato §10.7`). No se hacen commits, pushes, `dvc push`, instalaciones ni merges sin aprobación explícita.

## Cuándo detenerse

Detenerse y consultar antes de seguir si:
- la tarea exige una decisión que el contrato no fija;
- una cifra de aceptación no se reproduce;
- una prueba de aceptación parece incorrecta;
- aparece una posible fuga;
- una dependencia o un servicio externo falla;
- la tarea requiere tocar el legado.

No se corrige en silencio ni se optimizan decisiones metodológicas.
