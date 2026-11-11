@AGENTS.md

# Claude Code en DengueAlertas: rama `feature/forecasting-mvp`

Este archivo añade a `AGENTS.md` las reglas para implementar el MVP de pronóstico. Si `AGENTS.md` contradice este archivo o el contrato predictivo, prevalecen el contrato y este archivo. `AGENTS.md` se actualiza en la fase F0.

## Fuentes de verdad

| Documento | Uso |
|---|---|
| `docs/forecasting/contrato_predictivo.md` | Qué se construye. Es vinculante. En §10 están los 21 requisitos con sus pruebas |
| `docs/forecasting/especificacion_implementacion.md` | Cómo se construye: modo de trabajo, módulos, fases, pruebas de aceptación y configuración |
| `docs/forecasting/api_salida_d10.md` | Esquema de salida de la API |
| `docs/forecasting/registro_decisiones.md` | Solo cuando haga falta el porqué de una decisión |
| `docs/forecasting/verificacion/*.py` | Scripts de referencia con los que se calcularon las cifras del contrato |

Si la especificación contradice al contrato, manda el contrato. Estos documentos no se importan aquí para no cargarlos en cada sesión: lee las secciones que correspondan al planificar cada unidad. Las rutas `claude/...` que aparecen dentro del contrato equivalen a `docs/forecasting/...` (ver la especificación, §0).

**Lo que no se discute aquí:** la metodología (D0–D10) está cerrada. Si algo del contrato resulta inviable, se detiene la parte afectada, se explica el problema y se espera la decisión de Jesús. Toda desviación aprobada se registra en `docs/forecasting/desviaciones.md` antes de continuar.

## Reglas no negociables

1. **Periodo de prueba.** Ninguna métrica de ningún sistema (modelo, B1, B2 ni alerta del canal) se calcula sobre semanas objetivo de 2023–2025 sin que exista `configs/receta_congelada.yaml` aprobado por Jesús. El código lo hace cumplir (especificación §4). No se desactiva esa barrera.
2. **Sin fuga temporal.** Ninguna feature usa semanas posteriores a t; la prueba de truncamiento lo verifica. Si detectas una posible fuga, te detienes y lo informas.
3. **Comparaciones pareadas.** Todos los candidatos, las líneas base y la ablación usan las mismas filas y los mismos cortes.
4. **Legado congelado.** No se modifican `ewt`, `excess_label`, `pilot_weekly_labels.parquet`, `src/epidemiology/ewt.py`, `src/epidemiology/labels.py` ni los descriptores DVC existentes. El paquete `src/forecasting/` no lee el parquet de legado.
5. **Registro completo.** Se registran todos los candidatos, las líneas base, la ablación y las semillas, no solo el elegido.
6. **Nada se decide mirando la prueba.** No se ajustan umbrales, ventanas, k, W ni τ con datos fuera de lo que fija el contrato.

## Ciclo de trabajo por unidad

Una unidad cubre uno o dos requisitos de `contrato §10` y dura entre 30 y 90 minutos de trabajo.

1. **Plan (modo Plan).** Lee las secciones pertinentes y presenta:
   - qué vas a construir y por qué, citando la sección del contrato;
   - las funciones públicas, con sus entradas y salidas;
   - las pruebas, en lenguaje natural, con el valor esperado y su origen;
   - los riesgos y las dudas.

   No edites nada hasta que Jesús apruebe. Si pregunta, responde con el flujo real de los datos, no con una lista de archivos.
2. **Implementación.** Haz solo lo aprobado y ejecuta las pruebas.
3. **Informe de paso.** Termina siempre con esta estructura:
   - **Archivos tocados:** lista.
   - **Pruebas:** comando exacto y resultado (aprobadas, fallidas, omitidas). Una prueba de aceptación omitida no cuenta como aprobada.
   - **Referencia:** cifras obtenidas frente a las esperadas.
   - **Pruebas de aceptación intactas:** salida de `git diff --stat HEAD -- "tests/acceptance/test_f*_*.py"` y de `git log --oneline <commit de congelamiento>..HEAD -- "tests/acceptance/test_f*_*.py"`; ambas deben estar vacías. Los cambios en `tests/acceptance/conftest.py` se señalan aparte y solo pueden añadir opciones (con su fixture de ruta), nunca alterar las existentes.
   - **Dudas o desviaciones:** con la sección del contrato afectada.
   - **Fuera de alcance:** confirmación de que no tocaste nada más.
4. **Espera.** Jesús decide entre commit, recorrido del código o corrección. **No hagas `git commit`, `git push`, `dvc push`, instalaciones ni merges sin su aprobación explícita.**

## Pruebas

- **`tests/acceptance/`** contiene las pruebas de aceptación de cada fase (`test_fN_*.py`), con las cifras literales de la especificación §6. Se escriben al inicio de la fase, Jesús revisa sus afirmaciones y se congelan en un commit propio antes de implementar. Después se añade una regla `Edit(...)` para ese archivo en `.claude/settings.json`, con aprobación de Jesús. **Nunca las modifiques para que pasen.** Si una parece errónea, detente y explica por qué.
- **Pruebas unitarias:** van en `tests/forecasting/` y las escribes libremente dentro de cada unidad, con datos sintéticos.
- **Datos locales:** se sigue la convención existente en `tests/epidemiology/conftest.py`. Las pruebas con datos reales se activan con una opción explícita de pytest: sin la opción se omiten; con la opción y sin el archivo, fallan. En el informe de paso, las de aceptación siempre se ejecutan con la opción.

## Entorno: Windows, app de escritorio, sesión local

- **Python:** usa siempre el del entorno virtual del repositorio por su ruta: `.venv/Scripts/python.exe` (Python 3.11.0; ruta confirmada en F0). No uses el Python global aunque tenga paquetes instalados. No supongas que el entorno está activado.
  - Pruebas: `pytest.ini` ya aplica `--import-mode=importlib` (hay dos `test_local_regression.py` en el legado).
    - Suite completa: `.venv/Scripts/python.exe -B -m pytest tests/ -q -p no:cacheprovider`
    - Aceptación: `.venv/Scripts/python.exe -B -m pytest tests/acceptance -q -rA -p no:cacheprovider` más las opciones de datos de `tests/acceptance/conftest.py` (`--panel-parquet`, `--channel-parquet`, `--legacy-labels-parquet`). Las opciones de datos solo se reconocen si la carpeta cuyo `conftest.py` las define se pasa explícitamente como argumento (no basta `tests/`).
  - DVC: `.venv/Scripts/dvc.exe ...` (`status` y `pull` sin preguntar; `add` y `push` con aprobación).
- **Dependencias:** el conjunto de modelado está en `requirements/forecasting.txt`, que se crea en F0. No instales nada sin aprobación.
- **Credenciales:** `.env` contiene las credenciales de Cloudflare Access para MLflow. **No lo leas, no lo imprimas y no copies su contenido en mensajes, código, logs ni commits.** Los scripts las cargan con `python-dotenv`. Las credenciales de DVC son locales y nunca se escriben.
- **MLflow:** se usa solo a través de `src/forecasting/tracking.py`, que añade los encabezados de Cloudflare Access. Si `.env` falta o MLflow no responde, informa y usa el respaldo local de la especificación §9. No inventes credenciales ni desactives la verificación TLS.

## Convenciones

- **Código nuevo:** va en `src/forecasting/` y en `src/epidemiology/channel_epico.py`, con pruebas en `tests/forecasting/` y `tests/acceptance/`. Los cambios fuera de esas rutas (documentación, `.gitignore`, `requirements/`, `configs/`, API) se hacen solo si la unidad aprobada los incluye.
- **Nombres:**
  - Los que fija el contrato se usan tal cual: features (`zona_t`, `posicion_4s`, …), columnas (`epico_central`, `epico_upper`, `target_u1e`, `channel_zone`) y campos de la API.
  - Lo demás sigue el estilo del código existente: identificadores y docstrings en inglés, documentación en español.
- **Configuración:** los parámetros van en `configs/forecasting.yaml` y se validan con Pydantic al cargarse. Ningún parámetro del contrato se escribe en el código.
- **Datos y resultados:** los Parquet se gestionan con DVC. `.gitignore` ignora `*.csv` y `*.parquet`; las excepciones se añaden de forma explícita, como en `reports/epidemiology/evidence/`.
- **Errores:** valida las entradas y falla con mensajes claros. No captures excepciones para ocultarlas.
- **Commits:** formato `tipo(alcance): descripción` en español, por ejemplo `feat(canal): …`, `test(aceptacion): …`, `docs: …` o `chore: …`. Un commit por unidad, y el cuerpo cita el requisito (`contrato §10.7`).

## Cuándo detenerse

Detente y pregunta antes de seguir si:
- el plan exige una decisión que el contrato no fija;
- una cifra de aceptación no se reproduce;
- una prueba de aceptación parece incorrecta;
- aparece una posible fuga;
- una dependencia o servicio externo falla;
- la tarea requiere tocar el legado.

No corrijas en silencio ni optimices decisiones metodológicas.
