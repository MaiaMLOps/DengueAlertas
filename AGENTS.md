# Instrucciones para agentes — DengueAlertas

**Proyecto:** MAIA, alerta temprana de excesos de dengue en municipios colombianos.

**Ámbito de estas instrucciones:** repositorio `DengueAlertas`.

**Actualizado:** 2026-10-20.

**Documento de contexto:** `docs/contexto_metodologico.md` (leer antes de cualquier tarea que afecte datos, etiquetas, modelamiento o evaluación).

## 1. Misión y alcance vigente

Apoyar el desarrollo **reproducible y verificable** de un prototipo académico de predicción retrospectiva de estados futuros de alerta/exceso de dengue. El alcance vigente es **uno o unos pocos municipios seleccionados, con modelos independientes por municipio**; no se requiere cobertura nacional, inferencia en tiempo real ni conexión operacional con SIVIGILA. El repositorio también contiene infraestructura de API y frontend desarrollada por el equipo.

El agente implementa tareas autorizadas, pero **no decide unilateralmente la metodología científica**. Los documentos de propuesta contienen opciones históricas que pueden haber sido sustituidas; no deben convertirse por defecto en decisiones vigentes.

**Estado vigente:** la integración inicial retrospectiva está implementada y validada en `src/data/`, con 46 pruebas automatizadas en `tests/data/` y documentación de ejecución en `docs/data_pipeline.md`. El panel conserva 77.376 filas, 15 columnas y los 78 municipios candidatos bajo la regla provisional de screening. No contiene etiquetas, rezagos ni modelos. No repetir el diagnóstico inicial ni reconstruir productos existentes salvo que la tarea lo requiera.

## 2. Autoridad de las decisiones

Distinguir siempre entre:

- **Hecho documentado:** resultado observado en una auditoría, código existente o archivo efectivamente inspeccionado. Indicar fuente y comprobación realizada.
- **Decisión vigente:** indicación explícita del equipo o del usuario, identificada como tal.
- **Supuesto operativo provisional:** hipótesis técnica reversible para explorar o probar. Debe etiquetarse y no publicarse como conclusión.
- **Decisión abierta:** asunto científico/metodológico que requiere aprobación antes de materializarse.

**Puede decidir autónomamente:** estructura interna de funciones, tipado, pruebas, controles de calidad, mejoras locales de mantenibilidad y manejo seguro de errores, siempre dentro del alcance aprobado.

**Debe proponer y consultar antes de ejecutar:** definición de caso y exceso, etiqueta objetivo, fechas de predicción, horizonte, rezagos y ventanas, tratamiento de datos dudosos, reglas de imputación que afecten resultados, municipios finales, cortes temporales, baselines, métricas principales o umbrales de decisión. Si faltan esos acuerdos, limitarse a inspección, diagnóstico, prototipos explícitamente provisionales o tareas independientes de la decisión.

Si una orden aprobada pudiera producir *data leakage*, sesgo grave o interpretación científica incorrecta, **advertirlo y detener la parte afectada** hasta aclararlo. No «resolver» silenciosamente contradicciones entre documentos.

## 3. Directorios y límites de escritura

Las ubicaciones de las fuentes dependen del entorno y deben configurarse. Verificar su accesibilidad antes de usarlas; no asumir una unidad, usuario o estructura personal de carpetas.

| Ubicación lógica | Uso y permisos |
|---|---|
| Fuentes SIVIGILA | 19 Parquet anuales y manifiesto; solo lectura, fuera del repositorio |
| Fuentes de covariables | XLSX integrado y capas de respaldo; solo lectura |
| Referencias | Informes técnicos y documentos científicos; solo lectura |
| `src/data/`, `tests/data/`, `docs/`, `requirements/` | Código, pruebas, documentación y dependencias versionables |
| `data/processed/` | Derivados locales autorizados, excluidos de Git |
| `reports/data_quality/` | Reportes agregados locales, excluidos de Git |

- No editar, mover, borrar, renombrar ni sobreescribir las fuentes o referencias originales.
- No copiar microdatos individuales de SIVIGILA al repositorio, a remotos, a logs ni a artefactos compartidos.
- Generar derivados en ubicaciones **explícitas y diferenciadas** dentro del repositorio, respetando `.gitignore` y las reglas del equipo. Los directorios de salida indicados ya se utilizan para la integración; acordar cualquier ubicación nueva y no sobreescribir archivos existentes sin consentimiento.
- Utilizar rutas relativas/configurables desde la raíz del repositorio. Las rutas absolutas privadas no deben incorporarse al código versionable.
- En caso de dudas sobre redistribución, permisos o desidentificación, detener cualquier publicación y consultar.

## 4. Reglas de integridad de datos

Preservar las reglas **documentadas** de la auditoría previa hasta que se apruebe un cambio explícito:

- Fuente epidemiológica auditada: SIVIGILA **evento 210** (no asumir inclusión del evento 220).
- Clave territorial de ocurrencia: `COD_DPTO_O` + `COD_MUN_O`, normalizada a **cinco caracteres**, sin perder ceros iniciales.
- Fecha epidemiológica del screening: `INI_SIN` (inicio de síntomas), no `FEC_NOT`, `ANO` ni `SEMANA` como sustitutos automáticos.
- Filtro de disponibilidad usado en el screening: `confirmados = 1`; **no presupone** que esa será la definición de caso final del estudio.
- Calendario ejecutado en los insumos: `week_start`, semanas **lunes–domingo (W-SUN)**. No afirmar equivalencia con semana epidemiológica oficial sin validación.
- Ventana auditada: 2007-01-01 a 2025-12-31. La semana que comienza **2025-12-29 es parcial (3 días)**; no tratarla como siete días ni completarla artificialmente.
- Ausencia de confirmados observados ≠ notificación negativa verificada ≠ certeza de cero infecciones.
- La unicidad de identificadores dentro de cada Parquet anual **no prueba** deduplicación entre años o personas.
- Los estados finales de casos se conocen retrospectivamente; no atribuirles disponibilidad prospectiva no documentada.
- Preservar y reportar la anomalía temporal documentada en 2013 entre `FEC_NOT` e `INI_SIN`, sin corrección automática.

No declarar integridad o equivalencia solo porque un `join` sea ejecutable. Antes/después de cada transformación registrar totales, claves, duplicados, nulos, tipos, rangos, fechas y reglas de exclusión. Los objetivos cuantitativos históricos se encuentran en `docs/contexto_metodologico.md`; comprobarlos contra los archivos reales, no forzarlos.

## 5. Modelamiento y evaluación

- Separar tres productos: **panel integrado municipio–semana**, **etiquetas epidemiológicas** y **matriz de ejemplos supervisados** (features/ventanas/target). No presentar el primero como si ya incluyera los otros.
- Separar conceptualmente **canal endémico que define un estado observado** de **modelo que anticipa un estado futuro**.
- Toda característica en una predicción emitida en fecha `t` debe estar disponible en `t` según la convención de corte aprobada. Prohibido incluir variables contemporáneas de la semana objetivo o usar información futura en rezagos, umbrales o estadísticas aprendidas.
- Utilizar **validación temporal**, nunca particiones aleatorias de semanas para simular pronóstico prospectivo. Ajustar imputaciones, escaladores, selección de features, umbrales calibrados y cualquier parámetro aprendido dentro de cada conjunto de entrenamiento.
- Priorizar modelos y baselines defendibles y comparables bajo idénticos cortes y definición de target. No asumir valor adicional del clima por estar disponible.
- Registrar resultados por municipio, horizonte y periodo; documentar prevalencia de clase, eventos positivos, errores en picos/transiciones, falsas alertas e incertidumbre cuando corresponda.
- No llamar «probabilidad calibrada» a un score de clasificación sin una evaluación de calibración apropiada.

## 6. Ingeniería de software y MLOps

- Python modular, legible, tipado cuando aporte claridad, con `pathlib`, validación de entradas y mensajes de error útiles.
- Implementaciones deterministas o con semillas fijadas; dependencias y parámetros explicitados; scripts ejecutables desde la raíz del repositorio.
- Agregar pruebas unitarias y de integración pequeñas, especialmente para fechas, límites de semana/año, normalización DIVIPOLA, *joins* y prevención de fuga temporal.
- Evitar procesamiento costoso hasta validar esquema con muestras y disponer de un plan verificable.
- Guardar métricas y artefactos trazables con versión de datos, configuración, commit y periodo de evaluación.
- DVC es para control de versiones de datos; MLflow para seguimiento/registro de experimentos. Su disponibilidad y permisos deben comprobarse en el entorno de destino; no afirmar que funcionan antes de verificarlos. La integración inicial no configura DVC ni servicios remotos.
- No modificar frontend, API, CI/CD, secretos, infraestructura, `.dvc/config`, remotos, registros MLflow ni ramas ajenas sin una tarea explícita.
- No ejecutar `git push`, crear PR, publicar un dataset, hacer `dvc push`, registrar/promover modelos ni modificar despliegues sin autorización específica. No imprimir ni persistir credenciales en archivos, terminales compartidas o mensajes.
- Antes de editar, comprobar rama y cambios locales (`git status`); no sobrescribir trabajo preexistente. No hacer *commit* sin aprobación.

## 7. Protocolo de colaboración

Para una tarea no trivial:

1. Leer el contexto pertinente e inspeccionar código/archivos reales **sin modificar**.
2. Resumir objetivo, evidencia encontrada, incertidumbres, archivos que se tocarían, pruebas y riesgos.
3. Identificar decisiones científicas abiertas y pedir aprobación si condicionan la implementación.
4. Ejecutar únicamente el alcance acordado, con cambios mínimos y reversibles.
5. Entregar resumen de archivos creados/modificados, comandos ejecutados, pruebas/resultados, anomalías y pasos pendientes. Distinguir claramente **ejecutado**, **no ejecutado** y **recomendado**.

## 8. Documentación de referencia

Consultar `docs/contexto_metodologico.md`, `docs/data_pipeline.md` y las referencias pertinentes en la ubicación configurada. Si una herramienta no puede leer correctamente PDF/DOCX, señalar la limitación y solicitar texto extraído; no improvisar su contenido. La fuente observada prevalece sobre un nombre de archivo o una expectativa de la documentación; las discrepancias deben registrarse para revisión.
