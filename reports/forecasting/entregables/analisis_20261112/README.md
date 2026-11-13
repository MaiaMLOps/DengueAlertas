# Entregables del análisis integrado de resultados (corrida del 2026-11-12)

Versiones para lectura y presentación del análisis integrado de la prueba única 2023–2025 (contrato §8.8). Se generaron a partir de la corrida registrada en la ejecución 1 de `reports/forecasting/bitacora_prueba.jsonl` y de la evidencia de `prueba_20261112/`, `reportes_20261112/` y `exploratorio_20261112/`.

| Archivo | Qué es | Fuente |
|---|---|---|
| `analisis_resultados.docx` | El análisis integrado en Word, con figuras y tablas incrustadas | [`docs/forecasting/analisis_resultados.md`](../../../../docs/forecasting/analisis_resultados.md), convertido con pandoc |
| `presentacion_resultados.pdf` | Presentación de referencia para el equipo: complemento visual del análisis, en lenguaje para un lector externo | Resume el mismo análisis; sus figuras son versiones con términos simplificados de las de `figures/` |

Reglas de uso:

- **La fuente es el Markdown.** Si cambia `analisis_resultados.md`, estos archivos quedan desactualizados hasta que se regeneren. Se regeneran solo en hitos, no con cada corrección, para no acumular binarios en el historial de git.
- **No son vinculantes.** Las conclusiones de §8.4 están en la evidencia de la prueba. Estos archivos las presentan sin cambiarlas.
- **Las cifras no se recalculan aquí.** Cada cifra del análisis cita su archivo de evidencia en el Markdown. La presentación omite esas rutas a propósito, porque está escrita para quien no conoce el repositorio.
- **Lo exploratorio va marcado.** Lo marcado como exploratorio en ambos documentos se definió después de ver la prueba. Describe patrones, pero no los confirma.
