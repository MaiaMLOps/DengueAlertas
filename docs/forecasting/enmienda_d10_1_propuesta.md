# Propuesta de enmienda D10.1: regla de alerta del producto

**Estado:** texto propuesto el 2026-11-20, pendiente de aplicación por el responsable de la implementación. `contrato_predictivo.md` y `registro_decisiones.md` son documentos preregistrados y protegidos: este archivo no los modifica. Una vez aplicados, este archivo queda como registro de la propuesta.

**Alcance.** La regla de alerta que ve el usuario pasa a ser B01. No cambian la receta congelada (`configs/receta_congelada.yaml`, hash `de74d4ab…fda5`), τ = 0,05, k = 2, W = 12, la probabilidad, las referencias B1 y B2, la selección ni la evaluación preregistrada de prueba (§7.4 y §8.4 con la regla p ≥ τ). Registro de la desviación: `desviaciones.md`, DV-07. Justificación con cifras: `analisis_complementario.md`, sección «Justificación de la regla B01 en el producto».

Los bloques siguientes indican dónde va cada texto. Los marcados «Añadir» no reescriben texto existente; los marcados «Léase» siguen el formato de las erratas (el texto anterior se conserva y se lee el nuevo).

---

## 1. `contrato_predictivo.md`

### 1.1 Encabezado, «Última actualización» — Léase

> **Última actualización:** 2026-11-20 (enmienda D10.1: regla de alerta del producto, §7.4 y §9.3; la evaluación preregistrada no cambia). Versiones anteriores: 2026-11-12 (enmienda D7.2: el conjunto de variables pasa a ser un paso de selección, §6.6; y enmienda D9.1: análisis diagnóstico integrado, §8.8 y §10.22; solo reporte), 2026-11-11 (enmienda D7.1: el entrenamiento de cada ensayo se cierra en su primera emisión, §6.1) y 2026-11-10 (D1.1 y D3–D10 cerrados: contrato completo).

### 1.2 §0, tabla de decisiones — Añadir una fila después de D10

> | D10.1 | Regla de alerta del producto | [D] El producto alerta con B01 (p̂_t ≥ τ y zona observada en t distinta de epidemia), con los mismos τ, k y W; cuatro estados; la evaluación preregistrada (§7.4, §8.4) no cambia. Formulada después de la prueba: su evaluación en 2023–2025 no es confirmatoria | §7.4 y §9.3 |

### 1.3 §7.4 — Añadir al final de la sección, antes de §7.5

> **Regla de alerta del producto (D10.1, 2026-11-20) [D].** La evaluación de esta sección, la elección de τ (§7.5) y las conclusiones de §8.4 usan la emisión positiva del modelo p̂_t ≥ τ. El producto (§9) usa la regla B01:
>
> | Componente | Regla del producto (B01) |
> |---|---|
> | Emisión positiva | p̂_t ≥ τ **y** zona observada en t distinta de epidemia |
> | Alerta, fecha, ventana, anticipación y falsa alarma | Las de la tabla de definiciones de esta sección, sin cambios (k = 2, t_c, W = 12) |
> | Alertas de continuidad | No existen: una emisión en zona epidémica es negativa, de modo que la zona observada en t_c nunca es epidemia |
>
> - **Mismos parámetros.** τ, k y W son los de la receta congelada; B01 no ajusta ningún parámetro.
> - **Mecanismo.** Una racha de emisiones positivas que continúa durante un episodio y después de él no puede avisar el inicio siguiente, porque no confirma una alerta nueva dentro de la ventana. B01 corta la racha al entrar en zona epidémica: aplica a la emisión el principio de esta sección de contar aparte las alertas de continuidad.
> - **Estado de la evidencia.** B01 se formuló después de la evaluación preregistrada de prueba, a partir de un diagnóstico en desarrollo, y se evaluó una vez en 2023–2025 sin cambiar τ, k ni W (`analisis_complementario.md`; evidencia en `reports/forecasting/evidence/complementario_20261120/`). Esa evaluación no es confirmatoria; queda pendiente una validación prospectiva con datos posteriores a 2025.
> - **Lo que no cambia.** Las cifras preregistradas de eventos de §8.4 se siguen reportando con la regla p̂_t ≥ τ.

### 1.4 §9.1, «Principio» — Léase

> - **Principio:** el usuario ve la misma probabilidad y las mismas referencias que se evaluaron, y el desempeño medido en prueba. La regla de alerta es la del producto (§7.4, D10.1); la ficha muestra sus cifras de prueba como evaluación posterior no confirmatoria, junto al resultado preregistrado de la regla p̂_t ≥ τ.

### 1.5 §9.3, tabla de la vista principal — Léase (filas Probabilidad, Estado de alerta, Trayectoria y Ficha de desempeño)

> | Elemento | Contenido | Origen |
> |---|---|---|
> | Probabilidad | p̂ de zona epidémica en t + 8, en porcentaje entero, con la semana de emisión y la semana objetivo explícitas. Se etiqueta según el estado: «probabilidad de entrar en zona epidémica» fuera de continuidad epidémica y «probabilidad de seguir en zona epidémica» en continuidad epidémica, siempre con la fecha objetivo | §2.1; D10.1 |
> | Estado de alerta | Cuatro estados de la regla del producto, con τ y k congelados: **sin alerta**; **señal en observación** (una emisión ≥ τ fuera de zona epidémica); **alerta de inicio confirmada** (dos emisiones consecutivas ≥ τ fuera de zona epidémica); **continuidad epidémica** (la zona observada en t es epidemia y no hay alerta del modelo) | §7.4 (D10.1) |
> | Trayectoria | Probabilidades de las últimas 12 emisiones, con la línea de τ. En continuidad epidémica, además, la trayectoria de la probabilidad de seguir durante el episodio en curso y la referencia B2 de ese estado | §7.4; D10.1 |
> | Ficha de desempeño | Resultados de prueba (2023–2025) del municipio y globales (§9.4, vista 4): las cifras de la regla del producto, rotuladas «evaluación posterior no confirmatoria», junto al resultado preregistrado de la regla p̂_t ≥ τ y al resultado de continuidad | §7, §8; D10.1 |

> **Añadir debajo de la tabla:** No hay bandas de riesgo ni aviso de salida de la zona epidémica: no se evaluaron.

### 1.6 §9.4, vista 4 — Añadir al final de la lista

> - las cifras de prueba de la regla del producto (D10.1): inicios avisados, entradas avisadas, falsas alarmas, fracción de semanas bajo alerta y anticipación mediana (desde la alerta más reciente y desde la primera), rotuladas «evaluación posterior no confirmatoria»;
> - el resultado de continuidad: BSS frente a B2 en las semanas en zona epidémica, en prueba y, rotulado como tal, en desarrollo. Es la única cifra de desarrollo de la ficha, porque describe el estado de continuidad y no se usó para elegir nada.

### 1.7 §9.6, «Campos de `/predict`» — Añadir

> Además: regla de alerta (`umbral_fuera_de_epidemia`), etiqueta de la probabilidad según el estado y, en continuidad epidémica, la probabilidad de seguir, la referencia B2 de ese estado y la trayectoria durante el episodio. Detalle en `api_salida_d10.md` (versión de esquema 0.2).

---

## 2. `registro_decisiones.md` — Añadir antes de «Erratas (2026-11-13)»

> ## D10.1 — Regla de alerta del producto (2026-11-20; enmienda a D10)
>
> **Decisión:** el producto alerta con la regla B01: una emisión es positiva si p̂_t ≥ τ y la zona observada del canal en t no es epidemia. Se conservan τ = 0,05, k = 2, W = 12, la receta congelada, la probabilidad y las referencias. La vista principal tiene cuatro estados (sin alerta, señal en observación, alerta de inicio confirmada, continuidad epidémica) y etiqueta la probabilidad según el estado. La evaluación preregistrada (§7.4 con p̂_t ≥ τ, §7.5 y §8.4) no cambia. Texto vinculante en `contrato_predictivo.md` §7.4 y §9.3, con ajustes en §0, §9.1, §9.4 y §9.6.
>
> **Origen.** En el diagnóstico de desarrollo, dos inicios no se avisaron porque la alerta de la receta ya estaba encendida antes de la ventana y no se confirmó ninguna alerta nueva dentro de ella. La regla se formuló después de conocer la evaluación preregistrada de prueba y se evaluó una vez en 2023–2025 sin cambiar τ, k ni W.
>
> **Justificación** (cifras en `analisis_complementario.md`; evidencia en `reports/forecasting/evidence/complementario_20261120/` y `continuidad_20261120/`):
> - *Mecanismo sin parámetros ajustados.* Una alerta que sigue encendida después de un episodio no puede contar como aviso del inicio siguiente. B01 aplica a la emisión el principio de D8 de contar aparte las alertas de continuidad.
> - *Criterio del contrato.* En desarrollo y en prueba, B01 cumple la restricción de §7.5 (falsas alarmas por municipio-año no mayores que las del canal: 1,54 frente a 2,00 y 1,44 frente a 1,67) con la mayor capacidad de detección de los tres sistemas (10 de 10 y 9 de 10).
> - *Consistencia entre periodos.* B01 avisa más inicios que la regla p̂_t ≥ τ en los dos periodos; en prueba hay 5 inicios discrepantes, todos a favor de B01 (McNemar exacto de una cola, p = 0,03125).
> - *Costo explícito.* Más alertas y falsas alarmas que la regla p̂_t ≥ τ (en prueba, 38 y 26 frente a 26 y 18), pero menos semanas bajo alerta (20,5 % frente a 60,4 %); anticipación corta (mediana de 1,0 semanas desde la alerta más reciente).
>
> **Alternativas consideradas:**
> - **Mantener la regla p̂_t ≥ τ en el producto.** Conserva la equivalencia exacta con la evaluación preregistrada, pero en prueba avisa 4 de 10 inicios, menos que el canal (8 de 10).
> - **Cambiar τ, k o W.** Se descarta: se ajustarían con un periodo de prueba ya consultado.
> - **Mostrar la zona del canal como alerta.** Es la práctica actual (8 de 10 inicios en prueba, 30 falsas alarmas); no usa la probabilidad del modelo.
>
> **Lectura.** La adopción es una decisión de producto con evidencia posterior y no confirmatoria. El desempeño preregistrado de la receta sigue siendo el de §8.4.
>
> **Limitaciones declaradas:** pocos eventos (10 inicios por periodo); la evaluación de 2023–2025 no es independiente de la formulación de la regla; queda pendiente una validación prospectiva con datos posteriores a 2025.
>
> **Declarar en el artículo:** la regla de alerta del producto se formuló después de la evaluación preregistrada de prueba, a partir de un diagnóstico en desarrollo, sin ajustar parámetros; su evaluación en 2023–2025 no es confirmatoria.

---

## 3. Otros documentos afectados (no protegidos)

- `docs/forecasting/api_salida_d10.md`: versión de esquema 0.2 con los cuatro estados de D10.1 (actualizado en esta rama).
- `configs/forecasting.yaml`: sección `producto.regla_alerta` (actualizada en esta rama); `decision.regla_emision` queda fija en la regla preregistrada.
- `especificacion_implementacion.md` (preregistrada): la unidad de la API en F8 debe leer `producto.regla_alerta`. Si hace falta, se registra como errata fechada al final del documento.
