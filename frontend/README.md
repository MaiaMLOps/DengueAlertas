# Frontend de DengueAlertas

Interfaz web del prototipo académico **DengueAlertas**, desarrollado en la Maestría en Inteligencia Artificial de la Universidad de los Andes. Presenta pronósticos retrospectivos de zona epidémica del canal endémico de dengue para seis municipios piloto de Colombia: Barranquilla, Soledad, Neiva, Villavicencio, Cúcuta y Cali.

La aplicación tiene dos vistas independientes:

- **Dashboard (`/`):** selecciona municipio y semana de emisión, consulta la API y muestra la probabilidad de zona epidémica ocho semanas después, el estado de alerta, el canal endémico y la trayectoria de emisiones. El mapa representa el estado devuelto por la API y los municipios con modelos `enabled=true`.
- **Desempeño en prueba (`/desempeno`):** presenta las métricas y las series de la prueba única 2023–2025. Lee exclusivamente el archivo estático `public/desempeno_prueba.json`, con predicciones de origen móvil; no utiliza el modelo final, la API ni MLflow.

> Es un prototipo retrospectivo de investigación. No reemplaza la vigilancia oficial ni tiene aval del Instituto Nacional de Salud.

## Enlaces relevantes

- Demostración: <https://demo.arkoiot.com/>
- Staging: <https://staging.denguealertas.pages.dev/>
- MLflow: <https://platform.arkoiot.com/>
- Documentación general del repositorio: [`../README.md`](../README.md)
- Contrato predictivo: [`../docs/forecasting/contrato_predictivo.md`](../docs/forecasting/contrato_predictivo.md)
- Salida de la API v0.2: [`../docs/forecasting/api_salida_d10.md`](../docs/forecasting/api_salida_d10.md)
- Esquema de la vista de desempeño: [`../docs/forecasting/desempeno_prueba.md`](../docs/forecasting/desempeno_prueba.md)

## Dependencias y entorno de ejecución

- **Node.js 24**, recomendado y utilizado por el workflow de CI.
- **npm**, con dependencias fijadas en `package-lock.json`.
- Navegador moderno con JavaScript habilitado.
- Para el Dashboard, acceso a la API y a los modelos habilitados. La vista **Desempeño en prueba** no necesita Python, DVC, MLflow ni API.

Tecnologías principales:

| Herramienta | Uso |
|---|---|
| React 19 y React DOM | Componentes de la interfaz |
| React Router 7 | Rutas `/` y `/desempeno` |
| Vite 8 | Desarrollo y compilación |
| Tailwind CSS 4 | Estilos |
| Lucide React | Iconos |
| Framer Motion | Animaciones |
| Geist | Tipografía |
| Oxlint | Revisión estática |

No se requiere Docker para desarrollar o publicar este frontend. La interfaz sí puede ejecutarse sola en local, pero el Dashboard depende de la API y esta, en su implementación actual, descubre y carga los modelos desde el MLflow remoto: no usa directamente `modelos/*.joblib` como respaldo offline. Solo **Desempeño en prueba** funciona sin esos servicios.

## Uso básico en desarrollo

Desde la raíz del repositorio:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env
npm run dev
```

En macOS o Linux, el archivo de entorno se copia con:

```bash
cp .env.example .env
```

Vite muestra la URL local, normalmente <http://localhost:5173>. Para ejecutar también la API en el equipo, usar `VITE_API_URL=http://127.0.0.1:8000` y seguir la guía completa del [`README.md` general, §Ejecución local de la API y del frontend](../README.md#ejecución-local-de-la-api-y-del-frontend). Comandos disponibles:

```bash
npm run dev       # Servidor de desarrollo
npm run lint      # Revisión estática
npm run build     # Compilación de producción en dist/
npm run preview   # Vista previa local de dist/
```

### Ejemplo: consultar un pronóstico

1. Abrir **Dashboard**.
2. Elegir un municipio habilitado en el mapa o en su selector.
3. Seleccionar una **semana de emisión** dentro de la ventana disponible.
4. Pulsar **Consultar pronóstico**.
5. Revisar la probabilidad, las referencias B2/B1, el estado, el canal y las emisiones anteriores.
6. Abrir **¿Cómo leer este resultado?** para consultar la explicación de los estados.

La etiqueta superior y la tarjeta de vigilancia muestran el ordinal del canal y la fecha de emisión seleccionados. Si la API no está disponible, la interfaz no sustituye la respuesta con valores locales.

### Ejemplo: consultar el desempeño en prueba

1. Abrir la pestaña **Desempeño en prueba** o visitar `/desempeno`.
2. Revisar el resumen global.
3. Seleccionar un municipio para ver su ficha de la Tabla 6.
4. Comparar la probabilidad del modelo de origen móvil con B2 en cada semana objetivo.
5. Revisar las franjas de alerta de umbral, producto y canal, junto con los inicios de brote y si fueron avisados.

La pestaña ejecuta:

```js
fetch('/desempeno_prueba.json')
```

El JSON se genera desde la evidencia versionada y **no debe editarse manualmente**. Su procedencia y regeneración están documentadas en [`../docs/forecasting/desempeno_prueba.md`](../docs/forecasting/desempeno_prueba.md).

## Variables de entorno y credenciales de ejemplo

`VITE_API_URL` se incorpora al JavaScript **durante la compilación**; no es un secreto ni cambia en tiempo de ejecución. Si no se define, se utiliza `https://api.arkoiot.com`.

Ejemplo para usar la API de producción:

```dotenv
VITE_API_URL=https://api.arkoiot.com
```

Ejemplo **solo para desarrollo local** mediante el proxy `/api` de Vite:

```dotenv
VITE_API_URL=/api
CF_ACCESS_CLIENT_ID=cliente-local-de-ejemplo
CF_ACCESS_CLIENT_SECRET=secreto-local-de-ejemplo
```

Los dos últimos valores son ficticios y no permiten acceso. El responsable de infraestructura entrega los service tokens reales, que deben permanecer en archivos locales ignorados por git. **No deben incluirse en código, documentación, logs ni commits.**

Para el acceso interactivo a la demo se puede usar un correo con dominio `@uniandes.edu.co`, por ejemplo `nombre@uniandes.edu.co`. Cloudflare Access envía un código OTP al correo; no se utiliza una contraseña compartida. La sesión dura 72 horas.

## Compilación y despliegue

### 1. Verificar y compilar

```bash
npm ci
npm run lint
npm run build
```

Para definir explícitamente la API en PowerShell:

```powershell
$env:VITE_API_URL = "https://api.arkoiot.com"
npm run build
npm run preview
```

La salida queda en `dist/`, incluidos `desempeno_prueba.json` y `_redirects`. Este último dirige las rutas desconocidas a `index.html` para que `/desempeno` funcione también al recargar la página en Cloudflare Pages.

### 2. Publicar con los workflows del repositorio

El despliegue vigente usa **Cloudflare Pages** y GitHub Actions:

1. `CI Pipeline - Frontend` instala dependencias, ejecuta `npm run build` y publica el artefacto `frontend-build` con el contenido de `dist/`.
2. En GitHub Actions, abrir `CD Pipeline - Frontend Deployment` y seleccionar **Run workflow**.
3. Elegir `staging` o `production` y, opcionalmente, el `run_id` de CI que generó el artefacto.
4. El workflow descarga el artefacto y ejecuta Wrangler sobre el proyecto `denguealertas`:
   - `production` → rama `main`, <https://demo.arkoiot.com/> y <https://denguealertas.pages.dev/>.
   - `staging` → rama `staging`, <https://staging.denguealertas.pages.dev/>.
5. Comprobar `/`, `/desempeno` y `/desempeno_prueba.json` en el ambiente publicado.

Los secretos `CLOUDFLARE_API_TOKEN` y `CLOUDFLARE_ACCOUNT_ID` se configuran en GitHub Actions; `GITHUB_TOKEN` lo proporciona el workflow. Nunca se escriben en este archivo ni en `.env.example`.

> Estos pasos describen el proceso existente; no implican que una compilación local publique automáticamente cambios. El despliegue se realiza mediante el workflow autorizado.

## Estructura principal

```text
frontend/
├── public/
│   ├── desempeno_prueba.json  # Evidencia estática generada; no editar a mano
│   └── _redirects             # Respaldo SPA para las rutas
├── src/
│   ├── components/
│   │   ├── Dashboard.jsx
│   │   ├── ColombiaMap.jsx
│   │   ├── TestPerformance.jsx
│   │   └── TestPerformanceCharts.jsx
│   ├── App.jsx                # Rutas principales
│   └── main.jsx
├── .env.example
├── package.json
├── package-lock.json
└── vite.config.js
```
