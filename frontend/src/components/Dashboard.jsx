import React, { useState, useEffect } from 'react';
import {
    Activity,
    AlertTriangle,
    Layers,
    Zap,
    MapPin,
    CheckCircle2,
    Sliders,
    ChevronDown,
    Users
} from 'lucide-react';
import ColombiaMap, { MUNICIPIOS_ESTRATEGICOS } from './ColombiaMap';
import ChannelSeriesChart from './ChannelSeriesChart';
import EmissionTrajectoryTable from './EmissionTrajectoryTable';

// La URL se define por entorno al compilar (VITE_API_URL). El valor por defecto
// preserva el comportamiento actual; Staging debe compilar con su propia URL.
const API_BASE_URL = (import.meta.env.VITE_API_URL || 'https://api.arkoiot.com').replace(/\/+$/, '');

// Seis variables del conjunto "sin_clima" del modelo final
// (docs/forecasting/contrato_predictivo.md §5.2 y §6.6). Los rangos son del
// control de la interfaz, no restricciones impuestas por el modelo.
const FIELD_META = {
    zona_t: {
        label: 'Zona del canal en t (observada)', unit: '', type: 'select',
        options: ['seguridad', 'alerta', 'epidemia'], default: 'seguridad', icon: Activity,
        help: 'Dato observado en la semana de emisión t; no se predice.'
    },
    posicion_4s: {
        label: 'Posición media de 4 semanas', unit: '', min: -3, max: 2, step: 0.01, default: 0, icon: Activity,
        help: 'Media de log(C_s + 1) − log(U1e_s + 1) para s = t−3 … t.'
    },
    crecimiento_4s: {
        label: 'Crecimiento de 4 semanas', unit: '', min: -5, max: 5, step: 0.01, default: 0, icon: Activity,
        help: 'log(C_t + 1) − log(C_{t−4} + 1).'
    },
    cambio_umbral: {
        label: 'Cambio del umbral', unit: '', min: -2, max: 2, step: 0.01, default: 0, icon: Activity,
        help: 'log(U1e_{t+8} + 1) − log(U1e_t + 1).'
    },
    sem_sin: {
        label: 'Seno de la semana objetivo', unit: '', min: -1, max: 1, step: 0.01, default: 0, icon: Activity,
        help: 'seno de 2π·(w_{t+8} − 1)/52,18; debe ser coherente con sem_cos.'
    },
    sem_cos: {
        label: 'Coseno de la semana objetivo', unit: '', min: -1, max: 1, step: 0.01, default: 1, icon: Activity,
        help: 'coseno de 2π·(w_{t+8} − 1)/52,18; debe ser coherente con sem_sin.'
    }
};

// Modo v0.2 por defecto en `/`; ?new=false conserva el modo manual de DV-09.
const NEW_MODE = typeof window === 'undefined'
    || new URLSearchParams(window.location.search).get('new') !== 'false';

// Ventana vigente de api_salida_d10.md §4. La API valida la misma ventana contra el panel.
const V2_WEEK_OPTIONS = (() => {
    const options = [];
    const end = Date.parse('2025-12-22T00:00:00Z');
    for (let ms = Date.parse('2025-10-27T00:00:00Z'); ms <= end; ms += 7 * 24 * 60 * 60 * 1000) {
        options.push(new Date(ms).toISOString().slice(0, 10));
    }
    return options;
})();

const FEATURE_NAMES = Object.keys(FIELD_META);
const MODEL_INPUT_SCHEMA = FEATURE_NAMES.map(name => ({
    name,
    type: name === 'zona_t' ? 'string' : 'double',
    required: true
}));

// Extrae un mensaje legible del cuerpo de error de la API sin ocultar el código HTTP.
const apiErrorMessage = async (res, fallback) => {
    let detail = fallback;
    try {
        const body = await res.json();
        detail = body?.error?.mensaje || body?.detail?.mensaje || body?.detail || body?.error || detail;
    } catch {
        // El cuerpo no es JSON; se conserva el mensaje genérico.
    }
    if (typeof detail !== 'string') detail = JSON.stringify(detail);
    if (detail.length > 300) detail = `${detail.slice(0, 300)}…`;
    return `HTTP ${res.status} · ${detail}`;
};

const ALERT_STATE_LABELS = {
    sin_alerta: 'Sin alerta',
    senal_en_observacion: 'Señal en observación',
    alerta_inicio_confirmada: 'Alerta de inicio confirmada',
    continuidad_epidemica: 'Continuidad epidémica'
};
const ALERT_STATES = new Set(Object.keys(ALERT_STATE_LABELS));

// La misma respuesta alimenta el detalle del municipio y los colores del mapa.
// El frontend nunca deduce el estado a partir de la probabilidad.
const requestV2Prediction = async (codigo, semana, signal) => {
    const query = new URLSearchParams({ municipio: codigo, semana_emision: semana });
    const res = await fetch(`${API_BASE_URL}/predict?${query.toString()}`, {
        credentials: 'include',
        signal
    });
    if (!res.ok) {
        throw new Error(await apiErrorMessage(res, 'La API rechazó la consulta v0.2.'));
    }

    const data = await res.json();
    if (data.version_esquema !== '0.2' || typeof data.probabilidad !== 'number'
        || !Number.isFinite(data.probabilidad) || data.semana_emision !== semana
        || data.municipio?.codigo !== codigo || !ALERT_STATES.has(data.decision?.estado_alerta)) {
        throw new Error('La API no devolvió una respuesta v0.2 válida; no se muestra ningún valor sustituto.');
    }
    return data;
};

export default function Dashboard() {
    // Municipio Estratégico Seleccionado en el Mapa (por defecto Cali)
    const [selectedMunicipio, setSelectedMunicipio] = useState(MUNICIPIOS_ESTRATEGICOS[0]);
    // La fuente es GET /municipios: no se presupone ningún modelo habilitado.
    const [activeCodes, setActiveCodes] = useState([]);
    const [activeModelsStatus, setActiveModelsStatus] = useState('loading');
    const [showTechDetails, setShowTechDetails] = useState(false);

    // Estado del esquema y de la API
    const [schemaFields, setSchemaFields] = useState(MODEL_INPUT_SCHEMA);
    const [schemaWarning, setSchemaWarning] = useState(null);
    const [apiConnected, setApiConnected] = useState(false);
    const [apiError, setApiError] = useState(null);
    const [threshold, setThreshold] = useState(0.205);
    const [modelInfo, setModelInfo] = useState({
        stage: 'Production',
        version: '1.1.0',
        model_name: 'ModeloPrueba'
    });

    // Valores introducidos por el usuario para las seis variables del modelo.
    const [formValues, setFormValues] = useState(() => {
        const initial = {
            municipio: MUNICIPIOS_ESTRATEGICOS[0].codigo,
            municipality_code: MUNICIPIOS_ESTRATEGICOS[0].codigo
        };
        MODEL_INPUT_SCHEMA.forEach(f => {
            initial[f.name] = FIELD_META[f.name].default;
        });
        return initial;
    });

    // No hay predicción inicial: solo se muestra la que devuelve la API.
    const [prediccion, setPrediccion] = useState(null);
    const [isPredicting, setIsPredicting] = useState(false);

    // Modo v0.2: municipio + semana de emisión; la API calcula el resto.
    const [v2Week, setV2Week] = useState(V2_WEEK_OPTIONS[V2_WEEK_OPTIONS.length - 1]);
    const [v2Data, setV2Data] = useState(null);
    const [v2Error, setV2Error] = useState(null);
    const [isPredictingV2, setIsPredictingV2] = useState(false);
    const [mapStatuses, setMapStatuses] = useState(() => Object.fromEntries(
        MUNICIPIOS_ESTRATEGICOS.map(mun => [mun.codigo, { status: 'loading' }])
    ));
    const activeCodesKey = MUNICIPIOS_ESTRATEGICOS
        .filter(mun => activeCodes.includes(mun.codigo))
        .map(mun => mun.codigo)
        .join(',');

    // Consultar el esquema del modelo del municipio seleccionado.
    const fetchSchema = async (codigo) => {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 3000);

            const res = await fetch(`${API_BASE_URL}/schema?municipio=${encodeURIComponent(codigo)}`, {
                credentials: 'include',
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (!res.ok) {
                setApiConnected(false);
                setApiError(await apiErrorMessage(res, 'No se pudo consultar /schema.'));
                return;
            }

            const data = await res.json();
            const names = (data.input_schema || []).map(f => f.name);
            const sirveSeisVariables = FEATURE_NAMES.every(name => names.includes(name));

            if (sirveSeisVariables) {
                setSchemaFields(data.input_schema.filter(f => FEATURE_NAMES.includes(f.name)));
                setSchemaWarning(null);
            } else {
                setSchemaFields(MODEL_INPUT_SCHEMA);
                setSchemaWarning('El /schema de la API no coincide con las seis variables del modelo final. Se usa el esquema del contrato y se enviarán solo esas seis variables.');
            }
            setModelInfo({
                stage: data.stage || 'Production',
                version: data.version || '1.1.0',
                model_name: data.model_name || 'ModeloPrueba'
            });
            setApiConnected(true);
            setApiError(null);
        } catch (err) {
            setApiConnected(false);
            setApiError(err.name === 'AbortError'
                ? 'Tiempo de espera agotado al consultar /schema.'
                : `No se pudo conectar con ${API_BASE_URL}/schema (${err.message}).`);
        }
    };

    // /municipios descubre en la API los DengueAlertas_* con enabled=true en MLflow.
    // También se acepta una lista vacía: significa que no hay modelos habilitados.
    const fetchActiveMunicipios = async () => {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);
        try {
            const res = await fetch(`${API_BASE_URL}/municipios`, {
                credentials: 'include',
                signal: controller.signal
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            if (!Array.isArray(data.active_codes)) throw new Error('active_codes ausente o inválido');
            setActiveCodes(data.active_codes.filter(codigo =>
                MUNICIPIOS_ESTRATEGICOS.some(mun => mun.codigo === codigo)
            ));
            setActiveModelsStatus('ready');
        } catch {
            // Sin respuesta no se muestran modelos como activos por defecto.
            setActiveCodes([]);
            setActiveModelsStatus('error');
        } finally {
            clearTimeout(timeoutId);
        }
    };

    // Helper para capturar eventos y clics en la interfaz (Clickstream)
    const trackUiEvent = (action, target, section, metadata = {}) => {
        try {
            fetch(`${API_BASE_URL}/telemetry/event`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ action, target, section, metadata })
            }).catch(() => {});
        } catch {}
    };

    // Escuchador global de clics en la UI
    useEffect(() => {
        const handleGlobalClick = (e) => {
            const el = e.target.closest('button, input[type="range"], a');
            if (!el) return;
            const label = (el.innerText || el.getAttribute('aria-label') || el.name || el.id || el.tagName).trim().slice(0, 60);
            trackUiEvent('click', label, 'ui_interaction', { tag: el.tagName });
        };
        document.addEventListener('click', handleGlobalClick);
        return () => document.removeEventListener('click', handleGlobalClick);
    }, []);

    useEffect(() => {
        fetchSchema(selectedMunicipio.codigo);
        fetchActiveMunicipios();
        const interval = setInterval(() => {
            fetchSchema(selectedMunicipio.codigo);
            fetchActiveMunicipios();
        }, 60000);
        return () => clearInterval(interval);
    }, [selectedMunicipio.codigo]);

    // Consulta los estados de todos los pilotos activos para la semana elegida.
    // Las respuestas fallidas o pendientes quedan en gris, nunca toman un color supuesto.
    useEffect(() => {
        if (!NEW_MODE || !activeCodesKey) return;

        const codes = activeCodesKey.split(',');
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000);
        let cancelled = false;

        const requests = codes.map(async (codigo) => {
            try {
                const data = await requestV2Prediction(codigo, v2Week, controller.signal);
                if (!cancelled) {
                    setMapStatuses(prev => ({ ...prev, [codigo]: { status: 'ready', data } }));
                }
            } catch (error) {
                if (!cancelled) {
                    setMapStatuses(prev => ({
                        ...prev,
                        [codigo]: {
                            status: 'error',
                            message: error.name === 'AbortError'
                                ? 'Tiempo de espera agotado al consultar la API.'
                                : error.message
                        }
                    }));
                }
            }
        });
        Promise.allSettled(requests).then(() => clearTimeout(timeoutId));

        return () => {
            cancelled = true;
            clearTimeout(timeoutId);
            controller.abort();
        };
    }, [v2Week, activeCodesKey]);

    const handleSelectMunicipio = (mun) => {
        setSelectedMunicipio(mun);
        setFormValues(prev => ({
            ...prev,
            municipio: mun.codigo,
            municipality_code: mun.codigo
        }));
        setPrediccion(null);
        setV2Data(null);
        setV2Error(null);
        setApiError(null);
        setSchemaWarning(null);
        trackUiEvent('select_municipio', mun.nombre, 'colombia_map', {
            codigo: mun.codigo,
            departamento: mun.departamento,
            densidad: mun.densidad_poblacional
        });
    };

    const handleFieldChange = (name, value) => {
        setFormValues(prev => ({ ...prev, [name]: value }));
        setPrediccion(null);
    };

    const renderFieldInput = (field) => {
        const meta = FIELD_META[field.name];
        if (!meta) return null;
        const currentValue = formValues[field.name] ?? meta.default;
        const displayed = typeof currentValue === 'number' && !Number.isInteger(currentValue)
            ? currentValue.toFixed(2)
            : currentValue;

        if (meta.type === 'select') {
            return (
                <div key={field.name} className="space-y-1.5 p-2 rounded-xl hover:bg-neutral-50 dark:hover:bg-neutral-800/40 transition">
                    <div className="flex justify-between items-center text-xs font-medium">
                        <span className="flex items-center gap-1.5 text-neutral-700 dark:text-neutral-300">
                            <meta.icon className="w-3.5 h-3.5 text-emerald-500" />
                            {meta.label}
                        </span>
                        <span className="font-mono font-bold text-neutral-900 dark:text-white bg-neutral-100 dark:bg-neutral-800 px-2 py-0.5 rounded text-[11px]">
                            {displayed}
                        </span>
                    </div>
                    <select
                        aria-label={meta.label}
                        value={currentValue}
                        onChange={(e) => handleFieldChange(field.name, e.target.value)}
                        className="w-full px-2 py-1.5 rounded-lg text-xs bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-700 text-neutral-900 dark:text-white"
                    >
                        {meta.options.map(option => <option key={option} value={option}>{option}</option>)}
                    </select>
                    <p className="text-[10px] text-neutral-500">{meta.help}</p>
                </div>
            );
        }

        return (
            <div key={field.name} className="space-y-1.5 p-2 rounded-xl hover:bg-neutral-50 dark:hover:bg-neutral-800/40 transition">
                <div className="flex justify-between items-center text-xs font-medium">
                    <span className="flex items-center gap-1.5 text-neutral-700 dark:text-neutral-300">
                        <meta.icon className="w-3.5 h-3.5 text-emerald-500" />
                        {meta.label}
                    </span>
                    <span className="font-mono font-bold text-neutral-900 dark:text-white bg-neutral-100 dark:bg-neutral-800 px-2 py-0.5 rounded text-[11px]">
                        {displayed} {meta.unit}
                    </span>
                </div>
                <input
                    type="range"
                    aria-label={meta.label}
                    min={meta.min}
                    max={meta.max}
                    step={meta.step}
                    value={currentValue}
                    onChange={(e) => handleFieldChange(field.name, parseFloat(e.target.value))}
                    className="w-full h-1.5 bg-neutral-200 dark:bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
                <div className="flex justify-between text-[10px] text-neutral-400">
                    <span>{meta.min}</span>
                    <span>{meta.help}</span>
                    <span>{meta.max}</span>
                </div>
            </div>
        );
    };

    // Ejecutar evaluación: el número mostrado siempre viene de la API.
    const handlePredecir = async () => {
        setIsPredicting(true);
        setApiError(null);

        if (!apiConnected) {
            setPrediccion(null);
            setApiError('La API no está disponible. No se muestra ningún valor sustituto; revise VITE_API_URL, CORS/Cloudflare Access y el estado del servicio.');
            setIsPredicting(false);
            return;
        }

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 4000);
            const payload = {
                municipio: formValues.municipio,
                municipality_code: formValues.municipality_code,
                ...Object.fromEntries(FEATURE_NAMES.map(name => [name, formValues[name]]))
            };

            const res = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify(payload),
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (!res.ok) {
                setPrediccion(null);
                setApiError(await apiErrorMessage(res, 'La API rechazó la evaluación.'));
                return;
            }

            const data = await res.json();
            const resData = data.results || data;
            const probability = resData.probability ?? resData.probabilidad;
            if (typeof probability !== 'number' || !Number.isFinite(probability)) {
                setPrediccion(null);
                setApiError('La API respondió sin una probabilidad válida; no se muestra ningún valor sustituto.');
                return;
            }

            const apiThreshold = data.threshold_used ?? resData.threshold_used;
            if (typeof apiThreshold === 'number' && Number.isFinite(apiThreshold)) {
                setThreshold(apiThreshold);
            }
            setPrediccion({
                probabilidad: probability,
                alertaActiva: resData.alerta_activa ?? probability >= (apiThreshold ?? threshold),
                source: `API (${data.stage || 'Staging'} v${data.model_version || '1.0'})`,
                modelName: data.model_name || selectedMunicipio.modelo_activo || 'ModeloPrueba',
                modelVersion: data.model_version || 'v1.0',
                municipioEvaluado: data.municipio_evaluado || selectedMunicipio.nombre,
                timestamp: new Date().toLocaleTimeString()
            });
        } catch (e) {
            setPrediccion(null);
            setApiError(e.name === 'AbortError'
                ? 'Tiempo de espera agotado al consultar /predict; no se muestra ningún valor sustituto.'
                : `No se pudo conectar con ${API_BASE_URL}/predict (${e.message}).`);
        } finally {
            setIsPredicting(false);
        }
    };

    // Modo v0.2: la API deriva las seis variables y devuelve el esquema 0.2 completo.
    const handlePredictV2 = async () => {
        setIsPredictingV2(true);
        setV2Error(null);
        setV2Data(null);

        if (!apiConnected) {
            setV2Error('La API no está disponible. No se muestra ningún valor sustituto; revise VITE_API_URL, CORS/Cloudflare Access y el estado del servicio.');
            setIsPredictingV2(false);
            return;
        }

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 8000);
            let data;
            try {
                data = await requestV2Prediction(formValues.municipio, v2Week, controller.signal);
            } finally {
                clearTimeout(timeoutId);
            }
            setV2Data(data);
            setMapStatuses(prev => ({
                ...prev,
                [formValues.municipio]: { status: 'ready', data }
            }));
        } catch (e) {
            setV2Error(e.name === 'AbortError'
                ? 'Tiempo de espera agotado al consultar /predict; no se muestra ningún valor sustituto.'
                : e.message?.startsWith('HTTP ') || e.message?.startsWith('La API no devolvió')
                    ? e.message
                    : `No se pudo conectar con ${API_BASE_URL}/predict (${e.message}).`);
        } finally {
            setIsPredictingV2(false);
        }
    };

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            {/* Header / Hero */}
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 p-5 sm:p-6 rounded-2xl bg-gradient-to-r from-neutral-900 via-neutral-900 to-neutral-800 text-white shadow-xl border border-neutral-800">
                <div className="space-y-1">
                    <div className="flex flex-wrap items-center gap-2">
                        <span className="flex h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                        <span className="text-xs uppercase tracking-wider font-semibold text-emerald-400">
                            Vigilancia Epidemiológica de Precisión
                        </span>
                        <span className="text-xs px-2 py-0.5 rounded-full bg-neutral-800 text-neutral-300 border border-neutral-700">
                            Semana 42
                        </span>
                    </div>
                    <h2 className="text-xl sm:text-2xl lg:text-3xl font-extrabold tracking-tight">
                        Sistema de Alerta Temprana de Dengue
                    </h2>
                    <p className="text-xs sm:text-sm text-neutral-400 max-w-2xl">
                        Pronóstico por municipio a partir de seis variables observadas en la semana de emisión t; el resultado es la probabilidad devuelta por la API.
                    </p>
                </div>
                
                <div className="flex items-center gap-2 shrink-0">
                    <div className="px-3 py-1.5 rounded-xl bg-neutral-800/90 border border-neutral-700 text-xs text-neutral-300 flex items-center gap-2">
                        <span className={`w-2 h-2 rounded-full ${apiConnected ? 'bg-emerald-400' : 'bg-rose-500'}`} />
                        <span className="font-medium">{apiConnected ? 'API En Línea' : 'API No Disponible'}</span>
                    </div>
                </div>
            </div>

            {/* Indicadores Clave de Vigilancia Epidemiológica */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* 1. Municipio Activo */}
                <div className="p-4 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/50 shadow-sm">
                    <div className="flex items-center justify-between text-neutral-500 dark:text-neutral-400 mb-1">
                        <span className="text-[11px] font-semibold uppercase tracking-wider">Municipio en Monitoreo</span>
                        <MapPin className="w-4 h-4 text-emerald-500" />
                    </div>
                    <div className="text-xl font-extrabold text-neutral-900 dark:text-white truncate">
                        {selectedMunicipio.nombre}
                    </div>
                    <p className="text-xs text-neutral-500 mt-0.5 truncate">{selectedMunicipio.departamento} • DANE {selectedMunicipio.codigo}</p>
                </div>

                {/* 2. Frecuencia histórica en zona epidémica */}
                <div className="p-4 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/50 shadow-sm">
                    <div className="flex items-center justify-between text-neutral-500 dark:text-neutral-400 mb-1">
                        <span className="text-[11px] font-semibold uppercase tracking-wider">Semanas en zona epidémica</span>
                        <Activity className="w-4 h-4 text-rose-500" />
                    </div>
                    <div className="text-xl font-extrabold text-neutral-900 dark:text-white">
                        {selectedMunicipio.porcentaje_semanas_epidemia_2012_2025.toFixed(1).replace('.', ',')} %
                    </div>
                    <p className="text-xs text-neutral-500 mt-0.5">Histórico 2012–2025 · Tabla 1 del informe</p>
                </div>

                {/* 3. Densidad Poblacional (DANE) */}
                <div className="p-4 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/50 shadow-sm">
                    <div className="flex items-center justify-between text-neutral-500 dark:text-neutral-400 mb-1">
                        <span className="text-[11px] font-semibold uppercase tracking-wider">Densidad Poblacional</span>
                        <Users className="w-4 h-4 text-sky-500" />
                    </div>
                    <div className="flex items-baseline gap-1.5">
                        <span className="text-xl font-extrabold text-neutral-900 dark:text-white font-mono">
                            {selectedMunicipio.densidad_poblacional.toLocaleString()}
                        </span>
                        <span className="text-xs font-medium text-neutral-500">hab/km²</span>
                    </div>
                    <p className="text-xs text-neutral-500 mt-0.5">Asignada automáticamente por mapa</p>
                </div>

                {/* 4. Semana Epidemiológica */}
                <div className="p-4 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/50 shadow-sm">
                    <div className="flex items-center justify-between text-neutral-500 dark:text-neutral-400 mb-1">
                        <span className="text-[11px] font-semibold uppercase tracking-wider">Vigilancia SIVIGILA</span>
                        <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                    </div>
                    <div className="text-xl font-extrabold text-neutral-900 dark:text-white">
                        Semana 42
                    </div>
                    <p className="text-xs text-emerald-600 dark:text-emerald-400 mt-0.5">Monitoreo activo y sincronizado</p>
                </div>
            </div>

            {/* Main Interactive Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Formulario / Simulador (Left - 5 Cols) */}
                <div className="lg:col-span-5 p-5 sm:p-6 rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/40 shadow-sm space-y-5">
                  {NEW_MODE ? (
                    <div className="space-y-5">
                      <div className="border-b border-neutral-100 dark:border-neutral-800 pb-3">
                        <div className="flex items-center gap-2 mb-2">
                          <Layers className="w-4 h-4 text-emerald-500" />
                          <h3 className="font-bold text-base text-neutral-900 dark:text-white">
                            Modo v0.2 · captura automática
                          </h3>
                        </div>
                        <p className="text-[11px] text-neutral-500">
                          La API calcula las seis variables desde el panel consolidado. Solo elige el municipio en el mapa y la semana de emisión.
                        </p>
                      </div>

                      <div className="space-y-1.5">
                        <label htmlFor="semana-emision" className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
                          Semana de emisión
                        </label>
                        <select
                          id="semana-emision"
                          aria-label="Semana de emisión"
                          value={v2Week}
                          onChange={(e) => {
                            setV2Week(e.target.value);
                            setV2Data(null);
                            setV2Error(null);
                            setMapStatuses(Object.fromEntries(
                                activeCodesKey.split(',').filter(Boolean).map(codigo => [codigo, { status: 'loading' }])
                            ));
                          }}
                          className="w-full px-2 py-1.5 rounded-lg text-xs bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-700 text-neutral-900 dark:text-white"
                        >
                          {V2_WEEK_OPTIONS.map((week) => <option key={week} value={week}>{week}</option>)}
                        </select>
                        <p className="text-[10px] text-neutral-500">Lunes W-SUN, dentro de la ventana vigente de la API.</p>
                      </div>

                      {v2Error && (
                        <div role="alert" className="p-3 rounded-xl bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-900/60">
                          <div className="flex items-start gap-2">
                            <AlertTriangle className="w-4 h-4 shrink-0 text-rose-600 dark:text-rose-400 mt-0.5" />
                            <p className="font-mono text-[11px] text-rose-700 dark:text-rose-300 break-words">{v2Error}</p>
                          </div>
                        </div>
                      )}

                      <button
                        type="button"
                        onClick={handlePredictV2}
                        disabled={isPredictingV2}
                        className="w-full py-2.5 px-4 rounded-xl font-semibold text-sm transition flex items-center justify-center gap-2 shadow-sm bg-emerald-600 hover:bg-emerald-700 active:scale-[0.99] text-white cursor-pointer"
                      >
                        <Zap className={`w-4 h-4 ${isPredictingV2 ? 'animate-spin' : ''}`} />
                        {isPredictingV2 ? 'Consultando la API...' : '🔍 Consultar pronóstico'}
                      </button>

                      {v2Data ? (
                        <div className="p-4 rounded-xl border border-emerald-200 dark:border-emerald-900/50 bg-emerald-50/80 dark:bg-emerald-950/30 space-y-3">
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-bold uppercase tracking-wider text-neutral-600 dark:text-neutral-300">
                              Predicción v0.2 · semana objetivo {v2Data.semana_objetivo}
                            </span>
                            <span className="text-[10px] text-neutral-400">esquema {v2Data.version_esquema}</span>
                          </div>

                          <div>
                            <span className="text-3xl font-extrabold tracking-tight text-neutral-900 dark:text-white">
                              {(v2Data.probabilidad * 100).toFixed(1)}%
                            </span>
                            <p className="text-xs text-neutral-600 dark:text-neutral-300">
                              {v2Data.etiqueta_probabilidad === 'seguir_en_zona_epidemica'
                                ? 'Probabilidad de seguir en zona epidémica'
                                : 'Probabilidad de entrar en zona epidémica'}
                            </p>
                          </div>

                          <div className="grid grid-cols-2 gap-2 text-[11px]">
                            <div className="p-2 rounded-lg bg-white dark:bg-neutral-900/60">
                              <span className="block text-neutral-500">Referencia: lo esperable solo por la zona actual del canal (B2)</span>
                              <span className="font-mono font-bold">{(v2Data.referencias.b2_estado_canal * 100).toFixed(1)}%</span>
                            </div>
                            <div className="p-2 rounded-lg bg-white dark:bg-neutral-900/60">
                              <span className="block text-neutral-500">Referencia: lo esperable solo por la época del año (B1)</span>
                              <span className="font-mono font-bold">{(v2Data.referencias.b1_climatologia_estacional * 100).toFixed(1)}%</span>
                            </div>
                          </div>

                          <div className="text-[11px] text-neutral-700 dark:text-neutral-300 space-y-1 pt-2 border-t border-neutral-200/80 dark:border-neutral-800/80">
                            <p className="flex items-center gap-2">
                              <strong>Estado:</strong>
                              <span className={`rounded-full px-2 py-0.5 font-bold ${
                                v2Data.decision.estado_alerta === 'continuidad_epidemica'
                                  ? 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300'
                                  : v2Data.decision.estado_alerta === 'sin_alerta'
                                    ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300'
                                    : 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300'
                              }`}>
                                {ALERT_STATE_LABELS[v2Data.decision.estado_alerta] || String(v2Data.decision.estado_alerta).replaceAll('_', ' ')}
                              </span>
                            </p>

                            <details className="my-2 rounded-lg border border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900/60 p-2.5">
                              <summary className="cursor-pointer select-none font-semibold text-neutral-800 dark:text-neutral-200">
                                ¿Cómo leer este resultado?
                              </summary>
                              <div className="mt-2 overflow-x-auto">
                                <table className="w-full text-left text-[10px]">
                                  <thead className="bg-neutral-50 dark:bg-neutral-800/70 text-neutral-500">
                                    <tr>
                                      <th className="px-2 py-1.5 font-semibold">Estado</th>
                                      <th className="px-2 py-1.5 font-semibold">Qué significa</th>
                                    </tr>
                                  </thead>
                                  <tbody className="divide-y divide-neutral-100 dark:divide-neutral-800">
                                    <tr>
                                      <td className="px-2 py-1.5 font-semibold whitespace-nowrap">Sin alerta</td>
                                      <td className="px-2 py-1.5">No hay señal esta semana</td>
                                    </tr>
                                    <tr>
                                      <td className="px-2 py-1.5 font-semibold whitespace-nowrap">Señal en observación</td>
                                      <td className="px-2 py-1.5">Una semana con señal, fuera de zona epidémica</td>
                                    </tr>
                                    <tr>
                                      <td className="px-2 py-1.5 font-semibold whitespace-nowrap">Alerta de inicio confirmada</td>
                                      <td className="px-2 py-1.5">Dos semanas seguidas con señal: posible inicio de brote</td>
                                    </tr>
                                    <tr>
                                      <td className="px-2 py-1.5 font-semibold whitespace-nowrap">Continuidad epidémica</td>
                                      <td className="px-2 py-1.5">El municipio ya está en zona epidémica; la probabilidad es la de seguir en ella</td>
                                    </tr>
                                  </tbody>
                                </table>
                              </div>
                              <p className="mt-2 leading-relaxed text-neutral-500">
                                La probabilidad del modelo se compara con dos referencias. Si es mayor que la referencia, el modelo ve más riesgo del que indica el canal por sí solo.
                              </p>
                            </details>

                            <p><strong>Zona observada en t:</strong> {v2Data.canal_actual.zona} · <strong>Casos:</strong> {v2Data.canal_actual.casos}</p>
                            <p><strong>Tendencia central del canal (casos esperados):</strong> {v2Data.canal_actual.central}</p>
                            <p><strong>Límite superior del canal (por encima: zona epidémica):</strong> {v2Data.canal_actual.u1e}</p>
                            <p><strong>Umbral de señal:</strong> probabilidad de {(v2Data.decision.tau * 100).toLocaleString('es-CO')} % o más</p>
                            <p>La alerta se confirma con {v2Data.decision.k} señales seguidas</p>
                            <p><strong>Señales seguidas hasta hoy:</strong> {v2Data.decision.emisiones_positivas_consecutivas}</p>
                            <p><strong>Modelo:</strong> <span className="font-mono">{v2Data.versiones.modelo}</span></p>
                          </div>

                          <div className="pt-2 border-t border-neutral-200/80 dark:border-neutral-800/80 space-y-4">
                            <ChannelSeriesChart
                              serie={v2Data.serie_canal}
                              semanaEmision={v2Data.semana_emision}
                              semanaObjetivo={v2Data.semana_objetivo}
                            />
                            <EmissionTrajectoryTable
                              trayectoria={v2Data.trayectoria}
                              continuidad={v2Data.continuidad}
                              decision={v2Data.decision}
                            />
                            <details className="text-[10px] text-neutral-600 dark:text-neutral-400">
                              <summary className="cursor-pointer">Ver datos completos (JSON)</summary>
                              <pre className="mt-2 max-h-48 overflow-auto rounded-lg bg-white dark:bg-neutral-900/60 p-2 font-mono">
                                {JSON.stringify({ serie_canal: v2Data.serie_canal, trayectoria: v2Data.trayectoria, continuidad: v2Data.continuidad }, null, 2)}
                              </pre>
                            </details>
                          </div>

                          <ul className="text-[10px] text-neutral-500 list-disc list-inside space-y-0.5">
                            {v2Data.advertencias.map((item) => <li key={item.codigo}>{item.texto}</li>)}
                          </ul>
                        </div>
                      ) : (
                        <div className="p-4 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-900/50">
                          <p className="text-xs font-bold uppercase tracking-wider text-neutral-500">Predicción de la API</p>
                          <p className="mt-2 text-xs text-neutral-700 dark:text-neutral-300 leading-relaxed">
                            Sin evaluación. Elige la semana de emisión y pulsa «Consultar pronóstico».
                          </p>
                        </div>
                      )}
                    </div>
                  ) : (
                    <>
                    <div className="border-b border-neutral-100 dark:border-neutral-800 pb-3">
                        <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                                <Sliders className="w-4 h-4 text-emerald-500" />
                                <h3 className="font-bold text-base text-neutral-900 dark:text-white">
                                    Variables observadas en t
                                </h3>
                            </div>
                            <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 font-medium">
                                {selectedMunicipio.nombre}
                            </span>
                        </div>
                        <p className="text-[11px] text-neutral-500 mb-3">
                            Ingresa las seis variables <strong>observadas en la semana de emisión t</strong> —incluida la zona del canal, que no se predice— y consulta la API. El resultado es la probabilidad devuelta por la API para la semana objetivo.
                        </p>
                    </div>

                    {/* Inputs del Formulario */}
                    <div className="space-y-2.5">
                        {schemaFields.map((field) => renderFieldInput(field))}
                        {schemaWarning && (
                            <p className="p-2 rounded-lg text-[10px] bg-amber-50 dark:bg-amber-950/40 text-amber-800 dark:text-amber-300 border border-amber-200 dark:border-amber-900/60">
                                {schemaWarning}
                            </p>
                        )}
                    </div>

                    {/* Error visible: nunca se sustituye por un valor local */}
                    {apiError && (
                        <div role="alert" className="p-3 rounded-xl bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-900/60">
                            <div className="flex items-start gap-2">
                                <AlertTriangle className="w-4 h-4 shrink-0 text-rose-600 dark:text-rose-400 mt-0.5" />
                                <div className="min-w-0 space-y-1">
                                    <p className="text-xs font-bold text-rose-800 dark:text-rose-300">No se pudo obtener la predicción de la API</p>
                                    <p className="font-mono text-[11px] text-rose-700 dark:text-rose-300 break-words">{apiError}</p>
                                    <p className="text-[10px] text-rose-600/80 dark:text-rose-400/80">
                                        No se muestra ningún valor sustituto. Revise VITE_API_URL, CORS/Cloudflare Access y el esquema del modelo.
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Botón de Evaluación */}
                    <button
                        type="button"
                        onClick={handlePredecir}
                        disabled={isPredicting}
                        className="w-full py-2.5 px-4 rounded-xl font-semibold text-sm transition flex items-center justify-center gap-2 shadow-sm bg-emerald-600 hover:bg-emerald-700 active:scale-[0.99] text-white cursor-pointer"
                    >
                        <Zap className={`w-4 h-4 ${isPredicting ? 'animate-spin' : ''}`} />
                        {isPredicting ? 'Consultando la API...' : '🔍 Consultar predicción de la API'}
                    </button>

                    {/* Caja de Resultado: solo se muestra si la API respondió */}
                    {prediccion ? (
                        <div className={`p-4 rounded-xl border transition-all ${
                            prediccion.alertaActiva 
                                ? 'bg-rose-50/80 border-rose-200 dark:bg-rose-950/30 dark:border-rose-900/50' 
                                : 'bg-emerald-50/80 border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-900/50'
                        }`}>
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-xs font-bold uppercase tracking-wider text-neutral-600 dark:text-neutral-300">
                                    Predicción de la API · semana objetivo t+8
                                </span>
                                <span className="text-[10px] text-neutral-400">
                                    {selectedMunicipio.nombre} • {prediccion.source}
                                </span>
                            </div>
                            <p className="text-[10px] text-neutral-500 mb-2">
                                Probabilidad de que la semana objetivo esté en zona epidémica (C &gt; U1e). La API actual no informa la fecha de esa semana.
                            </p>
                            
                            <div className="flex items-baseline justify-between mb-2.5">
                                <div>
                                    <span className="text-3xl font-extrabold tracking-tight text-neutral-900 dark:text-white">
                                        {(prediccion.probabilidad * 100).toFixed(1)}%
                                    </span>
                                    <span className="text-xs text-neutral-500 ml-1">de zona epidémica en t+8</span>
                                </div>
                                <div>
                                    {prediccion.alertaActiva ? (
                                        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-bold bg-rose-600 text-white shadow-sm">
                                            <AlertTriangle className="w-3.5 h-3.5" /> SEÑAL ≥ UMBRAL (API)
                                        </span>
                                    ) : (
                                        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-bold bg-emerald-600 text-white shadow-sm">
                                            <CheckCircle2 className="w-3.5 h-3.5" /> SEÑAL &lt; UMBRAL (API)
                                        </span>
                                    )}
                                </div>
                            </div>

                            {/* Barra de Umbral Visual Clara */}
                            <div className="space-y-1">
                                <div className="relative w-full h-2.5 bg-neutral-200 dark:bg-neutral-800 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full rounded-full transition-all duration-300 ${prediccion.alertaActiva ? 'bg-rose-500' : 'bg-emerald-500'}`}
                                        style={{ width: `${Math.min(prediccion.probabilidad * 100, 100)}%` }}
                                    />
                                </div>
                                <div className="flex justify-between text-[10px] text-neutral-500">
                                    <span>Por debajo del umbral</span>
                                    <span className="font-semibold text-neutral-700 dark:text-neutral-300">
                                        Umbral de la API: {(threshold * 100).toFixed(1)}%
                                    </span>
                                    <span>Por encima</span>
                                </div>
                            </div>

                            {/* Alcance real de la señal: la API actual no aplica la regla D10.1 */}
                            <p className="text-[10px] text-neutral-600 dark:text-neutral-400 mt-3 pt-2.5 border-t border-neutral-200/80 dark:border-neutral-800/80 leading-relaxed">
                                La API actual decide esta señal solo con <code>probability ≥ umbral</code>: no aplica la regla del producto (zona observada distinta de epidemia y dos emisiones consecutivas) ni devuelve referencias B1/B2. Este número depende de los valores introducidos; no es una emisión validada del producto.
                            </p>

                            {/* Badge del Modelo MLflow Ejecutado */}
                            <div className="mt-2.5 pt-2 border-t border-neutral-200/60 dark:border-neutral-800/60 flex flex-wrap items-center justify-between gap-1 text-[11px]">
                                <div className="flex items-center gap-1.5 text-neutral-600 dark:text-neutral-400">
                                    <Layers className="w-3.5 h-3.5 text-emerald-500" />
                                    <span>Modelo MLflow:</span>
                                    <span className="font-mono font-bold text-neutral-900 dark:text-white">
                                        {prediccion.modelName || 'ModeloPrueba'}
                                    </span>
                                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-neutral-200 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300">
                                        v{prediccion.modelVersion || '1.0'}
                                    </span>
                                </div>
                                <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                                    prediccion.modelName?.startsWith('DengueAlertas_')
                                        ? 'bg-blue-100 dark:bg-blue-950/60 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800'
                                        : 'bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400'
                                }`}>
                                    {prediccion.modelName?.startsWith('DengueAlertas_') ? '🎯 Especializado Municipal' : '🌐 Modelo Base'}
                                </span>
                            </div>
                        </div>
                    ) : (
                        <div className="p-4 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-900/50">
                            <p className="text-xs font-bold uppercase tracking-wider text-neutral-500">Predicción de la API</p>
                            <p className="mt-2 text-xs text-neutral-700 dark:text-neutral-300 leading-relaxed">
                                Sin evaluación. Ingresa las seis variables observadas en t y pulsa «Consultar predicción de la API».
                            </p>
                        </div>
                    )}
                    </>
                  )}
                </div>

                {/* Mapa Epidemiológico de Colombia (Right - 7 Cols) */}
                <div className="lg:col-span-7 p-6 rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/40 shadow-sm space-y-6">
                    <ColombiaMap
                        selectedMunicipioId={selectedMunicipio.id}
                        onSelectMunicipio={handleSelectMunicipio}
                        activeCodes={activeCodes}
                        activeModelsStatus={activeModelsStatus}
                        mapStatuses={NEW_MODE ? mapStatuses : {}}
                        emissionWeek={NEW_MODE ? v2Week : undefined}
                    />
                </div>
            </div>

            {/* Trazabilidad MLOps (Colapsable para no sobrecargar al usuario) */}
            <div className="rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-900/40 overflow-hidden">
                <button
                    type="button"
                    onClick={() => setShowTechDetails(!showTechDetails)}
                    className="w-full p-4 flex items-center justify-between text-left text-xs font-semibold text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white transition cursor-pointer"
                >
                    <div className="flex items-center gap-2">
                        <Layers className="w-4 h-4 text-emerald-500" />
                        <span>Detalles Técnicos & Gobernanza MLOps</span>
                        <span className="text-[10px] px-2 py-0.5 rounded-full bg-neutral-200 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 font-mono">
                            {modelInfo.model_name} (v{modelInfo.version}) • {modelInfo.stage}
                        </span>
                    </div>
                    <ChevronDown className={`w-4 h-4 transition-transform duration-200 ${showTechDetails ? 'rotate-180' : ''}`} />
                </button>

                {showTechDetails && (
                    <div className="p-4 pt-0 border-t border-neutral-200/60 dark:border-neutral-800/60 animate-in fade-in duration-200">
                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs mt-3">
                            <div className="p-3 bg-white dark:bg-neutral-800/50 rounded-xl border border-neutral-200 dark:border-neutral-700/60">
                                <span className="text-neutral-400 block mb-1">API Backend</span>
                                <span className="font-mono font-semibold text-neutral-900 dark:text-emerald-400 break-all">{API_BASE_URL}</span>
                            </div>
                            <div className="p-3 bg-white dark:bg-neutral-800/50 rounded-xl border border-neutral-200 dark:border-neutral-700/60">
                                <span className="text-neutral-400 block mb-1">Servidor MLflow</span>
                                <span className="font-mono font-semibold text-neutral-900 dark:text-white">platform.arkoiot.com</span>
                            </div>
                            <div className="p-3 bg-white dark:bg-neutral-800/50 rounded-xl border border-neutral-200 dark:border-neutral-700/60">
                                <span className="text-neutral-400 block mb-1">Modelo Activo</span>
                                <span className="font-mono font-semibold text-neutral-900 dark:text-white">{modelInfo.model_name} (v{modelInfo.version})</span>
                            </div>
                            <div className="p-3 bg-white dark:bg-neutral-800/50 rounded-xl border border-neutral-200 dark:border-neutral-700/60">
                                <span className="text-neutral-400 block mb-1">Despliegue Edge</span>
                                <span className="font-mono font-semibold text-neutral-900 dark:text-white">Cloudflare Pages (Blue/Green)</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
