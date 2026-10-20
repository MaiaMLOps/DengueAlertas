import React, { useState, useEffect } from 'react';
import {
    Activity,
    AlertTriangle,
    Droplets,
    Thermometer,
    Layers,
    Zap,
    MapPin,
    CheckCircle2,
    Sliders,
    ArrowUpRight,
    Wifi,
    WifiOff,
    ChevronDown,
    Sparkles,
    Users
} from 'lucide-react';
import ColombiaMap, { MUNICIPIOS_ESTRATEGICOS } from './ColombiaMap';

// Metadatos amigables para variables de MLflow
const FIELD_META = {
    // Variables generales / Fallback
    temperatura: { label: "Temperatura Promedio", unit: "°C", min: 20, max: 38, step: 0.5, default: 28.5, icon: Thermometer },
    precipitacion: { label: "Precipitación Acumulada", unit: "mm", min: 10, max: 350, step: 5, default: 140, icon: Droplets },
    humedad: { label: "Humedad Relativa", unit: "%", min: 40, max: 98, step: 1, default: 82, icon: Activity },
    indice_aedico: { label: "Índice Aédico / Larvario", unit: "%", min: 1, max: 30, step: 0.5, default: 14.5, icon: Sliders },
    casos_previos: { label: "Casos Últimas 4 Semanas", unit: "casos", min: 0, max: 300, step: 1, default: 110, icon: AlertTriangle },
    densidad_poblacional: { label: "Densidad Poblacional", unit: "hab/km²", min: 100, max: 10000, step: 10, default: 3910, icon: MapPin },

    // Variables MLOps 15 Features
    casos_lag_1: { label: "Casos Semana Anterior (t-1)", unit: "casos", min: 0, max: 500, step: 1, default: 95, icon: AlertTriangle },
    casos_lag_2: { label: "Casos Hace 2 Semanas (t-2)", unit: "casos", min: 0, max: 500, step: 1, default: 85, icon: AlertTriangle },
    casos_lag_3: { label: "Casos Hace 3 Semanas (t-3)", unit: "casos", min: 0, max: 500, step: 1, default: 80, icon: AlertTriangle },
    casos_lag_4: { label: "Casos Hace 4 Semanas (t-4)", unit: "casos", min: 0, max: 500, step: 1, default: 70, icon: AlertTriangle },
    casos_rolling_mean_4w: { label: "Promedio Casos 4 Semanas", unit: "casos", min: 0, max: 500, step: 1, default: 82.5, icon: AlertTriangle },
    temperature_c: { label: "Temperatura Promedio", unit: "°C", min: 15, max: 40, step: 0.5, default: 26.5, icon: Thermometer },
    temperatura_lag_2: { label: "Temperatura Hace 2 Semanas", unit: "°C", min: 15, max: 40, step: 0.5, default: 26.0, icon: Thermometer },
    precipitation_mm: { label: "Precipitación Acumulada", unit: "mm", min: 0, max: 350, step: 5, default: 120, icon: Droplets },
    precipitacion_lag_2: { label: "Precipitación Hace 2 Semanas", unit: "mm", min: 0, max: 350, step: 5, default: 130, icon: Droplets },
    precipitacion_lag_4: { label: "Precipitación Hace 4 Semanas", unit: "mm", min: 0, max: 350, step: 5, default: 140, icon: Droplets },
    relative_humidity_pct: { label: "Humedad Relativa", unit: "%", min: 30, max: 100, step: 1, default: 84, icon: Activity },
    humedad_lag_2: { label: "Humedad Hace 2 Semanas", unit: "%", min: 30, max: 100, step: 1, default: 85, icon: Activity },
    population_density_km2: { label: "Densidad Poblacional", unit: "hab/km²", min: 100, max: 10000, step: 10, default: 3910, icon: MapPin },
    sin_week: { label: "Semana del Año (Seno)", unit: "", min: -1, max: 1, step: 0.05, default: 0.5, icon: Sliders },
    cos_week: { label: "Semana del Año (Coseno)", unit: "", min: -1, max: 1, step: 0.05, default: 0.8, icon: Sliders },
};

const DEFAULT_FALLBACK_SCHEMA = [
    { name: 'casos_previos', type: 'integer', required: true },
    { name: 'temperatura', type: 'double', required: true },
    { name: 'precipitacion', type: 'double', required: true },
    { name: 'humedad', type: 'double', required: true },
    { name: 'densidad_poblacional', type: 'double', required: true },
    { name: 'indice_aedico', type: 'double', required: false },
];

export default function Dashboard() {
    const API_BASE_URL = "https://api.arkoiot.com";

    // Municipio Estratégico Seleccionado en el Mapa (por defecto Cali)
    const [selectedMunicipio, setSelectedMunicipio] = useState(MUNICIPIOS_ESTRATEGICOS[0]);
    const [activeCodes, setActiveCodes] = useState(['76001', '68001', '13001', '73001', '50001']);
    const [showAdvancedConfig, setShowAdvancedConfig] = useState(false);
    const [showTechDetails, setShowTechDetails] = useState(false);

    // Estado del Schema y estado de la API
    const [schemaFields, setSchemaFields] = useState(DEFAULT_FALLBACK_SCHEMA);
    const [apiConnected, setApiConnected] = useState(false);
    const [modelInfo, setModelInfo] = useState({
        stage: 'Production',
        version: '1.1.0',
        model_name: 'ModeloPrueba'
    });

    const THRESHOLD = 0.205;

    // Estimación epidemiológica local de respaldo para respuesta inmediata y modo offline
    const computeLocalRisk = (values, mun) => {
        const temp = values.temperatura ?? values.temperature_c ?? mun.temp_sugerida ?? 26.5;
        const precip = values.precipitacion ?? values.precipitation_mm ?? mun.precip_sugerida ?? 120;
        const hum = values.humedad ?? values.relative_humidity_pct ?? mun.humedad_sugerida ?? 82;
        const casos = values.casos_previos ?? values.casos_lag_1 ?? 85;

        let score = mun.probabilidad_base;
        // Factor temperatura (óptimo biológico del vector 26°C - 32°C)
        if (temp >= 26 && temp <= 32) score += 0.06;
        else if (temp < 22) score -= 0.12;

        // Factor precipitación (estancamiento y criaderos)
        if (precip > 140) score += 0.08;
        else if (precip < 45) score -= 0.07;

        // Factor humedad y casos recientes
        if (hum > 80) score += 0.04;
        if (casos > 100) score += 0.06;

        const prob = Math.min(Math.max(score, 0.05), 0.95);
        return {
            probabilidad: prob,
            alertaActiva: prob >= THRESHOLD,
            source: 'Simulación Epidemiológica',
            timestamp: new Date().toLocaleTimeString()
        };
    };

    // Valores dinámicos del formulario (inicializados con los del municipio inicial)
    const [formValues, setFormValues] = useState(() => {
        const initial = {
            municipio: MUNICIPIOS_ESTRATEGICOS[0].codigo,
            municipality_code: MUNICIPIOS_ESTRATEGICOS[0].codigo,
            densidad_poblacional: MUNICIPIOS_ESTRATEGICOS[0].densidad_poblacional,
            population_density_km2: MUNICIPIOS_ESTRATEGICOS[0].densidad_poblacional,
            temperatura: MUNICIPIOS_ESTRATEGICOS[0].temp_sugerida,
            temperature_c: MUNICIPIOS_ESTRATEGICOS[0].temp_sugerida,
            precipitacion: MUNICIPIOS_ESTRATEGICOS[0].precip_sugerida,
            precipitation_mm: MUNICIPIOS_ESTRATEGICOS[0].precip_sugerida,
            humedad: MUNICIPIOS_ESTRATEGICOS[0].humedad_sugerida,
            relative_humidity_pct: MUNICIPIOS_ESTRATEGICOS[0].humedad_sugerida,
            casos_previos: 90
        };
        DEFAULT_FALLBACK_SCHEMA.forEach(f => {
            if (initial[f.name] === undefined) {
                initial[f.name] = FIELD_META[f.name]?.default ?? 10;
            }
        });
        return initial;
    });

    // Estado de la predicción con evaluación inicial activa
    const [prediccion, setPrediccion] = useState(() => {
        const baseRisk = computeLocalRisk({
            temperatura: MUNICIPIOS_ESTRATEGICOS[0].temp_sugerida,
            precipitacion: MUNICIPIOS_ESTRATEGICOS[0].precip_sugerida,
            humedad: MUNICIPIOS_ESTRATEGICOS[0].humedad_sugerida,
            casos_previos: 90
        }, MUNICIPIOS_ESTRATEGICOS[0]);
        return {
            ...baseRisk,
            modelName: MUNICIPIOS_ESTRATEGICOS[0].modelo_activo,
            modelVersion: 'v1.0'
        };
    });
    const [isPredicting, setIsPredicting] = useState(false);

    // Consultar el esquema a https://api.arkoiot.com/schema
    const fetchSchema = async () => {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 3000);

            const res = await fetch(`${API_BASE_URL}/schema`, { 
                credentials: 'include',
                signal: controller.signal 
            });
            clearTimeout(timeoutId);

            if (res.ok) {
                const data = await res.json();
                if (data.input_schema && Array.isArray(data.input_schema) && data.input_schema.length > 0) {
                    setSchemaFields(data.input_schema);
                    setModelInfo({
                        stage: data.stage || 'Production',
                        version: data.version || '1.1.0',
                        model_name: data.model_name || 'ModeloPrueba'
                    });

                    setFormValues(prev => {
                        const updated = { ...prev };
                        data.input_schema.forEach(f => {
                            if (updated[f.name] === undefined) {
                                updated[f.name] = FIELD_META[f.name]?.default ?? 10;
                            }
                        });
                        return updated;
                    });
                }
                setApiConnected(true);
            } else {
                setApiConnected(false);
            }
        } catch (err) {
            setApiConnected(false);
        }
    };

    // Consultar municipios activos con tag 'enabled == true' en MLflow
    const fetchActiveMunicipios = async () => {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 3000);
            const res = await fetch(`${API_BASE_URL}/municipios`, {
                credentials: 'include',
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            if (res.ok) {
                const data = await res.json();
                if (data.active_codes && Array.isArray(data.active_codes) && data.active_codes.length > 0) {
                    setActiveCodes(data.active_codes);
                }
            }
        } catch (e) {
            // Mantener configuracion inicial
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
        } catch (e) {}
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
        fetchSchema();
        fetchActiveMunicipios();
        const interval = setInterval(() => {
            fetchSchema();
            fetchActiveMunicipios();
        }, 60000);
        return () => clearInterval(interval);
    }, []);

    const handleSelectMunicipio = (mun) => {
        setSelectedMunicipio(mun);
        const updatedValues = {
            ...formValues,
            densidad_poblacional: mun.densidad_poblacional,
            population_density_km2: mun.densidad_poblacional,
            municipality_code: mun.codigo,
            temperatura: mun.temp_sugerida ?? formValues.temperatura,
            temperature_c: mun.temp_sugerida ?? formValues.temperature_c,
            precipitacion: mun.precip_sugerida ?? formValues.precipitacion,
            precipitation_mm: mun.precip_sugerida ?? formValues.precipitation_mm,
            humedad: mun.humedad_sugerida ?? formValues.humedad,
            relative_humidity_pct: mun.humedad_sugerida ?? formValues.relative_humidity_pct,
        };
        setFormValues(updatedValues);
        setPrediccion(computeLocalRisk(updatedValues, mun));
        trackUiEvent('select_municipio', mun.nombre, 'colombia_map', {
            codigo: mun.codigo,
            departamento: mun.departamento,
            densidad: mun.densidad_poblacional
        });
    };

    const handleFieldChange = (name, value) => {
        setFormValues(prev => {
            const updated = {
                ...prev,
                [name]: value
            };
            return updated;
        });
    };

    // Aplicar escenarios rápidos preconfigurados
    const applyPreset = (type) => {
        setFormValues(prev => {
            const updated = { ...prev };
            if (type === 'habitual') {
                updated.temperatura = selectedMunicipio.temp_sugerida;
                updated.temperature_c = selectedMunicipio.temp_sugerida;
                updated.precipitacion = selectedMunicipio.precip_sugerida;
                updated.precipitation_mm = selectedMunicipio.precip_sugerida;
                updated.humedad = selectedMunicipio.humedad_sugerida;
                updated.relative_humidity_pct = selectedMunicipio.humedad_sugerida;
                updated.casos_previos = 75;
            } else if (type === 'lluvias') {
                updated.precipitacion = Math.min((selectedMunicipio.precip_sugerida || 120) + 70, 320);
                updated.precipitation_mm = updated.precipitacion;
                updated.humedad = 88;
                updated.relative_humidity_pct = 88;
                updated.casos_previos = 130;
            } else if (type === 'calor') {
                updated.temperatura = 32.5;
                updated.temperature_c = 32.5;
                updated.precipitacion = 40;
                updated.precipitation_mm = 40;
                updated.humedad = 65;
                updated.relative_humidity_pct = 65;
            }
            setPrediccion(computeLocalRisk(updated, selectedMunicipio));
            return updated;
        });
    };

    // Clasificación de campos requeridos vs opcionales
    const isOptionalField = (field) => {
        if (field.required === false) return true;
        if (field.required === true) return false;
        const optionalNames = [
            'indice_aedico',
            'temperatura_lag_2',
            'precipitacion_lag_2',
            'precipitacion_lag_4',
            'humedad_lag_2',
            'sin_week',
            'cos_week'
        ];
        return optionalNames.includes(field.name);
    };

    const requiredFields = schemaFields.filter(f => !isOptionalField(f));
    const optionalFields = schemaFields.filter(f => isOptionalField(f));

    const renderFieldInput = (field) => {
        // Si el campo es densidad poblacional, presentarlo como dato territorial auto-asignado del DANE
        if (field.name === 'densidad_poblacional' || field.name === 'population_density_km2') {
            return (
                <div key={field.name} className="p-3 rounded-xl bg-emerald-500/10 dark:bg-emerald-950/30 border border-emerald-500/20 flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                        <MapPin className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                        <div>
                            <span className="font-semibold text-neutral-800 dark:text-neutral-200 block">Densidad Poblacional (DANE)</span>
                            <span className="text-[10px] text-neutral-500">Autoasignada: {selectedMunicipio.nombre}</span>
                        </div>
                    </div>
                    <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">
                        {selectedMunicipio.densidad_poblacional.toLocaleString()} hab/km²
                    </span>
                </div>
            );
        }

        const meta = FIELD_META[field.name] || {
            label: field.name.replace(/_/g, ' ').toUpperCase(),
            unit: '',
            min: 0,
            max: 100,
            step: 1,
            default: 10
        };
        const currentValue = formValues[field.name] ?? meta.default;
        const isInteger = field.type?.toLowerCase().includes('integer') || field.type?.toLowerCase().includes('int');

        return (
            <div key={field.name} className="space-y-1.5 p-2 rounded-xl hover:bg-neutral-50 dark:hover:bg-neutral-800/40 transition">
                <div className="flex justify-between items-center text-xs font-medium">
                    <span className="flex items-center gap-1.5 text-neutral-700 dark:text-neutral-300">
                        {meta.icon ? <meta.icon className="w-3.5 h-3.5 text-emerald-500" /> : <Activity className="w-3.5 h-3.5 text-neutral-400" />}
                        <span className="capitalize">{meta.label}</span>
                    </span>
                    <span className="font-mono font-bold text-neutral-900 dark:text-white bg-neutral-100 dark:bg-neutral-800 px-2 py-0.5 rounded text-[11px]">
                        {typeof currentValue === 'number' && !Number.isInteger(currentValue) ? currentValue.toFixed(1) : currentValue} {meta.unit}
                    </span>
                </div>
                <input
                    type="range"
                    min={meta.min}
                    max={meta.max}
                    step={meta.step}
                    value={currentValue}
                    onChange={(e) => {
                        const val = isInteger ? parseInt(e.target.value) : parseFloat(e.target.value);
                        handleFieldChange(field.name, val);
                    }}
                    className="w-full h-1.5 bg-neutral-200 dark:bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
                <div className="flex justify-between text-[10px] text-neutral-400">
                    <span>{meta.min} {meta.unit}</span>
                    <span>{meta.max} {meta.unit}</span>
                </div>
            </div>
        );
    };

    // Ejecutar evaluación de riesgo
    const handlePredecir = async () => {
        setIsPredicting(true);

        if (!apiConnected) {
            setTimeout(() => {
                setPrediccion(computeLocalRisk(formValues, selectedMunicipio));
                setIsPredicting(false);
            }, 300);
            return;
        }

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 4000);

            const res = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify(formValues),
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (res.ok) {
                const data = await res.json();
                const resData = data.results || data;
                setPrediccion({
                    probabilidad: resData.probability,
                    alertaActiva: resData.alerta_activa,
                    source: `API (${data.stage || 'Staging'} v${data.model_version || '1.0'})`,
                    modelName: data.model_name || selectedMunicipio.modelo_activo || 'ModeloPrueba',
                    modelVersion: data.model_version || 'v1.0',
                    municipioEvaluado: data.municipio_evaluado || selectedMunicipio.nombre,
                    timestamp: new Date().toLocaleTimeString()
                });
                setApiConnected(true);
            } else {
                setPrediccion(computeLocalRisk(formValues, selectedMunicipio));
            }
        } catch (e) {
            setPrediccion(computeLocalRisk(formValues, selectedMunicipio));
        } finally {
            setIsPredicting(false);
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
                        Monitoreo preventivo y simulación de brotes en municipios prioritarios de Colombia.
                    </p>
                </div>
                
                <div className="flex items-center gap-2 shrink-0">
                    <div className="px-3 py-1.5 rounded-xl bg-neutral-800/90 border border-neutral-700 text-xs text-neutral-300 flex items-center gap-2">
                        <span className={`w-2 h-2 rounded-full ${apiConnected ? 'bg-emerald-400' : 'bg-emerald-500'}`} />
                        <span className="font-medium">{apiConnected ? 'API En Línea' : 'Simulación Activa'}</span>
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

                {/* 2. Nivel de Riesgo Base */}
                <div className="p-4 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/50 shadow-sm">
                    <div className="flex items-center justify-between text-neutral-500 dark:text-neutral-400 mb-1">
                        <span className="text-[11px] font-semibold uppercase tracking-wider">Nivel de Alerta Base</span>
                        <Activity className="w-4 h-4 text-rose-500" />
                    </div>
                    <div className="flex items-baseline gap-2">
                        <span className={`text-xl font-extrabold ${
                            selectedMunicipio.riesgo === 'Crítico' ? 'text-rose-600' :
                            selectedMunicipio.riesgo === 'Alto' ? 'text-amber-600' :
                            'text-sky-600'
                        }`}>
                            Riesgo {selectedMunicipio.riesgo}
                        </span>
                    </div>
                    <p className="text-xs text-neutral-500 mt-0.5">Probabilidad histórica: {(selectedMunicipio.probabilidad_base * 100).toFixed(0)}%</p>
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
                    <div className="border-b border-neutral-100 dark:border-neutral-800 pb-3">
                        <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                                <Sliders className="w-4 h-4 text-emerald-500" />
                                <h3 className="font-bold text-base text-neutral-900 dark:text-white">
                                    Simulador de Riesgo
                                </h3>
                            </div>
                            <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 font-medium">
                                {selectedMunicipio.nombre}
                            </span>
                        </div>
                        <p className="text-[11px] text-neutral-500 mb-3">
                            Ajusta los factores ambientales o prueba un escenario rápido:
                        </p>

                        {/* Botones de Escenarios Rápidos */}
                        <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="text-[10px] text-neutral-400 font-semibold uppercase">Escenarios:</span>
                            <button
                                type="button"
                                onClick={() => applyPreset('habitual')}
                                className="px-2 py-1 rounded-lg text-xs font-medium bg-neutral-100 dark:bg-neutral-800 hover:bg-neutral-200 dark:hover:bg-neutral-700 text-neutral-700 dark:text-neutral-200 transition cursor-pointer"
                            >
                                🌤️ Típico
                            </button>
                            <button
                                type="button"
                                onClick={() => applyPreset('lluvias')}
                                className="px-2 py-1 rounded-lg text-xs font-medium bg-sky-50 dark:bg-sky-950/50 hover:bg-sky-100 dark:hover:bg-sky-900/60 text-sky-700 dark:text-sky-300 border border-sky-200 dark:border-sky-800/60 transition cursor-pointer"
                            >
                                🌧️ Lluvias
                            </button>
                            <button
                                type="button"
                                onClick={() => applyPreset('calor')}
                                className="px-2 py-1 rounded-lg text-xs font-medium bg-amber-50 dark:bg-amber-950/50 hover:bg-amber-100 dark:hover:bg-amber-900/60 text-amber-700 dark:text-amber-300 border border-amber-200 dark:border-amber-800/60 transition cursor-pointer"
                            >
                                ☀️ Ola de Calor
                            </button>
                        </div>
                    </div>

                    {/* Inputs del Formulario */}
                    <div className="space-y-2.5">
                        {/* Variables Principales / Requeridas */}
                        <div className="space-y-2.5">
                            {requiredFields.map((field) => renderFieldInput(field))}
                        </div>

                        {/* Variables Opcionales en Dropdown de Configuración Avanzada */}
                        {optionalFields.length > 0 && (
                            <div className="pt-2 border-t border-neutral-100 dark:border-neutral-800">
                                <button
                                    type="button"
                                    onClick={() => setShowAdvancedConfig(!showAdvancedConfig)}
                                    className="w-full flex items-center justify-between p-2.5 rounded-xl bg-neutral-50 dark:bg-neutral-800/40 hover:bg-neutral-100 dark:hover:bg-neutral-800/80 border border-neutral-200 dark:border-neutral-800 text-xs font-medium text-neutral-600 dark:text-neutral-400 transition cursor-pointer"
                                >
                                    <div className="flex items-center gap-2">
                                        <Sliders className="w-3.5 h-3.5 text-neutral-400" />
                                        <span>Configuración Avanzada</span>
                                        <span className="text-[10px] px-1.5 py-0.2 rounded bg-neutral-200 dark:bg-neutral-700 text-neutral-600 dark:text-neutral-400">
                                            {optionalFields.length} variables
                                        </span>
                                    </div>
                                    <ChevronDown className={`w-3.5 h-3.5 text-neutral-400 transition-transform duration-200 ${showAdvancedConfig ? 'rotate-180' : ''}`} />
                                </button>

                                {showAdvancedConfig && (
                                    <div className="mt-2.5 p-3 rounded-xl bg-neutral-50/70 dark:bg-neutral-900/60 border border-neutral-200 dark:border-neutral-800/80 space-y-2.5 animate-in fade-in duration-200">
                                        <div className="flex items-center gap-1.5 text-[11px] text-neutral-500 dark:text-neutral-400 pb-1 border-b border-neutral-200/60 dark:border-neutral-800">
                                            <Sparkles className="w-3.5 h-3.5 text-emerald-500" />
                                            <span>Variables complementarias históricas.</span>
                                        </div>
                                        {optionalFields.map((field) => renderFieldInput(field))}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Botón de Evaluación */}
                    <button
                        type="button"
                        onClick={handlePredecir}
                        disabled={isPredicting}
                        className="w-full py-2.5 px-4 rounded-xl font-semibold text-sm transition flex items-center justify-center gap-2 shadow-sm bg-emerald-600 hover:bg-emerald-700 active:scale-[0.99] text-white cursor-pointer"
                    >
                        <Zap className={`w-4 h-4 ${isPredicting ? 'animate-spin' : ''}`} />
                        {isPredicting ? 'Calculando Riesgo...' : '🔍 Evaluar Nivel de Riesgo'}
                    </button>

                    {/* Caja de Resultado Intuitiva */}
                    {prediccion && (
                        <div className={`p-4 rounded-xl border transition-all ${
                            prediccion.alertaActiva 
                                ? 'bg-rose-50/80 border-rose-200 dark:bg-rose-950/30 dark:border-rose-900/50' 
                                : 'bg-emerald-50/80 border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-900/50'
                        }`}>
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-xs font-bold uppercase tracking-wider text-neutral-600 dark:text-neutral-300">
                                    Diagnóstico de Riesgo
                                </span>
                                <span className="text-[10px] text-neutral-400">
                                    {selectedMunicipio.nombre} • SIVIGILA
                                </span>
                            </div>
                            
                            <div className="flex items-baseline justify-between mb-2.5">
                                <div>
                                    <span className="text-3xl font-extrabold tracking-tight text-neutral-900 dark:text-white">
                                        {(prediccion.probabilidad * 100).toFixed(1)}%
                                    </span>
                                    <span className="text-xs text-neutral-500 ml-1">probabilidad</span>
                                </div>
                                <div>
                                    {prediccion.alertaActiva ? (
                                        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-bold bg-rose-600 text-white shadow-sm">
                                            <AlertTriangle className="w-3.5 h-3.5" /> ALERTA TEMPRANA
                                        </span>
                                    ) : (
                                        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-bold bg-emerald-600 text-white shadow-sm">
                                            <CheckCircle2 className="w-3.5 h-3.5" /> RIESGO CONTROLADO
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
                                    <span>Bajo (&lt;20%)</span>
                                    <span className="font-semibold text-neutral-700 dark:text-neutral-300">
                                        Umbral de Alerta: 20.5%
                                    </span>
                                    <span>Alto</span>
                                </div>
                            </div>

                            {/* Recomendación Práctica en lenguaje claro */}
                            <p className="text-xs text-neutral-700 dark:text-neutral-300 mt-3 pt-2.5 border-t border-neutral-200/80 dark:border-neutral-800/80 leading-relaxed">
                                {prediccion.alertaActiva
                                    ? '⚠️ Recomendación: Intensificar búsqueda activa comunitaria, control biológico de larvas y jornadas de recolección de inservibles.'
                                    : '✅ Recomendación: Mantener vigilancia pasiva y control rutinario de criaderos.'}
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
                    )}
                </div>

                {/* Mapa Epidemiológico de Colombia (Right - 7 Cols) */}
                <div className="lg:col-span-7 p-6 rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/40 shadow-sm space-y-6">
                    <ColombiaMap
                        selectedMunicipioId={selectedMunicipio.id}
                        onSelectMunicipio={handleSelectMunicipio}
                        activeCodes={activeCodes}
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
                                <span className="font-mono font-semibold text-neutral-900 dark:text-emerald-400">api.arkoiot.com</span>
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
