import React, { useState, useEffect } from 'react';
import {
    Activity,
    AlertTriangle,
    TrendingUp,
    Droplets,
    Thermometer,
    ShieldCheck,
    Layers,
    Cpu,
    Zap,
    MapPin,
    CheckCircle2,
    Sliders,
    ArrowUpRight,
    Search,
    Wifi,
    WifiOff
} from 'lucide-react';

const MUNICIPIOS_MOCK = [
    { id: 1, nombre: 'Cali - Comuna 13', riesgo: 'Crítico', probabilidad: 0.82, casosEst: 340, humedad: '84%', temp: '29°C', estado: 'Fumigación Prioritaria' },
    { id: 2, nombre: 'Palmira - Sector Norte', riesgo: 'Alto', probabilidad: 0.68, casosEst: 185, humedad: '79%', temp: '28°C', estado: 'Inspección Larvaria' },
    { id: 3, nombre: 'Buenaventura - Cascajal', riesgo: 'Alto', probabilidad: 0.65, casosEst: 210, humedad: '88%', temp: '30°C', estado: 'Monitoreo Clave' },
    { id: 4, nombre: 'Tuluá - Zona Centro', riesgo: 'Moderado', probabilidad: 0.38, casosEst: 92, humedad: '72%', temp: '27°C', estado: 'Vigilancia Regular' },
    { id: 5, nombre: 'Buga - Rural', riesgo: 'Bajo', probabilidad: 0.12, casosEst: 24, humedad: '65%', temp: '26°C', estado: 'Control Estable' },
    { id: 6, nombre: 'Cartago - Comuna 4', riesgo: 'Moderado', probabilidad: 0.44, casosEst: 88, humedad: '75%', temp: '28°C', estado: 'Vigilancia Regular' },
];

// Metadatos amigables para variables de MLflow
const FIELD_META = {
    temperatura: { label: "Temperatura Promedio", unit: "°C", min: 20, max: 38, step: 0.5, default: 28.5, icon: Thermometer },
    precipitacion: { label: "Precipitación Acumulada", unit: "mm", min: 10, max: 350, step: 5, default: 140, icon: Droplets },
    humedad: { label: "Humedad Relativa", unit: "%", min: 40, max: 98, step: 1, default: 82, icon: Activity },
    indice_aedico: { label: "Índice Aédico / Larvario", unit: "%", min: 1, max: 30, step: 0.5, default: 14.5, icon: Sliders },
    casos_previos: { label: "Casos Últimas 4 Semanas", unit: "casos", min: 0, max: 300, step: 1, default: 110, icon: AlertTriangle },
    densidad_poblacional: { label: "Densidad Poblacional", unit: "hab/km²", min: 500, max: 10000, step: 100, default: 3500, icon: MapPin },
};

const DEFAULT_FALLBACK_SCHEMA = [
    { name: 'temperatura', type: 'double' },
    { name: 'precipitacion', type: 'double' },
    { name: 'humedad', type: 'double' },
    { name: 'indice_aedico', type: 'double' },
    { name: 'casos_previos', type: 'integer' },
];

export default function Dashboard() {
    const API_BASE_URL = "/api";

    // Estado del Schema y estado de la API
    const [schemaFields, setSchemaFields] = useState(DEFAULT_FALLBACK_SCHEMA);
    const [apiConnected, setApiConnected] = useState(false);
    const [modelInfo, setModelInfo] = useState({
        stage: 'Production',
        version: '1.1.0',
        model_name: 'ModeloPrueba'
    });

    // Valores dinámicos del formulario
    const [formValues, setFormValues] = useState(() => {
        const initial = {};
        DEFAULT_FALLBACK_SCHEMA.forEach(f => {
            initial[f.name] = FIELD_META[f.name]?.default ?? 10;
        });
        return initial;
    });

    const THRESHOLD = 0.205;

    // Estado de la predicción
    const [prediccion, setPrediccion] = useState(null);
    const [isPredicting, setIsPredicting] = useState(false);

    // Filtros de tabla
    const [filtroRiesgo, setFiltroRiesgo] = useState('Todos');
    const [busqueda, setBusqueda] = useState('');

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

    useEffect(() => {
        fetchSchema();
        // Reintentar periódicamente cada 15 segundos para auto-habilitar el formulario al conectar la API
        const interval = setInterval(fetchSchema, 15000);
        return () => clearInterval(interval);
    }, []);

    const handleFieldChange = (name, value) => {
        if (!apiConnected) return;
        setFormValues(prev => ({
            ...prev,
            [name]: value
        }));
    };

    // Ejecutar predicción
    const handlePredecir = async () => {
        if (!apiConnected) return;
        setIsPredicting(true);

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
                    source: `API (${data.stage} v${data.model_version})`,
                    timestamp: new Date().toLocaleTimeString()
                });
                setApiConnected(true);
            } else {
                setApiConnected(false);
            }
        } catch (e) {
            setApiConnected(false);
        } finally {
            setIsPredicting(false);
        }
    };

    const municipiosFiltrados = MUNICIPIOS_MOCK.filter(m => {
        const coincideFiltro = filtroRiesgo === 'Todos' || m.riesgo === filtroRiesgo;
        const coincideBusqueda = m.nombre.toLowerCase().includes(busqueda.toLowerCase());
        return coincideFiltro && coincideBusqueda;
    });

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            {/* Header / Hero */}
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 p-6 rounded-2xl bg-gradient-to-r from-neutral-900 via-neutral-900 to-neutral-800 text-white shadow-xl border border-neutral-800">
                <div className="space-y-1">
                    <div className="flex flex-wrap items-center gap-2">
                        <span className={`flex h-2.5 w-2.5 rounded-full ${apiConnected ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
                        <span className="text-xs uppercase tracking-wider font-semibold text-emerald-400">
                            Modelo {modelInfo.model_name} (v{modelInfo.version})
                        </span>
                        <span className="text-xs px-2 py-0.5 rounded-full bg-neutral-800 text-neutral-300 border border-neutral-700">
                            Stage: {modelInfo.stage}
                        </span>
                    </div>
                    <h2 className="text-2xl md:text-3xl font-extrabold tracking-tight">
                        Sistema de Alerta Temprana de Dengue
                    </h2>
                    <p className="text-sm text-neutral-400 max-w-2xl">
                        Esquema e inferencia sincronizados en caliente con el servidor MLflow a través de <span className="text-emerald-300 font-mono">api.arkoiot.com</span>.
                    </p>
                </div>
                
                <div className="flex flex-wrap items-center gap-3">
                    <div className="px-3.5 py-2 rounded-xl bg-neutral-800/80 border border-neutral-700 text-xs text-neutral-300 flex items-center gap-2">
                        {apiConnected ? (
                            <>
                                <Wifi className="w-4 h-4 text-emerald-400" />
                                <span className="text-emerald-400 font-medium">API Conectada</span>
                            </>
                        ) : (
                            <>
                                <WifiOff className="w-4 h-4 text-amber-400" />
                                <span className="text-amber-300 font-medium">API Desconectada</span>
                            </>
                        )}
                    </div>
                </div>
            </div>

            {/* Metricas Principales (KPIs) */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="p-5 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/50 shadow-sm hover:shadow-md transition">
                    <div className="flex items-center justify-between text-neutral-500 dark:text-neutral-400 mb-2">
                        <span className="text-xs font-semibold uppercase tracking-wider">Riesgo Global</span>
                        <Activity className="w-4 h-4 text-rose-500" />
                    </div>
                    <div className="flex items-baseline gap-2">
                        <span className="text-2xl font-bold text-neutral-900 dark:text-white">68.4%</span>
                        <span className="text-xs font-medium text-rose-600 flex items-center">
                            +6.2% <ArrowUpRight className="w-3 h-3" />
                        </span>
                    </div>
                    <p className="text-xs text-neutral-500 mt-1">Nivel Epidemiológico Alto (Semana 41)</p>
                </div>

                <div className="p-5 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/50 shadow-sm hover:shadow-md transition">
                    <div className="flex items-center justify-between text-neutral-500 dark:text-neutral-400 mb-2">
                        <span className="text-xs font-semibold uppercase tracking-wider">Variables en Modelo</span>
                        <Layers className="w-4 h-4 text-sky-500" />
                    </div>
                    <div className="flex items-baseline gap-2">
                        <span className="text-2xl font-bold text-neutral-900 dark:text-white">{schemaFields.length}</span>
                        <span className="text-xs font-medium text-sky-600">features</span>
                    </div>
                    <p className="text-xs text-neutral-500 mt-1">
                        {apiConnected ? 'Esquema activo desde /schema' : 'Esperando conexión API'}
                    </p>
                </div>

                <div className="p-5 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/50 shadow-sm hover:shadow-md transition">
                    <div className="flex items-center justify-between text-neutral-500 dark:text-neutral-400 mb-2">
                        <span className="text-xs font-semibold uppercase tracking-wider">Métricas del Modelo</span>
                        <Zap className="w-4 h-4 text-emerald-500" />
                    </div>
                    <div className="flex items-baseline gap-2">
                        <span className="text-2xl font-bold text-neutral-900 dark:text-white">0.983</span>
                        <span className="text-xs font-medium text-emerald-600 font-mono">Accuracy</span>
                    </div>
                    <p className="text-xs text-neutral-500 mt-1">F1-Score 0.991 • Versión {modelInfo.version}</p>
                </div>

                <div className="p-5 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/50 shadow-sm hover:shadow-md transition">
                    <div className="flex items-center justify-between text-neutral-500 dark:text-neutral-400 mb-2">
                        <span className="text-xs font-semibold uppercase tracking-wider">Umbral Óptimo</span>
                        <AlertTriangle className="w-4 h-4 text-rose-500" />
                    </div>
                    <div className="flex items-baseline gap-2">
                        <span className="text-2xl font-bold text-rose-600 font-mono">{THRESHOLD}</span>
                        <span className="text-xs font-medium text-neutral-500">corte</span>
                    </div>
                    <p className="text-xs text-neutral-500 mt-1">Optimizado para maximizar recall de brotes</p>
                </div>
            </div>

            {/* Main Interactive Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Formulario Dinámico (Left - 5 Cols) */}
                <div className="lg:col-span-5 p-6 rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/40 shadow-sm space-y-6">
                    <div className="flex items-center justify-between border-b border-neutral-100 dark:border-neutral-800 pb-4">
                        <div className="flex items-center gap-2">
                            <Sliders className="w-5 h-5 text-emerald-500" />
                            <div>
                                <h3 className="font-bold text-lg text-neutral-900 dark:text-white">Formulario Dinámico</h3>
                                <p className="text-[11px] text-neutral-500">
                                    {apiConnected ? `Esquema de /schema (${schemaFields.length} variables)` : 'Inhabilitado: API offline'}
                                </p>
                            </div>
                        </div>
                        <span className="text-xs px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 font-medium font-mono">
                            {modelInfo.stage}
                        </span>
                    </div>

                    {/* Banner de API Offline si no hay conexión */}
                    {!apiConnected && (
                        <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-700 dark:text-amber-300 text-xs flex items-start gap-2.5 animate-in fade-in">
                            <AlertTriangle className="w-4 h-4 shrink-0 text-amber-500 mt-0.5" />
                            <div>
                                <p className="font-semibold">API de Inferencia Desconectada</p>
                                <p className="text-[11px] opacity-80 mt-0.5">
                                    El backend en <span className="font-mono">/api/schema</span> no está disponible. Toda la sección del formulario se encuentra inhabilitada hasta que la API esté en línea.
                                </p>
                            </div>
                        </div>
                    )}

                    {/* Inputs del Formulario (Deshabilitados si !apiConnected) */}
                    <div className={`space-y-4 transition-opacity ${!apiConnected ? 'opacity-40 select-none' : 'opacity-100'}`}>
                        {schemaFields.map((field) => {
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
                                <div key={field.name} className="space-y-1.5">
                                    <div className="flex justify-between text-xs font-medium">
                                        <span className="flex items-center gap-1.5 text-neutral-600 dark:text-neutral-400 capitalize">
                                            {meta.icon ? <meta.icon className="w-3.5 h-3.5 text-emerald-500" /> : <Activity className="w-3.5 h-3.5 text-neutral-400" />}
                                            {meta.label}
                                        </span>
                                        <span className="font-mono font-bold text-neutral-900 dark:text-white">
                                            {currentValue} {meta.unit}
                                        </span>
                                    </div>
                                    <input
                                        type="range"
                                        disabled={!apiConnected}
                                        min={meta.min}
                                        max={meta.max}
                                        step={meta.step}
                                        value={currentValue}
                                        onChange={(e) => {
                                            const val = isInteger ? parseInt(e.target.value) : parseFloat(e.target.value);
                                            handleFieldChange(field.name, val);
                                        }}
                                        className="w-full h-1.5 bg-neutral-200 dark:bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-emerald-500 disabled:cursor-not-allowed"
                                    />
                                    <div className="flex justify-between text-[10px] text-neutral-400">
                                        <span>Min: {meta.min}</span>
                                        <span className="font-mono">{field.type}</span>
                                        <span>Max: {meta.max}</span>
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* Botón de Inferencia (Deshabilitado si !apiConnected) */}
                    <button
                        onClick={handlePredecir}
                        disabled={!apiConnected || isPredicting}
                        className={`w-full py-2.5 px-4 rounded-xl font-medium text-sm transition flex items-center justify-center gap-2 shadow ${
                            apiConnected
                                ? 'bg-neutral-900 hover:bg-neutral-800 text-white dark:bg-emerald-600 dark:hover:bg-emerald-500 cursor-pointer'
                                : 'bg-neutral-200 dark:bg-neutral-800 text-neutral-400 dark:text-neutral-500 cursor-not-allowed border border-neutral-300 dark:border-neutral-700'
                        }`}
                    >
                        {apiConnected ? (
                            <>
                                <Zap className={`w-4 h-4 ${isPredicting ? 'animate-spin' : ''}`} />
                                {isPredicting ? 'Calculando Inferencia...' : 'Ejecutar Inferencia de Riesgo'}
                            </>
                        ) : (
                            <>
                                <WifiOff className="w-4 h-4" />
                                API Desconectada — Inferencia Inhabilitada
                            </>
                        )}
                    </button>

                    {/* Caja de Resultado */}
                    {apiConnected && prediccion ? (
                        <div className={`p-4 rounded-xl border transition-all ${prediccion.alertaActiva ? 'bg-rose-50 border-rose-200 dark:bg-rose-950/30 dark:border-rose-900/50' : 'bg-emerald-50 border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-900/50'}`}>
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-xs font-semibold uppercase tracking-wider text-neutral-600 dark:text-neutral-300">
                                    Resultado de Inferencia
                                </span>
                                <span className="text-[11px] text-neutral-500 font-mono">
                                    {prediccion.source}
                                </span>
                            </div>
                            
                            <div className="flex items-baseline justify-between mb-2">
                                <div>
                                    <span className="text-3xl font-extrabold tracking-tight text-neutral-900 dark:text-white">
                                        {(prediccion.probabilidad * 100).toFixed(1)}%
                                    </span>
                                    <span className="text-xs text-neutral-500 ml-1">probabilidad</span>
                                </div>
                                <div className="text-right">
                                    {prediccion.alertaActiva ? (
                                        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-bold bg-rose-600 text-white">
                                            <AlertTriangle className="w-3.5 h-3.5" /> ALERTA ACTIVA
                                        </span>
                                    ) : (
                                        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-bold bg-emerald-600 text-white">
                                            <CheckCircle2 className="w-3.5 h-3.5" /> RIESGO BAJO
                                        </span>
                                    )}
                                </div>
                            </div>

                            {/* Barra de Umbral */}
                            <div className="space-y-1">
                                <div className="relative w-full h-2.5 bg-neutral-200 dark:bg-neutral-800 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full rounded-full transition-all duration-300 ${prediccion.alertaActiva ? 'bg-rose-500' : 'bg-emerald-500'}`}
                                        style={{ width: `${prediccion.probabilidad * 100}%` }}
                                    />
                                </div>
                                <div className="flex justify-between text-[11px] text-neutral-500">
                                    <span>0%</span>
                                    <span className="font-mono text-neutral-700 dark:text-neutral-300 font-semibold">
                                        Umbral de corte: {THRESHOLD}
                                    </span>
                                    <span>100%</span>
                                </div>
                            </div>

                            <p className="text-xs text-neutral-600 dark:text-neutral-400 mt-3">
                                {prediccion.alertaActiva
                                    ? '⚠️ Protocolo: Disparar alerta a secretarías de salud y desplegar brigada de fumigación.'
                                    : '✅ Protocolo: Mantener monitoreo pasivo y reporte semanal rutinario.'}
                            </p>
                        </div>
                    ) : (
                        <div className="p-4 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-900/30 text-neutral-400 text-center text-xs py-5">
                            <WifiOff className="w-5 h-5 mx-auto mb-1.5 opacity-40 text-neutral-400" />
                            <p className="font-medium text-neutral-500">Módulo en Espera</p>
                            <p className="text-[11px] text-neutral-400 mt-0.5">La predicción se activará automáticamente al conectar la API</p>
                        </div>
                    )}
                </div>

                {/* Monitoreo Regional y Zonas (Right - 7 Cols) */}
                <div className="lg:col-span-7 p-6 rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/40 shadow-sm space-y-6">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-neutral-100 dark:border-neutral-800 pb-4">
                        <div className="flex items-center gap-2">
                            <MapPin className="w-5 h-5 text-emerald-500" />
                            <h3 className="font-bold text-lg text-neutral-900 dark:text-white">Vigilancia por Zonas</h3>
                        </div>

                        <div className="flex items-center gap-2">
                            <div className="relative">
                                <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-neutral-400" />
                                <input
                                    type="text"
                                    placeholder="Buscar zona..."
                                    value={busqueda}
                                    onChange={(e) => setBusqueda(e.target.value)}
                                    className="pl-8 pr-3 py-1 text-xs rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-800/50 text-neutral-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-emerald-500"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Filtros rápidos */}
                    <div className="flex items-center gap-2 overflow-x-auto pb-1 text-xs">
                        <span className="text-neutral-400 text-[11px] font-medium mr-1">Filtrar:</span>
                        {['Todos', 'Crítico', 'Alto', 'Moderado', 'Bajo'].map(filtro => (
                            <button
                                key={filtro}
                                onClick={() => setFiltroRiesgo(filtro)}
                                className={`px-2.5 py-1 rounded-lg font-medium transition ${filtroRiesgo === filtro ? 'bg-neutral-900 text-white dark:bg-white dark:text-neutral-900' : 'bg-neutral-100 text-neutral-600 dark:bg-neutral-800 dark:text-neutral-400 hover:bg-neutral-200'}`}
                            >
                                {filtro}
                            </button>
                        ))}
                    </div>

                    {/* Tabla de Zonas */}
                    <div className="overflow-x-auto">
                        <table className="w-full text-left text-xs">
                            <thead>
                                <tr className="border-b border-neutral-100 dark:border-neutral-800 text-neutral-400 uppercase tracking-wider text-[11px]">
                                    <th className="pb-3 font-semibold">Sector</th>
                                    <th className="pb-3 font-semibold">Nivel</th>
                                    <th className="pb-3 font-semibold">Probabilidad</th>
                                    <th className="pb-3 font-semibold">Casos Est.</th>
                                    <th className="pb-3 font-semibold">Acción Protocolar</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-neutral-100 dark:divide-neutral-800/60">
                                {municipiosFiltrados.map(m => (
                                    <tr key={m.id} className="hover:bg-neutral-50 dark:hover:bg-neutral-800/30 transition">
                                        <td className="py-3 font-medium text-neutral-900 dark:text-white">
                                            {m.nombre}
                                            <div className="text-[11px] text-neutral-400">{m.temp} • {m.humedad} hum.</div>
                                        </td>
                                        <td className="py-3">
                                            <span className={`px-2 py-0.5 rounded-full text-[11px] font-semibold ${
                                                m.riesgo === 'Crítico' ? 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300' :
                                                m.riesgo === 'Alto' ? 'bg-amber-100 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300' :
                                                m.riesgo === 'Moderado' ? 'bg-sky-100 text-sky-700 dark:bg-sky-950/60 dark:text-sky-300' :
                                                'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300'
                                            }`}>
                                                {m.riesgo}
                                            </span>
                                        </td>
                                        <td className="py-3 font-mono font-medium text-neutral-700 dark:text-neutral-300">
                                            {(m.probabilidad * 100).toFixed(0)}%
                                        </td>
                                        <td className="py-3 font-medium text-neutral-900 dark:text-white">
                                            {m.casosEst}
                                        </td>
                                        <td className="py-3 text-neutral-600 dark:text-neutral-400">
                                            {m.estado}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            {/* MLOps Architecture & Tracking Card */}
            <div className="p-6 rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-900/60 space-y-4">
                <div className="flex items-center gap-2 text-neutral-900 dark:text-white font-bold text-sm">
                    <Layers className="w-4 h-4 text-emerald-500" />
                    <span>Trazabilidad MLOps & Arquitectura de Producción</span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
                    <div className="p-3 bg-white dark:bg-neutral-800/50 rounded-xl border border-neutral-200 dark:border-neutral-700">
                        <span className="text-neutral-400 block mb-1">API Backend</span>
                        <span className="font-mono font-semibold text-neutral-900 dark:text-emerald-400">api.arkoiot.com</span>
                    </div>
                    <div className="p-3 bg-white dark:bg-neutral-800/50 rounded-xl border border-neutral-200 dark:border-neutral-700">
                        <span className="text-neutral-400 block mb-1">Servidor MLflow</span>
                        <span className="font-mono font-semibold text-neutral-900 dark:text-white">platform.arkoiot.com</span>
                    </div>
                    <div className="p-3 bg-white dark:bg-neutral-800/50 rounded-xl border border-neutral-200 dark:border-neutral-700">
                        <span className="text-neutral-400 block mb-1">Modelo Activo</span>
                        <span className="font-mono font-semibold text-neutral-900 dark:text-white">{modelInfo.model_name} (v{modelInfo.version})</span>
                    </div>
                    <div className="p-3 bg-white dark:bg-neutral-800/50 rounded-xl border border-neutral-200 dark:border-neutral-700">
                        <span className="text-neutral-400 block mb-1">Despliegue Edge</span>
                        <span className="font-mono font-semibold text-neutral-900 dark:text-white">Cloudflare Pages (Blue/Green)</span>
                    </div>
                </div>
            </div>
        </div>
    );
}
