import React, { useState } from 'react';
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
    Search
} from 'lucide-react';

const MUNICIPIOS_MOCK = [
    { id: 1, nombre: 'Cali - Comuna 13', riesgo: 'Crítico', probabilidad: 0.82, casosEst: 340, humedad: '84%', temp: '29°C', estado: 'Fumigación Prioritaria' },
    { id: 2, nombre: 'Palmira - Sector Norte', riesgo: 'Alto', probabilidad: 0.68, casosEst: 185, humedad: '79%', temp: '28°C', estado: 'Inspección Larvaria' },
    { id: 3, nombre: 'Buenaventura - Cascajal', riesgo: 'Alto', probabilidad: 0.65, casosEst: 210, humedad: '88%', temp: '30°C', estado: 'Monitoreo Clave' },
    { id: 4, nombre: 'Tuluá - Zona Centro', riesgo: 'Moderado', probabilidad: 0.38, casosEst: 92, humedad: '72%', temp: '27°C', estado: 'Vigilancia Regular' },
    { id: 5, nombre: 'Buga - Rural', riesgo: 'Bajo', probabilidad: 0.12, casosEst: 24, humedad: '65%', temp: '26°C', estado: 'Control Estable' },
    { id: 6, nombre: 'Cartago - Comuna 4', riesgo: 'Moderado', probabilidad: 0.44, casosEst: 88, humedad: '75%', temp: '28°C', estado: 'Vigilancia Regular' },
];

export default function Dashboard() {
    // Estado del Simulador de Inferencia
    const [temp, setTemp] = useState(28.5);
    const [precipitacion, setPrecipitacion] = useState(140);
    const [humedad, setHumedad] = useState(82);
    const [indiceAedico, setIndiceAedico] = useState(14.5);
    const [casosPrevios, setCasosPrevios] = useState(110);
    const [filtroRiesgo, setFiltroRiesgo] = useState('Todos');
    const [busqueda, setBusqueda] = useState('');

    // Umbral calibrado de MLOps (configurado en CI/CD: 0.205)
    const THRESHOLD = 0.205;

    // Cálculo simulado de inferencia del modelo
    const calcularScore = () => {
        // Normalización aproximada inspirada en los coeficientes epidemiológicos
        const score = (
            (temp - 20) * 0.025 +
            (precipitacion / 300) * 0.3 +
            (humedad / 100) * 0.25 +
            (indiceAedico / 30) * 0.35 +
            Math.min(casosPrevios / 200, 1) * 0.15
        );
        return Math.min(Math.max(score, 0.05), 0.98);
    };

    const [prediccion, setPrediccion] = useState(() => {
        const p = calcularScore();
        return {
            probabilidad: p,
            alertaActiva: p >= THRESHOLD,
            timestamp: new Date().toLocaleTimeString()
        };
    });

    const handlePredecir = () => {
        const score = calcularScore();
        setPrediccion({
            probabilidad: score,
            alertaActiva: score >= THRESHOLD,
            timestamp: new Date().toLocaleTimeString()
        });
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
                    <div className="flex items-center gap-2">
                        <span className="flex h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse" />
                        <span className="text-xs uppercase tracking-wider font-semibold text-emerald-400">Sistema en Producción v1.2</span>
                        <span className="text-xs px-2 py-0.5 rounded-full bg-neutral-800 text-neutral-300 border border-neutral-700">MLflow Tracking Active</span>
                    </div>
                    <h2 className="text-2xl md:text-3xl font-extrabold tracking-tight">
                        Sistema de Alerta Temprana de Dengue
                    </h2>
                    <p className="text-sm text-neutral-400 max-w-2xl">
                        Monitor epidemiológico basado en Machine Learning (Random Forest) calibrado con umbral óptimo (<span className="text-emerald-300 font-mono">0.205</span>) para prevención comunitaria.
                    </p>
                </div>
                <div className="flex flex-wrap items-center gap-3">
                    <div className="px-3.5 py-2 rounded-xl bg-neutral-800/80 border border-neutral-700 text-xs text-neutral-300 flex items-center gap-2">
                        <Cpu className="w-4 h-4 text-emerald-400" />
                        <span>Inferencia: <strong>&lt; 45ms</strong></span>
                    </div>
                    <div className="px-3.5 py-2 rounded-xl bg-neutral-800/80 border border-neutral-700 text-xs text-neutral-300 flex items-center gap-2">
                        <ShieldCheck className="w-4 h-4 text-sky-400" />
                        <span>Cloudflare Edge Zero Trust</span>
                    </div>
                </div>
            </div>

            {/* Metricas Principales (KPIs) */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* KPI 1 */}
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

                {/* KPI 2 */}
                <div className="p-5 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/50 shadow-sm hover:shadow-md transition">
                    <div className="flex items-center justify-between text-neutral-500 dark:text-neutral-400 mb-2">
                        <span className="text-xs font-semibold uppercase tracking-wider">Casos Proyectados</span>
                        <TrendingUp className="w-4 h-4 text-amber-500" />
                    </div>
                    <div className="flex items-baseline gap-2">
                        <span className="text-2xl font-bold text-neutral-900 dark:text-white">1,420</span>
                        <span className="text-xs font-medium text-amber-600">Estimación a 14 días</span>
                    </div>
                    <p className="text-xs text-neutral-500 mt-1">Con base en clima y vector histórico</p>
                </div>

                {/* KPI 3 */}
                <div className="p-5 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/50 shadow-sm hover:shadow-md transition">
                    <div className="flex items-center justify-between text-neutral-500 dark:text-neutral-400 mb-2">
                        <span className="text-xs font-semibold uppercase tracking-wider">Métricas del Modelo</span>
                        <Zap className="w-4 h-4 text-emerald-500" />
                    </div>
                    <div className="flex items-baseline gap-2">
                        <span className="text-2xl font-bold text-neutral-900 dark:text-white">0.892</span>
                        <span className="text-xs font-medium text-emerald-600 font-mono">AUC-ROC</span>
                    </div>
                    <p className="text-xs text-neutral-500 mt-1">Recall 88.4% • F1-Score 0.841</p>
                </div>

                {/* KPI 4 */}
                <div className="p-5 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/50 shadow-sm hover:shadow-md transition">
                    <div className="flex items-center justify-between text-neutral-500 dark:text-neutral-400 mb-2">
                        <span className="text-xs font-semibold uppercase tracking-wider">Zonas en Alerta</span>
                        <AlertTriangle className="w-4 h-4 text-rose-500" />
                    </div>
                    <div className="flex items-baseline gap-2">
                        <span className="text-2xl font-bold text-rose-600">8 de 14</span>
                        <span className="text-xs font-medium text-neutral-500">Comunas</span>
                    </div>
                    <p className="text-xs text-neutral-500 mt-1">Superan el umbral de decisión 0.205</p>
                </div>
            </div>

            {/* Main Interactive Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Simulador Interactivo (Left - 5 Cols) */}
                <div className="lg:col-span-5 p-6 rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/40 shadow-sm space-y-6">
                    <div className="flex items-center justify-between border-b border-neutral-100 dark:border-neutral-800 pb-4">
                        <div className="flex items-center gap-2">
                            <Sliders className="w-5 h-5 text-emerald-500" />
                            <h3 className="font-bold text-lg text-neutral-900 dark:text-white">Simulador de Inferencia</h3>
                        </div>
                        <span className="text-xs px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 font-medium">
                            En Vivo
                        </span>
                    </div>

                    <div className="space-y-4">
                        {/* Control 1: Temperatura */}
                        <div>
                            <div className="flex justify-between text-xs font-medium mb-1.5">
                                <span className="flex items-center gap-1.5 text-neutral-600 dark:text-neutral-400">
                                    <Thermometer className="w-3.5 h-3.5 text-amber-500" /> Temperatura Promedio
                                </span>
                                <span className="font-mono font-bold text-neutral-900 dark:text-white">{temp} °C</span>
                            </div>
                            <input
                                type="range"
                                min="20"
                                max="36"
                                step="0.5"
                                value={temp}
                                onChange={(e) => setTemp(parseFloat(e.target.value))}
                                className="w-full h-1.5 bg-neutral-200 dark:bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                            />
                        </div>

                        {/* Control 2: Precipitación */}
                        <div>
                            <div className="flex justify-between text-xs font-medium mb-1.5">
                                <span className="flex items-center gap-1.5 text-neutral-600 dark:text-neutral-400">
                                    <Droplets className="w-3.5 h-3.5 text-sky-500" /> Precipitación Acumulada
                                </span>
                                <span className="font-mono font-bold text-neutral-900 dark:text-white">{precipitacion} mm</span>
                            </div>
                            <input
                                type="range"
                                min="10"
                                max="350"
                                step="5"
                                value={precipitacion}
                                onChange={(e) => setPrecipitacion(parseInt(e.target.value))}
                                className="w-full h-1.5 bg-neutral-200 dark:bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                            />
                        </div>

                        {/* Control 3: Humedad */}
                        <div>
                            <div className="flex justify-between text-xs font-medium mb-1.5">
                                <span className="text-neutral-600 dark:text-neutral-400">Humedad Relativa</span>
                                <span className="font-mono font-bold text-neutral-900 dark:text-white">{humedad} %</span>
                            </div>
                            <input
                                type="range"
                                min="40"
                                max="98"
                                step="1"
                                value={humedad}
                                onChange={(e) => setHumedad(parseInt(e.target.value))}
                                className="w-full h-1.5 bg-neutral-200 dark:bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                            />
                        </div>

                        {/* Control 4: Índice Aédico */}
                        <div>
                            <div className="flex justify-between text-xs font-medium mb-1.5">
                                <span className="text-neutral-600 dark:text-neutral-400">Índice Larvario / Aédico (Breteau)</span>
                                <span className="font-mono font-bold text-neutral-900 dark:text-white">{indiceAedico} %</span>
                            </div>
                            <input
                                type="range"
                                min="1"
                                max="30"
                                step="0.5"
                                value={indiceAedico}
                                onChange={(e) => setIndiceAedico(parseFloat(e.target.value))}
                                className="w-full h-1.5 bg-neutral-200 dark:bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                            />
                        </div>

                        {/* Control 5: Casos Previos */}
                        <div>
                            <div className="flex justify-between text-xs font-medium mb-1.5">
                                <span className="text-neutral-600 dark:text-neutral-400">Casos Notificados (Últ. 4 Semanas)</span>
                                <span className="font-mono font-bold text-neutral-900 dark:text-white">{casosPrevios} casos</span>
                            </div>
                            <input
                                type="range"
                                min="0"
                                max="300"
                                step="5"
                                value={casosPrevios}
                                onChange={(e) => setCasosPrevios(parseInt(e.target.value))}
                                className="w-full h-1.5 bg-neutral-200 dark:bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                            />
                        </div>
                    </div>

                    <button
                        onClick={handlePredecir}
                        className="w-full py-2.5 px-4 rounded-xl font-medium text-sm bg-neutral-900 hover:bg-neutral-800 text-white dark:bg-emerald-600 dark:hover:bg-emerald-500 transition flex items-center justify-center gap-2 shadow"
                    >
                        <Zap className="w-4 h-4" /> Ejecutar Inferencia de Riesgo
                    </button>

                    {/* Caja de Resultado */}
                    <div className={`p-4 rounded-xl border transition-all ${prediccion.alertaActiva ? 'bg-rose-50 border-rose-200 dark:bg-rose-950/30 dark:border-rose-900/50' : 'bg-emerald-50 border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-900/50'}`}>
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-semibold uppercase tracking-wider text-neutral-600 dark:text-neutral-300">
                                Resultado de Inferencia
                            </span>
                            <span className="text-[11px] text-neutral-500 font-mono">
                                Calc: {prediccion.timestamp}
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
                                ? '⚠️ Recomendación: Iniciar fumigación focalizada y activar plan de contingencia hospitalaria.'
                                : '✅ Recomendación: Mantener vigilancia pasiva y control larvario rutinario.'}
                        </p>
                    </div>
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
                        <span className="text-neutral-400 block mb-1">Servidor MLflow</span>
                        <span className="font-mono font-semibold text-neutral-900 dark:text-emerald-400">platform.arkoiot.com</span>
                    </div>
                    <div className="p-3 bg-white dark:bg-neutral-800/50 rounded-xl border border-neutral-200 dark:border-neutral-700">
                        <span className="text-neutral-400 block mb-1">Versionado de Datos</span>
                        <span className="font-mono font-semibold text-neutral-900 dark:text-white">DVC + AWS S3</span>
                    </div>
                    <div className="p-3 bg-white dark:bg-neutral-800/50 rounded-xl border border-neutral-200 dark:border-neutral-700">
                        <span className="text-neutral-400 block mb-1">Entrenamiento Remoto</span>
                        <span className="font-mono font-semibold text-neutral-900 dark:text-white">Lightning AI SDK</span>
                    </div>
                    <div className="p-3 bg-white dark:bg-neutral-800/50 rounded-xl border border-neutral-200 dark:border-neutral-700">
                        <span className="text-neutral-400 block mb-1">Despliegue Edge</span>
                        <span className="font-mono font-semibold text-neutral-900 dark:text-white">Cloudflare Pages + Zero Trust</span>
                    </div>
                </div>
            </div>
        </div>
    );
}
