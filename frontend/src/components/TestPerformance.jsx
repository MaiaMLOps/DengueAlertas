import React, { useEffect, useState } from 'react';
import { AlertTriangle, Database, LineChart, ShieldCheck } from 'lucide-react';
import { AlertStripsChart, ProbabilityTestChart } from './TestPerformanceCharts';

const RULES = [
    { key: 'umbral', label: 'Regla de umbral' },
    { key: 'producto', label: 'Regla del producto' },
    { key: 'canal', label: 'Canal' },
];

const LEVELS = {
    favorable: 'Favorable',
    mixto: 'Mixto',
    desfavorable: 'Desfavorable',
    tendencia_favorable: 'Tendencia favorable',
    mejora_consistente: 'Mejora consistente',
};

const LEVEL_STYLES = {
    favorable: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300',
    mixto: 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300',
    desfavorable: 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300',
};

const formatNumber = (value, digits = 3) => Number(value).toLocaleString('es-CO', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
});
const formatSigned = (value, digits = 3) => `${value > 0 ? '+' : ''}${formatNumber(value, digits)}`;
const formatLevel = (value) => LEVELS[value] || String(value).replaceAll('_', ' ');

function EventsTable({ events, detailed = false }) {
    return (
        <div className="overflow-x-auto rounded-xl border border-neutral-200 dark:border-neutral-800">
            <table className={`w-full ${detailed ? 'min-w-[760px]' : 'min-w-[360px]'} text-left text-[11px] text-neutral-700 dark:text-neutral-300`}>
                <thead className="bg-neutral-50 dark:bg-neutral-900/70 text-[10px] uppercase tracking-wide text-neutral-500">
                    <tr>
                        <th className="px-3 py-2 font-semibold">Sistema</th>
                        <th className="px-3 py-2 font-semibold">Inicios avisados</th>
                        {detailed && <th className="px-3 py-2 font-semibold">Entradas avisadas</th>}
                        <th className="px-3 py-2 font-semibold">Falsas alarmas</th>
                        {detailed && <th className="px-3 py-2 font-semibold">Semanas bajo alerta</th>}
                        {detailed && <th className="px-3 py-2 font-semibold">Anticipación mediana</th>}
                    </tr>
                </thead>
                <tbody className="divide-y divide-neutral-100 dark:divide-neutral-800">
                    {RULES.map(rule => {
                        const event = events[rule.key];
                        return (
                            <tr key={rule.key}>
                                <td className="px-3 py-2.5 font-semibold text-neutral-900 dark:text-neutral-100">
                                    {rule.label}{event.posterior ? ' †' : ''}
                                </td>
                                <td className="px-3 py-2.5">{event.inicios_avisados} de {event.inicios_evaluables}</td>
                                {detailed && <td className="px-3 py-2.5">{event.entradas_avisadas} de {event.entradas_evaluables}</td>}
                                <td className="px-3 py-2.5">{event.falsas_alarmas}</td>
                                {detailed && <td className="px-3 py-2.5">{event.semanas_bajo_alerta} de {event.semanas}</td>}
                                {detailed && <td className="px-3 py-2.5">{event.anticipacion_mediana === null ? '—' : `${formatNumber(event.anticipacion_mediana, 1)} semanas`}</td>}
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

function GlobalMetric({ label, value, detail }) {
    return (
        <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/50 p-4 shadow-sm">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500">{label}</p>
            <p className="mt-1 text-xl font-extrabold text-neutral-900 dark:text-white">{value}</p>
            <p className="mt-0.5 text-[11px] text-neutral-500">{detail}</p>
        </div>
    );
}

function MunicipalityCard({ municipality }) {
    const { ficha, eventos } = municipality;
    return (
        <section className="rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/40 p-5 shadow-sm space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500">Ficha por municipio · Tabla 6</p>
                    <h3 className="text-lg font-bold text-neutral-900 dark:text-white">{municipality.nombre}</h3>
                </div>
                <span className={`rounded-full px-3 py-1 text-xs font-bold ${LEVEL_STYLES[ficha.nivel] || 'bg-neutral-100 text-neutral-600'}`}>
                    {formatLevel(ficha.nivel)}
                </span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="rounded-xl bg-neutral-50 dark:bg-neutral-900/70 p-3">
                    <p className="text-[10px] uppercase tracking-wide text-neutral-500">BSS frente a B2</p>
                    <p className="mt-1 text-xl font-extrabold text-neutral-900 dark:text-white">{formatSigned(ficha.bss_vs_b2)}</p>
                    <p className="text-[10px] text-neutral-500">Positivo es mejor</p>
                </div>
                <div className="rounded-xl bg-neutral-50 dark:bg-neutral-900/70 p-3">
                    <p className="text-[10px] uppercase tracking-wide text-neutral-500">Pares ganados frente a B2</p>
                    <p className="mt-1 text-xl font-extrabold text-neutral-900 dark:text-white">{ficha.pares_ganados} de {ficha.pares_total}</p>
                    <p className="text-[10px] text-neutral-500">Pares municipio-año</p>
                </div>
                <div className="rounded-xl bg-neutral-50 dark:bg-neutral-900/70 p-3">
                    <p className="text-[10px] uppercase tracking-wide text-neutral-500">Precisión promedio (AP)</p>
                    <p className="mt-1 text-xl font-extrabold text-neutral-900 dark:text-white">{formatNumber(ficha.ap.modelo, 2)} / {formatNumber(ficha.ap.b2, 2)}</p>
                    <p className="text-[10px] text-neutral-500">Modelo / referencia B2</p>
                </div>
            </div>
            <EventsTable events={eventos} />
        </section>
    );
}

export default function TestPerformance() {
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);
    const [selectedCode, setSelectedCode] = useState('41001');

    useEffect(() => {
        const controller = new AbortController();
        fetch('/desempeno_prueba.json', { signal: controller.signal })
            .then(async response => {
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return response.json();
            })
            .then(body => {
                if (body.version_esquema !== '1.0' || !body.municipios || !body.global) {
                    throw new Error('El archivo no tiene el esquema 1.0 esperado.');
                }
                setData(body);
                setSelectedCode(previous => body.municipios[previous] ? previous : Object.keys(body.municipios)[0]);
            })
            .catch(reason => {
                if (reason.name !== 'AbortError') setError(`No se pudo cargar /desempeno_prueba.json (${reason.message}).`);
            });
        return () => controller.abort();
    }, []);

    if (error) {
        return (
            <div role="alert" className="rounded-2xl border border-rose-200 dark:border-rose-900/60 bg-rose-50 dark:bg-rose-950/30 p-5 text-sm text-rose-700 dark:text-rose-300 flex items-start gap-3">
                <AlertTriangle className="h-5 w-5 shrink-0" />
                <p>{error} La vista no sustituye los resultados de la evidencia.</p>
            </div>
        );
    }

    if (!data) {
        return (
            <div className="rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/40 p-8 text-center text-sm text-neutral-500">
                Cargando la evidencia estática de la prueba…
            </div>
        );
    }

    const globalProbability = data.global.probabilidad;
    const municipality = data.municipios[selectedCode];

    return (
        <div className="space-y-7 animate-in fade-in duration-500">
            <section className="rounded-2xl border border-neutral-800 bg-gradient-to-r from-neutral-900 to-neutral-800 p-5 sm:p-6 text-white shadow-xl">
                <div className="flex items-start gap-3">
                    <div className="rounded-xl bg-white/10 p-2.5"><ShieldCheck className="h-5 w-5 text-emerald-400" /></div>
                    <div>
                        <div className="flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-wider font-semibold text-emerald-400">
                            <span>Prueba única · 2023–2025</span>
                            <span className="rounded-full border border-neutral-700 bg-neutral-800 px-2 py-0.5 text-neutral-300">{data.global.eventos.canal.semanas} emisiones</span>
                        </div>
                        <h2 className="mt-1 text-2xl sm:text-3xl font-extrabold tracking-tight">Desempeño en prueba</h2>
                        <p className="mt-1 max-w-3xl text-xs sm:text-sm text-neutral-400">
                            Resultados de la receta congelada con predicciones de origen móvil. Esta vista lee solamente la evidencia estática y no consulta el modelo final ni la API.
                        </p>
                    </div>
                </div>
            </section>

            <section className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    <GlobalMetric label="BSS frente a B2" value={formatSigned(globalProbability.bss_vs_b2, 4)} detail={formatLevel(globalProbability.nivel_vs_b2)} />
                    <GlobalMetric label="BSS frente a B1" value={formatSigned(globalProbability.bss_vs_b1, 4)} detail={formatLevel(globalProbability.nivel_vs_b1)} />
                    <GlobalMetric label="AP · modelo / B2" value={`${formatNumber(globalProbability.ap.modelo, 2)} / ${formatNumber(globalProbability.ap.b2, 2)}`} detail="Precisión promedio global" />
                    <GlobalMetric label="Pares ganados frente a B2" value={`${globalProbability.pares_ganados_vs_b2} de ${globalProbability.pares_total}`} detail="Pares municipio-año con menor Brier" />
                </div>
                <div className="rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/40 p-4 sm:p-5 space-y-3">
                    <div>
                        <h3 className="text-sm font-bold text-neutral-900 dark:text-white">Resumen global de alertas</h3>
                        <p className="text-[11px] text-neutral-500">Resultados de las tres reglas sobre las mismas emisiones de prueba.</p>
                    </div>
                    <EventsTable events={data.global.eventos} detailed />
                </div>
            </section>

            <section className="space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3">
                    <div>
                        <p className="text-[10px] font-semibold uppercase tracking-wider text-neutral-500">Resultados por municipio</p>
                        <h3 className="text-lg font-bold text-neutral-900 dark:text-white">Selecciona un municipio piloto</h3>
                    </div>
                    <div className="flex flex-wrap gap-1.5 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-100 dark:bg-neutral-900/70 p-1">
                        {Object.entries(data.municipios).map(([code, item]) => (
                            <button
                                key={code}
                                type="button"
                                onClick={() => setSelectedCode(code)}
                                aria-pressed={selectedCode === code}
                                className={`rounded-lg px-3 py-1.5 text-xs font-medium transition cursor-pointer ${selectedCode === code
                                    ? 'bg-white dark:bg-neutral-800 text-neutral-900 dark:text-white font-bold shadow-sm ring-1 ring-neutral-300 dark:ring-neutral-700'
                                    : 'text-neutral-600 dark:text-neutral-400 hover:bg-white/60 dark:hover:bg-neutral-800'}`}
                            >
                                {item.nombre}
                            </button>
                        ))}
                    </div>
                </div>

                <MunicipalityCard municipality={municipality} />

                <section className="rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/40 p-5 shadow-sm space-y-4">
                    <div className="flex items-start gap-2.5">
                        <LineChart className="mt-0.5 h-4 w-4 text-emerald-500" />
                        <div>
                            <h3 className="text-sm font-bold text-neutral-900 dark:text-white">Probabilidad semana a semana · {municipality.nombre}</h3>
                            <p className="text-[11px] text-neutral-500">Modelo de origen móvil frente a B2 en la semana objetivo. El sombreado marca las semanas que realmente terminaron en zona epidémica.</p>
                        </div>
                    </div>
                    <ProbabilityTestChart serie={municipality.serie} tau={data.receta.tau} />
                </section>

                <section className="rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/40 p-5 shadow-sm space-y-4">
                    <div>
                        <h3 className="text-sm font-bold text-neutral-900 dark:text-white">Franjas de alerta e inicios de brote · {municipality.nombre}</h3>
                        <p className="text-[11px] text-neutral-500">Semanas de emisión bajo alerta con la regla de umbral, la regla del producto y el canal, como en la Fig. 3 del informe.</p>
                    </div>
                    <AlertStripsChart serie={municipality.serie} inicios={municipality.inicios} W={data.receta.W} />
                </section>
            </section>

            <footer className="rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-900/40 p-4 space-y-3">
                <ul className="list-disc list-inside space-y-1 text-[11px] leading-relaxed text-neutral-600 dark:text-neutral-400">
                    {data.notas.map(nota => <li key={nota}>{nota}</li>)}
                </ul>
                <p className="text-[10px] text-neutral-500">† Regla del producto: evaluación posterior descriptiva, no confirmatoria.</p>
                <details className="text-[10px] text-neutral-500">
                    <summary className="cursor-pointer flex items-center gap-1.5"><Database className="h-3 w-3" /> Fuentes de evidencia y trazabilidad</summary>
                    <ul className="mt-2 space-y-1.5">
                        {data.fuentes.map(fuente => (
                            <li key={fuente.ruta} className="break-all rounded bg-white dark:bg-neutral-950/40 p-2">
                                <span className="font-mono">{fuente.ruta}</span><br />
                                <span className="text-neutral-400">SHA-256: {fuente.sha256}</span>
                            </li>
                        ))}
                    </ul>
                </details>
            </footer>
        </div>
    );
}
