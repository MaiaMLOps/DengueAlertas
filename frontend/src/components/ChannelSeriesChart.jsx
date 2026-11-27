import React, { useEffect, useMemo, useState } from 'react';
import { Maximize2, X } from 'lucide-react';

const ZONE_COLORS = {
    seguridad: '#10b981',
    alerta: '#f59e0b',
    epidemia: '#ef4444',
};

const WIDTH = 720;
const HEIGHT = 250;
const PAD = { left: 38, right: 14, top: 18, bottom: 34 };
const INNER_W = WIDTH - PAD.left - PAD.right;
const INNER_H = HEIGHT - PAD.top - PAD.bottom;

function ChartCanvas({ serie, semanaEmision, semanaObjetivo }) {
    const [hover, setHover] = useState(null);

    const chart = useMemo(() => {
        if (!serie.length) return null;
        const n = serie.length;
        const maxValue = Math.max(
            50,
            ...serie.map((row) => Math.max(row.casos ?? 0, row.u1e ?? 0)),
        );
        const maxY = Math.ceil(maxValue / 50) * 50;
        const x = (i) => PAD.left + (i / Math.max(1, n - 1)) * INNER_W;
        const y = (value) => PAD.top + INNER_H - (value / maxY) * INNER_H;
        const path = (key) => serie.map((row, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(row[key]).toFixed(1)}`).join(' ');
        const upperReversed = [...serie].reverse().map((row, reverseIndex) => {
            const i = n - 1 - reverseIndex;
            return `L${x(i).toFixed(1)},${y(row.u1e).toFixed(1)}`;
        }).join(' ');
        const areaPath = `${path('central')} ${upperReversed} Z`;
        const futureStart = serie.findIndex((row) => row.tipo === 'futura');
        const emissionIndex = serie.findIndex((row) => row.semana === semanaEmision);
        const targetIndex = serie.findIndex((row) => row.semana === semanaObjetivo);
        const step = INNER_W / Math.max(1, n - 1);
        return { n, maxY, x, y, path, areaPath, futureStart, emissionIndex, targetIndex, step };
    }, [serie, semanaEmision, semanaObjetivo]);

    if (!chart) return null;
    const { n, maxY, x, y, path, areaPath, futureStart, emissionIndex, targetIndex, step } = chart;
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((fraction) => fraction * maxY);
    const hovered = hover === null ? null : serie[hover];

    return (
        <div className="relative w-full overflow-hidden rounded-lg border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900/50">
            <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="block w-full" role="img" aria-label="Serie del canal con casos observados, tendencia central y límite superior">
                {futureStart >= 0 && (
                    <rect
                        x={x(futureStart) - step / 2}
                        y={PAD.top}
                        width={WIDTH - PAD.right - (x(futureStart) - step / 2)}
                        height={INNER_H}
                        fill="#94a3b8"
                        opacity="0.10"
                    />
                )}

                {ticks.map((tick) => (
                    <g key={tick}>
                        <line x1={PAD.left} x2={WIDTH - PAD.right} y1={y(tick)} y2={y(tick)} stroke="currentColor" strokeOpacity="0.10" />
                        <text x={PAD.left - 6} y={y(tick) + 3} textAnchor="end" className="fill-neutral-400 text-[9px]">{tick}</text>
                    </g>
                ))}

                <path d={areaPath} fill="#38bdf8" opacity="0.12" />
                <path d={path('central')} fill="none" stroke="#0ea5e9" strokeWidth="1.8" />
                <path d={path('u1e')} fill="none" stroke="#ef4444" strokeWidth="1.6" strokeDasharray="5 3" opacity="0.85" />

                {serie.map((row, i) => row.casos === null ? null : (
                    <rect
                        key={row.semana}
                        x={x(i) - 3}
                        y={y(row.casos)}
                        width="6"
                        height={PAD.top + INNER_H - y(row.casos)}
                        rx="1"
                        fill={ZONE_COLORS[row.zona] ?? '#94a3b8'}
                        opacity="0.85"
                    />
                ))}

                {emissionIndex >= 0 && (
                    <g>
                        <line x1={x(emissionIndex)} x2={x(emissionIndex)} y1={PAD.top} y2={PAD.top + INNER_H} stroke="#0f172a" strokeOpacity="0.35" strokeDasharray="3 3" />
                        <text x={x(emissionIndex) + 3} y={PAD.top + 10} className="fill-neutral-500 text-[9px]">emisión t</text>
                    </g>
                )}
                {targetIndex >= 0 && (
                    <g>
                        <line x1={x(targetIndex)} x2={x(targetIndex)} y1={PAD.top} y2={PAD.top + INNER_H} stroke="#7c3aed" strokeOpacity="0.55" strokeDasharray="3 3" />
                        <text x={x(targetIndex) - 3} y={PAD.top + 10} textAnchor="end" className="fill-violet-600 dark:fill-violet-400 text-[9px]">t+8</text>
                    </g>
                )}

                <line x1={PAD.left} x2={WIDTH - PAD.right} y1={PAD.top + INNER_H} y2={PAD.top + INNER_H} stroke="currentColor" strokeOpacity="0.20" />
                {serie.filter((_, i) => i % 8 === 0 || i === n - 1).map((row) => {
                    const i = serie.indexOf(row);
                    return (
                        <text key={row.semana} x={x(i)} y={HEIGHT - 12} textAnchor="middle" className="fill-neutral-400 text-[8px]">
                            {row.semana.slice(5)}
                        </text>
                    );
                })}

                {serie.map((row, i) => (
                    <rect
                        key={`hit-${row.semana}`}
                        x={x(i) - step / 2}
                        y={PAD.top}
                        width={step}
                        height={INNER_H}
                        fill="transparent"
                        onMouseEnter={() => setHover(i)}
                        onMouseLeave={() => setHover(null)}
                    />
                ))}
            </svg>

            {hovered && (
                <div
                    className="pointer-events-none absolute top-2 z-10 w-64 rounded-lg border border-neutral-200 dark:border-neutral-700 bg-white/95 dark:bg-neutral-900/95 p-2 text-[10px] shadow-lg"
                    style={{
                        left: `${(x(hover) / WIDTH) * 100}%`,
                        transform: hover > n / 2 ? 'translateX(-100%)' : 'none',
                    }}
                >
                    <p className="font-bold text-neutral-800 dark:text-neutral-100">{hovered.semana}</p>
                    <p className="text-neutral-500">{hovered.tipo === 'observada' ? `Casos: ${hovered.casos}` : 'Sin observación'}</p>
                    <p className="text-sky-600 dark:text-sky-400">Tendencia central del canal (casos esperados): {hovered.central}</p>
                    <p className="text-rose-600 dark:text-rose-400">Límite superior del canal (por encima: zona epidémica): {hovered.u1e}</p>
                    {hovered.zona && (
                        <p className="mt-1 font-semibold" style={{ color: ZONE_COLORS[hovered.zona] }}>
                            {hovered.zona}
                        </p>
                    )}
                </div>
            )}
        </div>
    );
}

function Legend() {
    return (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[9px] text-neutral-500">
            <span className="flex items-center gap-1"><i className="h-2 w-2 rounded-sm bg-emerald-500" /> seguridad</span>
            <span className="flex items-center gap-1"><i className="h-2 w-2 rounded-sm bg-amber-500" /> alerta</span>
            <span className="flex items-center gap-1"><i className="h-2 w-2 rounded-sm bg-rose-500" /> epidemia</span>
            <span className="flex items-center gap-1"><i className="h-0.5 w-4 bg-sky-500" /> Tendencia central del canal (casos esperados)</span>
            <span className="flex items-center gap-1"><i className="h-0.5 w-4 border-t border-dashed border-rose-500" /> Límite superior del canal (por encima: zona epidémica)</span>
            <span>Las semanas futuras muestran la referencia estacional, no casos predichos.</span>
        </div>
    );
}

/** Serie del canal epiCo con vista ampliada en overlay. */
export default function ChannelSeriesChart({ serie = [], semanaEmision, semanaObjetivo }) {
    const [expanded, setExpanded] = useState(false);

    useEffect(() => {
        if (!expanded) return undefined;
        const closeOnEscape = (event) => {
            if (event.key === 'Escape') setExpanded(false);
        };
        window.addEventListener('keydown', closeOnEscape);
        return () => window.removeEventListener('keydown', closeOnEscape);
    }, [expanded]);

    if (!serie.length) return null;

    return (
        <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
                <p className="text-[11px] font-bold uppercase tracking-wider text-neutral-600 dark:text-neutral-300">
                    Serie del canal epiCo
                </p>
                <div className="flex items-center gap-2">
                    <span className="text-[10px] text-neutral-400">52 observadas · 8 futuras</span>
                    <button
                        type="button"
                        onClick={() => setExpanded(true)}
                        aria-label="Ampliar serie del canal"
                        className="inline-flex items-center gap-1 rounded-lg border border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-2 py-1 text-[10px] font-semibold text-neutral-600 dark:text-neutral-300 transition hover:border-emerald-400 hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer"
                    >
                        <Maximize2 className="h-3 w-3" />
                        Ampliar
                    </button>
                </div>
            </div>

            <ChartCanvas serie={serie} semanaEmision={semanaEmision} semanaObjetivo={semanaObjetivo} />
            <Legend />

            {expanded && (
                <div
                    className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
                    onClick={(event) => {
                        if (event.target === event.currentTarget) setExpanded(false);
                    }}
                >
                    <div
                        role="dialog"
                        aria-modal="true"
                        aria-label="Serie del canal epiCo ampliada"
                        className="w-full max-w-5xl space-y-3 rounded-2xl border border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900 p-4 shadow-2xl"
                        onClick={(event) => event.stopPropagation()}
                    >
                        <div className="flex items-center justify-between gap-3">
                            <div>
                                <p className="text-sm font-bold text-neutral-900 dark:text-white">Serie del canal epiCo</p>
                                <p className="text-[10px] text-neutral-500">52 semanas observadas · 8 semanas futuras · emisión {semanaEmision}</p>
                            </div>
                            <button
                                type="button"
                                onClick={() => setExpanded(false)}
                                aria-label="Cerrar vista ampliada"
                                className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-neutral-200 dark:border-neutral-700 text-neutral-500 transition hover:bg-rose-50 hover:text-rose-600 dark:hover:bg-rose-950/40 dark:hover:text-rose-400 cursor-pointer"
                            >
                                <X className="h-4 w-4" />
                            </button>
                        </div>

                        <ChartCanvas serie={serie} semanaEmision={semanaEmision} semanaObjetivo={semanaObjetivo} />
                        <Legend />
                        <p className="text-[10px] text-neutral-400">Cierra con la X, haciendo clic fuera del gráfico o pulsando Escape.</p>
                    </div>
                </div>
            )}
        </div>
    );
}
