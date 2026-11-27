import React from 'react';

const WIDTH = 1100;
const WEEK_MS = 7 * 24 * 60 * 60 * 1000;
const asTime = (date) => Date.parse(`${date}T00:00:00Z`);
const formatPercent = (value, digits = 1) => `${(value * 100).toFixed(digits).replace('.', ',')} %`;
const formatNumber = (value) => Number(value).toLocaleString('es-CO', { maximumFractionDigits: 4 });

function segments(serie, predicate) {
    const result = [];
    let start = null;
    serie.forEach((row, index) => {
        if (predicate(row) && start === null) start = index;
        if (start !== null && (!predicate(row) || index === serie.length - 1)) {
            result.push({ start, end: predicate(row) ? index : index - 1 });
            start = null;
        }
    });
    return result;
}

function yearTicks(start, end) {
    const firstYear = new Date(start).getUTCFullYear();
    const lastYear = new Date(end).getUTCFullYear();
    const ticks = [{ time: start, label: String(firstYear) }];
    for (let year = firstYear + 1; year <= lastYear; year += 1) {
        const time = Date.UTC(year, 0, 1);
        if (time <= end) ticks.push({ time, label: String(year) });
    }
    return ticks;
}

function SegmentBands({ items, serie, dateKey, x, step, y, height, fill, opacity = 1, stroke, className }) {
    return items.map(({ start, end }) => {
        const left = x(asTime(serie[start][dateKey])) - step / 2;
        const right = x(asTime(serie[end][dateKey])) + step / 2;
        return (
            <rect
                key={`${start}-${end}`}
                x={left}
                y={y}
                width={Math.max(1, right - left)}
                height={height}
                fill={fill}
                opacity={opacity}
                stroke={stroke}
                strokeWidth={stroke ? 1 : undefined}
                className={className}
            />
        );
    });
}

export function ProbabilityTestChart({ serie, tau }) {
    const height = 330;
    const pad = { left: 56, right: 22, top: 28, bottom: 50 };
    const innerWidth = WIDTH - pad.left - pad.right;
    const innerHeight = height - pad.top - pad.bottom;
    const first = asTime(serie[0].semana_objetivo);
    const last = asTime(serie[serie.length - 1].semana_objetivo);
    const x = (time) => pad.left + ((time - first) / Math.max(1, last - first)) * innerWidth;
    const y = (value) => pad.top + innerHeight - value * innerHeight;
    const step = innerWidth / Math.max(1, serie.length - 1);
    const ticks = yearTicks(first, last);
    const modelPath = serie.map((row, index) => `${index ? 'L' : 'M'}${x(asTime(row.semana_objetivo)).toFixed(2)},${y(row.probabilidad).toFixed(2)}`).join(' ');
    const b2Path = serie.map((row, index) => `${index ? 'L' : 'M'}${x(asTime(row.semana_objetivo)).toFixed(2)},${y(row.b2_estado_canal).toFixed(2)}`).join(' ');
    const epidemicSegments = segments(serie, row => row.zona_epidemica_observada);

    return (
        <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-[10px] text-neutral-500">
                <span className="flex items-center gap-1.5"><i className="h-0.5 w-5 bg-emerald-500" /> Modelo de origen móvil</span>
                <span className="flex items-center gap-1.5"><i className="h-0.5 w-5 bg-sky-500" /> Referencia B2</span>
                <span className="flex items-center gap-1.5"><i className="h-3 w-4 bg-rose-400/20 border border-rose-200" /> Semana objetivo que terminó en zona epidémica</span>
                <span className="flex items-center gap-1.5"><i className="h-0.5 w-5 border-t border-dashed border-amber-500" /> Umbral de señal: {formatPercent(tau, 0)}</span>
            </div>
            <div className="overflow-x-auto rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950/30">
                <svg viewBox={`0 0 ${WIDTH} ${height}`} className="block h-auto w-full min-w-[760px]" role="img" aria-label="Probabilidad del modelo y de B2 por semana objetivo, con las semanas observadas en zona epidémica">
                    {ticks.map(tick => (
                        <g key={tick.label}>
                            <line x1={x(tick.time)} x2={x(tick.time)} y1={pad.top} y2={pad.top + innerHeight} stroke="currentColor" className="text-neutral-300 dark:text-neutral-700" strokeOpacity="0.35" />
                            <text x={x(tick.time)} y={height - 18} textAnchor="middle" className="fill-neutral-500 text-[11px]">{tick.label}</text>
                        </g>
                    ))}
                    {[0, 0.25, 0.5, 0.75, 1].map(tick => (
                        <g key={tick}>
                            <line x1={pad.left} x2={WIDTH - pad.right} y1={y(tick)} y2={y(tick)} stroke="currentColor" className="text-neutral-300 dark:text-neutral-700" strokeOpacity="0.3" />
                            <text x={pad.left - 9} y={y(tick) + 4} textAnchor="end" className="fill-neutral-500 text-[10px]">{tick * 100} %</text>
                        </g>
                    ))}
                    <SegmentBands
                        items={epidemicSegments}
                        serie={serie}
                        dateKey="semana_objetivo"
                        x={x}
                        step={step}
                        y={pad.top}
                        height={innerHeight}
                        fill="#fb7185"
                        opacity={0.13}
                    />
                    <line x1={pad.left} x2={WIDTH - pad.right} y1={y(tau)} y2={y(tau)} stroke="#f59e0b" strokeDasharray="6 4" strokeWidth="1.2" />
                    <path d={b2Path} fill="none" stroke="#0ea5e9" strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round" />
                    <path d={modelPath} fill="none" stroke="#10b981" strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" />
                    {serie.map(row => {
                        const center = x(asTime(row.semana_objetivo));
                        return (
                            <rect key={row.semana_emision} x={center - step / 2} y={pad.top} width={Math.max(step, 2)} height={innerHeight} fill="transparent">
                                <title>{`Semana objetivo: ${row.semana_objetivo}\nEmisión: ${row.semana_emision}\nModelo: ${formatPercent(row.probabilidad, 2)}\nB2: ${formatPercent(row.b2_estado_canal, 2)}\nTerminó en zona epidémica: ${row.zona_epidemica_observada ? 'sí' : 'no'}\nCasos: ${formatNumber(row.casos_objetivo)}\nTendencia central: ${formatNumber(row.central_objetivo)}\nLímite superior U1e: ${formatNumber(row.u1e_objetivo)}`}</title>
                            </rect>
                        );
                    })}
                    <text x={WIDTH / 2} y={height - 3} textAnchor="middle" className="fill-neutral-500 text-[10px]">Semana objetivo (t + 8)</text>
                </svg>
            </div>
        </div>
    );
}

const RULES = [
    { key: 'umbral', label: 'Regla de umbral', fill: 'currentColor', className: 'text-neutral-900 dark:text-neutral-100' },
    { key: 'producto', label: 'Regla del producto †', fill: 'currentColor', className: 'text-neutral-500 dark:text-neutral-400' },
    { key: 'canal', label: 'Canal', fill: '#ffffff', stroke: 'currentColor', className: 'text-neutral-600 dark:text-neutral-300' },
];

export function AlertStripsChart({ serie, inicios, W }) {
    const height = 245;
    const pad = { left: 142, right: 24, top: 58, bottom: 42 };
    const innerWidth = WIDTH - pad.left - pad.right;
    const rowGap = 45;
    const rowHeight = 22;
    const plotBottom = pad.top + 2 * rowGap + rowHeight;
    const first = asTime(serie[0].semana_emision);
    const lastSerie = asTime(serie[serie.length - 1].semana_emision);
    const last = Math.max(lastSerie, ...inicios.map(inicio => asTime(inicio.semana)));
    const x = (time) => pad.left + ((time - first) / Math.max(1, last - first)) * innerWidth;
    const step = (WEEK_MS / Math.max(1, last - first)) * innerWidth;
    const ticks = yearTicks(first, last);
    const epidemicSegments = segments(serie, row => row.zona_t === 'epidemia');

    return (
        <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-[10px] text-neutral-500">
                <span className="flex items-center gap-1.5"><i className="h-3 w-4 bg-neutral-900 dark:bg-neutral-100" /> Regla de umbral</span>
                <span className="flex items-center gap-1.5"><i className="h-3 w-4 bg-neutral-500" /> Regla del producto †</span>
                <span className="flex items-center gap-1.5"><i className="h-3 w-4 bg-white border border-neutral-600" /> Canal</span>
                <span className="flex items-center gap-1.5"><i className="h-3 w-4 bg-neutral-300" /> Zona epidémica en t</span>
                <span>▼ Inicio de brote y ventana de aviso de {W} semanas</span>
                <span className="flex items-center gap-1.5"><i className="h-2.5 w-2.5 rounded-full bg-neutral-800 dark:bg-neutral-100" /> Avisado</span>
                <span className="flex items-center gap-1.5"><i className="h-2.5 w-2.5 rounded-full border border-neutral-600 bg-white" /> No avisado</span>
            </div>
            <div className="overflow-x-auto rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950/30">
                <svg viewBox={`0 0 ${WIDTH} ${height}`} className="block h-auto w-full min-w-[760px]" role="img" aria-label="Semanas bajo alerta de la regla de umbral, la regla del producto y el canal, con inicios de brote">
                    {ticks.map(tick => (
                        <g key={tick.label}>
                            <line x1={x(tick.time)} x2={x(tick.time)} y1={pad.top - 7} y2={plotBottom + 7} stroke="currentColor" className="text-neutral-300 dark:text-neutral-700" strokeOpacity="0.4" />
                            <text x={x(tick.time)} y={height - 18} textAnchor="middle" className="fill-neutral-500 text-[11px]">{tick.label}</text>
                        </g>
                    ))}
                    <SegmentBands
                        items={epidemicSegments}
                        serie={serie}
                        dateKey="semana_emision"
                        x={x}
                        step={step}
                        y={pad.top}
                        height={plotBottom - pad.top}
                        fill="#d4d4d4"
                        opacity={0.55}
                    />
                    {RULES.map((rule, ruleIndex) => {
                        const y = pad.top + ruleIndex * rowGap;
                        const ruleSegments = segments(serie, row => Boolean(row.bajo_alerta?.[rule.key]));
                        return (
                            <g key={rule.key} className={rule.className}>
                                <text x={pad.left - 12} y={y + 15} textAnchor="end" className="fill-neutral-700 dark:fill-neutral-300 text-[11px] font-semibold">{rule.label}</text>
                                <SegmentBands
                                    items={ruleSegments}
                                    serie={serie}
                                    dateKey="semana_emision"
                                    x={x}
                                    step={step}
                                    y={y}
                                    height={rowHeight}
                                    fill={rule.fill}
                                    opacity={rule.key === 'canal' ? 0.85 : 0.95}
                                    stroke={rule.stroke}
                                    className={rule.className}
                                />
                            </g>
                        );
                    })}
                    {inicios.map((inicio, index) => {
                        const time = asTime(inicio.semana);
                        const triangleY = 16 + (index % 2) * 11;
                        const center = x(time);
                        const windowStart = Math.max(pad.left, x(time - W * WEEK_MS));
                        return (
                            <g key={inicio.semana} className="text-neutral-900 dark:text-neutral-100">
                                <line x1={windowStart} x2={center} y1={triangleY} y2={triangleY} stroke="currentColor" strokeWidth="1.2">
                                    <title>{`Inicio de brote: ${inicio.semana}. Ventana de aviso: ${W} semanas.`}</title>
                                </line>
                                <path d={`M${center},${triangleY + 6} L${center - 5},${triangleY - 3} L${center + 5},${triangleY - 3} Z`} fill="currentColor">
                                    <title>{`Inicio de brote: ${inicio.semana}`}</title>
                                </path>
                                {RULES.map((rule, ruleIndex) => {
                                    const centerY = pad.top + ruleIndex * rowGap + rowHeight / 2;
                                    const avisado = Boolean(inicio.avisado?.[rule.key]);
                                    return (
                                        <circle key={rule.key} cx={center} cy={centerY} r="4.5" fill={avisado ? 'currentColor' : '#ffffff'} stroke="currentColor" strokeWidth="1.3">
                                            <title>{`${inicio.semana} · ${rule.label}: ${avisado ? 'avisado' : 'no avisado'}`}</title>
                                        </circle>
                                    );
                                })}
                            </g>
                        );
                    })}
                    <text x={(pad.left + WIDTH - pad.right) / 2} y={height - 3} textAnchor="middle" className="fill-neutral-500 text-[10px]">Semana de emisión t · los inicios se ubican en su semana s</text>
                </svg>
            </div>
            {inicios.length === 0 && (
                <p className="rounded-lg bg-neutral-50 dark:bg-neutral-900/60 px-3 py-2 text-xs text-neutral-500">Sin inicios de brote en la prueba.</p>
            )}
        </div>
    );
}
