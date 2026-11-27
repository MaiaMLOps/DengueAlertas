import React from 'react';

const ZONE_STYLES = {
    seguridad: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300',
    alerta: 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300',
    epidemia: 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300',
};

/** Trayectoria de las 12 emisiones y, cuando aplica, el episodio de continuidad. */
export default function EmissionTrajectoryTable({ trayectoria = [], continuidad, decision }) {
    if (!trayectoria.length) return null;
    const tau = decision?.tau ?? 0.05;
    const episode = continuidad?.trayectoria_episodio ?? [];
    const maxEpisodeP = Math.max(0.01, ...episode.map((row) => row.probabilidad));

    return (
        <div className="space-y-3">
            <div className="flex items-center justify-between gap-2">
                <p className="text-[11px] font-bold uppercase tracking-wider text-neutral-600 dark:text-neutral-300">
                    Trayectoria de emisiones
                </p>
                <span className="text-[10px] text-neutral-400">
                    {trayectoria.length} emisiones · Umbral de señal: probabilidad de {(tau * 100).toLocaleString('es-CO')} % o más
                </span>
            </div>

            <div className="overflow-x-auto rounded-lg border border-neutral-200 dark:border-neutral-800">
                <table className="w-full min-w-[460px] text-left text-[10px] text-neutral-700 dark:text-neutral-300">
                    <thead className="bg-neutral-50 dark:bg-neutral-900/60 text-[9px] uppercase tracking-wide text-neutral-500">
                        <tr>
                            <th className="px-2 py-1.5 font-semibold">Emisión</th>
                            <th className="px-2 py-1.5 font-semibold">Objetivo</th>
                            <th className="px-2 py-1.5 font-semibold">Prob.</th>
                            <th className="px-2 py-1.5 font-semibold">Umbral de señal</th>
                            <th className="px-2 py-1.5 font-semibold">Zona</th>
                            <th className="px-2 py-1.5 font-semibold">Emisión positiva</th>
                            <th className="px-2 py-1.5 font-semibold">Origen</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-neutral-100 dark:divide-neutral-800">
                        {trayectoria.map((row, index) => {
                            const isLast = index === trayectoria.length - 1;
                            return (
                                <tr key={row.semana_emision} className={isLast ? 'bg-emerald-50/70 dark:bg-emerald-950/30' : ''}>
                                    <td className="px-2 py-1.5 font-mono whitespace-nowrap">
                                        {row.semana_emision}
                                        {isLast && <span className="ml-1 font-sans font-bold text-emerald-600 dark:text-emerald-400">actual</span>}
                                    </td>
                                    <td className="px-2 py-1.5 font-mono whitespace-nowrap">{row.semana_objetivo}</td>
                                    <td className="px-2 py-1.5">
                                        <div className="flex items-center gap-1.5">
                                            <span className="w-10 font-mono font-bold">{(row.probabilidad * 100).toFixed(1)}%</span>
                                            <span className="relative block h-1.5 w-14 rounded-full bg-neutral-200 dark:bg-neutral-800">
                                                <span
                                                    className={`absolute inset-y-0 left-0 rounded-full ${row.sobre_tau ? 'bg-rose-500' : 'bg-emerald-500'}`}
                                                    style={{ width: `${Math.min(100, row.probabilidad * 100)}%` }}
                                                />
                                            </span>
                                        </div>
                                    </td>
                                    <td className="px-2 py-1.5">
                                        <span className={`rounded px-1.5 py-0.5 font-semibold ${row.sobre_tau
                                            ? 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300'
                                            : 'bg-neutral-100 text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400'}`}>
                                            {row.sobre_tau ? 'sí' : 'no'}
                                        </span>
                                    </td>
                                    <td className="px-2 py-1.5">
                                        <span className={`rounded px-1.5 py-0.5 font-semibold ${ZONE_STYLES[row.zona] ?? ''}`}>{row.zona}</span>
                                    </td>
                                    <td className="px-2 py-1.5">
                                        <span className={row.emision_positiva ? 'font-bold text-rose-600 dark:text-rose-400' : 'text-neutral-400'}>
                                            {row.emision_positiva ? '✓' : '—'}
                                        </span>
                                    </td>
                                    <td className="px-2 py-1.5 font-mono text-[9px] text-neutral-500">{row.origen}</td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            <p className="text-[9px] leading-relaxed text-neutral-500">
                «Umbral de señal» usa solo la probabilidad: indica si es de {(tau * 100).toLocaleString('es-CO')} % o más.
                «Emisión positiva» aplica la regla {decision?.regla_alerta}: en zona<em> epidemia</em> la emisión se marca
                negativa aunque la probabilidad supere ese umbral.
            </p>

            {continuidad && (
                <div className="rounded-lg border border-rose-200 dark:border-rose-900/60 bg-rose-50/70 dark:bg-rose-950/25 p-3 space-y-2">
                    <div className="flex flex-wrap items-center justify-between gap-2 text-[10px]">
                        <span className="font-bold text-neutral-700 dark:text-neutral-200">
                            Episodio epidémico desde {continuidad.semana_inicio_episodio}
                        </span>
                        <span className="font-mono text-neutral-600 dark:text-neutral-300">
                            Probabilidad de seguir: {(continuidad.probabilidad_seguir * 100).toFixed(1)}% · Referencia: lo esperable solo por la zona actual del canal (B2): {(continuidad.b2_estado_epidemia * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div className="flex h-10 items-end gap-1.5">
                        {episode.map((row) => (
                            <div key={row.semana_emision} className="flex w-14 flex-col items-center gap-1" title={`${row.semana_emision}: ${(row.probabilidad * 100).toFixed(2)}%`}>
                                <span
                                    className="w-full rounded-t bg-rose-500/80"
                                    style={{ height: `${Math.max(8, (row.probabilidad / maxEpisodeP) * 100)}%` }}
                                />
                                <span className="text-[8px] text-neutral-500">{row.semana_emision.slice(5)}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
