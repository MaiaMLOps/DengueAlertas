import React, { useState } from 'react';
import {
    CheckCircle2,
    Compass,
    Lock
} from 'lucide-react';
import departmentsData from '../data/colombia_simplemaps.json';

// Los porcentajes 2012–2025 provienen de la Tabla 1 del informe; no son
// probabilidades del modelo ni niveles de riesgo calculados por el frontend.
export const MUNICIPIOS_ESTRATEGICOS = [
    {
        id: '76001',
        codigo: '76001',
        nombre: 'Cali',
        departamento: 'Valle del Cauca',
        deptoId: 'COVAC',
        densidad_poblacional: 3910,
        area_km2: 562,
        poblacion_aprox: '2.2 millones hab.',
        temp_sugerida: 26.5,
        precip_sugerida: 115,
        humedad_sugerida: 86,
        porcentaje_semanas_epidemia_2012_2025: 17.0,
        modelo_activo: 'DengueAlertas_Cali',
        pin: { x: 395, y: 552 },
        descripcion: 'Distrito Especial del Valle del Cauca. Alta densidad urbana en comunas periféricas y presencia endémica de serotipos DENV-1 y DENV-2.'
    },
    {
        id: '08001',
        codigo: '08001',
        nombre: 'Barranquilla',
        departamento: 'Atlántico',
        deptoId: 'COATL',
        densidad_poblacional: 7320,
        area_km2: 154,
        poblacion_aprox: '1.3 millones hab.',
        temp_sugerida: 28.5,
        precip_sugerida: 110,
        humedad_sugerida: 80,
        porcentaje_semanas_epidemia_2012_2025: 23.4,
        modelo_activo: 'DengueAlertas_Barranquilla',
        pin: { x: 468, y: 190 },
        descripcion: 'Distrito Especial, Industrial y Portuario de Barranquilla. Clima tropical cálido y alta densidad urbana en el Caribe.'
    },
    {
        id: '08758',
        codigo: '08758',
        nombre: 'Soledad',
        departamento: 'Atlántico',
        deptoId: 'COATL',
        densidad_poblacional: 9850,
        area_km2: 67,
        poblacion_aprox: '690 mil hab.',
        temp_sugerida: 28.5,
        precip_sugerida: 115,
        humedad_sugerida: 82,
        porcentaje_semanas_epidemia_2012_2025: 27.9,
        modelo_activo: 'DengueAlertas_Soledad',
        pin: { x: 474, y: 202 },
        descripcion: 'Municipio del área metropolitana de Barranquilla. Máxima densidad poblacional y constante flujo poblacional metropolitano.'
    },
    {
        id: '41001',
        codigo: '41001',
        nombre: 'Neiva',
        departamento: 'Huila',
        deptoId: 'COHUI',
        densidad_poblacional: 215,
        area_km2: 1553,
        poblacion_aprox: '360 mil hab.',
        temp_sugerida: 27.8,
        precip_sugerida: 105,
        humedad_sugerida: 72,
        porcentaje_semanas_epidemia_2012_2025: 20.8,
        modelo_activo: 'DengueAlertas_Neiva',
        pin: { x: 442, y: 618 },
        descripcion: 'Capital del departamento del Huila, sobre el valle del río Magdalena. Temperaturas cálidas y transmisión endémica sostenida.'
    },
    {
        id: '50001',
        codigo: '50001',
        nombre: 'Villavicencio',
        departamento: 'Meta',
        deptoId: 'COMET',
        densidad_poblacional: 395,
        area_km2: 1328,
        poblacion_aprox: '530 mil hab.',
        temp_sugerida: 27.0,
        precip_sugerida: 180,
        humedad_sugerida: 75,
        porcentaje_semanas_epidemia_2012_2025: 22.6,
        modelo_activo: 'DengueAlertas_Villavicencio',
        pin: { x: 506, y: 508 },
        descripcion: 'Puerta a la Orinoquía colombiana (piedemonte llanero). Alto régimen pluviométrico y temperaturas propicias para el vector.'
    },
    {
        id: '54001',
        codigo: '54001',
        nombre: 'Cúcuta',
        departamento: 'Norte de Santander',
        deptoId: 'CONSA',
        densidad_poblacional: 625,
        area_km2: 1176,
        poblacion_aprox: '780 mil hab.',
        temp_sugerida: 28.2,
        precip_sugerida: 90,
        humedad_sugerida: 76,
        porcentaje_semanas_epidemia_2012_2025: 17.7,
        modelo_activo: 'DengueAlertas_Cucuta',
        pin: { x: 575, y: 345 },
        descripcion: 'San José de Cúcuta, capital fronteriza de Norte de Santander. Valle cálido con alta dinámica de transmisión vectorial.'
    }
];

const ALERT_STATE_STYLES = {
    sin_alerta: {
        label: 'Sin alerta',
        color: '#10b981',
        badge: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300'
    },
    senal_en_observacion: {
        label: 'Señal en observación',
        color: '#f59e0b',
        badge: 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300'
    },
    alerta_inicio_confirmada: {
        label: 'Alerta de inicio confirmada',
        color: '#f97316',
        badge: 'bg-orange-100 text-orange-800 dark:bg-orange-950 dark:text-orange-300'
    },
    continuidad_epidemica: {
        label: 'Continuidad epidémica',
        color: '#f43f5e',
        badge: 'bg-rose-100 text-rose-700 dark:bg-rose-950 dark:text-rose-300'
    }
};

const NEUTRAL_STATE_STYLE = {
    label: 'Sin respuesta de la API',
    color: '#94a3b8',
    badge: 'bg-neutral-200 text-neutral-600 dark:bg-neutral-800 dark:text-neutral-300'
};

const formatHistoricalPercentage = (value) => `${value.toFixed(1).replace('.', ',')} %`;

export default function ColombiaMap({
    selectedMunicipioId,
    onSelectMunicipio,
    activeCodes = [],
    activeModelsStatus = 'loading',
    mapStatuses = {},
    emissionWeek
}) {
    const [hoveredItem, setHoveredItem] = useState(null);
    const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

    const isMunActive = (mun) => mun && activeCodes.includes(mun.codigo);
    const enabledMunicipios = MUNICIPIOS_ESTRATEGICOS.filter(isMunActive);
    const selectedMunicipio = MUNICIPIOS_ESTRATEGICOS.find(m => m.id === selectedMunicipioId) || enabledMunicipios[0] || MUNICIPIOS_ESTRATEGICOS[0];
    const getAlertState = (mun) => {
        const result = mapStatuses[mun?.codigo];
        if (result?.status !== 'ready' || result.data.semana_emision !== emissionWeek) return null;
        return result.data.decision?.estado_alerta || null;
    };
    const getVisualStyle = (mun) => {
        const state = getAlertState(mun);
        if (ALERT_STATE_STYLES[state]) return ALERT_STATE_STYLES[state];
        if (mapStatuses[mun?.codigo]?.status === 'loading') {
            return { ...NEUTRAL_STATE_STYLE, label: 'Consultando la API…' };
        }
        return NEUTRAL_STATE_STYLE;
    };
    const selectedVisualStyle = getVisualStyle(selectedMunicipio);

    const handleMouseMove = (e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        setTooltipPos({
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        });
    };

    return (
        <div className="space-y-5">
            {/* Header del Mapa */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-neutral-100 dark:border-neutral-800 pb-3">
                <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-lg bg-emerald-500/10 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 flex items-center justify-center">
                        <Compass className="w-4 h-4" />
                    </div>
                    <div>
                        <h3 className="font-bold text-base text-neutral-900 dark:text-white flex items-center gap-2">
                            Mapa de Monitoreo Territorial
                            <span
                                className="text-[10px] px-2 py-0.5 rounded-full font-semibold bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800"
                                title="Según GET /municipios, que descubre los modelos DengueAlertas_* con enabled=true en MLflow"
                            >
                                {activeModelsStatus === 'loading'
                                    ? 'Consultando modelos…'
                                    : activeModelsStatus === 'error'
                                        ? 'Modelos no disponibles'
                                        : `${enabledMunicipios.length} ${enabledMunicipios.length === 1 ? 'modelo activo' : 'modelos activos'} · enabled: true`}
                            </span>
                        </h3>
                        <p className="text-[11px] text-neutral-500">
                            {emissionWeek
                                ? <>Marcadores según el estado de la API para la semana de emisión <strong>{emissionWeek}</strong>; gris si no hay respuesta.</>
                                : 'Marcadores neutros: consulta la API para conocer el estado de alerta.'}
                        </p>
                    </div>
                </div>

                {/* Selector rápido tipo pastillas compactas */}
                <div className="flex flex-wrap items-center gap-1.5 p-1 rounded-xl bg-neutral-100 dark:bg-neutral-800/60 border border-neutral-200 dark:border-neutral-700/60">
                    {MUNICIPIOS_ESTRATEGICOS.map(mun => {
                        const isSelected = mun.id === selectedMunicipio.id;
                        const isEnabled = isMunActive(mun);
                        const visualStyle = getVisualStyle(mun);
                        return (
                            <button
                                key={mun.id}
                                disabled={!isEnabled}
                                onClick={() => isEnabled && onSelectMunicipio(mun)}
                                title={isEnabled ? `${mun.nombre}: ${visualStyle.label}` : `Modelo deshabilitado en MLflow platform`}
                                className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                                    !isEnabled
                                        ? 'opacity-35 cursor-not-allowed bg-neutral-200/50 dark:bg-neutral-800/40 text-neutral-400'
                                        : isSelected
                                            ? 'bg-white dark:bg-neutral-900 text-neutral-900 dark:text-white font-bold shadow-sm ring-1 ring-neutral-300 dark:ring-neutral-700 cursor-pointer'
                                            : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white hover:bg-white/60 dark:hover:bg-neutral-800 cursor-pointer'
                                }`}
                            >
                                <span
                                    className="w-1.5 h-1.5 rounded-full"
                                    style={{ backgroundColor: isEnabled ? visualStyle.color : '#a3a3a3' }}
                                />
                                <span>{mun.nombre}</span>
                                {!isEnabled && <Lock className="w-2.5 h-2.5 text-neutral-400" />}
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* Contenedor del Mapa SVG con los 33 Departamentos Oficiales */}
            <div
                className="relative rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-neutral-900/5 dark:bg-neutral-950 p-3 sm:p-5 overflow-hidden shadow-inner"
                onMouseMove={handleMouseMove}
            >
                {/* Leyenda del estado devuelto por la API */}
                <div className="absolute top-4 right-4 z-10 flex flex-col gap-1.5 p-2.5 rounded-xl bg-white/95 dark:bg-neutral-900/95 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 text-[10px] shadow-sm">
                    <span className="font-semibold text-neutral-800 dark:text-neutral-200">Estado API · marcadores</span>
                    <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
                        {Object.entries(ALERT_STATE_STYLES).map(([state, style]) => (
                            <div key={state} className="flex items-center gap-1.5" title={style.label}>
                                <span className="w-2.5 h-2.5 shrink-0 rounded-full" style={{ backgroundColor: style.color }} />
                                <span className="text-neutral-600 dark:text-neutral-300">
                                    {state === 'sin_alerta' ? 'Sin alerta' :
                                        state === 'senal_en_observacion' ? 'En observación' :
                                            state === 'alerta_inicio_confirmada' ? 'Alerta confirmada' : 'Continuidad epidémica'}
                                </span>
                            </div>
                        ))}
                    </div>
                    <div className="flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: NEUTRAL_STATE_STYLE.color }} />
                        <span className="text-neutral-500 dark:text-neutral-400">Sin respuesta de la API</span>
                    </div>
                    <div className="flex items-center gap-1.5 opacity-70">
                        <span className="w-2.5 h-2.5 rounded-sm bg-neutral-300 dark:bg-neutral-800 border border-dashed border-neutral-400 dark:border-neutral-600" />
                        <span className="text-neutral-500 dark:text-neutral-400">Municipio no disponible</span>
                    </div>
                </div>

                {/* SVG Vectorial con Proyección Oficial de Colombia */}
                <svg
                    viewBox="200 60 590 870"
                    className="w-full h-[440px] sm:h-[530px] select-none"
                >
                    <defs>
                        {/* Patrón de líneas diagonales para máscara de departamento no disponible */}
                        <pattern
                            id="co-mask-stripes"
                            width="8"
                            height="8"
                            patternTransform="rotate(45 0 0)"
                            patternUnits="userSpaceOnUse"
                        >
                            <line
                                x1="0"
                                y1="0"
                                x2="0"
                                y2="8"
                                stroke="currentColor"
                                strokeWidth="1.2"
                                className="text-neutral-300 dark:text-neutral-800"
                            />
                        </pattern>

                        {/* Filtro de resplandor para departamento activo */}
                        <filter id="glow-filter" x="-20%" y="-20%" width="140%" height="140%">
                            <feDropShadow dx="0" dy="0" stdDeviation="5" floodColor="#a8a29e" floodOpacity="0.7" />
                        </filter>
                    </defs>

                    {/* 1. Capa de Departamentos NO DISPONIBLES / DESHABILITADOS */}
                    <g id="deptos-no-disponibles">
                        {departmentsData.filter(d => !MUNICIPIOS_ESTRATEGICOS.some(
                            m => m.deptoId === d.id && isMunActive(m)
                        )).map(depto => (
                            <path
                                key={depto.id}
                                id={depto.id}
                                d={depto.d}
                                fill="url(#co-mask-stripes)"
                                stroke="currentColor"
                                strokeWidth="0.8"
                                strokeDasharray="3 2"
                                className="text-neutral-300 dark:text-neutral-800 transition-colors duration-200 hover:text-neutral-400 dark:hover:text-neutral-700 cursor-not-allowed"
                                onMouseEnter={() => setHoveredItem({
                                    id: depto.id,
                                    name: depto.name,
                                    isAvailable: false
                                })}
                                onMouseLeave={() => setHoveredItem(null)}
                            />
                        ))}
                    </g>

                    {/* 2. Fondo neutro: el estado municipal se representa solo en los marcadores. */}
                    <g id="deptos-disponibles">
                        {departmentsData.filter(d => MUNICIPIOS_ESTRATEGICOS.some(
                            m => m.deptoId === d.id && isMunActive(m)
                        )).map(depto => {
                            const municipios = enabledMunicipios.filter(m => m.deptoId === depto.id);
                            const targetMun = municipios.find(m => m.id === selectedMunicipio.id) || municipios[0];
                            const isSelected = targetMun?.id === selectedMunicipio.id;

                            return (
                                <path
                                    key={depto.id}
                                    id={depto.id}
                                    d={depto.d}
                                    fill="#a8a29e"
                                    fillOpacity={isSelected ? 0.28 : 0.14}
                                    stroke="#a8a29e"
                                    strokeWidth={isSelected ? "2.6" : "1.6"}
                                    filter={isSelected ? "url(#glow-filter)" : undefined}
                                    className="cursor-pointer transition-all duration-300 hover:fill-opacity-35"
                                    onClick={() => targetMun && onSelectMunicipio(targetMun)}
                                    onMouseEnter={() => setHoveredItem({
                                        id: depto.id,
                                        name: `${depto.name} (${municipios.map(m => m.nombre).join(' / ')})`,
                                        isAvailable: true,
                                        municipios
                                    })}
                                    onMouseLeave={() => setHoveredItem(null)}
                                />
                            );
                        })}
                    </g>

                    {/* 3. Pines y Marcadores Interactivos de los Municipios ACTIVOS */}
                    {enabledMunicipios.map(mun => {
                        const isSelected = mun.id === selectedMunicipio.id;
                        const visualStyle = getVisualStyle(mun);
                        return (
                            <g
                                key={mun.id}
                                transform={`translate(${mun.pin.x}, ${mun.pin.y})`}
                                className="cursor-pointer group"
                                onClick={() => onSelectMunicipio(mun)}
                                onMouseEnter={() => setHoveredItem({
                                    id: mun.id,
                                    name: `${mun.nombre} (${mun.departamento})`,
                                    isAvailable: true,
                                    municipios: [mun]
                                })}
                                onMouseLeave={() => setHoveredItem(null)}
                            >
                                {/* Onda de radar pulsante */}
                                {isSelected && (
                                    <circle
                                        r="18"
                                        fill={visualStyle.color}
                                        fillOpacity="0.3"
                                        className="animate-ping"
                                    />
                                )}

                                {/* Anillo exterior del pin */}
                                <circle
                                    r={isSelected ? "9" : "6.5"}
                                    fill={visualStyle.color}
                                    className={`transition-all duration-300 stroke-white dark:stroke-neutral-900 ${isSelected ? 'stroke-2' : 'stroke-1.5 group-hover:opacity-80'}`}
                                />

                                {/* Centro blanco */}
                                <circle
                                    r={isSelected ? "3.5" : "2"}
                                    fill="#ffffff"
                                />

                                {/* Etiqueta del municipio */}
                                <g transform="translate(10, 4)">
                                    <rect
                                        x="-2"
                                        y="-12"
                                        width={mun.nombre.length * 8 + 14}
                                        height="17"
                                        rx="4"
                                        fill={isSelected ? visualStyle.color : undefined}
                                        className={`transition-all ${isSelected
                                                ? 'text-white shadow-md'
                                                : 'fill-white/95 dark:fill-neutral-900/95 text-neutral-800 dark:text-neutral-200 stroke border border-neutral-200 dark:border-neutral-700'
                                            }`}
                                    />
                                    <text
                                        x="5"
                                        y="0"
                                        fontSize="9.5"
                                        fontWeight={isSelected ? "bold" : "600"}
                                        fill={isSelected ? "#ffffff" : "currentColor"}
                                        className="pointer-events-none"
                                    >
                                        {mun.nombre}
                                    </text>
                                </g>
                            </g>
                        );
                    })}
                </svg>

                {/* Tooltip Dinámico Flotante */}
                {hoveredItem && (
                    <div
                        className="pointer-events-none absolute z-20 px-3.5 py-2.5 rounded-xl bg-neutral-900/95 text-white backdrop-blur-md shadow-2xl border border-neutral-700 text-xs transition-opacity duration-150 animate-in fade-in"
                        style={{
                            left: Math.min(Math.max(tooltipPos.x + 10, 10), 300),
                            top: Math.max(tooltipPos.y - 50, 10)
                        }}
                    >
                        <div className="flex items-center gap-1.5 font-bold">
                            {hoveredItem.isAvailable ? (
                                <>
                                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                                    <span>{hoveredItem.name}</span>
                                    <span className="text-[9px] px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-300 font-mono">
                                        Disponible
                                    </span>
                                </>
                            ) : (
                                <>
                                    <Lock className="w-3.5 h-3.5 text-amber-400" />
                                    <span className="text-neutral-300">{hoveredItem.name}</span>
                                    <span className="text-[9px] px-1.5 py-0.2 rounded bg-rose-500/20 text-rose-300 font-mono">
                                        No disponible
                                    </span>
                                </>
                            )}
                        </div>
                        {hoveredItem.isAvailable ? (
                            <div className="mt-1.5 space-y-2 text-[10px] text-neutral-300">
                                {hoveredItem.municipios.map(mun => {
                                    const visualStyle = getVisualStyle(mun);
                                    return (
                                        <div key={mun.codigo} className="space-y-1">
                                            <p className="flex items-center gap-1.5">
                                                <span
                                                    className="w-2 h-2 shrink-0 rounded-full"
                                                    style={{ backgroundColor: visualStyle.color }}
                                                />
                                                <span>{mun.nombre}: <strong>{visualStyle.label}</strong></span>
                                            </p>
                                            <p>
                                                Zona epidémica 2012–2025: <strong>{formatHistoricalPercentage(mun.porcentaje_semanas_epidemia_2012_2025)}</strong>
                                            </p>
                                        </div>
                                    );
                                })}
                                <p className="text-neutral-400">Haz clic para seleccionar el municipio.</p>
                            </div>
                        ) : (
                            <p className="text-[10px] text-neutral-400 mt-1">
                                No disponible para inferencia en esta fase.
                            </p>
                        )}
                    </div>
                )}
            </div>

            {/* Ficha Informativa del Municipio Seleccionado */}
            <div className="p-3.5 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50/70 dark:bg-neutral-900/50 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                <div className="space-y-1">
                    <div className="flex items-center gap-2">
                        <span className="text-sm font-bold text-neutral-900 dark:text-white">
                            📍 {selectedMunicipio.nombre}
                        </span>
                        <span className="text-neutral-500">
                            ({selectedMunicipio.departamento})
                        </span>
                        <span className={`px-2 py-0.5 rounded-full font-bold text-[10px] ${selectedVisualStyle.badge}`}>
                            {selectedVisualStyle.label}
                        </span>
                    </div>
                    <p className="text-[11px] text-neutral-500 line-clamp-1">
                        {selectedMunicipio.descripcion}
                    </p>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                    <div className="px-2.5 py-1 rounded-lg bg-white dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 text-right">
                        <span className="text-[9px] text-neutral-400 block uppercase font-medium">Densidad DANE</span>
                        <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">
                            {selectedMunicipio.densidad_poblacional.toLocaleString()} hab/km²
                        </span>
                    </div>
                    <div className="px-2.5 py-1 rounded-lg bg-white dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 text-right">
                        <span className="text-[9px] text-neutral-400 block uppercase font-medium">Población</span>
                        <span className="font-semibold text-neutral-800 dark:text-neutral-200">
                            {selectedMunicipio.poblacion_aprox}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
}
