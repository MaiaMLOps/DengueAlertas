import React, { useState } from 'react';
import {
    CheckCircle2,
    Compass,
    Lock
} from 'lucide-react';
import departmentsData from '../data/colombia_simplemaps.json';

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
        riesgo: 'Alto',
        probabilidad_base: 0.72,
        modelo_activo: 'LightGBM_Cali',
        pin: { x: 395, y: 552 },
        descripcion: 'Distrito Especial del Valle del Cauca. Alta densidad urbana en comunas periféricas y presencia endémica de serotipos DENV-1 y DENV-2.'
    },
    {
        id: '68001',
        codigo: '68001',
        nombre: 'Bucaramanga',
        departamento: 'Santander',
        deptoId: 'COSAN',
        densidad_poblacional: 3655,
        area_km2: 165,
        poblacion_aprox: '610 mil hab.',
        temp_sugerida: 24.2,
        precip_sugerida: 95,
        humedad_sugerida: 87,
        riesgo: 'Moderado',
        probabilidad_base: 0.48,
        modelo_activo: 'LightGBM_Bucaramanga',
        pin: { x: 546, y: 366 },
        descripcion: 'Capital de Santander. Meseta urbana con microclimas cálidos en el área metropolitana (Girón, Piedecuesta y Floridablanca).'
    },
    {
        id: '13001',
        codigo: '13001',
        nombre: 'Cartagena',
        departamento: 'Bolívar',
        deptoId: 'COBOL',
        densidad_poblacional: 1581,
        area_km2: 609,
        poblacion_aprox: '980 mil hab.',
        temp_sugerida: 29.5,
        precip_sugerida: 130,
        humedad_sugerida: 82,
        riesgo: 'Crítico',
        probabilidad_base: 0.84,
        modelo_activo: 'LightGBM_Cartagena',
        pin: { x: 458, y: 202 },
        descripcion: 'Distrito costero Caribe. Alta humedad constante, temperatura superior a 29°C y gran presión poblacional y turística.'
    },
    {
        id: '73001',
        codigo: '73001',
        nombre: 'Ibagué',
        departamento: 'Tolima',
        deptoId: 'COTOL',
        densidad_poblacional: 377,
        area_km2: 1439,
        poblacion_aprox: '540 mil hab.',
        temp_sugerida: 23.8,
        precip_sugerida: 145,
        humedad_sugerida: 81,
        riesgo: 'Moderado',
        probabilidad_base: 0.42,
        modelo_activo: 'LightGBM_Ibague',
        pin: { x: 446, y: 504 },
        descripcion: 'Valle interandino sobre la Cordillera Central. Cuencas de los ríos Combeima y Chipalo con criaderos estacionales.'
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
        riesgo: 'Alto',
        probabilidad_base: 0.69,
        modelo_activo: 'LightGBM_Villavicencio',
        pin: { x: 506, y: 508 },
        descripcion: 'Puerta a la Orinoquía colombiana (piedemonte llanero). Alto régimen pluviométrico y temperaturas propicias para el vector.'
    }
];

export default function ColombiaMap({ selectedMunicipioId, onSelectMunicipio, activeCodes = ['76001', '68001', '13001', '73001', '50001'] }) {
    const [hoveredItem, setHoveredItem] = useState(null);
    const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

    const isMunActive = (mun) => mun && activeCodes.includes(mun.codigo);
    const enabledMunicipios = MUNICIPIOS_ESTRATEGICOS.filter(isMunActive);
    const selectedMunicipio = MUNICIPIOS_ESTRATEGICOS.find(m => m.id === selectedMunicipioId) || enabledMunicipios[0] || MUNICIPIOS_ESTRATEGICOS[0];

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
                            <span className="text-[10px] px-2 py-0.5 rounded-full font-semibold bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800">
                                {enabledMunicipios.length} {enabledMunicipios.length === 1 ? 'Municipio Activo' : 'Municipios Activos'} (MLflow)
                            </span>
                        </h3>
                        <p className="text-[11px] text-neutral-500">
                            Municipios con modelo <code className="text-emerald-600 dark:text-emerald-400 font-mono">DengueAlertas_*</code> y tag <code className="text-emerald-600 dark:text-emerald-400 font-mono">enabled: true</code> en MLflow.
                        </p>
                    </div>
                </div>

                {/* Selector rápido tipo pastillas compactas */}
                <div className="flex flex-wrap items-center gap-1.5 p-1 rounded-xl bg-neutral-100 dark:bg-neutral-800/60 border border-neutral-200 dark:border-neutral-700/60">
                    {MUNICIPIOS_ESTRATEGICOS.map(mun => {
                        const isSelected = mun.id === selectedMunicipio.id;
                        const isEnabled = isMunActive(mun);
                        return (
                            <button
                                key={mun.id}
                                disabled={!isEnabled}
                                onClick={() => isEnabled && onSelectMunicipio(mun)}
                                title={isEnabled ? `Modelo MLflow habilitado (${mun.nombre})` : `Modelo deshabilitado en MLflow platform`}
                                className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                                    !isEnabled
                                        ? 'opacity-35 cursor-not-allowed bg-neutral-200/50 dark:bg-neutral-800/40 text-neutral-400'
                                        : isSelected
                                            ? 'bg-white dark:bg-neutral-900 text-neutral-900 dark:text-white font-bold shadow-sm ring-1 ring-neutral-300 dark:ring-neutral-700 cursor-pointer'
                                            : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white hover:bg-white/60 dark:hover:bg-neutral-800 cursor-pointer'
                                }`}
                            >
                                <span className={`w-1.5 h-1.5 rounded-full ${
                                    !isEnabled ? 'bg-neutral-400' :
                                    mun.riesgo === 'Crítico' ? 'bg-rose-500' :
                                    mun.riesgo === 'Alto' ? 'bg-amber-500' : 'bg-sky-500'
                                }`} />
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
                {/* Leyenda flotante */}
                <div className="absolute top-4 right-4 z-10 flex flex-col gap-1.5 p-2.5 rounded-xl bg-white/95 dark:bg-neutral-900/95 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 text-[10px] shadow-sm">
                    <div className="flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
                        <span className="font-semibold text-neutral-800 dark:text-neutral-200">
                            Modelo Activo (5)
                        </span>
                    </div>
                    <div className="flex items-center gap-1.5 opacity-70">
                        <span className="w-2.5 h-2.5 rounded-sm bg-neutral-300 dark:bg-neutral-800 border border-dashed border-neutral-400 dark:border-neutral-600" />
                        <span className="text-neutral-500 dark:text-neutral-400">
                            No disponible (Fase 2)
                        </span>
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
                            <feDropShadow dx="0" dy="0" stdDeviation="5" floodColor="#10b981" floodOpacity="0.7" />
                        </filter>
                    </defs>

                    {/* 1. Capa de Departamentos NO DISPONIBLES / DESHABILITADOS */}
                    <g id="deptos-no-disponibles">
                        {departmentsData.filter(d => {
                            const mun = MUNICIPIOS_ESTRATEGICOS.find(m => m.deptoId === d.id);
                            return !isMunActive(mun);
                        }).map(depto => (
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

                    {/* 2. Capa de Departamentos DISPONIBLES (con modelo MLflow enabled) */}
                    <g id="deptos-disponibles">
                        {departmentsData.filter(d => {
                            const mun = MUNICIPIOS_ESTRATEGICOS.find(m => m.deptoId === d.id);
                            return isMunActive(mun);
                        }).map(depto => {
                            const mun = MUNICIPIOS_ESTRATEGICOS.find(m => m.deptoId === depto.id);
                            const isSelected = mun && mun.id === selectedMunicipio.id;

                            return (
                                <path
                                    key={depto.id}
                                    id={depto.id}
                                    d={depto.d}
                                    stroke="currentColor"
                                    strokeWidth={isSelected ? "2.6" : "1.6"}
                                    filter={isSelected ? "url(#glow-filter)" : undefined}
                                    className={`cursor-pointer transition-all duration-300 ${isSelected
                                            ? 'fill-emerald-500/40 text-emerald-500 dark:fill-emerald-500/50 dark:text-emerald-400'
                                            : 'fill-emerald-500/15 text-emerald-600/70 hover:fill-emerald-500/30 hover:text-emerald-500 dark:fill-emerald-500/25 dark:text-emerald-500/70'
                                        }`}
                                    onClick={() => mun && onSelectMunicipio(mun)}
                                    onMouseEnter={() => setHoveredItem({
                                        id: depto.id,
                                        name: `${depto.name} (${mun?.nombre})`,
                                        isAvailable: true,
                                        mun
                                    })}
                                    onMouseLeave={() => setHoveredItem(null)}
                                />
                            );
                        })}
                    </g>

                    {/* 3. Pines y Marcadores Interactivos de los Municipios ACTIVOS */}
                    {enabledMunicipios.map(mun => {
                        const isSelected = mun.id === selectedMunicipio.id;
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
                                    mun
                                })}
                                onMouseLeave={() => setHoveredItem(null)}
                            >
                                {/* Onda de radar pulsante */}
                                {isSelected && (
                                    <circle
                                        r="18"
                                        className="fill-emerald-400/30 animate-ping"
                                    />
                                )}

                                {/* Anillo exterior del pin */}
                                <circle
                                    r={isSelected ? "9" : "6.5"}
                                    className={`transition-all duration-300 ${isSelected
                                            ? 'fill-emerald-500 stroke-white dark:stroke-neutral-900 stroke-2'
                                            : 'fill-emerald-600 hover:fill-emerald-400 stroke-white dark:stroke-neutral-800 stroke-1.5'
                                        }`}
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
                                        className={`transition-all ${isSelected
                                                ? 'fill-emerald-600 text-white shadow-md'
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
                            <p className="text-[10px] text-neutral-300 mt-1">
                                Haz clic para fijar la densidad y parámetros de este municipio.
                            </p>
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
                        <span className={`px-2 py-0.5 rounded-full font-bold text-[10px] ${selectedMunicipio.riesgo === 'Crítico' ? 'bg-rose-100 text-rose-700 dark:bg-rose-950 dark:text-rose-300' :
                                selectedMunicipio.riesgo === 'Alto' ? 'bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300' :
                                    'bg-sky-100 text-sky-700 dark:bg-sky-950 dark:text-sky-300'
                            }`}>
                            Riesgo {selectedMunicipio.riesgo}
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
