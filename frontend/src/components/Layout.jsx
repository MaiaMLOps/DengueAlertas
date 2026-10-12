import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { Activity, ShieldAlert, BarChart3, Database } from 'lucide-react';

export default function Layout() {
    return (
        <div className="min-h-screen flex flex-col bg-neutral-50 dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 antialiased selection:bg-emerald-500 selection:text-white">
            {/* Top Navigation Bar */}
            <header className="bg-white/80 dark:bg-neutral-900/80 backdrop-blur-md border-b border-neutral-200 dark:border-neutral-800 sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex justify-between items-center">
                    {/* Brand / Logo */}
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 flex items-center justify-center text-white shadow-md shadow-emerald-500/20">
                            <Activity className="w-5 h-5" />
                        </div>
                        <div>
                            <div className="flex items-center gap-2">
                                <span className="font-extrabold text-base tracking-tight text-neutral-900 dark:text-white">
                                    DengueAlertas
                                </span>
                                <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300">
                                    MLOps
                                </span>
                            </div>
                            <p className="text-[11px] text-neutral-500 dark:text-neutral-400">
                                Vigilancia Epidemiológica & Alerta Temprana
                            </p>
                        </div>
                    </div>

                    {/* Nav Links */}
                    <nav className="hidden md:flex items-center gap-1 text-xs font-medium text-neutral-600 dark:text-neutral-400">
                        <NavLink
                            to="/"
                            className={({ isActive }) =>
                                `flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition ${
                                    isActive
                                        ? 'bg-neutral-100 dark:bg-neutral-800 text-neutral-900 dark:text-white font-semibold'
                                        : 'hover:bg-neutral-50 dark:hover:bg-neutral-800/50'
                                }`
                            }
                        >
                            <BarChart3 className="w-4 h-4" />
                            Dashboard
                        </NavLink>
                    </nav>

                    {/* Right cluster */}
                    <div className="flex items-center gap-3">
                        <div className="flex items-center gap-2 text-xs text-neutral-500 border border-neutral-200 dark:border-neutral-800 px-3 py-1.5 rounded-full bg-neutral-50 dark:bg-neutral-800/40">
                            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                            <span className="font-mono text-[11px]">API: Healthy (200)</span>
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Content Area */}
            <main className="flex-grow p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto w-full">
                <Outlet />
            </main>

            {/* Footer */}
            <footer className="border-t border-neutral-200 dark:border-neutral-800 bg-white/50 dark:bg-neutral-900/50 py-4 text-center text-xs text-neutral-500">
                <p>
                    DengueAlertas MLOps Platform • Potenciado con Lightning AI, MLflow & Cloudflare Edge
                </p>
            </footer>
        </div>
    );
}