"""Static descriptive figures of saved counts, EWT, annual labels and run lengths."""

import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pilot_selection.config import PILOTS

from .config import COUNT
from .validation import require

STYLE = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
         'axes.spines.right': False, 'axes.edgecolor': '#AEB8BF', 'axes.linewidth': .6,
         'figure.facecolor': 'white', 'axes.facecolor': 'white', 'text.color': '#233442',
         'axes.labelcolor': '#233442', 'path.simplify': False,
         'svg.fonttype': 'path', 'svg.hashsalt': 'denguealertas-epidemiology-characterization'}


def weekly_figure(frame: pd.DataFrame, eligible_period_only: bool = False):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharey=False)
        fig.subplots_adjust(left=.065, right=.98, top=.85, bottom=.10, hspace=.40, wspace=.18)
        checks = []
        for ax, pilot in zip(axes.flat, PILOTS):
            group = frame.loc[frame.municipality_code.eq(pilot.code)].sort_values('week_start')
            require(len(group) > 0, 'Missing pilot in weekly figure.')
            if eligible_period_only:
                # Keep ineligible positions inside this period as gaps in both series.
                group = group.loc[group.week_start.dt.year.ge(2012)]
            visible = group.label_eligible if eligible_period_only else ~group.partial_week
            counts = group[COUNT].where(visible).to_numpy(dtype=float)
            ewt = group.ewt.to_numpy(dtype=float)
            cases_line, = ax.plot(group.week_start.to_numpy(), counts, color='#18657A', linewidth=.8, label='Conteo observado')
            ewt_line, = ax.plot(group.week_start.to_numpy(), ewt, color='#BD562B', linewidth=1.05, label='EWT histórico')
            cases_line.set_gid('counts_' + pilot.code)
            ewt_line.set_gid('ewt_' + pilot.code)
            require(np.array_equal(cases_line.get_ydata(), counts, equal_nan=True)
                    and np.array_equal(ewt_line.get_ydata(), ewt, equal_nan=True), 'Figure changes source values.')
            ax.set_title(f'{pilot.name} · {pilot.code}', loc='left', fontsize=12, fontweight='bold')
            ax.set_ylabel('Registros por semana')
            ax.set_ylim(0, max(1, float(np.nanmax(counts)), float(np.nanmax(ewt))) * 1.08)
            ax.set_xlim(pd.Timestamp('2012-01-01' if eligible_period_only else '2007-01-01'), pd.Timestamp('2025-12-31'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2 if eligible_period_only else 3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.grid(axis='y', color='#E2E8ED', linewidth=.5)
            ax.set_axisbelow(True)
            checks.append({'municipality_code': pilot.code, 'plotted_count_positions': int(np.isfinite(counts).sum()),
                           'plotted_ewt_positions': int(np.isfinite(ewt).sum()),
                           'count_total': int(np.nansum(counts)), 'ewt_sum': float(np.nansum(ewt))})
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.98, .958), ncol=2, frameon=False)
        fig.suptitle('Conteos observados y referencia EWT', x=.065, y=.97, ha='left', fontsize=19, fontweight='bold')
        period = '2012–2025 · Período con etiquetas' if eligible_period_only else '2007–2025 · Histórico completo'
        fig.text(.065, .917, period + ' · Escalas verticales independientes · Sin suavizado', color='#526573')
        fig.text(.5, .052, 'Fecha del lunes de inicio de semana', ha='center')
        fig.text(.065, .015, 'EWT central retrospectivo; no declara brotes. Semana parcial final omitida de las líneas. EWT nulo antes de 2012.',
                 fontsize=9, color='#526573')
        return fig, checks


def annual_figure(tables: dict):
    annual = tables['labels_by_municipality_year']
    years = list(range(2012, 2026))
    values = np.array([annual.loc[annual.municipality_code.eq(p.code)].set_index('year').positive_pct.reindex(years)
                       for p in PILOTS], dtype=float)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(13, 4.6))
        fig.subplots_adjust(left=.14, right=.90, top=.78, bottom=.14)
        cells = ax.imshow(values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)
        ax.set_xticks(range(len(years)), years)
        ax.set_yticks(range(len(PILOTS)), [p.name for p in PILOTS])
        for i in range(len(PILOTS)):
            for j in range(len(years)):
                ax.text(j, i, f'{values[i,j]:.0f}', ha='center', va='center', fontsize=9,
                        color='white' if values[i,j] >= 65 else '#263746')
        bar = fig.colorbar(cells, ax=ax, fraction=.025, pad=.025)
        bar.set_label('% de semanas elegibles positivas')
        fig.suptitle('Frecuencia anual de etiquetas positivas', x=.14, y=.96, ha='left', fontsize=17, fontweight='bold')
        fig.text(.14, .86, 'Año del lunes inicial · Porcentaje sobre semanas elegibles de cada municipio y año', color='#526573')
        return fig


def duration_figure(tables: dict):
    runs = tables['positive_runs']
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 5.8))
        fig.subplots_adjust(left=.08, right=.98, top=.78, bottom=.17)
        data = [runs.loc[runs.municipality_code.eq(p.code), 'duration_weeks'].to_numpy() for p in PILOTS]
        ax.boxplot(data, tick_labels=[p.name for p in PILOTS], showmeans=True,
                   medianprops={'color': '#BD562B', 'linewidth': 1.5},
                   meanprops={'marker': 'D', 'markerfacecolor': '#18657A', 'markeredgecolor': '#18657A', 'markersize': 4})
        ax.set_ylabel('Duración de la racha (semanas)')
        ax.grid(axis='y', color='#E2E8ED', linewidth=.5)
        ax.set_ylim(bottom=0)
        fig.suptitle('Duración de rachas de etiquetas positivas', x=.08, y=.96, ha='left', fontsize=17, fontweight='bold')
        fig.text(.08, .86, 'Secuencias semanales observadas · Mediana en naranja y media con rombo · Sin nueva definición de episodio', fontsize=10, color='#526573')
        fig.text(.08, .035, 'Las negativas, semanas no elegibles y discontinuidades cortan las rachas; el cambio de año no las corta.', fontsize=9, color='#526573')
        return fig


def figure_bytes(frame: pd.DataFrame, tables: dict) -> tuple[dict[str, bytes], dict]:
    files, checks = {}, {}
    for name, zoom in [('weekly_cases_ewt_full', False), ('weekly_cases_ewt_eligible_period', True)]:
        fig, audit = weekly_figure(frame, zoom)
        checks[name] = audit
        try:
            with plt.rc_context(STYLE):
                for extension in ['png', 'svg']:
                    stream = io.BytesIO()
                    fig.savefig(stream, format=extension, dpi=200,
                                metadata={'Date': None} if extension == 'svg' else {})
                    files[name + '.' + extension] = stream.getvalue()
        finally:
            plt.close(fig)
    for name, builder in [('annual_positive_percentage', annual_figure), ('positive_run_durations', duration_figure)]:
        fig = builder(tables)
        try:
            stream = io.BytesIO()
            fig.savefig(stream, format='png', dpi=200)
            files[name + '.png'] = stream.getvalue()
        finally:
            plt.close(fig)
    return files, checks
