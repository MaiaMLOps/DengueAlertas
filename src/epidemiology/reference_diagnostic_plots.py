"""Descriptive plots; quartile limits never change epidemiological labels."""

import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from src.pilot_selection.config import PILOTS
from .characterization_plots import STYLE
from .config import COUNT


def zero_figure(frame):
    groups = [(n, frame.loc[frame.historical_zero_count.eq(n), 'ewt'].to_numpy()) for n in range(6)]
    nonempty = [(n, values) for n, values in groups if len(values)]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 6))
        fig.subplots_adjust(left=.09, right=.97, bottom=.16, top=.77)
        ax.boxplot([values for _, values in nonempty], tick_labels=[f'{n}\n(n={len(values)})' for n, values in nonempty],
                   medianprops={'color': '#BD562B', 'linewidth': 1.6}, showfliers=True)
        if frame.ewt.gt(0).all():
            ax.set_yscale('log')
            ax.set_ylabel('EWT guardado (escala logarítmica)')
        else:
            ax.set_ylabel('EWT guardado (escala lineal, incluye ceros)')
        ax.set_xlabel('Número de ceros entre las cinco referencias históricas')
        ax.grid(axis='y', alpha=.25)
        fig.suptitle('EWT según ceros históricos', x=.09, y=.96, ha='left', fontsize=18, fontweight='bold')
        fig.text(.09, .85, 'Solo semanas elegibles · Distribuciones observadas, sin modificar el EWT', color='#526573')
        absent = ', '.join(str(n) for n, values in groups if not len(values))
        fig.text(.09, .035, 'Grupos sin observaciones: ' + (absent if absent else 'ninguno') + '. Los tamaños de grupo son muy diferentes.', fontsize=10)
        return fig


def quartile_figure(tables):
    grid = tables['quartile_grid_global']
    rates = grid.pivot(index='ewt_quartile', columns='observed_quartile', values='positive_pct_in_cell').to_numpy(dtype=float)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        fig.subplots_adjust(left=.15, right=.88, bottom=.17, top=.76)
        image = ax.imshow(rates, vmin=0, vmax=100, cmap='YlOrRd', aspect='auto')
        for row in grid.itertuples():
            ax.text(row.observed_quartile - 1, row.ewt_quartile - 1,
                    f'{row.positive_weeks}/{row.eligible_weeks}\n{row.positive_pct_in_cell:.1f} %' if row.eligible_weeks else 'Sin semanas',
                    ha='center', va='center', color='white' if row.positive_pct_in_cell >= 65 else '#263746', fontsize=11)
        ax.set_xticks(range(4), ['Q1 (menor)', 'Q2', 'Q3', 'Q4 (mayor)'])
        ax.set_yticks(range(4), ['Q1 (menor)', 'Q2', 'Q3', 'Q4 (mayor)'])
        ax.set_xlabel('Cuartil del conteo observado, dentro de cada municipio')
        ax.set_ylabel('Cuartil del EWT, dentro de cada municipio')
        fig.colorbar(image, ax=ax, fraction=.04, pad=.04).set_label('% de etiquetas positivas en la celda')
        fig.suptitle('Combinaciones de EWT y conteos observados', x=.15, y=.96, ha='left', fontsize=16, fontweight='bold')
        fig.text(.15, .86, 'Positivas / semanas elegibles · Cuartiles municipales calculados con 2012–2025', color='#526573', fontsize=10)
        fig.text(.15, .035, 'Rangos descriptivos; no son nuevos umbrales epidemiológicos ni predictores. Las igualdades quedan en el intervalo inferior.', fontsize=8)
        return fig


def long_runs_figure(frame, table):
    names = {pilot.code: pilot.name for pilot in PILOTS}
    ncols = 4
    nrows = max(1, int(np.ceil(len(table) / ncols)))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3 * nrows + 2), squeeze=False)
        fig.subplots_adjust(left=.05, right=.985, top=.88, bottom=.055, hspace=.65, wspace=.30)
        handles = None
        for ax, run in zip(axes.flat, table.itertuples()):
            selected = frame.loc[frame.municipality_code.eq(run.municipality_code)
                                 & frame.week_start.between(run.start_week, run.end_week)].sort_values('week_start')
            cases, = ax.plot(selected.week_start, selected[COUNT], color='#18657A', linewidth=.8, label='Conteo observado')
            ewt, = ax.plot(selected.week_start, selected.ewt, color='#BD562B', linewidth=1., label='EWT guardado')
            zeros = selected.loc[selected.historical_zero_count.gt(0)]
            dots = ax.scatter(zeros.week_start, zeros.ewt, s=10, color='#7B2465', zorder=3, label='EWT con algún cero histórico')
            handles = [cases, ewt, dots]
            ax.set_title(f'{run.run_id} · {names.get(run.municipality_code, run.municipality)} · {run.duration_weeks} semanas', loc='left', fontsize=10, fontweight='bold')
            locator = mdates.AutoDateLocator(minticks=3, maxticks=4)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.tick_params(labelsize=8)
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', alpha=.25)
        for ax in list(axes.flat)[len(table):]:
            ax.set_visible(False)
        fig.suptitle('Las 20 rachas positivas de al menos 52 semanas', x=.05, y=.975, ha='left', fontsize=22, fontweight='bold')
        fig.text(.05, .94, 'Fechas reales · Escalas verticales independientes · Sin suavizado ni nueva definición de episodio', color='#526573', fontsize=12)
        if handles:
            fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.045, .925), ncol=3, frameon=False)
        fig.text(.05, .015, 'Los puntos destacan referencias con ceros; no indican causalidad. Las rachas se definen exclusivamente con las etiquetas semanales guardadas.', fontsize=11)
        return fig


def figures(frame, tables):
    result = {}
    for name, builder in [('ewt_by_historical_zeros', lambda: zero_figure(frame)),
                          ('municipal_quartile_combinations', lambda: quartile_figure(tables)),
                          ('long_runs_cases_ewt', lambda: long_runs_figure(frame, tables['long_runs_detail']))]:
        fig = builder()
        try:
            stream = io.BytesIO()
            fig.savefig(stream, format='png', dpi=170)
            result[name + '.png'] = stream.getvalue()
        finally:
            plt.close(fig)
    return result
