"""The approved 3 x 2 weekly comparison; no transformed or smoothed counts."""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from .analysis import require
from .config import COUNT, KEYS, PILOTS


def weekly_figure(complete: pd.DataFrame):
    """Return the figure and a reconciliation for the data actually plotted."""
    style = {
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12,
        "axes.labelsize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.edgecolor": "#AEB8BF", "axes.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": "#1C2E3A", "axes.labelcolor": "#3E505D",
        "xtick.color": "#3E505D", "ytick.color": "#3E505D",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "path.simplify": False, "svg.fonttype": "path",
        "svg.hashsalt": "denguealertas-six-municipalities-horizontal-weekly",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 2, figsize=(12.8, 8.1), sharex=False, sharey=False)
        try:
            fig.subplots_adjust(left=.073, right=.985, bottom=.115, top=.86, hspace=.55, wspace=.20)
            checks = []
            for index, pilot in enumerate(PILOTS):
                group = complete.loc[complete.municipality_code.eq(pilot.code)].sort_values(KEYS)
                require(len(group) > 0 and not group.partial_week.any(), "Invalid pilot figure input.")
                require(group.week_start.diff().dropna().eq(pd.Timedelta(days=7)).all(),
                        "Pilot figure contains missing weeks.")
                ax = axes.flat[index]
                line, = ax.plot(group.week_start.to_numpy(), group[COUNT].to_numpy(),
                                color=pilot.color, linewidth=.85, solid_capstyle="round")
                line.set_gid("serie_" + pilot.code)
                require(np.array_equal(line.get_ydata(), group[COUNT].to_numpy()), "Plot changes counts.")
                require(np.array_equal(line.get_xdata(), group.week_start.to_numpy()), "Plot changes dates.")
                total = int(group[COUNT].sum())
                require(int(line.get_ydata().sum()) == total, "Plot total disagrees with panel.")
                ax.set_title(f"({chr(97 + index)}) {pilot.name} · {pilot.code}",
                             loc="left", fontweight="bold", pad=8)
                ax.set_xlim(pd.Timestamp("2007-01-01"), pd.Timestamp("2025-12-31"))
                ax.set_xticks([pd.Timestamp(year=year, month=1, day=1)
                               for year in (2007, 2010, 2013, 2016, 2019, 2022, 2025)])
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                ax.set_ylim(0, max(1, float(group[COUNT].max()) * 1.10))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
                ax.set_ylabel("Registros semanales", labelpad=5)
                ax.grid(axis="y", color="#E3E8EB", linewidth=.55)
                ax.set_axisbelow(True)
                ax.tick_params(axis="both", length=3, width=.6, pad=3)
                checks.append({"municipality_code": pilot.code, "pilot_name": pilot.name,
                               "weeks": len(group), "source_total": total,
                               "plotted_total": int(line.get_ydata().sum())})
            fig.suptitle("Evolución semanal de registros confirmados elegibles", x=.073, y=.971,
                         ha="left", fontsize=17, fontweight="bold")
            fig.text(.073, .928, "2007–2025 · Semanas completas · Escalas verticales independientes y lineales",
                     ha="left", fontsize=10.8, color="#50616E")
            fig.text(.53, .057, "Fecha de inicio de semana", ha="center", fontsize=10.5, color="#3E505D")
            fig.text(.073, .018,
                     "Fuente: master_weekly_screening.parquet · Semana parcial excluida · Registros retrospectivos bajo la regla provisional de screening.",
                     ha="left", fontsize=8.1, color="#50616E")
            # Store export-only settings; rc_context does not remain active after return.
            fig._pilot_export_style = style
            return fig, checks
        except Exception:
            plt.close(fig)
            raise


def figure_bytes(complete: pd.DataFrame) -> tuple[dict[str, bytes], list[dict]]:
    fig, checks = weekly_figure(complete)
    try:
        files = {}
        metadata = {"Title": "Evolución semanal en seis municipios piloto, 2007–2025",
                    "Description": "Registros retrospectivos elegibles; semanas completas; sin suavizado."}
        with plt.rc_context(fig._pilot_export_style):
            for extension in ("png", "svg"):
                buffer = io.BytesIO()
                extra = {"Date": None} if extension == "svg" else {}
                fig.savefig(buffer, format=extension, dpi=300, facecolor="white", metadata={**metadata, **extra})
                files["weekly_comparison_6." + extension] = buffer.getvalue()
        return files, checks
    finally:
        plt.close(fig)
