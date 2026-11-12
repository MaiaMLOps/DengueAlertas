"""Figures of the derivable reports (unit U7.5), drawn only from the tables of reports.py and the saved predictions.
Static PNG and SVG for the article (matplotlib, Agg backend, light surface). Development figures carry the label
«usado para elegir»; the test AMOC curve is descriptive and does not change tau (contrato §7.5).

Colour by job (reference palette of the dataviz method, first three categorical slots, validated all-pairs): the model
(recipe) is slot 1 blue, the channel slot 2 orange, the ablation slot 3 aqua; B2 is the muted grey reference. The BSS
map is diverging blue (model better) <-> red (B2 better) around a grey zero. Identity is never colour alone: every
figure has a legend or direct labels.

Figures (FIGURES): confiabilidad_<periodo>, amoc_<periodo>, mapa_bss_<periodo>, serie_<code> (2019–2025, one per
municipality) and resumen_municipios.
"""

from datetime import date, timedelta
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.epidemiology.validation import require  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, NEUTRAL, RED = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f0efec", "#e34948"
NAMES = {"08001": "Barranquilla", "08758": "Soledad", "41001": "Neiva", "50001": "Villavicencio", "54001": "Cúcuta",
         "76001": "Cali"}
PERIODS = ("desarrollo", "prueba")
PERIOD_TITLE = {"desarrollo": "Desarrollo 2019–2022 (usado para elegir)", "prueba": "Prueba 2023–2025"}
FIGURES = tuple([f"confiabilidad_{p}" for p in PERIODS] + [f"amoc_{p}" for p in PERIODS]
                + [f"mapa_bss_{p}" for p in PERIODS] + [f"serie_{code}" for code in NAMES] + ["resumen_municipios"])
HAT = r"$\hat{p}$"
SERIES_NOTE = (HAT + " y las alertas se dibujan en la semana de emisión t y se refieren a la semana objetivo t + 8; "
               "casos, U1e e inicios, en su propia semana.")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "svg.hashsalt": "u75-reportes", "lines.linewidth": 2.0,
})


def _save(fig, folder: Path, name: str) -> list[Path]:
    paths = []
    for suffix, metadata in (("png", {"Software": None}), ("svg", {"Date": None, "Creator": None})):
        path = folder / f"{name}.{suffix}"
        require(not path.exists(), f"{path} exists and is never overwritten.")
        fig.savefig(path, dpi=200 if suffix == "png" else None, bbox_inches="tight", metadata=metadata)
        paths.append(path)
    plt.close(fig)
    return paths


def _period(frame: pd.DataFrame, periodo: str) -> pd.DataFrame:
    return frame[frame["periodo"] == periodo]


# --------------------------------------------------------------------------- reliability
def reliability_figure(bins: pd.DataFrame, periodo: str):
    own = _period(bins, periodo)
    own = own[own["nivel"] == "global"]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--", label="Calibración perfecta")
    for system, color, label in (("modelo", BLUE, "Receta (sin clima)"), ("ablacion", AQUA, "Ablación (completo)")):
        part = own[own["sistema"] == system]
        ax.plot(part["p_media"], part["frecuencia_observada"], color=color, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1, label=label)
    n = own[own["sistema"] == "modelo"]["n"]
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=HAT + " media del intervalo", ylabel="Frecuencia observada de y = 1",
           title=f"Diagrama de confiabilidad — {PERIOD_TITLE[periodo]}")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.text(0.98, 0.02, f"{own['intervalo'].nunique()} intervalos de igual frecuencia de {HAT}\n"
                        f"({int(n.min())}–{int(n.max())} semanas por intervalo, receta)",
            transform=ax.transAxes, ha="right", va="bottom", color=INK_2, fontsize=8)
    return fig


# --------------------------------------------------------------------------- AMOC
_CURVE_STYLE = {"eleccion_tau_receta": ("-", "Receta logistica_05 en los 4 años (curva que eligió τ)"),
                "comun_fuera_de_anio": ("--", "Configuración común fuera de año (tablas de desarrollo)"),
                "receta": ("-", "Receta congelada")}


def amoc_figure(curves: pd.DataFrame, periodo: str, tau: float):
    own = _period(curves, periodo)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for configuration in [c for c in _CURVE_STYLE if c in set(own["configuracion"])]:
        part = own[(own["configuracion"] == configuration) & (own["sistema"] == "modelo")]
        style, label = _CURVE_STYLE[configuration]
        for ax, column in zip(axes, ("deteccion", "anticipacion_mediana")):
            ax.plot(part["falsas_por_municipio_anio"], part[column], color=BLUE, linestyle=style, marker="o",
                    markersize=3.5, label=label)
            chosen = part[np.isclose(part["tau"], tau)]
            ax.plot(chosen["falsas_por_municipio_anio"], chosen[column], linestyle="none", marker="o", markersize=9,
                    markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.2)
    channel = own[own["sistema"] == "canal"].iloc[0]
    for ax, column, ylabel in zip(axes, ("deteccion", "anticipacion_mediana"),
                                  ("Capacidad de detección (inicios avisados / evaluables)",
                                   "Anticipación mediana (semanas)")):
        ax.plot([channel["falsas_por_municipio_anio"]], [channel[column]], linestyle="none", marker="D",
                markersize=8, color=ORANGE, markeredgecolor=SURFACE, label="Canal (alerta de zona)")
        ax.set(xlabel="Falsas alarmas por municipio-año", ylabel=ylabel)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].plot([], [], linestyle="none", marker="o", markersize=9, markerfacecolor="none", markeredgecolor=INK,
                 label=f"τ = {tau:.2f} (receta)")
    axes[0].legend(loc="lower right", fontsize=8)
    note = ("Curva descriptiva: no cambia τ (contrato §7.5)." if periodo == "prueba"
            else "Rejilla τ = 0,05…0,95; inicios evaluables = 10.")
    fig.suptitle(f"Curva tipo AMOC — {PERIOD_TITLE[periodo]}", color=INK, fontsize=11)
    fig.text(0.5, -0.02, note, ha="center", color=INK_2, fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- BSS map
_DIVERGING = LinearSegmentedColormap.from_list("bss", [RED, NEUTRAL, BLUE])


def bss_map_figure(cells: pd.DataFrame, periodo: str):
    own = _period(cells, periodo)
    table = own.pivot(index="municipality_code", columns="anio", values="bss_modelo_vs_B2")
    fig, ax = plt.subplots(figsize=(1.3 * table.shape[1] + 2.6, 3.6))
    limit = 0.5
    image = ax.imshow(table.to_numpy(dtype=float), cmap=_DIVERGING,
                      norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(table.shape[1]), [str(year) for year in table.columns])
    ax.set_yticks(range(table.shape[0]), [NAMES[code] for code in table.index])
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iloc[i, j]
            ax.text(j, i, f"{value:+.2f}".replace(".", ","), ha="center", va="center", fontsize=8,
                    color=INK if abs(value) < 0.35 else "#ffffff")
    bar = fig.colorbar(image, ax=ax, extend="both", shrink=0.9)
    bar.set_label("BSS frente a B2 (> 0: el modelo mejora al canal)", color=INK_2)
    ax.set_title(f"BSS del modelo frente a B2 por casilla — {PERIOD_TITLE[periodo]}")
    fig.text(0.5, -0.02, f"Color recortado a ±{limit:.1f}; la cifra es el valor exacto.".replace(".", ",", 1), ha="center", color=INK_2,
             fontsize=8)
    return fig


# --------------------------------------------------------------------------- weekly series
def _spans(ax, alerts: pd.DataFrame, code: str, y0: float, y1: float, color: str, label: str) -> None:
    own = alerts[alerts["municipality_code"] == code]
    for i, alert in enumerate(own.itertuples()):
        ax.axvspan(alert.confirmation, alert.end + timedelta(days=6), ymin=y0, ymax=y1, color=color, alpha=0.55,
                   linewidth=0, label=label if i == 0 else None)


def series_figure(code: str, systems: dict, observed: pd.DataFrame, events: dict, tau: float):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    weeks = observed[(observed["municipality_code"] == code)]
    first = min(systems[p].rows["target_week"].min() for p in PERIODS)
    last = max(systems[p].rows["target_week"].max() for p in PERIODS)
    start = min(systems[p].rows.loc[systems[p].rows["municipality_code"] == code, "t"].min() for p in PERIODS)
    weeks = weeks[(weeks["week_start"] >= start) & (weeks["week_start"] <= last)].sort_values("week_start")
    top.plot(weeks["week_start"], weeks["confirmed_records_screening"], color=INK_2, linewidth=1.2,
             label="Casos semanales (C)")
    top.plot(weeks["week_start"], weeks["epico_upper"], color=RED, linewidth=1.4, label="Límite U1e del canal")
    top.set_ylabel("Casos confirmados")

    for periodo in PERIODS:
        rows, p = systems[periodo].rows, systems[periodo].p
        mask = (rows["municipality_code"] == code).to_numpy()
        bottom.plot(rows.loc[mask, "t"], p["modelo"][mask], color=BLUE, linewidth=1.3,
                    label=HAT + " del modelo" if periodo == PERIODS[0] else None)
        bottom.plot(rows.loc[mask, "t"], p["B2"][mask], color=MUTED, linewidth=1.0, linestyle=":",
                    label="B2 (canal como probabilidad)" if periodo == PERIODS[0] else None)
        first_period = periodo == PERIODS[0]
        _spans(bottom, events[periodo]["modelo"].alerts, code, 0.93, 1.0, BLUE,
               "Alerta del modelo (de t_c al fin)" if first_period else None)
        _spans(bottom, events[periodo]["canal"].alerts, code, 0.85, 0.92, ORANGE,
               "Alerta del canal (de t_c al fin)" if first_period else None)
        onsets = events[periodo]["modelo"].onsets
        onsets = onsets[(onsets["municipality_code"] == code) & (onsets["kind"] == "inicio") & onsets["evaluable"]]
        for i, week in enumerate(onsets["week"]):
            for ax in (top, bottom):
                ax.axvline(week, color=INK, linewidth=0.9, linestyle="--",
                           label="Inicio de brote evaluable" if (ax is top and first_period and i == 0) else None)
    bottom.axhline(tau, color=INK, linewidth=0.9, label=f"τ = {tau:.2f}")
    bottom.set(ylim=(0, 1.0), ylabel="Probabilidad")
    test_start = systems["prueba"].rows.loc[systems["prueba"].rows["municipality_code"] == code, "t"].min()
    for ax in (top, bottom):
        ax.axvline(test_start, color=MUTED, linewidth=1.2)
    top.text(test_start, 1.01, "  prueba →", transform=top.get_xaxis_transform(), color=INK_2, fontsize=8)
    top.text(test_start, 1.01, "← desarrollo (usado para elegir)  ", transform=top.get_xaxis_transform(),
             color=INK_2, fontsize=8, ha="right")
    top.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.01, 1.0))
    bottom.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.01, 1.0))
    top.set_title(f"{NAMES[code]} ({code}) — serie semanal {first.year}–{last.year}", pad=16)
    fig.text(0.5, -0.01, SERIES_NOTE, ha="center", color=INK_2, fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- municipality summary
def summary_figure(cells: pd.DataFrame, events: pd.DataFrame):
    codes = list(NAMES)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    y = np.arange(len(codes))
    offsets = {"desarrollo": -0.14, "prueba": 0.14}
    for periodo in PERIODS:
        own = _period(cells, periodo)
        bss = []
        for code in codes:
            part = own[own["municipality_code"] == code]
            pooled = lambda column: float((part[column] * part["filas"]).sum() / part["filas"].sum())  # noqa: E731
            bss.append(1 - pooled("brier_modelo") / pooled("brier_B2"))
        filled = periodo == "prueba"
        axes[0].plot(bss, y + offsets[periodo], linestyle="none", marker="o", markersize=7, color=BLUE,
                     markerfacecolor=BLUE if filled else SURFACE, markeredgewidth=1.5,
                     label=f"Modelo, {periodo}")
        own_events = _period(events, periodo)
        own_events = own_events[own_events["nivel"] == "municipio"].set_index(["sistema", "grupo"])
        for system, color, shift in (("modelo", BLUE, -0.06), ("canal", ORANGE, 0.06)):
            warned = [own_events.loc[(system, code), "inicios_avisados"] for code in codes]
            total = [own_events.loc[(system, code), "inicios_evaluables"] for code in codes]
            rate = [own_events.loc[(system, code), "falsas_por_municipio_anio"] for code in codes]
            kwargs = dict(linestyle="none", marker="o" if system == "modelo" else "D", markersize=6.5, color=color,
                          markerfacecolor=color if filled else SURFACE, markeredgewidth=1.5,
                          label=f"{'Modelo' if system == 'modelo' else 'Canal'}, {periodo}")
            axes[1].plot(warned, y + offsets[periodo] + shift, **kwargs)
            axes[2].plot(rate, y + offsets[periodo] + shift, **kwargs)
    axes[0].axvline(0, color=AXIS, linewidth=1)
    evaluable = {periodo: _period(events, periodo).query("nivel == 'municipio' and sistema == 'modelo'")
                 .set_index("grupo")["inicios_evaluables"] for periodo in PERIODS}
    labels = [f"{NAMES[code]}\nevaluables: {int(evaluable['desarrollo'][code])} des. · "
              f"{int(evaluable['prueba'][code])} prueba" for code in codes]
    axes[0].set(xlabel="BSS frente a B2 (filas del municipio juntas)", yticks=y, yticklabels=labels,
                title="Estimación")
    axes[1].set(xlabel="Inicios avisados", title="Detección", xlim=(-0.3, 3.3), xticks=[0, 1, 2, 3])
    axes[2].set(xlabel="Falsas alarmas por municipio-año", title="Costo")
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=7)
    axes[2].legend(loc="lower right", fontsize=7)
    fig.suptitle("Resumen por municipio: hueco = desarrollo (usado para elegir), lleno = prueba", color=INK,
                 fontsize=11)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- all
def write_figures(tables: dict, systems: dict, observed: pd.DataFrame, events: dict, *, folder: Path,
                  tau: float) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for periodo in PERIODS:
        paths += _save(reliability_figure(tables["confiabilidad"], periodo), folder, f"confiabilidad_{periodo}")
        paths += _save(amoc_figure(tables["curva_amoc"], periodo, tau), folder, f"amoc_{periodo}")
        paths += _save(bss_map_figure(tables["casillas"], periodo), folder, f"mapa_bss_{periodo}")
    for code in NAMES:
        paths += _save(series_figure(code, systems, observed, events, tau), folder, f"serie_{code}")
    paths += _save(summary_figure(tables["casillas"], tables["eventos_razon"]), folder, "resumen_municipios")
    return paths
