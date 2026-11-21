#!/usr/bin/env python3
"""Analisis grafico punta a punta de una corrida DengueAlertas EWT.

Consume directamente la carpeta de salida creada por el pipeline ewt-final-v1 y
produce:
  * imagenes PNG auditables;
  * un informe PDF;
  * un informe Word (DOCX);
  * tablas y JSON de hallazgos derivados.

El script es tolerante a artefactos opcionales ausentes: registra la omision en
el manifiesto, pero nunca inventa resultados.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import textwrap
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image as PILImage
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve, auc

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor
except ImportError:  # pragma: no cover
    Document = None

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        BaseDocTemplate, Frame, Image as RLImage, KeepTogether, PageBreak,
        PageTemplate, Paragraph, Spacer, Table, TableStyle,
    )
except ImportError:  # pragma: no cover
    BaseDocTemplate = None

VERSION = "1.0.0"
COUNT_COL = "confirmed_records_screening"
TASK_LABELS = {
    "zone_future": "Zona epidemica a 8 semanas",
    "pre_onset": "Inicio de episodio en 8 semanas",
}
MUNICIPALITY_FALLBACK = {
    "54001": "Cucuta", "08001": "Barranquilla", "41001": "Neiva",
    "50001": "Villavicencio", "76001": "Cali", "08758": "Soledad",
}
FAMILY_LABELS = {
    "epi": "Epidemiologia", "ch": "Canal EWT", "cal": "Calendario",
    "clim": "Clima", "demo": "Demografia", "other": "Otra",
}
FAMILY_COLORS = {
    "epi": "#0078D4", "ch": "#D83B01", "cal": "#8764B8",
    "clim": "#107C10", "demo": "#FFB900", "other": "#737373",
}
COLORS = {
    "blue": "#0078D4", "navy": "#004578", "orange": "#D83B01",
    "green": "#107C10", "gold": "#FFB900", "purple": "#8764B8",
    "red": "#C50F1F", "gray": "#737373", "light": "#F3F2F1",
    "black": "#201F1E", "cyan": "#00B7C3",
}

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.labelsize": 10, "legend.fontsize": 8,
    "font.family": "DejaVu Sans", "savefig.bbox": "tight",
})


@dataclass
class ArtifactStatus:
    name: str
    path: str
    required: bool
    found: bool
    rows: int | None = None
    note: str = ""


@dataclass
class FigureItem:
    key: str
    path: Path
    title: str
    section: str
    caption: str
    task: str | None = None
    municipality: str | None = None


@dataclass
class RunData:
    root: Path
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    baselines: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    episodes: pd.DataFrame = field(default_factory=pd.DataFrame)
    family_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)
    panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)
    importances: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)
    metrics_json: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    artifacts: list[ArtifactStatus] = field(default_factory=list)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: Any, digits: int = 3, pct: bool = False) -> str:
    value = safe_float(value)
    if math.isnan(value):
        return "NA"
    if pct:
        return f"{100 * value:.{digits}f}%"
    return f"{value:.{digits}f}"


def normalize_code(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5)


def normalize_codes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "municipality_code" not in frame:
        return frame
    frame = frame.copy()
    frame["municipality_code"] = frame["municipality_code"].map(normalize_code)
    return frame


def read_csv_safe(path: Path, **kwargs: Any) -> pd.DataFrame:
    try:
        return normalize_codes(pd.read_csv(path, low_memory=False, **kwargs))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def resolve_run_dir(input_path: Path, tmp_root: Path) -> Path:
    if input_path.is_dir():
        if (input_path / "results.csv").exists():
            return input_path
        matches = list(input_path.rglob("results.csv"))
        if len(matches) == 1:
            return matches[0].parent
        raise FileNotFoundError(
            f"No se encontro results.csv en {input_path}; use la carpeta final_run."
        )
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        target = tmp_root / "unzipped_run"
        with zipfile.ZipFile(input_path) as archive:
            archive.extractall(target)
        matches = list(target.rglob("results.csv"))
        if not matches:
            raise FileNotFoundError("El ZIP no contiene una carpeta de resultados con results.csv")
        preferred = [p for p in matches if p.parent.name == "final_run"]
        return (preferred or matches)[0].parent
    raise FileNotFoundError(f"Ruta de corrida inexistente o no soportada: {input_path}")


def load_run(root: Path) -> RunData:
    data = RunData(root=root)
    specs = [
        ("results", "results.csv", True),
        ("baselines", "baselines_test.csv", False),
        ("candidates", "candidates_validation.csv", False),
        ("episodes", "episodes.csv", False),
        ("family_importance", "feature_family_importance.csv", False),
        ("skipped", "skipped_models.csv", False),
        ("panel", "prepared_supervised_panel.csv.gz", False),
    ]
    for attr, rel, required in specs:
        path = root / rel
        if path.exists():
            frame = read_csv_safe(path)
            setattr(data, attr, frame)
            data.artifacts.append(ArtifactStatus(attr, rel, required, True, len(frame)))
        else:
            data.artifacts.append(ArtifactStatus(attr, rel, required, False, note="No encontrado"))
            if required:
                raise FileNotFoundError(f"Falta el artefacto obligatorio: {path}")

    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        data.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data.artifacts.append(ArtifactStatus("manifest", "run_manifest.json", False, True))
    else:
        data.artifacts.append(ArtifactStatus("manifest", "run_manifest.json", False, False))

    config_candidates = [root / "pipeline.yaml", root.parent.parent / "configs" / "pipeline.yaml"]
    config_path = next((p for p in config_candidates if p.exists()), None)
    if config_path and yaml:
        data.config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        data.artifacts.append(ArtifactStatus("config", str(config_path), False, True))

    model_root = root / "models"
    if model_root.exists():
        for model_dir in sorted(p for p in model_root.glob("*/*") if p.is_dir()):
            task, code = model_dir.parent.name, normalize_code(model_dir.name)
            key = (task, code)
            pred = model_dir / "predictions_test.csv"
            imp = model_dir / "feature_importance.csv"
            met = model_dir / "metrics.json"
            if pred.exists():
                frame = read_csv_safe(pred)
                if "week_start" in frame:
                    frame["week_start"] = pd.to_datetime(frame["week_start"], errors="coerce")
                data.predictions[key] = frame
            if imp.exists():
                data.importances[key] = read_csv_safe(imp)
            if met.exists():
                data.metrics_json[key] = json.loads(met.read_text(encoding="utf-8"))
    data.artifacts.append(ArtifactStatus(
        "model_directories", "models/<task>/<DIVIPOLA>", False, model_root.exists(),
        rows=len(data.predictions), note=f"{len(data.predictions)} predicciones; {len(data.importances)} importancias",
    ))

    for frame_name in ("results", "baselines", "candidates", "skipped", "panel", "episodes"):
        frame = getattr(data, frame_name)
        if not frame.empty:
            setattr(data, frame_name, normalize_codes(frame))
    if not data.panel.empty and "week_start" in data.panel:
        data.panel["week_start"] = pd.to_datetime(data.panel["week_start"], errors="coerce")
    for date_col in ("start_week", "end_week"):
        if not data.episodes.empty and date_col in data.episodes:
            data.episodes[date_col] = pd.to_datetime(data.episodes[date_col], errors="coerce")
    return data


def municipality_name(data: RunData, code: str) -> str:
    code = normalize_code(code)
    for frame in (data.results, data.panel, data.baselines, data.candidates):
        if not frame.empty and {"municipality_code", "municipality"}.issubset(frame.columns):
            values = frame.loc[frame["municipality_code"] == code, "municipality"].dropna()
            if not values.empty:
                return str(values.iloc[0])
    return MUNICIPALITY_FALLBACK.get(code, code)


def feature_family(name: str) -> str:
    for prefix in ("epi", "ch", "cal", "clim", "demo"):
        if str(name).startswith(prefix + "_"):
            return prefix
    return "other"


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def placeholder_figure(title: str, message: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")
    ax.text(0.5, 0.60, title, ha="center", va="center", fontsize=18, weight="bold")
    ax.text(0.5, 0.40, textwrap.fill(message, 100), ha="center", va="center", fontsize=12, color=COLORS["gray"])
    return fig


def add_kpi(fig: plt.Figure, x: float, y: float, title: str, value: str, note: str = "") -> None:
    fig.text(x, y, value, ha="center", va="center", fontsize=24, weight="bold", color=COLORS["blue"])
    fig.text(x, y - 0.06, title, ha="center", va="center", fontsize=10, color=COLORS["black"])
    if note:
        fig.text(x, y - 0.095, note, ha="center", va="center", fontsize=7.5, color=COLORS["gray"])


def executive_dashboard(data: RunData, out: Path, dpi: int) -> FigureItem:
    results = data.results.copy()
    primary = results[results["task"] == "zone_future"] if "task" in results else results
    n_models = len(results)
    n_muni = primary["municipality_code"].nunique() if "municipality_code" in primary else 0
    ap = primary.get("test_average_precision", pd.Series(dtype=float)).mean()
    brier = primary.get("test_brier", pd.Series(dtype=float)).mean()
    prev = primary.get("test_prevalence", pd.Series(dtype=float)).mean()
    spec = primary.get("test_specificity", pd.Series(dtype=float)).mean()

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("DengueAlertas EWT - tablero ejecutivo de la corrida", fontsize=22, weight="bold", y=0.96)
    fig.text(0.5, 0.91, "Lectura conjunta de discriminacion, calibracion, carga de alertas y estabilidad", ha="center", color=COLORS["gray"])
    add_kpi(fig, 0.13, 0.80, "Modelos evaluados", str(n_models), "incluye tarea secundaria")
    add_kpi(fig, 0.37, 0.80, "Municipios (principal)", str(n_muni))
    add_kpi(fig, 0.61, 0.80, "AUC-PR macro", fmt(ap))
    add_kpi(fig, 0.85, 0.80, "Brier macro", fmt(brier), "menor es mejor")
    add_kpi(fig, 0.25, 0.64, "Prevalencia macro", fmt(prev, pct=True))
    add_kpi(fig, 0.50, 0.64, "Especificidad macro", fmt(spec, pct=True))

    baseline = data.baselines
    wins_ap = wins_brier = None
    if not baseline.empty and not primary.empty:
        p = baseline[(baseline["task"] == "zone_future") & (baseline["baseline"] == "persistence")]
        merged = primary.merge(p, on=["task", "municipality_code"], suffixes=("_model", "_base"))
        if not merged.empty:
            wins_ap = int((merged["test_average_precision"] > merged["average_precision"]).sum())
            wins_brier = int((merged["test_brier"] < merged["brier"]).sum())
            total = len(merged)
            add_kpi(fig, 0.75, 0.64, "Mejoras vs persistencia", f"AP {wins_ap}/{total} | Brier {wins_brier}/{total}")

    ax = fig.add_axes([0.08, 0.10, 0.84, 0.40])
    if not primary.empty:
        view = primary[["municipality", "test_average_precision", "test_brier", "test_specificity", "test_prevalence"]].copy()
        view = view.sort_values("test_average_precision")
        y = np.arange(len(view))
        ax.barh(y - 0.18, view["test_average_precision"], height=0.18, label="AUC-PR", color=COLORS["blue"])
        ax.barh(y, view["test_prevalence"], height=0.18, label="Prevalencia", color=COLORS["gold"])
        ax.barh(y + 0.18, view["test_specificity"], height=0.18, label="Especificidad", color=COLORS["green"])
        ax.set_yticks(y, view["municipality"])
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Escala 0-1")
        ax.set_title("La AUC-PR debe interpretarse junto con prevalencia y especificidad")
        ax.legend(ncol=3, loc="lower right")
    else:
        ax.axis("off")
    path = out / "00_tablero_ejecutivo.png"
    save_figure(fig, path, dpi)
    return FigureItem("dashboard", path, "Tablero ejecutivo", "Resumen ejecutivo",
                      "Sintesis de la tarea principal; Brier menor es mejor.")


def plot_data_coverage(data: RunData, out: Path, dpi: int) -> FigureItem | None:
    if data.panel.empty or not {"municipality_code", "week_start"}.issubset(data.panel.columns):
        return None
    panel = data.panel.copy()
    rows = []
    for code, group in panel.groupby("municipality_code"):
        rows.append({
            "code": code, "municipality": municipality_name(data, code), "start": group["week_start"].min(),
            "end": group["week_start"].max(), "weeks": len(group),
            "labeled": int(group.get("ch_ewt", pd.Series(index=group.index, dtype=float)).notna().sum()),
            "missing_climate": int(group[[c for c in ("temperature_c", "precipitation_mm", "relative_humidity_pct") if c in group]].isna().any(axis=1).sum()) if any(c in group for c in ("temperature_c", "precipitation_mm", "relative_humidity_pct")) else 0,
        })
    cov = pd.DataFrame(rows).sort_values("municipality")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
    for i, row in cov.reset_index(drop=True).iterrows():
        axes[0].plot([row["start"], row["end"]], [i, i], lw=8, color=COLORS["blue"], solid_capstyle="round")
        axes[0].text(row["start"], i + 0.18, str(row["weeks"]), fontsize=8)
    axes[0].set_yticks(range(len(cov)), cov["municipality"])
    axes[0].xaxis.set_major_locator(mdates.YearLocator(3))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].set_title("Cobertura temporal y numero de semanas")
    axes[0].set_xlabel("Periodo disponible")
    width = 0.36
    x = np.arange(len(cov))
    axes[1].bar(x - width/2, cov["weeks"], width, label="Filas", color=COLORS["gray"])
    axes[1].bar(x + width/2, cov["labeled"], width, label="Canal EWT disponible", color=COLORS["green"])
    axes[1].set_xticks(x, cov["municipality"], rotation=35, ha="right")
    axes[1].set_title("Completitud para evaluacion")
    axes[1].legend()
    fig.suptitle("Perfil de datos de la corrida", fontsize=16, weight="bold")
    fig.tight_layout()
    path = out / "01_cobertura_datos.png"
    save_figure(fig, path, dpi)
    return FigureItem("coverage", path, "Cobertura y completitud", "Datos y canal EWT",
                      "Periodo observado y semanas con canal EWT calculable.")


def plot_series_channel(data: RunData, out: Path, dpi: int, code: str) -> FigureItem | None:
    required = {"municipality_code", "week_start", COUNT_COL, "ch_ewt"}
    if data.panel.empty or not required.issubset(data.panel.columns):
        return None
    group = data.panel[data.panel["municipality_code"] == code].sort_values("week_start").copy()
    if group.empty:
        return None
    name = municipality_name(data, code)
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax = axes[0]
    ax.plot(group["week_start"], group[COUNT_COL], color=COLORS["navy"], lw=1.1, label="Casos confirmados")
    ax.plot(group["week_start"], group["ch_ewt"], color=COLORS["orange"], lw=1.0, label="EWT")
    if "ch_upper" in group:
        ax.plot(group["week_start"], group["ch_upper"], color=COLORS["gold"], lw=0.9, label="Limite superior")
        ax.fill_between(group["week_start"], group["ch_ewt"], group["ch_upper"], color=COLORS["gold"], alpha=0.10)
    if "excess_flag_ewt" in group:
        flags = group["excess_flag_ewt"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        ax.scatter(group.loc[flags, "week_start"], group.loc[flags, COUNT_COL], s=8, color=COLORS["red"], label="Exceso EWT", zorder=4)
    if not data.episodes.empty:
        episodes = data.episodes[data.episodes["municipality_code"] == code]
        for _, ep in episodes.iterrows():
            ax.axvspan(ep["start_week"], ep["end_week"], color=COLORS["red"], alpha=0.06)
    test_start = pd.to_datetime(data.results.loc[data.results["municipality_code"] == code, "test_start"], errors="coerce").min() if "test_start" in data.results else pd.NaT
    if pd.notna(test_start):
        ax.axvline(test_start, color=COLORS["purple"], ls="--", lw=1.2, label="Inicio prueba")
    ax.set_ylabel("Casos semanales")
    ax.set_title(f"Serie epidemiologica y canal EWT - {name} ({code})")
    ax.legend(ncol=5, loc="upper left")

    ax2 = axes[1]
    ratio = group[COUNT_COL] / pd.to_numeric(group.get("ch_upper"), errors="coerce").replace(0, np.nan)
    ax2.plot(group["week_start"], ratio, color=COLORS["cyan"], lw=1.0, label="Casos / limite superior")
    ax2.axhline(1, color=COLORS["red"], ls="--", lw=1)
    if "ch_gsd" in group:
        z = (pd.to_numeric(group[COUNT_COL], errors="coerce") - pd.to_numeric(group["ch_ewt"], errors="coerce"))
        z = z / pd.to_numeric(group["ch_gsd"], errors="coerce").replace(0, np.nan)
        ax2b = ax2.twinx()
        ax2b.plot(group["week_start"], z.clip(-20, 20), color=COLORS["gray"], alpha=0.45, lw=0.7, label="Desviacion relativa (recortada)")
        ax2b.set_ylabel("Desviacion relativa", color=COLORS["gray"])
    ax2.set_ylabel("Relacion")
    ax2.set_xlabel("Semana de emision")
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    path = out / "series" / f"serie_canal_{code}.png"
    save_figure(fig, path, dpi)
    return FigureItem(f"series_{code}", path, f"Serie y canal - {name}", "Datos y canal EWT",
                      "Casos, referencia EWT, banda superior, semanas en exceso y episodios.", municipality=code)


def metric_comparison(data: RunData, out: Path, dpi: int, task: str) -> FigureItem | None:
    result = data.results[data.results["task"] == task].copy() if "task" in data.results else pd.DataFrame()
    if result.empty:
        return None
    metrics = [
        ("average_precision", "AUC-PR", False), ("roc_auc", "ROC-AUC", False),
        ("brier", "Brier (menor es mejor)", True), ("specificity", "Especificidad", False),
    ]
    long_rows = []
    for _, row in result.iterrows():
        for key, label, _ in metrics:
            long_rows.append({"municipality": row["municipality"], "metric": label, "source": "Modelo", "value": row.get(f"test_{key}")})
    if not data.baselines.empty:
        base = data.baselines[data.baselines["task"] == task]
        for _, row in base.iterrows():
            source = "Persistencia" if row["baseline"] == "persistence" else "Climatologia"
            for key, label, _ in metrics:
                long_rows.append({"municipality": row["municipality"], "metric": label, "source": source, "value": row.get(key)})
    long = pd.DataFrame(long_rows)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    palette = {"Modelo": COLORS["blue"], "Persistencia": COLORS["orange"], "Climatologia": COLORS["green"]}
    for ax, (_, label, lower) in zip(axes.flat, metrics):
        view = long[long["metric"] == label].dropna(subset=["value"])
        sns.barplot(data=view, x="municipality", y="value", hue="source", palette=palette, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        if not lower:
            ax.set_ylim(0, 1.05)
        if ax is not axes.flat[0] and ax.get_legend() is not None:
            ax.get_legend().remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[0].get_legend().remove()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(f"Modelo frente a referencias - {TASK_LABELS.get(task, task)}", fontsize=17, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = out / "metricas" / f"comparacion_referencias_{task}.png"
    save_figure(fig, path, dpi)
    return FigureItem(f"metrics_{task}", path, f"Comparacion de metricas - {TASK_LABELS.get(task, task)}",
                      "Desempeno del modelo", "Comparacion contra persistencia y climatologia; Brier se minimiza.", task=task)


def metric_delta(data: RunData, out: Path, dpi: int, task: str) -> FigureItem | None:
    if data.baselines.empty:
        return None
    result = data.results[data.results["task"] == task].copy()
    base = data.baselines[(data.baselines["task"] == task) & (data.baselines["baseline"] == "persistence")].copy()
    merged = result.merge(base, on=["task", "municipality_code"], suffixes=("_model", "_base"))
    if merged.empty:
        return None
    delta = pd.DataFrame({
        "municipality": merged["municipality_model"],
        "AUC-PR": merged["test_average_precision"] - merged["average_precision"],
        "ROC-AUC": merged["test_roc_auc"] - merged["roc_auc"],
        "Brier (mejora)": merged["brier"] - merged["test_brier"],
        "Especificidad": merged["test_specificity"] - merged["specificity"],
    }).set_index("municipality")
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.7 * len(delta))))
    cmap = LinearSegmentedColormap.from_list("delta", [COLORS["red"], "white", COLORS["green"]])
    vmax = np.nanmax(np.abs(delta.to_numpy(dtype=float))) or 1
    sns.heatmap(delta, annot=True, fmt="+.3f", center=0, cmap=cmap, vmin=-vmax, vmax=vmax,
                linewidths=0.5, cbar_kws={"label": "Cambio favorable (+) / desfavorable (-)"}, ax=ax)
    ax.set_title(f"Cambio del modelo frente a persistencia - {TASK_LABELS.get(task, task)}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    path = out / "metricas" / f"deltas_vs_persistencia_{task}.png"
    save_figure(fig, path, dpi)
    return FigureItem(f"delta_{task}", path, f"Deltas frente a persistencia - {TASK_LABELS.get(task, task)}",
                      "Desempeno del modelo", "Valores positivos representan mejora; para Brier se invierte la diferencia.", task=task)


def candidate_ablation(data: RunData, out: Path, dpi: int, task: str) -> FigureItem | None:
    if data.candidates.empty:
        return None
    view = data.candidates[(data.candidates["task"] == task) & (data.candidates.get("status", "evaluated") == "evaluated")].copy()
    if view.empty:
        return None
    municipalities = list(view["municipality"].drop_duplicates())
    ncols = 3
    nrows = math.ceil(len(municipalities) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)
    for ax, name in zip(axes.flat, municipalities):
        sub = view[view["municipality"] == name].copy()
        sns.pointplot(data=sub, x="feature_set", y="average_precision", hue="algorithm", dodge=0.25,
                      markers=["o", "s", "D"], palette="tab10", ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Familia de variables")
        ax.set_ylabel("AUC-PR validacion")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(max(0, sub["average_precision"].min() - 0.08), min(1.02, sub["average_precision"].max() + 0.05))
        if ax is not axes.flat[0] and ax.get_legend() is not None:
            ax.get_legend().remove()
    for ax in axes.flat[len(municipalities):]:
        ax.axis("off")
    if axes.flat[0].get_legend() is not None:
        axes.flat[0].legend(title="Algoritmo", fontsize=8)
    fig.suptitle(f"Ablacion de variables y algoritmos - {TASK_LABELS.get(task, task)}", fontsize=17, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out / "modelos" / f"ablacion_validacion_{task}.png"
    save_figure(fig, path, dpi)
    return FigureItem(f"ablation_{task}", path, f"Ablacion - {TASK_LABELS.get(task, task)}",
                      "Seleccion y ablacion", "AUC-PR fuera de muestra durante seleccion; la prueba no interviene.", task=task)


def confusion_matrices(data: RunData, out: Path, dpi: int, task: str) -> FigureItem | None:
    view = data.results[data.results["task"] == task].copy()
    required = {"test_tn", "test_fp", "test_fn", "test_tp"}
    if view.empty or not required.issubset(view.columns):
        return None
    n = len(view)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows), squeeze=False)
    for ax, (_, row) in zip(axes.flat, view.iterrows()):
        cm = np.array([[row["test_tn"], row["test_fp"]], [row["test_fn"], row["test_tp"]]], dtype=float)
        sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", cbar=False, square=True,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["Real 0", "Real 1"], ax=ax)
        ax.set_title(f"{row['municipality']} | umbral {fmt(row.get('test_threshold'))}")
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"Matrices de confusion - {TASK_LABELS.get(task, task)}", fontsize=17, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out / "metricas" / f"matrices_confusion_{task}.png"
    save_figure(fig, path, dpi)
    return FigureItem(f"cm_{task}", path, f"Matrices de confusion - {TASK_LABELS.get(task, task)}",
                      "Desempeno del modelo", "Conteos en prueba bloqueada al umbral fijado en validacion.", task=task)


def probability_timeline(data: RunData, out: Path, dpi: int, task: str) -> FigureItem | None:
    keys = [(t, c) for (t, c) in data.predictions if t == task]
    if not keys:
        return None
    fig, axes = plt.subplots(len(keys), 1, figsize=(15, max(4, 2.7 * len(keys))), squeeze=False, sharex=False)
    for ax, (_, code) in zip(axes.flat, keys):
        pred = data.predictions[(task, code)].sort_values("week_start")
        name = municipality_name(data, code)
        ax.plot(pred["week_start"], pred["probability"], color=COLORS["blue"], lw=1.2, label="Probabilidad calibrada")
        threshold = pd.to_numeric(pred.get("threshold"), errors="coerce").dropna()
        th = threshold.iloc[0] if not threshold.empty else np.nan
        if pd.notna(th):
            ax.axhline(th, color=COLORS["orange"], ls="--", lw=1, label=f"Umbral {th:.3f}")
        positives = pd.to_numeric(pred.get("y_true"), errors="coerce") == 1
        ax.scatter(pred.loc[positives, "week_start"], pred.loc[positives, "probability"], color=COLORS["red"], s=18, label="Positivo real", zorder=3)
        alerts = pred.get("alert", pd.Series(False, index=pred.index)).astype(str).str.lower().isin(["true", "1"])
        ax.fill_between(pred["week_start"], 0, 1, where=alerts.to_numpy(), color=COLORS["gold"], alpha=0.09, step="mid")
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel(name)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle(f"Probabilidades, umbrales y alertas - {TASK_LABELS.get(task, task)}", fontsize=17, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out / "predicciones" / f"probabilidades_alertas_{task}.png"
    save_figure(fig, path, dpi)
    return FigureItem(f"prob_{task}", path, f"Probabilidades y alertas - {TASK_LABELS.get(task, task)}",
                      "Predicciones y alertas", "Sombreado amarillo: semanas alertadas; puntos rojos: positivos reales.", task=task)


def calibration_plot(data: RunData, out: Path, dpi: int, task: str) -> FigureItem | None:
    keys = [(t, c) for (t, c) in data.predictions if t == task]
    if not keys:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1]})
    axes[0].plot([0, 1], [0, 1], "--", color=COLORS["gray"], label="Calibracion perfecta")
    any_curve = False
    for _, code in keys:
        pred = data.predictions[(task, code)].dropna(subset=["y_true", "probability"])
        if len(pred) < 5 or pred["y_true"].nunique() < 2:
            continue
        bins = max(3, min(8, len(pred) // 10))
        true_rate, mean_pred = calibration_curve(pred["y_true"].astype(int), pred["probability"], n_bins=bins, strategy="quantile")
        axes[0].plot(mean_pred, true_rate, marker="o", lw=1.1, label=municipality_name(data, code))
        axes[1].hist(pred["probability"], bins=np.linspace(0, 1, 11), histtype="step", lw=1.2, label=municipality_name(data, code))
        any_curve = True
    if not any_curve:
        plt.close(fig)
        return None
    axes[0].set(xlabel="Probabilidad media predicha", ylabel="Frecuencia observada", xlim=(0, 1), ylim=(0, 1), title="Curva de confiabilidad")
    axes[1].set(xlabel="Probabilidad predicha", ylabel="Observaciones", xlim=(0, 1), title="Distribucion de probabilidades")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.suptitle(f"Calibracion en prueba - {TASK_LABELS.get(task, task)}", fontsize=17, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out / "metricas" / f"calibracion_{task}.png"
    save_figure(fig, path, dpi)
    return FigureItem(f"calibration_{task}", path, f"Calibracion - {TASK_LABELS.get(task, task)}",
                      "Calibracion y discriminacion", "La diagonal representa probabilidades perfectamente calibradas.", task=task)


def discrimination_curves(data: RunData, out: Path, dpi: int, task: str) -> FigureItem | None:
    keys = [(t, c) for (t, c) in data.predictions if t == task]
    if not keys:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    any_curve = False
    for _, code in keys:
        pred = data.predictions[(task, code)].dropna(subset=["y_true", "probability"])
        if pred["y_true"].nunique() < 2:
            continue
        y = pred["y_true"].astype(int).to_numpy()
        p = pred["probability"].astype(float).to_numpy()
        precision, recall, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        name = municipality_name(data, code)
        axes[0].plot(recall, precision, lw=1.3, label=f"{name} ({auc(recall[::-1], precision[::-1]):.3f})")
        axes[1].plot(fpr, tpr, lw=1.3, label=f"{name} ({auc(fpr, tpr):.3f})")
        axes[0].axhline(y.mean(), color=COLORS["gray"], alpha=0.12, lw=0.6)
        any_curve = True
    if not any_curve:
        plt.close(fig)
        return None
    axes[0].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="Precision-Recall")
    axes[1].plot([0, 1], [0, 1], "--", color=COLORS["gray"])
    axes[1].set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", xlim=(0, 1), ylim=(0, 1.02), title="ROC")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.suptitle(f"Curvas de discriminacion - {TASK_LABELS.get(task, task)}", fontsize=17, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out / "metricas" / f"curvas_pr_roc_{task}.png"
    save_figure(fig, path, dpi)
    return FigureItem(f"curves_{task}", path, f"Curvas PR y ROC - {TASK_LABELS.get(task, task)}",
                      "Calibracion y discriminacion", "Solo se incluyen pruebas con ambas clases.", task=task)


def feature_importance_plot(data: RunData, out: Path, dpi: int, task: str, code: str, top_n: int) -> FigureItem | None:
    frame = data.importances.get((task, code), pd.DataFrame()).copy()
    if frame.empty or not {"feature", "importance"}.issubset(frame.columns):
        return None
    frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce").abs()
    frame = frame.nlargest(top_n, "importance").sort_values("importance")
    frame["family"] = frame["feature"].map(feature_family)
    name = municipality_name(data, code)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.34 * len(frame) + 1.5)))
    ax.barh(frame["feature"], frame["importance"], color=frame["family"].map(FAMILY_COLORS))
    ax.set_xlabel("Importancia absoluta / ganancia del modelo")
    ax.set_title(f"Top {len(frame)} variables - {TASK_LABELS.get(task, task)} - {name}")
    handles = [plt.Line2D([0], [0], color=color, lw=6, label=FAMILY_LABELS[fam]) for fam, color in FAMILY_COLORS.items() if fam in set(frame["family"])]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    if "coefficient" in frame and frame["coefficient"].notna().any():
        for i, (_, row) in enumerate(frame.iterrows()):
            sign = "+" if row["coefficient"] >= 0 else "-"
            ax.text(row["importance"], i, f" {sign}", va="center", fontsize=8)
        ax.text(0.99, 0.01, "+/- indica signo del coeficiente estandarizado", transform=ax.transAxes,
                ha="right", fontsize=8, color=COLORS["gray"])
    fig.tight_layout()
    path = out / "importancias" / f"importancia_{task}_{code}.png"
    save_figure(fig, path, dpi)
    return FigureItem(f"importance_{task}_{code}", path, f"Importancia - {name} - {TASK_LABELS.get(task, task)}",
                      "Variables e interpretabilidad", "Importancia predictiva; no implica causalidad.", task=task, municipality=code)


def family_importance_plot(data: RunData, out: Path, dpi: int) -> FigureItem | None:
    if data.family_importance.empty:
        rows = []
        for (task, _), frame in data.importances.items():
            if frame.empty:
                continue
            temp = frame.nlargest(15, "importance").copy()
            temp["family"] = temp["feature"].map(feature_family)
            for family, count in temp["family"].value_counts().items():
                rows.append({"task": task, "family": family, "top15_appearances": count})
        if not rows:
            return None
        view = pd.DataFrame(rows).groupby(["task", "family"], as_index=False)["top15_appearances"].sum()
    else:
        view = data.family_importance.copy()
    view["family_label"] = view["family"].map(FAMILY_LABELS).fillna(view["family"])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    family_palette = {FAMILY_LABELS[key]: value for key, value in FAMILY_COLORS.items()}
    sns.barplot(data=view, x="task", y="top15_appearances", hue="family_label", palette=family_palette, ax=ax)
    ax.set_xlabel("Tarea")
    ax.set_ylabel("Apariciones en top 15")
    ax.set_title("Contribucion agregada de familias de variables")
    ax.legend(title="Familia", ncol=3)
    fig.tight_layout()
    path = out / "importancias" / "familias_top15.png"
    save_figure(fig, path, dpi)
    return FigureItem("family_importance", path, "Familias de variables", "Variables e interpretabilidad",
                      "Frecuencia de aparicion entre las variables mas importantes de modelos ganadores.")


def episode_plot(data: RunData, out: Path, dpi: int) -> FigureItem | None:
    if data.episodes.empty or "duration_weeks" not in data.episodes:
        return None
    eps = data.episodes.copy()
    eps["municipality"] = eps["municipality_code"].map(lambda c: municipality_name(data, c))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.boxplot(data=eps, x="municipality", y="duration_weeks", color=COLORS["blue"], ax=axes[0])
    sns.stripplot(data=eps, x="municipality", y="duration_weeks", color=COLORS["navy"], alpha=0.5, ax=axes[0])
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Duracion (semanas)")
    axes[0].set_title("Duracion de episodios por municipio")
    counts = eps.groupby("municipality").size().sort_values()
    axes[1].barh(counts.index, counts.values, color=COLORS["orange"])
    axes[1].set_xlabel("Numero de episodios")
    axes[1].set_title("Frecuencia de episodios")
    fig.suptitle("Episodios derivados del exceso EWT", fontsize=17, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out / "episodios" / "episodios_resumen.png"
    save_figure(fig, path, dpi)
    return FigureItem("episodes", path, "Resumen de episodios", "Episodios y alertas",
                      "Distribucion de duraciones y numero de episodios EWT.")


def alert_burden_plot(data: RunData, out: Path, dpi: int, task: str) -> FigureItem | None:
    view = data.results[data.results["task"] == task].copy()
    cols = {"test_alert_runs_per_year", "test_false_alert_runs_per_year"}
    if view.empty or not cols.issubset(view.columns):
        return None
    long = view.melt(id_vars=["municipality"], value_vars=list(cols), var_name="kind", value_name="runs")
    long["kind"] = long["kind"].map({
        "test_alert_runs_per_year": "Rachas de alerta / ano",
        "test_false_alert_runs_per_year": "Rachas falsas / ano",
    })
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=long, x="municipality", y="runs", hue="kind", palette=[COLORS["blue"], COLORS["red"]], ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Rachas por ano")
    ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"Carga operativa de alertas - {TASK_LABELS.get(task, task)}")
    ax.legend(title="")
    fig.tight_layout()
    path = out / "episodios" / f"carga_alertas_{task}.png"
    save_figure(fig, path, dpi)
    return FigureItem(f"alert_burden_{task}", path, f"Carga de alertas - {TASK_LABELS.get(task, task)}",
                      "Episodios y alertas", "Numero anualizado de rachas, no numero de semanas.", task=task)


def derive_insights(data: RunData) -> dict[str, Any]:
    results = data.results.copy()
    insights: dict[str, Any] = {
        "version": VERSION,
        "run_root": str(data.root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": {},
        "warnings": [],
        "recommendations": [],
    }
    for task in sorted(results["task"].dropna().unique()) if "task" in results else []:
        view = results[results["task"] == task]
        task_summary: dict[str, Any] = {
            "models": int(len(view)),
            "municipalities": sorted(view["municipality"].astype(str).unique().tolist()),
        }
        for col in ("test_average_precision", "test_roc_auc", "test_brier", "test_recall", "test_specificity", "test_prevalence"):
            if col in view:
                task_summary[f"macro_{col.removeprefix('test_')}"] = safe_float(view[col].mean())
        if "winner_algorithm" in view:
            task_summary["algorithm_wins"] = view["winner_algorithm"].value_counts().to_dict()
        if "winner_feature_set" in view:
            task_summary["feature_set_wins"] = view["winner_feature_set"].value_counts().to_dict()
        if not data.baselines.empty:
            persistence = data.baselines[(data.baselines["task"] == task) & (data.baselines["baseline"] == "persistence")]
            merged = view.merge(persistence, on=["task", "municipality_code"], suffixes=("_model", "_base"))
            if not merged.empty:
                task_summary["vs_persistence"] = {
                    "n": int(len(merged)),
                    "auc_pr_wins": int((merged["test_average_precision"] > merged["average_precision"]).sum()),
                    "brier_wins": int((merged["test_brier"] < merged["brier"]).sum()),
                    "specificity_wins": int((merged["test_specificity"] > merged["specificity"]).sum()),
                    "auc_pr_macro_model": safe_float(merged["test_average_precision"].mean()),
                    "auc_pr_macro_persistence": safe_float(merged["average_precision"].mean()),
                    "brier_macro_model": safe_float(merged["test_brier"].mean()),
                    "brier_macro_persistence": safe_float(merged["brier"].mean()),
                }
        insights["tasks"][task] = task_summary

    primary = results[results["task"] == "zone_future"] if "task" in results else pd.DataFrame()
    if not primary.empty:
        zero_spec = primary.loc[primary.get("test_specificity", 1) <= 0.001, "municipality"].astype(str).tolist()
        if zero_spec:
            insights["warnings"].append(f"Especificidad cero en zone_future: {', '.join(zero_spec)}.")
        extreme = primary.loc[(primary["test_prevalence"] < 0.10) | (primary["test_prevalence"] > 0.90), ["municipality", "test_prevalence"]]
        for _, row in extreme.iterrows():
            insights["warnings"].append(f"Prevalencia extrema en {row['municipality']}: {row['test_prevalence']:.3f}; interpretar AUC-PR con cautela.")
        weak = primary.loc[primary["test_roc_auc"] < 0.5, ["municipality", "test_roc_auc"]]
        for _, row in weak.iterrows():
            insights["warnings"].append(f"ROC-AUC menor que 0.5 en {row['municipality']}: {row['test_roc_auc']:.3f}.")
    if not data.skipped.empty:
        insights["warnings"].append(f"Hay {len(data.skipped)} combinaciones no evaluables; no deben imputarse como resultados negativos.")
    insights["recommendations"] = [
        "Priorizar zone_future para la conclusion principal y mantener pre_onset como sensibilidad cuando su muestra sea escasa.",
        "Exigir mejora simultanea frente a persistencia en AUC-PR y Brier antes de considerar uso operativo.",
        "Acordar una especificidad minima y un limite de carga de alertas; el costo 5:1 puede inducir alertas casi permanentes.",
        "Repetir una simulacion prospectiva con retrasos reales de notificacion y disponibilidad climatica.",
        "Interpretar importancias como evidencia predictiva, no causal.",
    ]
    profile = data.manifest.get("profile", {}) if data.manifest else {}
    insights["data_profile"] = profile
    return insights


def summary_table(data: RunData) -> pd.DataFrame:
    cols = [
        "task", "municipality_code", "municipality", "winner_algorithm", "winner_feature_set",
        "test_n", "test_positives", "test_prevalence", "test_average_precision", "test_roc_auc",
        "test_brier", "test_recall", "test_specificity", "test_precision", "test_f1",
        "test_threshold", "test_tn", "test_fp", "test_fn", "test_tp",
    ]
    cols = [c for c in cols if c in data.results]
    return data.results[cols].sort_values([c for c in ("task", "municipality") if c in cols])


def build_all_figures(data: RunData, image_dir: Path, tasks: Sequence[str], municipalities: Sequence[str] | None,
                      dpi: int, top_features: int) -> list[FigureItem]:
    figures: list[FigureItem] = []
    figures.append(executive_dashboard(data, image_dir, dpi))
    item = plot_data_coverage(data, image_dir, dpi)
    if item: figures.append(item)

    available_codes = sorted(set(data.panel.get("municipality_code", pd.Series(dtype=str)).dropna().astype(str)) |
                             set(data.results.get("municipality_code", pd.Series(dtype=str)).dropna().astype(str)))
    selected_codes = [normalize_code(c) for c in municipalities] if municipalities else available_codes
    for code in selected_codes:
        item = plot_series_channel(data, image_dir, dpi, code)
        if item: figures.append(item)

    for task in tasks:
        for fn in (metric_comparison, metric_delta, candidate_ablation, confusion_matrices,
                   probability_timeline, calibration_plot, discrimination_curves, alert_burden_plot):
            item = fn(data, image_dir, dpi, task)
            if item: figures.append(item)
        for code in selected_codes:
            item = feature_importance_plot(data, image_dir, dpi, task, code, top_features)
            if item: figures.append(item)
    for fn in (family_importance_plot, episode_plot):
        item = fn(data, image_dir, dpi)
        if item: figures.append(item)
    return figures


def human_insight_paragraphs(insights: dict[str, Any]) -> list[str]:
    paragraphs: list[str] = []
    zone = insights.get("tasks", {}).get("zone_future", {})
    if zone:
        text = (
            f"La tarea principal cubrio {zone.get('models', 0)} municipios/modelos. "
            f"Su AUC-PR macro fue {fmt(zone.get('macro_average_precision'))}, el Brier macro "
            f"{fmt(zone.get('macro_brier'))}, la prevalencia macro {fmt(zone.get('macro_prevalence'), pct=True)} "
            f"y la especificidad macro {fmt(zone.get('macro_specificity'), pct=True)}."
        )
        paragraphs.append(text)
        comp = zone.get("vs_persistence", {})
        if comp:
            paragraphs.append(
                f"Frente a persistencia, el modelo mejoro AUC-PR en {comp.get('auc_pr_wins', 0)}/{comp.get('n', 0)} "
                f"municipios, pero Brier solo en {comp.get('brier_wins', 0)}/{comp.get('n', 0)}. "
                f"La lectura operativa exige ambas dimensiones, no solo discriminacion."
            )
    onset = insights.get("tasks", {}).get("pre_onset", {})
    if onset:
        paragraphs.append(
            f"La sensibilidad pre_onset produjo {onset.get('models', 0)} modelos evaluables. "
            "Los tamanos de prueba y el numero de eventos deben revisarse antes de comparar metricas."
        )
    return paragraphs


def docx_picture_width(path: Path, max_width: float = 9.5, max_height: float = 5.55) -> float:
    """Ancho en pulgadas que respeta simultaneamente ancho y alto utilizables."""
    with PILImage.open(path) as image:
        width_px, height_px = image.size
    if width_px <= 0 or height_px <= 0:
        return max_width
    return min(max_width, max_height * width_px / height_px)


def add_docx_table(document: Any, frame: pd.DataFrame, max_rows: int = 30) -> None:
    if frame.empty:
        document.add_paragraph("Sin filas disponibles.")
        return
    view = frame.head(max_rows).copy()
    table = document.add_table(rows=1, cols=len(view.columns))
    table.style = "Table Grid"
    for i, col in enumerate(view.columns):
        table.rows[0].cells[i].text = str(col)
    for _, row in view.iterrows():
        cells = table.add_row().cells
        for i, col in enumerate(view.columns):
            val = row[col]
            if isinstance(val, (float, np.floating)):
                cells[i].text = fmt(val)
            else:
                cells[i].text = "" if pd.isna(val) else str(val)
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(7)


def create_docx(data: RunData, figures: list[FigureItem], insights: dict[str, Any], path: Path) -> None:
    if Document is None:
        raise RuntimeError("python-docx no esta instalado")
    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)

    styles = doc.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(9)
    styles["Title"].font.name = "Aptos Display"
    styles["Title"].font.size = Pt(24)
    styles["Title"].font.color.rgb = RGBColor(0, 120, 212)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Analisis grafico punta a punta\nDengueAlertas EWT")
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run(f"Generado por analyze_ewt_results.py v{VERSION}\n").bold = True
    sub.add_run(f"Corrida: {data.root}\nFecha UTC: {insights['generated_at_utc']}")

    dashboard = next((f for f in figures if f.key == "dashboard"), None)
    if dashboard:
        width = docx_picture_width(dashboard.path, max_width=8.6, max_height=4.65)
        doc.add_picture(str(dashboard.path), width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_page_break()
    doc.add_heading("1. Resumen ejecutivo", level=1)
    for p in human_insight_paragraphs(insights):
        doc.add_paragraph(p)
    for warning in insights.get("warnings", []):
        p = doc.add_paragraph(style="List Bullet")
        p.add_run("Advertencia: ").bold = True
        p.add_run(warning)

    doc.add_heading("2. Metodologia de lectura", level=1)
    doc.add_paragraph(
        "El informe separa datos/canal, seleccion, prueba bloqueada, calibracion, carga de alertas e interpretabilidad. "
        "AUC-PR se interpreta junto con prevalencia; Brier y log-loss miden calidad probabilistica; especificidad y "
        "falsas alertas aproximan la carga operativa. Las importancias no se interpretan causalmente."
    )

    doc.add_heading("3. Resultados tabulares", level=1)
    table = summary_table(data).rename(columns={
        "winner_algorithm": "modelo", "winner_feature_set": "features",
        "test_average_precision": "auc_pr", "test_roc_auc": "roc_auc",
        "test_specificity": "especificidad", "test_prevalence": "prevalencia",
    })
    add_docx_table(doc, table, max_rows=50)

    grouped: dict[str, list[FigureItem]] = defaultdict(list)
    for fig in figures:
        if fig.key != "dashboard":
            grouped[fig.section].append(fig)
    section_number = 4
    for section_name, items in grouped.items():
        doc.add_page_break()
        doc.add_heading(f"{section_number}. {section_name}", level=1)
        for item in items:
            heading = doc.add_heading(item.title, level=2)
            heading.paragraph_format.keep_with_next = True
            width = docx_picture_width(item.path, max_width=9.5, max_height=5.45)
            doc.add_picture(str(item.path), width=Inches(width))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph(item.caption)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.runs[0].italic = True
            cap.runs[0].font.size = Pt(8)
        section_number += 1

    doc.add_page_break()
    doc.add_heading(f"{section_number}. Recomendaciones", level=1)
    for rec in insights.get("recommendations", []):
        doc.add_paragraph(rec, style="List Number")
    doc.add_heading("Limitaciones de este reporte", level=2)
    doc.add_paragraph(
        "El reporte describe los artefactos disponibles en la carpeta de corrida. No reentrena modelos, no altera "
        "umbrales y no completa archivos ausentes. Curvas PR/ROC y calibracion se omiten cuando la prueba no contiene "
        "ambas clases o no hay suficientes observaciones."
    )

    doc.add_heading("Inventario de artefactos", level=2)
    inv = pd.DataFrame([a.__dict__ for a in data.artifacts])
    add_docx_table(doc, inv, max_rows=100)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)


def register_pdf_fonts() -> tuple[str, str]:
    regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if os.path.exists(regular) and os.path.exists(bold):
        pdfmetrics.registerFont(TTFont("DejaVu", regular))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", bold))
        return "DejaVu", "DejaVu-Bold"
    return "Helvetica", "Helvetica-Bold"


def pdf_header_footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont(getattr(doc, "base_font", "Helvetica"), 7)
    canvas.setFillColor(colors.HexColor("#605E5C"))
    canvas.drawString(1.2 * cm, 0.7 * cm, "DengueAlertas EWT - analisis grafico reproducible")
    canvas.drawRightString(landscape(A4)[0] - 1.2 * cm, 0.7 * cm, f"Pagina {doc.page}")
    canvas.restoreState()


def create_pdf(data: RunData, figures: list[FigureItem], insights: dict[str, Any], path: Path) -> None:
    if BaseDocTemplate is None:
        raise RuntimeError("reportlab no esta instalado")
    base, bold = register_pdf_fonts()
    page = landscape(A4)
    doc = BaseDocTemplate(str(path), pagesize=page, rightMargin=1.2*cm, leftMargin=1.2*cm,
                          topMargin=1.1*cm, bottomMargin=1.2*cm, title="Analisis grafico DengueAlertas EWT")
    doc.base_font = base
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=pdf_header_footer))
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleEWT", parent=styles["Title"], fontName=bold, fontSize=23,
                                 leading=27, textColor=colors.HexColor(COLORS["blue"]), alignment=TA_CENTER)
    h1 = ParagraphStyle("H1EWT", parent=styles["Heading1"], fontName=bold, fontSize=16,
                        textColor=colors.HexColor(COLORS["navy"]), spaceAfter=8)
    h2 = ParagraphStyle("H2EWT", parent=styles["Heading2"], fontName=bold, fontSize=12,
                        textColor=colors.HexColor(COLORS["blue"]), spaceBefore=6, spaceAfter=5)
    body = ParagraphStyle("BodyEWT", parent=styles["BodyText"], fontName=base, fontSize=9, leading=12)
    caption = ParagraphStyle("CaptionEWT", parent=body, fontSize=7.5, leading=9,
                             textColor=colors.HexColor(COLORS["gray"]), alignment=TA_CENTER)
    bullet = ParagraphStyle("BulletEWT", parent=body, leftIndent=14, firstLineIndent=-7, bulletIndent=4)
    story: list[Any] = []
    story += [Spacer(1, 1.3*cm), Paragraph("Analisis grafico punta a punta<br/>DengueAlertas EWT", title_style),
              Spacer(1, 0.5*cm), Paragraph(f"Script v{VERSION} | {insights['generated_at_utc']}", caption), Spacer(1, 0.4*cm)]
    dashboard = next((f for f in figures if f.key == "dashboard"), None)
    if dashboard:
        story.append(RLImage(str(dashboard.path), width=21*cm, height=12*cm))
    story.append(PageBreak())
    story.append(Paragraph("1. Resumen ejecutivo", h1))
    for p in human_insight_paragraphs(insights):
        story += [Paragraph(p, body), Spacer(1, 0.15*cm)]
    for warning in insights.get("warnings", []):
        story.append(Paragraph(f"• <b>Advertencia:</b> {warning}", bullet))
    story += [Spacer(1, 0.25*cm), Paragraph("Resultados tabulares", h2)]
    table_df = summary_table(data)
    display_cols = [c for c in ["task", "municipality", "winner_algorithm", "winner_feature_set", "test_n",
                                  "test_prevalence", "test_average_precision", "test_roc_auc", "test_brier",
                                  "test_recall", "test_specificity", "test_threshold"] if c in table_df]
    table_df = table_df[display_cols].copy()
    for col in table_df.columns:
        if pd.api.types.is_numeric_dtype(table_df[col]):
            table_df[col] = table_df[col].map(lambda x: fmt(x))
    rows = [list(table_df.columns)] + table_df.astype(str).values.tolist()
    table = Table(rows, repeatRows=1, colWidths=[doc.width / max(1, len(display_cols))] * len(display_cols))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(COLORS["blue"])),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), bold), ("FONTNAME", (0, 1), (-1, -1), base),
        ("FONTSIZE", (0, 0), (-1, -1), 6.2), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F2F1")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(table)

    grouped: dict[str, list[FigureItem]] = defaultdict(list)
    for item in figures:
        if item.key != "dashboard":
            grouped[item.section].append(item)
    sec_num = 2
    for section_name, items in grouped.items():
        story.append(PageBreak())
        section_heading = Paragraph(f"{sec_num}. {section_name}", h1)
        for idx, item in enumerate(items):
            if idx:
                story.append(PageBreak())
            img = RLImage(str(item.path))
            max_w, max_h = doc.width, doc.height - 4.2*cm
            ratio = min(max_w / img.imageWidth, max_h / img.imageHeight)
            img.drawWidth = img.imageWidth * ratio
            img.drawHeight = img.imageHeight * ratio
            block = [Paragraph(item.title, h2), img, Spacer(1, 0.15*cm), Paragraph(item.caption, caption)]
            if idx == 0:
                block.insert(0, section_heading)
            story.append(KeepTogether(block))
        sec_num += 1

    story += [PageBreak(), Paragraph(f"{sec_num}. Recomendaciones y trazabilidad", h1)]
    for rec in insights.get("recommendations", []):
        story.append(Paragraph(f"• {rec}", bullet))
    story += [Spacer(1, 0.25*cm), Paragraph("Inventario de insumos", h2)]
    inv_rows = [["Artefacto", "Ruta", "Obligatorio", "Encontrado", "Filas / nota"]]
    for a in data.artifacts:
        inv_rows.append([a.name, a.path, str(a.required), str(a.found), str(a.rows if a.rows is not None else a.note)])
    inv = Table(inv_rows, repeatRows=1, colWidths=[3.2*cm, 12*cm, 2.5*cm, 2.5*cm, 5*cm])
    inv.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(COLORS["navy"])),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), bold),
        ("FONTNAME", (0, 1), (-1, -1), base), ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(inv)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)


def write_manifest(data: RunData, figures: list[FigureItem], insights: dict[str, Any], out_dir: Path,
                   formats: Sequence[str]) -> None:
    payload = {
        "script_version": VERSION,
        "generated_at_utc": insights["generated_at_utc"],
        "run_root": str(data.root),
        "requested_formats": list(formats),
        "figures": [{**f.__dict__, "path": str(f.path.relative_to(out_dir))} for f in figures],
        "inputs": [a.__dict__ for a in data.artifacts],
        "methodological_guardrails": [
            "No reentrena ni recalibra modelos.",
            "No usa la prueba para seleccionar modelos o umbrales.",
            "Omite curvas que requieren ambas clases cuando no son calculables.",
            "No interpreta importancias como causalidad.",
        ],
    }
    (out_dir / "manifest_analisis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera analisis grafico PDF, Word e imagenes desde outputs/final_run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True, type=Path, help="Carpeta final_run o ZIP que la contenga")
    parser.add_argument("--output", type=Path, default=None, help="Carpeta de salida; por defecto <run-dir>/analysis")
    parser.add_argument("--formats", nargs="+", default=["all"], choices=["all", "pdf", "docx", "images"], help="Formatos a producir")
    parser.add_argument("--tasks", nargs="+", default=None, help="Tareas a incluir; por defecto las presentes en results.csv")
    parser.add_argument("--municipalities", nargs="+", default=None, help="Codigos DIVIPOLA a incluir")
    parser.add_argument("--top-features", type=int, default=15, help="Numero de variables por grafico")
    parser.add_argument("--dpi", type=int, default=180, help="Resolucion de PNG")
    parser.add_argument("--overwrite", action="store_true", help="Reemplaza la carpeta de salida si existe")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.top_features < 1 or args.dpi < 72:
        raise SystemExit("--top-features debe ser positivo y --dpi al menos 72")
    formats = ["pdf", "docx", "images"] if "all" in args.formats else list(dict.fromkeys(args.formats))
    with tempfile.TemporaryDirectory(prefix="ewt_analysis_") as temp:
        run_root = resolve_run_dir(args.run_dir.expanduser().resolve(), Path(temp))
        output = args.output.expanduser().resolve() if args.output else run_root / "analysis"
        if output.exists() and args.overwrite:
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        image_dir = output / "images"
        image_dir.mkdir(exist_ok=True)
        data = load_run(run_root)
        available_tasks = sorted(data.results["task"].dropna().astype(str).unique())
        tasks = args.tasks or available_tasks
        unknown = sorted(set(tasks) - set(available_tasks))
        if unknown:
            raise SystemExit(f"Tareas no presentes en results.csv: {unknown}; disponibles: {available_tasks}")
        figures = build_all_figures(data, image_dir, tasks, args.municipalities, args.dpi, args.top_features)
        insights = derive_insights(data)
        (output / "resumen_hallazgos.json").write_text(json.dumps(insights, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        summary_table(data).to_csv(output / "resumen_metricas.csv", index=False)
        if data.skipped.empty:
            pd.DataFrame(columns=["task", "municipality_code", "municipality", "reason"]).to_csv(output / "modelos_no_evaluables.csv", index=False)
        else:
            data.skipped.to_csv(output / "modelos_no_evaluables.csv", index=False)
        if "docx" in formats:
            create_docx(data, figures, insights, output / "analisis_grafico_ewt.docx")
        if "pdf" in formats:
            create_pdf(data, figures, insights, output / "analisis_grafico_ewt.pdf")
        write_manifest(data, figures, insights, output, formats)
        print(json.dumps({
            "status": "ok", "run_dir": str(run_root), "output": str(output),
            "formats": formats, "figures": len(figures),
            "pdf": str(output / "analisis_grafico_ewt.pdf") if "pdf" in formats else None,
            "docx": str(output / "analisis_grafico_ewt.docx") if "docx" in formats else None,
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
