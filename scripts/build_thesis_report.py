"""Genera tablas y gráficas consolidadas de contraste de modelos (desarrollo 2019-2022) para la tesis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MUNICIPIOS = {
    "08001": "Barranquilla",
    "08758": "Soledad",
    "41001": "Neiva",
    "50001": "Villavicencio",
    "54001": "Cúcuta",
    "76001": "Cali",
}

ETIQUETA_FAMILIA = {
    "logistica": "Regresión Logística L2",
    "svm": "SVM (RBF)",
    "mlp": "MLP Tabular",
    "gru": "GRU Secuencial",
    "cnn1d": "CNN-1D Secuencial",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}

TIPO_FAMILIA = {
    "logistica": "Lineal / Kernel",
    "svm": "Lineal / Kernel",
    "mlp": "Deep Learning",
    "gru": "Deep Learning",
    "cnn1d": "Deep Learning",
    "catboost": "Ensambles de Árboles",
    "lightgbm": "Ensambles de Árboles",
    "xgboost": "Ensambles de Árboles",
}

COLORES_TIPO = {
    "Lineal / Kernel": "#1f4e79",
    "Deep Learning": "#c55a11",
    "Ensambles de Árboles": "#2e75b6",
}


def cargar_tablas(base_dir: Path) -> pd.DataFrame:
    archivos = sorted(base_dir.rglob("tabla_unica_desarrollo.parquet"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos tabla_unica_desarrollo.parquet en {base_dir}")

    frames = []
    for ruta in archivos:
        df = pd.read_parquet(ruta).copy()
        if "familia" not in df.columns or "mejora_b2" not in df.columns:
            continue
        df["familia"] = df["familia"].astype(str).str.lower().str.strip()
        df = df[df["familia"].isin(ETIQUETA_FAMILIA.keys())].copy()
        if df.empty:
            continue
        if "variante" not in df.columns:
            df["variante"] = "sin_clima"
        if "brier" not in df.columns:
            df["brier"] = np.nan
        df["municipality_code"] = df["municipality_code"].astype(str).str.zfill(5)
        df["corrida"] = ruta.parent.name
        frames.append(df)
        print(f"Cargado: {ruta} | familias={sorted(df['familia'].unique())} | filas={len(df)}")

    if not frames:
        raise RuntimeError("No se encontraron registros válidos de las familias esperadas.")
    return pd.concat(frames, ignore_index=True)


def seleccionar_ganadores(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    resumen = (
        df.groupby(["variante", "familia", "corrida", "candidato", "parametros"], as_index=False)
        .agg(
            mejora_b2_media=("mejora_b2", "mean"),
            mejora_b2_std=("mejora_b2", "std"),
            logloss_medio=("logloss", "mean"),
            brier_medio=("brier", "mean"),
            casillas_ganadas=("mejora_b2", lambda s: int((s > 0).sum())),
            n_casillas=("mejora_b2", "count"),
        )
        .sort_values(["mejora_b2_media", "logloss_medio"], ascending=[False, True])
    )
    # Exigir las 24 casillas completas si existen
    completos = resumen[resumen["n_casillas"] == 24]
    base = completos if not completos.empty else resumen
    ganadores = (
        base.groupby(["variante", "familia"], as_index=False)
        .first()
        .sort_values(["variante", "mejora_b2_media"], ascending=[False, False])
        .reset_index(drop=True)
    )
    ganadores["modelo"] = ganadores["familia"].map(ETIQUETA_FAMILIA)
    ganadores["categoria"] = ganadores["familia"].map(TIPO_FAMILIA)
    ganadores["mejora_b2_pct"] = ganadores["mejora_b2_media"] * 100.0

    # Extraer filas de detalle (24 casillas) de cada ganador
    llaves = ganadores[["variante", "familia", "corrida", "candidato"]]
    detalle = df.merge(llaves, on=["variante", "familia", "corrida", "candidato"], how="inner")
    detalle["municipio"] = detalle["municipality_code"].map(MUNICIPIOS).fillna(detalle["municipality_code"])
    detalle["modelo"] = detalle["familia"].map(ETIQUETA_FAMILIA)
    return ganadores, detalle


def guardar_tabla(df: pd.DataFrame, salida: Path, nombre: str) -> None:
    df.to_csv(salida / f"{nombre}.csv", index=False)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = df.to_string(index=False)
    (salida / f"{nombre}.md").write_text(md + "\n", encoding="utf-8")


def configurar_estilo() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def graficar_ranking(sin_clima: pd.DataFrame, salida: Path) -> None:
    d = sin_clima.sort_values("mejora_b2_pct", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colores = [COLORES_TIPO[c] for c in d["categoria"]]
    bars = ax.barh(d["modelo"], d["mejora_b2_pct"], color=colores, edgecolor="#222222", linewidth=0.5, height=0.65)
    ax.axvline(0, color="#222222", linewidth=1.1, linestyle="--")

    x_min = min(d["mejora_b2_pct"].min() - 6.0, -8.0)
    x_max = d["mejora_b2_pct"].max() + 9.5
    ax.set_xlim(x_min, x_max)

    for bar, (_, row) in zip(bars, d.iterrows()):
        val = row["mejora_b2_pct"]
        etq = f"{val:+.2f}%  ({int(row['casillas_ganadas'])}/24)"
        offset = 0.6 if val >= 0 else -0.6
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2, etq, va="center", ha=ha, fontsize=9, fontweight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, color=col, label=cat) for cat, col in COLORES_TIPO.items()]
    ax.legend(handles=handles, frameon=True, loc="lower right", title="Enfoque")
    ax.set_xlabel("Mejora media en Log-Loss frente a la línea base B2 (%)")
    ax.set_title("Contraste global de 8 familias de modelos (sin_clima, desarrollo 2019–2022)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for ext in ("png", "pdf"):
        fig.savefig(salida / f"fig_01_ranking_familias_b2.{ext}")
    plt.close(fig)


def graficar_calibracion(sin_clima: pd.DataFrame, salida: Path) -> None:
    d = sin_clima.dropna(subset=["logloss_medio", "brier_medio"]).copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for cat, g in d.groupby("categoria"):
        ax.scatter(
            g["logloss_medio"],
            g["brier_medio"],
            s=110,
            color=COLORES_TIPO[cat],
            edgecolor="#222222",
            linewidth=0.7,
            label=cat,
            zorder=3,
        )
    for _, row in d.iterrows():
        ax.annotate(
            row["modelo"],
            (row["logloss_medio"], row["brier_medio"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8.8,
        )

    ax.set_xlabel("Log-Loss medio (menor es mejor)")
    ax.set_ylabel("Brier Score medio (menor es mejor)")
    ax.set_title("Calibración probabilística por familia (sin_clima, 2019–2022)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, loc="upper left", title="Enfoque")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for ext in ("png", "pdf"):
        fig.savefig(salida / f"fig_02_calibracion_logloss_brier.{ext}")
    plt.close(fig)


def graficar_heatmap_municipal(detalle_sc: pd.DataFrame, orden_modelos: list[str], salida: Path) -> pd.DataFrame:
    agg = (
        detalle_sc.groupby(["municipio", "modelo"], as_index=False)
        .agg(
            mejora_pct=("mejora_b2", lambda s: float(s.mean() * 100.0)),
            anios=("mejora_b2", lambda s: int((s > 0).sum())),
        )
    )
    piv_val = agg.pivot(index="municipio", columns="modelo", values="mejora_pct")[orden_modelos]
    piv_an = agg.pivot(index="municipio", columns="modelo", values="anios")[orden_modelos]

    # Ordenar municipios por desempeño promedio
    orden_mun = piv_val.mean(axis=1).sort_values(ascending=False).index
    piv_val = piv_val.loc[orden_mun]
    piv_an = piv_an.loc[orden_mun]

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    vmax = max(abs(np.nanmin(piv_val.values)), abs(np.nanmax(piv_val.values)), 25.0)
    im = ax.imshow(piv_val.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(piv_val.columns)))
    ax.set_xticklabels(piv_val.columns, rotation=28, ha="right", fontsize=9)
    ax.set_yticks(range(len(piv_val.index)))
    ax.set_yticklabels(piv_val.index, fontsize=9.5)

    for i in range(piv_val.shape[0]):
        for j in range(piv_val.shape[1]):
            val = piv_val.iat[i, j]
            an = piv_an.iat[i, j]
            if pd.notna(val):
                color_txt = "white" if abs(val) > vmax * 0.65 else "#111111"
                ax.text(j, i, f"{val:+.1f}%\n({int(an)}/4)", ha="center", va="center", fontsize=8.2, color=color_txt, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mejora media vs. B2 (%)")
    ax.set_title("Desempeño municipal por familia: mejora frente a B2 (%) y años ganados (2019–2022)")

    for ext in ("png", "pdf"):
        fig.savefig(salida / f"fig_03_heatmap_municipal.{ext}")
    plt.close(fig)

    tabla_mun = piv_val.round(2).reset_index()
    return tabla_mun


def graficar_ablacion_dl(ganadores: pd.DataFrame, salida: Path) -> pd.DataFrame:
    dl = ganadores[ganadores["familia"].isin(["gru", "cnn1d"])].copy()
    if dl["variante"].nunique() < 2:
        return dl

    fams = ["gru", "cnn1d"]
    etqs = [ETIQUETA_FAMILIA[f] for f in fams]
    sin_v = [float(dl[(dl["familia"] == f) & (dl["variante"] == "sin_clima")]["mejora_b2_pct"].iloc[0]) for f in fams]
    con_v = [float(dl[(dl["familia"] == f) & (dl["variante"] == "con_clima")]["mejora_b2_pct"].iloc[0]) for f in fams]

    x = np.arange(len(fams))
    w = 0.34
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    b1 = ax.bar(x - w / 2, sin_v, width=w, label="Sin clima (5 canales)", color="#1f4e79", edgecolor="#222222")
    b2 = ax.bar(x + w / 2, con_v, width=w, label="Con clima semanal (8 canales)", color="#c55a11", edgecolor="#222222")
    ax.axhline(0, color="#222222", linewidth=1.0, linestyle="--")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            dy = 0.5 if h >= 0 else -0.6
            ax.text(bar.get_x() + bar.get_width() / 2, h + dy, f"{h:+.2f}%", ha="center", va=va, fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(etqs, fontsize=10)
    ax.set_ylabel("Mejora media en Log-Loss frente a B2 (%)")
    ax.set_title("Ablación climática en Deep Learning secuencial (12 semanas, 2019–2022)")
    ax.set_ylim(min(min(sin_v), min(con_v)) - 5.0, max(max(sin_v), max(con_v)) + 5.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(frameon=True, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for ext in ("png", "pdf"):
        fig.savefig(salida / f"fig_04_ablacion_clima_dl.{ext}")
    plt.close(fig)
    return dl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", type=Path, default=Path("data/processed/forecasting"))
    parser.add_argument("--salida", type=Path, default=Path("reports/tesis_contraste_modelos"))
    args = parser.parse_args()

    args.salida.mkdir(parents=True, exist_ok=True)
    configurar_estilo()

    df = cargar_tablas(args.entrada)
    ganadores, detalle = seleccionar_ganadores(df)

    sin_clima = (
        ganadores[ganadores["variante"] == "sin_clima"]
        .sort_values("mejora_b2_media", ascending=False)
        .reset_index(drop=True)
    )
    sin_clima.insert(0, "puesto", range(1, len(sin_clima) + 1))

    cols_global = [
        "puesto", "modelo", "categoria", "candidato", "corrida",
        "mejora_b2_pct", "mejora_b2_std", "logloss_medio", "brier_medio",
        "casillas_ganadas", "parametros",
    ]
    tabla_global = sin_clima[cols_global].copy()
    tabla_global["mejora_b2_pct"] = tabla_global["mejora_b2_pct"].round(2)
    tabla_global["mejora_b2_std"] = (tabla_global["mejora_b2_std"] * 100.0).round(2)
    tabla_global["logloss_medio"] = tabla_global["logloss_medio"].round(4)
    tabla_global["brier_medio"] = tabla_global["brier_medio"].round(4)
    guardar_tabla(tabla_global, args.salida, "tabla_01_ranking_global_sin_clima")

    # Gráficas y tablas derivadas
    graficar_ranking(sin_clima, args.salida)
    graficar_calibracion(sin_clima, args.salida)

    detalle_sc = detalle[detalle["variante"] == "sin_clima"].copy()
    orden_modelos = list(sin_clima["modelo"])
    tabla_mun = graficar_heatmap_municipal(detalle_sc, orden_modelos, args.salida)
    guardar_tabla(tabla_mun, args.salida, "tabla_03_desglose_municipal")

    dl_ab = graficar_ablacion_dl(ganadores, args.salida)
    if not dl_ab.empty:
        t_ab = dl_ab[["variante", "modelo", "candidato", "mejora_b2_pct", "logloss_medio", "brier_medio", "casillas_ganadas", "parametros"]].copy()
        t_ab["mejora_b2_pct"] = t_ab["mejora_b2_pct"].round(2)
        t_ab["logloss_medio"] = t_ab["logloss_medio"].round(4)
        t_ab["brier_medio"] = t_ab["brier_medio"].round(4)
        guardar_tabla(t_ab, args.salida, "tabla_02_ablacion_clima_dl")

    # Detalle anual de las 4 mejores familias
    top4 = orden_modelos[:4]
    col_anio = "year" if "year" in detalle_sc.columns else "anio"
    if col_anio in detalle_sc.columns:
        det_anual = (
            detalle_sc[detalle_sc["modelo"].isin(top4)]
            .groupby([col_anio, "modelo"], as_index=False)
            .agg(
                mejora_b2_pct=("mejora_b2", lambda s: round(float(s.mean() * 100.0), 2)),
                logloss_medio=("logloss", lambda s: round(float(s.mean()), 4)),
                municipios_ganados=("mejora_b2", lambda s: int((s > 0).sum())),
            )
            .sort_values([col_anio, "mejora_b2_pct"], ascending=[True, False])
        )
        guardar_tabla(det_anual, args.salida, "tabla_04_detalle_anual_top4")

    print("\n=== TABLA GLOBAL CONSOLIDADA (sin_clima) ===")
    print(tabla_global[["puesto", "modelo", "candidato", "corrida", "mejora_b2_pct", "logloss_medio", "brier_medio", "casillas_ganadas"]].to_string(index=False))
    print(f"\nArchivos generados en: {args.salida}")
    for f in sorted(args.salida.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
