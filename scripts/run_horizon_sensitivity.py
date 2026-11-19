"""Contraste 3: Sensibilidad del horizonte temporal (h in {2, 4, 8, 12}).

Evalúa las 5 recetas ganadoras (sin_clima) con sus hiperparámetros fijos del desarrollo
2019-2022 y recalcula B2 para cada horizonte h sin fuga de datos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts.run_seq_dl import fit_and_predict, standardize_fold
from src.forecasting.baselines import b2_predict
from src.forecasting.config import load_config
from src.forecasting.dataset import (
    FEATURE_SETS,
    ZONES,
    build_examples,
    build_gru_examples,
    load_channel_product,
)
from src.forecasting.guard import RecipeAccess
from src.forecasting.metrics import brier, improvement, log_loss
from src.forecasting.models import fit_model, predict_positive
from src.forecasting.splits import yearly_trials

MODELO_A_FAMILIA = {
    "Regresión Logística L2": "logistica",
    "SVM (RBF)": "svm",
    "MLP Tabular": "mlp",
    "CatBoost": "catboost",
    "GRU Secuencial": "gru",
}

COLORES = {
    "Regresión Logística L2": "#1f4e79",
    "SVM (RBF)": "#2e75b6",
    "MLP Tabular": "#c55a11",
    "CatBoost": "#548235",
    "GRU Secuencial": "#7030a0",
    "Línea base B2": "#333333",
}


def cargar_recetas_ganadoras(tabla_global: Path) -> list[dict]:
    df = pd.read_csv(tabla_global)
    recetas = []
    for _, row in df.iterrows():
        modelo = str(row["modelo"])
        if modelo in MODELO_A_FAMILIA:
            recetas.append({
                "modelo": modelo,
                "familia": MODELO_A_FAMILIA[modelo],
                "candidato": str(row["candidato"]),
                "params": json.loads(str(row["parametros"])),
            })
    return recetas


def graficar_sensibilidad(res: pd.DataFrame, b2_res: pd.DataFrame, salida: Path) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    horizontes = sorted(res["horizonte_h"].unique())

    # Panel izquierdo: Log-Loss absoluto (incluyendo B2)
    ax1.plot(
        b2_res["horizonte_h"],
        b2_res["logloss_medio"],
        linestyle="--",
        marker="s",
        linewidth=2.0,
        color=COLORES["Línea base B2"],
        label="Línea base B2",
    )
    for modelo in MODELO_A_FAMILIA.keys():
        sub = res[res["modelo"] == modelo].sort_values("horizonte_h")
        if sub.empty:
            continue
        ax1.plot(
            sub["horizonte_h"],
            sub["logloss_medio"],
            marker="o",
            linewidth=2.0,
            color=COLORES[modelo],
            label=modelo,
        )
    ax1.axvline(8, color="#888888", linestyle=":", linewidth=1.1)
    ax1.set_xticks(horizontes)
    ax1.set_xlabel("Horizonte de pronóstico h (semanas)")
    ax1.set_ylabel("Log-Loss medio (menor es mejor)")
    ax1.set_title("A. Degradación probabilística frente al horizonte")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel derecho: Mejora relativa frente a B2 (%)
    ax2.axhline(0, color="#222222", linestyle="--", linewidth=1.1)
    ax2.axvline(8, color="#888888", linestyle=":", linewidth=1.1)
    for modelo in MODELO_A_FAMILIA.keys():
        sub = res[res["modelo"] == modelo].sort_values("horizonte_h")
        if sub.empty:
            continue
        ax2.plot(
            sub["horizonte_h"],
            sub["mejora_b2_pct"],
            marker="o",
            linewidth=2.0,
            color=COLORES[modelo],
            label=modelo,
        )
    ax2.set_xticks(horizontes)
    ax2.set_xlabel("Horizonte de pronóstico h (semanas)")
    ax2.set_ylabel("Mejora media en Log-Loss frente a B2 (%)")
    ax2.set_title("B. Ganancia relativa sobre la inercia del canal (B2)")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.09))
    fig.suptitle("Sensibilidad del desempeño al horizonte temporal (sin_clima, desarrollo 2019–2022)", y=1.02, fontweight="bold")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(salida / f"fig_05_sensibilidad_horizonte.{ext}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizontes", nargs="+", type=int, default=[2, 4, 8, 12])
    parser.add_argument(
        "--tabla-ganadores",
        type=Path,
        default=Path("reports/tesis_contraste_modelos/tabla_01_ranking_global_sin_clima.csv"),
    )
    parser.add_argument("--salida-datos", type=Path, default=Path("data/processed/forecasting/oscar_sensibilidad_horizonte"))
    parser.add_argument("--salida-reporte", type=Path, default=Path("reports/tesis_contraste_modelos"))
    parser.add_argument("--semilla", type=int, default=20261112)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    access = RecipeAccess.from_config(cfg)
    prod = load_channel_product(cfg.datos.canal, expected_md5=cfg.datos.version_dvc_canal)
    dev_years = range(cfg.validacion.desarrollo[0], cfg.validacion.desarrollo[1] + 1)
    features_sc = FEATURE_SETS["sin_clima"]

    recetas = cargar_recetas_ganadoras(args.tabla_ganadores)
    recetas_tab = [r for r in recetas if r["familia"] != "gru"]
    recetas_gru = [r for r in recetas if r["familia"] == "gru"]
    print(f"Dispositivo: {device} | Recetas: {[r['candidato'] for r in recetas]}", flush=True)

    rows = []
    b2_rows = []
    t_global = time.perf_counter()

    for h in args.horizontes:
        t_h = time.perf_counter()
        print(f"\n--- Evaluando horizonte h = {h} semanas ---", flush=True)

        # 1. Modelos tabulares + línea base B2
        examples = build_examples(prod, h=h, canal=cfg.canal, features=cfg.features)
        trials = yearly_trials(examples, dev_years, first_train_year=cfg.validacion.primer_anio_entrenamiento)

        b2_cache = {}
        for tr in trials:
            tr_df = examples.loc[tr.train]
            ev_df = examples.loc[tr.evaluate]
            y_ev = ev_df["y"].to_numpy(dtype=int)
            p_b2 = b2_predict(tr_df, ev_df, access=access)
            ll_b2 = log_loss(y_ev, p_b2, ev_df["target_week"], access=access, clip=cfg.metricas.recorte_log_loss)
            br_b2 = brier(y_ev, p_b2, ev_df["target_week"], access=access)
            b2_cache[(str(tr.municipality_code), int(tr.year))] = (float(ll_b2), float(br_b2))
            b2_rows.append({
                "horizonte_h": int(h),
                "municipality_code": str(tr.municipality_code),
                "year": int(tr.year),
                "logloss_b2": float(ll_b2),
                "brier_b2": float(br_b2),
            })

        for r in recetas_tab:
            imps = []
            for tr in trials:
                tr_df = examples.loc[tr.train]
                ev_df = examples.loc[tr.evaluate]
                y_ev = ev_df["y"].to_numpy(dtype=int)
                ll_b2, _ = b2_cache[(str(tr.municipality_code), int(tr.year))]

                model = fit_model(r["familia"], r["params"], tr_df, features=features_sc)
                probs = predict_positive(model, ev_df)
                ll = log_loss(y_ev, probs, ev_df["target_week"], access=access, clip=cfg.metricas.recorte_log_loss)
                br = brier(y_ev, probs, ev_df["target_week"], access=access)
                imp = improvement(ll, ll_b2)
                imps.append(imp)
                rows.append({
                    "horizonte_h": int(h),
                    "modelo": r["modelo"],
                    "familia": r["familia"],
                    "candidato": r["candidato"],
                    "municipality_code": str(tr.municipality_code),
                    "year": int(tr.year),
                    "logloss": float(ll),
                    "brier": float(br),
                    "logloss_b2": float(ll_b2),
                    "mejora_b2": float(imp),
                })
            print(f"  h={h:2d} | {r['modelo']:22s} ({r['candidato']}) -> mejora_b2={np.mean(imps)*100:+.2f}%", flush=True)

        # 2. GRU Secuencial (sin_clima, 5 canales)
        if recetas_gru:
            gru_data = build_gru_examples(prod, h=h, canal=cfg.canal, features=cfg.features)
            keys = gru_data.keys.reset_index(drop=True)
            keys["zona_t"] = [ZONES[i] for i in np.argmax(gru_data.sequence[:, -1, 2:5], axis=1)]
            trials_gru = yearly_trials(keys, dev_years, first_train_year=cfg.validacion.primer_anio_entrenamiento)

            for r in recetas_gru:
                imps = []
                for tr in trials_gru:
                    tr_idx = keys.index.get_indexer(tr.train)
                    ev_idx = keys.index.get_indexer(tr.evaluate)
                    tr_df, ev_df = keys.loc[tr.train], keys.loc[tr.evaluate]
                    y_tr = tr_df["y"].to_numpy(dtype=np.float32)
                    y_ev = ev_df["y"].to_numpy(dtype=int)
                    ll_b2, _ = b2_cache[(str(tr.municipality_code), int(tr.year))]

                    s_tr, s_ev, st_tr, st_ev = standardize_fold(
                        gru_data.sequence[tr_idx][:, :, :5],
                        gru_data.sequence[ev_idx][:, :, :5],
                        gru_data.static[tr_idx],
                        gru_data.static[ev_idx],
                        [0, 1],
                    )
                    probs = fit_and_predict("gru", r["params"], s_tr, st_tr, y_tr, s_ev, st_ev, device, args.semilla)
                    ll = log_loss(y_ev, probs, ev_df["target_week"], access=access, clip=cfg.metricas.recorte_log_loss)
                    br = brier(y_ev, probs, ev_df["target_week"], access=access)
                    imp = improvement(ll, ll_b2)
                    imps.append(imp)
                    rows.append({
                        "horizonte_h": int(h),
                        "modelo": r["modelo"],
                        "familia": r["familia"],
                        "candidato": r["candidato"],
                        "municipality_code": str(tr.municipality_code),
                        "year": int(tr.year),
                        "logloss": float(ll),
                        "brier": float(br),
                        "logloss_b2": float(ll_b2),
                        "mejora_b2": float(imp),
                    })
                print(f"  h={h:2d} | {r['modelo']:22s} ({r['candidato']}) -> mejora_b2={np.mean(imps)*100:+.2f}%", flush=True)

        print(f"  Horizonte h={h} completado en {time.perf_counter() - t_h:.1f}s", flush=True)

    # Guardar resultados crudos y resumen
    args.salida_datos.mkdir(parents=True, exist_ok=True)
    args.salida_reporte.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(args.salida_datos / "tabla_sensibilidad_horizonte.parquet", index=False)

    b2_df = pd.DataFrame(b2_rows)
    b2_res = (
        b2_df.groupby("horizonte_h", as_index=False)
        .agg(logloss_medio=("logloss_b2", "mean"), brier_medio=("brier_b2", "mean"))
    )

    res = (
        out_df.groupby(["horizonte_h", "modelo", "candidato"], as_index=False)
        .agg(
            mejora_b2_pct=("mejora_b2", lambda s: round(float(s.mean() * 100.0), 2)),
            logloss_medio=("logloss", lambda s: round(float(s.mean()), 4)),
            logloss_b2_medio=("logloss_b2", lambda s: round(float(s.mean()), 4)),
            brier_medio=("brier", lambda s: round(float(s.mean()), 4)),
            casillas_ganadas=("mejora_b2", lambda s: int((s > 0).sum())),
        )
        .sort_values(["horizonte_h", "mejora_b2_pct"], ascending=[True, False])
        .reset_index(drop=True)
    )

    res.to_csv(args.salida_reporte / "tabla_05_sensibilidad_horizonte.csv", index=False)
    try:
        md = res.to_markdown(index=False)
    except Exception:
        md = res.to_string(index=False)
    (args.salida_reporte / "tabla_05_sensibilidad_horizonte.md").write_text(md + "\n", encoding="utf-8")

    graficar_sensibilidad(res, b2_res, args.salida_reporte)

    print(f"\n=== RESUMEN SENSIBILIDAD AL HORIZONTE ({time.perf_counter() - t_global:.1f}s) ===")
    print(res.to_string(index=False))
    print("\n=== LÍNEA BASE B2 POR HORIZONTE ===")
    print(b2_res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
