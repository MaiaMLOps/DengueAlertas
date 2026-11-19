"""Sensibilidad de horizonte (h in {2, 4, 8, 12}) y ablación climática para las 8 familias.

Reutiliza el parquet previo de las 5 familias ya evaluadas, calcula únicamente las faltantes
(LightGBM, XGBoost y CNN-1D), regenera tabla_05/fig_05 y tabla_06/fig_06 con las 8 familias
y registra solo las corridas faltantes en MLflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
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
from src.forecasting.tracking import configure_tracking

MODELO_A_FAMILIA = {
    "Regresión Logística L2": "logistica",
    "SVM (RBF)": "svm",
    "MLP Tabular": "mlp",
    "CatBoost": "catboost",
    "LightGBM": "lightgbm",
    "XGBoost": "xgboost",
    "GRU Secuencial": "gru",
    "CNN-1D Secuencial": "cnn1d",
}

COLORES = {
    "Regresión Logística L2": "#1b4f72",
    "SVM (RBF)": "#2874a6",
    "MLP Tabular": "#117864",
    "CatBoost": "#b9770e",
    "LightGBM": "#d35400",
    "XGBoost": "#7d3c98",
    "GRU Secuencial": "#6c3483",
    "CNN-1D Secuencial": "#922b21",
}


def buscar_params_en_manifiesto(candidato: str) -> dict | None:
    rutas = [
        Path("data/processed/forecasting/sin_clima/busqueda_manifest.json"),
        Path("data/processed/forecasting/busqueda_manifest.json"),
    ]
    def _buscar(obj):
        if isinstance(obj, dict):
            if obj.get("candidato") == candidato or obj.get("id") == candidato or obj.get("name") == candidato:
                for k in ("params", "parametros", "hyperparameters"):
                    if isinstance(obj.get(k), dict):
                        return dict(obj[k])
            if candidato in obj and isinstance(obj[candidato], dict):
                val = obj[candidato]
                return dict(val.get("params", val.get("parametros", val)))
            for v in obj.values():
                res = _buscar(v)
                if res is not None:
                    return res
        elif isinstance(obj, list):
            for item in obj:
                res = _buscar(item)
                if res is not None:
                    return res
        return None

    for ruta in rutas:
        if ruta.exists():
            data = json.loads(ruta.read_text(encoding="utf-8"))
            res = _buscar(data)
            if res is not None:
                return res
    return None


def cargar_todas_las_recetas(tabla_global: Path) -> list[dict]:
    df = pd.read_csv(tabla_global)
    recetas = []
    for _, row in df.iterrows():
        modelo = str(row["modelo"]).strip()
        if modelo not in MODELO_A_FAMILIA:
            continue
        fam = MODELO_A_FAMILIA[modelo]
        cand = str(row["candidato"]).strip()
        params = json.loads(str(row["parametros"]))
        if fam not in ("gru", "cnn1d"):
            man_params = buscar_params_en_manifiesto(cand)
            if man_params:
                man_params.update(params)
                params = man_params
        if fam == "xgboost":
            params.setdefault("reg_alpha", 0.0)
            params.setdefault("gamma", 0.0)
        recetas.append({
            "modelo": modelo,
            "familia": fam,
            "candidato": cand,
            "params": params,
        })
    return recetas


def guardar_tabla(df: pd.DataFrame, salida: Path, nombre: str) -> None:
    df.to_csv(salida / f"{nombre}.csv", index=False)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = df.to_string(index=False)
    (salida / f"{nombre}.md").write_text(md + "\n", encoding="utf-8")


def graficar_sensibilidad(res_h: pd.DataFrame, b2_res: pd.DataFrame, salida: Path) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    horizontes = sorted(res_h["horizonte_h"].unique())
    modelos = [m for m in MODELO_A_FAMILIA if m in set(res_h["modelo"])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax1.axhline(0, color="#222222", linewidth=1.0, linestyle="--", label="Línea base B2 (0.0%)")

    for m in modelos:
        sub = res_h[res_h["modelo"] == m].sort_values("horizonte_h")
        ax1.plot(sub["horizonte_h"], sub["mejora_b2_pct"], marker="o", linewidth=2.0, color=COLORES[m], label=m)

    ax1.set_xticks(horizontes)
    ax1.set_xticklabels([f"h = {h}" for h in horizontes])
    ax1.set_xlabel("Horizonte de pronóstico (semanas)")
    ax1.set_ylabel("Mejora relativa frente a B2 (%)")
    ax1.set_title("A. Mejora porcentual sobre B2 por horizonte (sin_clima)")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    b2_s = b2_res.sort_values("horizonte_h")
    ax2.plot(b2_s["horizonte_h"], b2_s["logloss_medio"], marker="s", linewidth=2.2, linestyle="--", color="#222222", label="Línea base B2")
    for m in modelos:
        sub = res_h[res_h["modelo"] == m].sort_values("horizonte_h")
        ax2.plot(sub["horizonte_h"], sub["logloss_medio"], marker="o", linewidth=2.0, color=COLORES[m], label=m)

    ax2.set_xticks(horizontes)
    ax2.set_xticklabels([f"h = {h}" for h in horizontes])
    ax2.set_xlabel("Horizonte de pronóstico (semanas)")
    ax2.set_ylabel("Log-loss medio (menor es mejor)")
    ax2.set_title("B. Log-loss absoluto por horizonte (2019–2022)")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(frameon=True, fontsize=8.2, ncol=2, loc="upper left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for ext in ("png", "pdf"):
        fig.savefig(salida / f"fig_05_sensibilidad_horizonte.{ext}")
    plt.close(fig)


def graficar_clima_por_horizonte(piv: pd.DataFrame, salida: Path) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    modelos = ["Regresión Logística L2", "SVM (RBF)", "MLP Tabular", "CatBoost", "LightGBM", "XGBoost"]
    horizontes = sorted(piv["horizonte_h"].unique())
    x = np.arange(len(horizontes))
    w = 0.13

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.axhline(0, color="#222222", linewidth=1.0, linestyle="--")

    for idx, modelo in enumerate(modelos):
        sub = piv[piv["modelo"] == modelo].sort_values("horizonte_h")
        if sub.empty:
            continue
        offset = (idx - 2.5) * w
        bars = ax.bar(x + offset, sub["impacto_clima_pp"], width=w, color=COLORES[modelo], edgecolor="#222222", linewidth=0.4, label=modelo)
        for bar in bars:
            h_val = bar.get_height()
            va = "top" if h_val < 0 else "bottom"
            dy = -0.9 if h_val < 0 else 0.6
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h_val + dy,
                f"{h_val:.1f}",
                ha="center",
                va=va,
                fontsize=7.0,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"h = {h} sem." for h in horizontes])
    ax.set_ylabel("Cambio en mejora frente a B2 al incluir clima (pp)")
    ax.set_title("Impacto de las variables climáticas por horizonte (completo vs. sin_clima, 2019–2022)")
    ax.set_ylim(piv["impacto_clima_pp"].min() - 4.5, max(3.5, piv["impacto_clima_pp"].max() + 3.5))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(frameon=True, loc="lower left", ncol=3, fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for ext in ("png", "pdf"):
        fig.savefig(salida / f"fig_06_clima_por_horizonte.{ext}")
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

    cfg = load_config()
    uri = configure_tracking(cfg)
    exp_cfg = getattr(cfg.seguimiento, "experimentos", None)
    exp_name = getattr(exp_cfg, "desarrollo", None) or "dengue_h8_oscar_contraste"
    mlflow.set_experiment(exp_name)
    print(f"MLflow conectado: {uri} | Experimento: {exp_name}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    access = RecipeAccess.from_config(cfg)
    prod = load_channel_product(cfg.datos.canal, expected_md5=cfg.datos.version_dvc_canal)
    dev_years = range(cfg.validacion.desarrollo[0], cfg.validacion.desarrollo[1] + 1)

    recetas = cargar_todas_las_recetas(args.tabla_ganadores)
    cache_path = args.salida_datos / "tabla_sensibilidad_horizonte_y_clima.parquet"
    if cache_path.exists():
        prev_df = pd.read_parquet(cache_path)
        ya_hechas = set(prev_df["familia"].unique())
        print(f"Reutilizando caché previa ({len(prev_df)} filas) con familias: {sorted(ya_hechas)}", flush=True)
    else:
        prev_df = pd.DataFrame()
        ya_hechas = set()

    pendientes = [r for r in recetas if r["familia"] not in ya_hechas]
    recetas_tab = [r for r in pendientes if r["familia"] not in ("gru", "cnn1d")]
    recetas_seq = [r for r in pendientes if r["familia"] in ("gru", "cnn1d")]
    print(f"Dispositivo: {device} | Familias a ejecutar ahora: {[r['candidato'] for r in pendientes]}", flush=True)

    rows = []
    b2_rows = []
    t0 = time.perf_counter()

    for h in args.horizontes:
        print(f"\n--- Horizonte h = {h} semanas ---", flush=True)
        examples = build_examples(prod, h=h, canal=cfg.canal, features=cfg.features)
        trials = yearly_trials(examples, dev_years, first_train_year=cfg.validacion.primer_anio_entrenamiento)

        b2_cache = {}
        for tr in trials:
            tr_df, ev_df = examples.loc[tr.train], examples.loc[tr.evaluate]
            y_ev = ev_df["y"].to_numpy(dtype=int)
            p_b2 = b2_predict(tr_df, ev_df, access=access)
            ll_b2 = float(log_loss(y_ev, p_b2, ev_df["target_week"], access=access, clip=cfg.metricas.recorte_log_loss))
            br_b2 = float(brier(y_ev, p_b2, ev_df["target_week"], access=access))
            b2_cache[(str(tr.municipality_code), int(tr.year))] = (ll_b2, br_b2)
            b2_rows.append({"horizonte_h": int(h), "municipality_code": str(tr.municipality_code), "year": int(tr.year), "logloss_b2": ll_b2, "brier_b2": br_b2})

        for variante in ("sin_clima", "completo"):
            feats = FEATURE_SETS[variante]
            for r in recetas_tab:
                for tr in trials:
                    tr_df, ev_df = examples.loc[tr.train], examples.loc[tr.evaluate]
                    y_ev = ev_df["y"].to_numpy(dtype=int)
                    ll_b2, _ = b2_cache[(str(tr.municipality_code), int(tr.year))]
                    model = fit_model(r["familia"], r["params"], tr_df, features=feats)
                    probs = predict_positive(model, ev_df)
                    ll = float(log_loss(y_ev, probs, ev_df["target_week"], access=access, clip=cfg.metricas.recorte_log_loss))
                    br = float(brier(y_ev, probs, ev_df["target_week"], access=access))
                    rows.append({
                        "horizonte_h": int(h),
                        "variante": variante,
                        "modelo": r["modelo"],
                        "familia": r["familia"],
                        "candidato": r["candidato"],
                        "params": json.dumps(r["params"], sort_keys=True),
                        "municipality_code": str(tr.municipality_code),
                        "year": int(tr.year),
                        "logloss": ll,
                        "brier": br,
                        "logloss_b2": ll_b2,
                        "mejora_b2": float(improvement(ll, ll_b2)),
                    })

        if recetas_seq:
            seq_data = build_gru_examples(prod, h=h, canal=cfg.canal, features=cfg.features)
            keys = seq_data.keys.reset_index(drop=True)
            keys["zona_t"] = [ZONES[i] for i in np.argmax(seq_data.sequence[:, -1, 2:5], axis=1)]
            trials_seq = yearly_trials(keys, dev_years, first_train_year=cfg.validacion.primer_anio_entrenamiento)

            for r in recetas_seq:
                for tr in trials_seq:
                    tr_idx = keys.index.get_indexer(tr.train)
                    ev_idx = keys.index.get_indexer(tr.evaluate)
                    tr_df, ev_df = keys.loc[tr.train], keys.loc[tr.evaluate]
                    y_tr = tr_df["y"].to_numpy(dtype=np.float32)
                    y_ev = ev_df["y"].to_numpy(dtype=int)
                    ll_b2, _ = b2_cache[(str(tr.municipality_code), int(tr.year))]
                    s_tr, s_ev, st_tr, st_ev = standardize_fold(
                        seq_data.sequence[tr_idx][:, :, :5],
                        seq_data.sequence[ev_idx][:, :, :5],
                        seq_data.static[tr_idx],
                        seq_data.static[ev_idx],
                        [0, 1],
                    )
                    probs = fit_and_predict(r["familia"], r["params"], s_tr, st_tr, y_tr, s_ev, st_ev, device, args.semilla)
                    ll = float(log_loss(y_ev, probs, ev_df["target_week"], access=access, clip=cfg.metricas.recorte_log_loss))
                    br = float(brier(y_ev, probs, ev_df["target_week"], access=access))
                    rows.append({
                        "horizonte_h": int(h),
                        "variante": "sin_clima",
                        "modelo": r["modelo"],
                        "familia": r["familia"],
                        "candidato": r["candidato"],
                        "params": json.dumps(r["params"], sort_keys=True),
                        "municipality_code": str(tr.municipality_code),
                        "year": int(tr.year),
                        "logloss": ll,
                        "brier": br,
                        "logloss_b2": ll_b2,
                        "mejora_b2": float(improvement(ll, ll_b2)),
                    })

    args.salida_datos.mkdir(parents=True, exist_ok=True)
    args.salida_reporte.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(rows)
    out_df = pd.concat([prev_df, new_df], ignore_index=True) if not prev_df.empty else new_df
    out_df.to_parquet(args.salida_datos / "tabla_sensibilidad_horizonte_y_clima_8familias.parquet", index=False)

    b2_df = pd.DataFrame(b2_rows)
    b2_res = b2_df.groupby("horizonte_h", as_index=False).agg(logloss_medio=("logloss_b2", "mean"), brier_medio=("brier_b2", "mean"))

    sc_df = out_df[out_df["variante"] == "sin_clima"]
    res_h = (
        sc_df.groupby(["horizonte_h", "modelo", "candidato"], as_index=False)
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
    guardar_tabla(res_h, args.salida_reporte, "tabla_05_sensibilidad_horizonte")
    graficar_sensibilidad(res_h, b2_res, args.salida_reporte)

    tab_df = out_df[out_df["familia"].isin(["logistica", "svm", "mlp", "catboost", "lightgbm", "xgboost"])]
    agg_clima = (
        tab_df.groupby(["horizonte_h", "modelo", "candidato", "variante"], as_index=False)
        .agg(mejora_b2_pct=("mejora_b2", lambda s: round(float(s.mean() * 100.0), 2)))
    )
    piv = agg_clima.pivot(index=["horizonte_h", "modelo", "candidato"], columns="variante", values="mejora_b2_pct").reset_index()
    piv["impacto_clima_pp"] = (piv["completo"] - piv["sin_clima"]).round(2)
    piv = piv.rename(columns={"sin_clima": "mejora_sin_clima_pct", "completo": "mejora_completo_pct"})
    piv = piv[["horizonte_h", "modelo", "candidato", "mejora_sin_clima_pct", "mejora_completo_pct", "impacto_clima_pp"]].sort_values(
        ["horizonte_h", "mejora_sin_clima_pct"], ascending=[True, False]
    )
    guardar_tabla(piv, args.salida_reporte, "tabla_06_clima_por_horizonte")
    graficar_clima_por_horizonte(piv, args.salida_reporte)

    if not new_df.empty:
        grupos = list(new_df.groupby(["horizonte_h", "variante", "familia", "modelo", "candidato", "params"], sort=True))
        print(f"\nRegistrando {len(grupos)} corridas faltantes en MLflow...", flush=True)
        for (h, var, fam, mod, cand, p_json), g in grupos:
            run_name = f"horizonte_h{h:02d}__{var}__{fam}__{cand}"
            metrics = {
                "mejora_b2_media": float(g["mejora_b2"].mean()),
                "logloss_medio": float(g["logloss"].mean()),
                "brier_medio": float(g["brier"].mean()),
                "logloss_b2_medio": float(g["logloss_b2"].mean()),
                "casillas_ganadas_b2": float((g["mejora_b2"] > 0).sum()),
            }
            for muni, gm in g.groupby("municipality_code"):
                metrics[f"mejora_b2_{muni}"] = float(gm["mejora_b2"].mean())
                metrics[f"logloss_{muni}"] = float(gm["logloss"].mean())

            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags({
                    "modelo_tipo": "sensibilidad_horizonte",
                    "horizonte_h": str(h),
                    "variante": str(var),
                    "familia": str(fam),
                    "modelo": str(mod),
                    "candidato": str(cand),
                })
                mlflow.log_params({**json.loads(p_json), "horizonte_h": str(h), "variante": str(var)})
                mlflow.log_metrics(metrics)
        print("Registro en MLflow completado.", flush=True)

    print(f"\n=== TABLA 05 (8 FAMILIAS, SIN CLIMA) ({time.perf_counter() - t0:.1f}s) ===")
    print(res_h.to_string(index=False))
    print("\n=== TABLA 06 (6 FAMILIAS TABULARES, IMPACTO DEL CLIMA) ===")
    print(piv.to_string(index=False))


if __name__ == "__main__":
    main()
