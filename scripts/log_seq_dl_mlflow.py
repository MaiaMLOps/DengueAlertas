"""Registra en MLflow los resultados de GRU y CNN-1D desde tabla_unica_desarrollo.parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd

from src.forecasting.config import load_config
from src.forecasting.tracking import configure_tracking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tabla",
        type=Path,
        default=Path("data/processed/forecasting/oscar_dl_secuencial/tabla_unica_desarrollo.parquet"),
    )
    parser.add_argument("--experimento", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    uri = configure_tracking(cfg)

    exp_cfg = getattr(cfg.seguimiento, "experimentos", None)
    exp_name = args.experimento or getattr(exp_cfg, "desarrollo", None) or "dengue_h8_oscar_contraste"
    mlflow.set_experiment(exp_name)
    print(f"Conectado a MLflow: {uri} | Experimento: {exp_name}", flush=True)

    df = pd.read_parquet(args.tabla)
    grupos = list(df.groupby(["variante", "familia", "candidato", "parametros"], sort=True))
    total = len(grupos)
    print(f"Registrando {total} candidatos secuenciales...", flush=True)

    for idx, ((variante, familia, candidato, params_json), g) in enumerate(grupos, start=1):
        run_name = f"dl_seq__{variante}__{familia}__{candidato}"
        params = json.loads(params_json)

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

        tags = {
            "modelo_tipo": "dl_secuencial",
            "familia": str(familia),
            "variante": str(variante),
            "candidato": str(candidato),
            "origen": str(args.tabla),
        }

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(tags)
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics(metrics)

        if idx % 10 == 0 or idx == total:
            print(f"  [{idx:3d}/{total}] {run_name} -> mejora_b2={metrics['mejora_b2_media']:+.4f}", flush=True)

    print("Registro en MLflow completado.", flush=True)


if __name__ == "__main__":
    main()
