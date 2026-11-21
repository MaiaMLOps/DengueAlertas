"""Log the GRU and CNN-1D results of the sequential contrast run to MLflow (análisis complementario).

Reads the single development table written by scripts/run_seq_dl.py and creates one run per (variant, family,
candidate) with its mean metrics and per-municipality metrics. The experiment is `experimento` of
configs/contraste_familias.yaml; the runs logged before this configuration existed are in the experiment whose ID is
`experimento_registrado_id`. Existing experiments and runs are never deleted or renamed. MLflow is reached only
through src/forecasting/tracking.py.

Run from the repository root:
    .venv/Scripts/python.exe -B scripts/log_seq_dl_mlflow.py --config configs/contraste_familias.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forecasting.config import (  # noqa: E402
    CONTRAST_CONFIG_PATH,
    contrast_base_config,
    load_contrast_config,
    resolve_path,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log the sequential contrast table to MLflow.")
    parser.add_argument("--config", type=Path, default=CONTRAST_CONFIG_PATH,
                        help="Family contrast configuration (configs/contraste_familias.yaml).")
    parser.add_argument("--tabla", type=Path, default=None,
                        help="Single development table (default: the sequential run's carpeta).")
    args = parser.parse_args(argv)

    import mlflow

    from src.forecasting.tracking import configure_tracking

    contrast = load_contrast_config(args.config)
    run_name_cfg, _ = contrast.secuencial
    table = args.tabla or resolve_path(contrast.corridas[run_name_cfg].carpeta) / "tabla_unica_desarrollo.parquet"
    uri = configure_tracking(contrast_base_config(contrast))
    mlflow.set_experiment(contrast.experimento)
    print(f"MLflow: {uri} | experiment: {contrast.experimento}", flush=True)

    df = pd.read_parquet(table)
    groups = list(df.groupby(["variante", "familia", "candidato", "parametros"], sort=True))
    print(f"Logging {len(groups)} sequential candidates...", flush=True)
    for idx, ((variante, familia, candidato, params_json), g) in enumerate(groups, start=1):
        run_name = f"dl_seq__{variante}__{familia}__{candidato}"
        metrics = {"mejora_b2_media": float(g["mejora_b2"].mean()), "logloss_medio": float(g["logloss"].mean()),
                   "brier_medio": float(g["brier"].mean()), "logloss_b2_medio": float(g["logloss_b2"].mean()),
                   "casillas_ganadas_b2": float((g["mejora_b2"] > 0).sum())}
        for muni, gm in g.groupby("municipality_code"):
            metrics[f"mejora_b2_{muni}"] = float(gm["mejora_b2"].mean())
            metrics[f"logloss_{muni}"] = float(gm["logloss"].mean())
        tags = {"modelo_tipo": "dl_secuencial", "familia": str(familia), "variante": str(variante),
                "candidato": str(candidato), "corrida": run_name_cfg, "analisis": "contraste_familias"}
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(tags)
            mlflow.log_params({k: str(v) for k, v in json.loads(params_json).items()})
            mlflow.log_metrics(metrics)
        if idx % 10 == 0 or idx == len(groups):
            print(f"  [{idx:3d}/{len(groups)}] {run_name} -> mejora_b2={metrics['mejora_b2_media']:+.4f}", flush=True)
    print("MLflow logging finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
