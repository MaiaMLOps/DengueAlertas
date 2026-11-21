"""CLI del pipeline EWT final."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_pipeline


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Pipeline EWT: procesar, entrenar, validar, probar y publicar")
    command.add_argument("--input", required=True, help="Panel semanal .parquet o .csv")
    command.add_argument("--config", default="configs/pipeline.yaml", help="Configuración YAML")
    command.add_argument("--output", default="outputs/final_run", help="Directorio de salida")
    command.add_argument("--no-mlflow", action="store_true", help="Ejecuta sin intentar publicación en MLflow")
    return command


def main() -> None:
    args = parser().parse_args()
    result = run_pipeline(
        input_path=Path(args.input),
        config_path=Path(args.config),
        output_dir=Path(args.output),
        enable_mlflow=not args.no_mlflow,
    )
    print(json.dumps({"output_dir": result["output_dir"], "mlflow": result["mlflow"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
