"""Run from repository root: python -m src.data --help."""

import argparse
from pathlib import Path
import sys

from .config import PipelineConfig
from .pipeline import run_pipeline


def main() -> int:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Integrate the authorized retrospective historical screening panel, without labels or lags.")
    for name in ["sivigila_dir", "covariates", "output", "report"]:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=getattr(defaults, name))
    args = parser.parse_args()
    try:
        report = run_pipeline(PipelineConfig(**vars(args)))
    except (ValueError, FileExistsError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Integration passed: {report['output']['rows']} rows, {report['output']['columns']} columns.")
    print(f"Dataset: {report['configuration']['output']}")
    print(f"Quality report: {report['configuration']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
