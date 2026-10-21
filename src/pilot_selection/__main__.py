"""Run from the repository root with python -m src.pilot_selection."""

import argparse
from pathlib import Path
import sys

from .config import AnalysisValidationError, RunConfig
from .reporting import run_analysis


def main(argv: list[str] | None = None) -> int:
    defaults = RunConfig()
    parser = argparse.ArgumentParser(description="Descriptive retrospective evidence for team-selected dengue pilots.")
    parser.add_argument("--input", type=Path, default=defaults.input)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--figures", action="store_true", help="Also export the approved 3 x 2 PNG/SVG comparison.")
    args = parser.parse_args(argv)
    try:
        result = run_analysis(RunConfig(**vars(args)))
    except (AnalysisValidationError, FileExistsError, FileNotFoundError) as exc:
        # Only messages constructed by our validation are surfaced; filesystem errors may contain private paths.
        if isinstance(exc, OSError):
            print("Analysis stopped: input unavailable or output already exists; choose a new destination.", file=sys.stderr)
        else:
            print(f"Analysis stopped: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Analysis failed ({type(exc).__name__}); no new completed run is certified. Inspect locally.", file=sys.stderr)
        return 1
    print(f"Analysis passed: {result['controls']['complete']['municipalities']} municipalities.")
    print(f"Output: {result['output_dir']}")
    print(f"Artifacts: {len(result['artifacts'])}; completion marker: run_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
