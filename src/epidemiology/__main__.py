"""Explicit local entry point; output directory is always supplied by the user."""

import argparse
from pathlib import Path
import sys

from .config import DEFAULT_INPUT, EpidemiologyValidationError
from .pipeline import export_labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preliminary retrospective EWT and weekly excess labels.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = export_labels(args.output_dir, args.input)
    except (EpidemiologyValidationError, FileExistsError) as exc:
        message = str(exc) if isinstance(exc, EpidemiologyValidationError) else "Output exists; choose a new directory."
        print(f"Epidemiology stopped: {message}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Epidemiology failed ({type(exc).__name__}); no completion certified; inspect locally.", file=sys.stderr)
        return 1
    result = manifest["diagnostics"]["validation"]
    print(f"Preliminary product: {result['rows']} rows, {result['columns']} columns; "
          f"{result['eligible']} eligible, {result['ineligible']} ineligible.")
    print(f"Output: {manifest['output']['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
