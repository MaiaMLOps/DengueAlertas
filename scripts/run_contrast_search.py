"""Tabular search of one run of the family contrast (análisis complementario de desarrollo, 2019–2022).

Reuses search.run_search unchanged: same rows, trials, baselines, metrics, manifest and MLflow logging as the
pre-registered search, with the search section of the run in configs/contraste_familias.yaml, the sin_clima feature
set and the contrast experiment (`experimento`). The output folder is the run's `carpeta`; the folders of the
pre-registered searches are never written. Nothing here feeds the selection, the frozen recipe or the test period.

Run from the repository root (optional dependencies: requirements/contraste.txt):
    .venv/Scripts/python.exe -B scripts/run_contrast_search.py --config configs/contraste_familias.yaml --corrida svm_ampliado
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.epidemiology.validation import require  # noqa: E402
from src.forecasting.config import (  # noqa: E402
    CONTRAST_CONFIG_PATH,
    contrast_base_config,
    contrast_search_config,
    load_contrast_config,
    resolve_path,
)
from src.forecasting.dataset import build_examples, load_channel_product  # noqa: E402
from src.forecasting.search import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    _dirty_paths,
    candidates_from_config,
    logger,
    run_search,
)
from src.forecasting.tracking import configure_tracking  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tabular search of one family-contrast run (development only).")
    parser.add_argument("--config", type=Path, default=CONTRAST_CONFIG_PATH,
                        help="Family contrast configuration (configs/contraste_familias.yaml).")
    parser.add_argument("--corrida", required=True, help="Name of a run with a busqueda section.")
    parser.add_argument("--permitir-cambios", action="store_true",
                        help="Run with uncommitted changes in src/ or configs/ (smoke runs only).")
    args = parser.parse_args(argv)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    try:
        contrast = load_contrast_config(args.config)
        base = contrast_base_config(contrast)
        config = contrast_search_config(base, contrast, args.corrida)
        output_dir = resolve_path(contrast.corridas[args.corrida].carpeta)
        protected = {DEFAULT_OUTPUT_DIR.resolve(), (DEFAULT_OUTPUT_DIR / "sin_clima").resolve(),
                     (DEFAULT_OUTPUT_DIR / "ablacion_clima").resolve()}
        require(output_dir.resolve() not in protected,
                f"{output_dir} is a folder of the pre-registered searches; a contrast run never writes there.")
        dirty = _dirty_paths()
        require(not dirty or args.permitir_cambios,
                "src/ or configs/ have uncommitted changes; commit them so the table matches a commit, or pass "
                f"--permitir-cambios for a smoke run. Changes: {dirty[:10]}")
        candidates = candidates_from_config(config.busqueda)
        product = load_channel_product(resolve_path(config.datos.canal), expected_md5=config.datos.version_dvc_canal)
        examples = build_examples(product, h=config.horizonte.h, canal=config.canal, features=config.features)
        uri = configure_tracking(config)
        logger.info("Contrast run %s -> %s (experiment %s).", args.corrida, output_dir,
                    config.seguimiento.experimentos.desarrollo)
        run_search(examples, candidates, config=config, output_dir=output_dir, tracking_uri=uri,
                   variante=contrast.variante)
    except KeyboardInterrupt:
        print("Search interrupted; completed units are kept. Relaunch the same command to continue.",
              file=sys.stderr)
        return 130
    except Exception as exc:   # reported with its traceback
        traceback.print_exc()
        print(f"Search stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        logger.removeHandler(console)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
