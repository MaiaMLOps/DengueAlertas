"""Entrada sin instalación: python run_pipeline.py --input ..."""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ewt_final.cli import main

if __name__ == "__main__":
    main()
