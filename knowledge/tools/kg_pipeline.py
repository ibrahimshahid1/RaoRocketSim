"""Compatibility entry point; the CLI and this script use the same pipeline."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kg.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
