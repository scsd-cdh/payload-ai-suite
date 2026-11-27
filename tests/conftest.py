"""Shared pytest setup.

Modules in src/ import each other by bare name (e.g. `import preprocess`),
so src/ is put on the import path here instead of turning it into a package.
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
