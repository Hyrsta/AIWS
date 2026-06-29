"""Pytest path setup shared by every test in this directory.

Puts the repo root and ``scripts/`` on ``sys.path`` so tests can do
``from gui.backend import app`` and ``import cadrille_seeding`` without each
file hand-rolling its own ``sys.path`` insertion. This is what lets CI use
plain ``pytest tests/`` discovery (new test files run automatically) instead of
a hand-maintained file allowlist.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

for path in (REPO_ROOT, SCRIPTS_DIR):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)
