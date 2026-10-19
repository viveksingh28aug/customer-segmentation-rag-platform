"""Make the repository's local packages importable by Streamlit."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def ensure_project_root() -> None:
    """Add both the repository root and its ``src`` directory to ``sys.path``.

    Streamlit executes page files as scripts, so an uninstalled src-layout
    package is otherwise invisible even when the repository root is importable.
    Keeping this bootstrap here also makes the documented dashboard command work
    without requiring an editable install first.
    """
    for path in (PROJECT_ROOT, SRC_DIR):
        path_string = str(path)
        if path_string not in sys.path:
            sys.path.insert(0, path_string)
