from __future__ import annotations

import atexit
import os
import shutil
import sys
from pathlib import Path
from typing import Iterator, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = PROJECT_ROOT / ".tmp"
TMP_LAYERS_ROOT = TMP_ROOT / "layers"


def tmp_table_dir(owner: str, table: str) -> Path:
    return TMP_LAYERS_ROOT / str(owner) / str(table)


def is_tmp_table_path(path: Path) -> bool:
    try:
        p = Path(str(path)).resolve()
        return str(p).startswith(str(TMP_LAYERS_ROOT.resolve()))
    except Exception:
        return False


def iter_tmp_tables() -> Iterator[Tuple[str, str, Path]]:
    """Yield (owner, table, dir) for tmp tables that have data_table.pkl."""
    root = TMP_LAYERS_ROOT
    if not root.exists():
        return
    for udir in root.iterdir():
        if not udir.is_dir():
            continue
        owner = udir.name
        for tdir in udir.iterdir():
            if tdir.is_dir() and (tdir / "data_table.pkl").exists():
                yield owner, tdir.name, tdir


def _rm_tree(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def init_tmp_lifecycle_cleanup() -> None:
    """Delete repo-level .tmp on startup and on process exit.

    Guarded to avoid running in Django's autoreload parent process.
    """
    run_main = os.environ.get("RUN_MAIN")
    if run_main != "true" and any(arg == "runserver" for arg in sys.argv):
        return

    _rm_tree(TMP_ROOT)
    TMP_LAYERS_ROOT.mkdir(parents=True, exist_ok=True)
    atexit.register(lambda: _rm_tree(TMP_ROOT))

