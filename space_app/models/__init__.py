"""Model package bootstrap and backward-compat shims for pickled data."""

import sys

from . import table as _table
from . import index as _index

# Provide legacy module names so older pickles (saved when modules were imported as
# `table` / `index`) can still be unpickled after we package-qualify imports.
sys.modules.setdefault("table", _table)
sys.modules.setdefault("index", _index)

__all__ = ["_table", "_index"]
