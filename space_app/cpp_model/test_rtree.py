"""Functional tests for rtree_module bindings.

The bound methods return a list of dict with shape:
{
    "id": int,
    "type": "point"|"polyline"|"polygon",
    "bbox": {"minx": float, "miny": float, "maxx": float, "maxy": float},
    "attributes": {<string>: <string>}
}
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"

# Ensure Python can import the compiled extension from the build dir.
sys.path.insert(0, str(BUILD))

import importlib.util
import sys
import sysconfig
from pathlib import Path


def _load_rtree_module():
    build_dir = Path(__file__).resolve().parent / "build"
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    so_path = build_dir / f"rtree_module{ext}"
    if not so_path.exists():
        raise ImportError(f"rtree_module build not found: {so_path}")
    spec = importlib.util.spec_from_file_location("rtree_module", so_path)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to create spec for rtree_module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rtree_module"] = module
    spec.loader.exec_module(module)
    return module


rtree_module = _load_rtree_module()  # type: ignore


class RTreeBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = rtree_module.RTreeIndex()
        # Seed a few geometries with unique attribute "name".
        self.index.insert("point", [[0.0, 0.0]], {"name": "pt"})
        self.index.insert("polyline", [[0.0, 0.0], [1.0, 1.0]], {"name": "ln"})
        self.index.insert(
            "polygon",
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
            {"name": "pg"},
        )

    def test_box_query_with_type_filter(self) -> None:
        results = self.index.query_box(-1, -1, 2, 2, "polyline")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["attributes"].get("name"), "ln")

    def test_circle_query_hits_point(self) -> None:
        results = self.index.query_circle(0.0, 0.0, 0.5, "point")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["type"], "point")

    def test_delete_by_attribute(self) -> None:
        deleted = self.index.delete_by_attribute("name", "ln")
        self.assertTrue(deleted)
        # The removed line should no longer be returned.
        results = self.index.query_box(-1, -1, 2, 2, "polyline")
        self.assertEqual(len(results), 0)

    def test_delete_nonexistent_attribute_returns_false(self) -> None:
        self.assertFalse(self.index.delete_by_attribute("name", "nope"))


if __name__ == "__main__":
    unittest.main()
