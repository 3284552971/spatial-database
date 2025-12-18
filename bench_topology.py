#!/usr/bin/env python
"""Topology speed benchmark (Python vs C++).

Usage:
  python bench_topology.py --n 5000 --repeat 5

Notes:
- Uses the same feature list for both implementations.
- "Python" path is forced by temporarily disabling native module inside space_app.algorithms.topology.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_features(path: Path, n: int) -> List[Dict[str, Any]]:
    fc = json.loads(path.read_text(encoding="utf-8"))
    feats = fc.get("features") or []
    if not isinstance(feats, list):
        raise ValueError("GeoJSON FeatureCollection.features 不是 list")
    if n > 0:
        feats = feats[:n]
    return feats


def _time_one(fn, features, bbox, ndigits) -> Tuple[float, Any]:
    t0 = time.perf_counter()
    out = fn(features, bbox, ndigits)
    t1 = time.perf_counter()
    return (t1 - t0), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        type=str,
        default="space_app/shenzhen/road_sz.geojson",
        help="GeoJSON 路径（默认：space_app/shenzhen/road_sz.geojson）",
    )
    ap.add_argument("--n", type=int, default=2000, help="使用前 N 个要素（0 表示全部）")
    ap.add_argument("--repeat", type=int, default=5, help="重复次数（取平均）")
    ap.add_argument("--warmup", type=int, default=1, help="预热次数（不计入统计）")
    ap.add_argument("--ndigits", type=int, default=6)
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"找不到文件：{path}")

    features = _load_features(path, args.n)

    from space_app.algorithms import topology

    print(f"数据：{path}  features={len(features)}  repeat={args.repeat}  warmup={args.warmup}")

    # Native (C++)
    if not getattr(topology, "_NATIVE_AVAILABLE", False) or getattr(topology, "_topology_cpp", None) is None:
        raise SystemExit("native 扩展不可用：请先编译 space_app/cpp_model/build 并确保 topology_cpp 可加载")

    native_fn = topology._topology_cpp.check_topology_layer

    def python_fn(feats, bbox, nd):
        prev_avail = topology._NATIVE_AVAILABLE
        prev_mod = topology._topology_cpp
        try:
            topology._NATIVE_AVAILABLE = False
            topology._topology_cpp = None
            return topology.check_topology_layer(feats, bbox, nd)
        finally:
            topology._NATIVE_AVAILABLE = prev_avail
            topology._topology_cpp = prev_mod

    # Warmup
    for _ in range(args.warmup):
        _time_one(native_fn, features, None, args.ndigits)
        _time_one(python_fn, features, None, args.ndigits)

    gc.collect()

    native_times: List[float] = []
    py_times: List[float] = []

    native_issues = None
    py_issues = None

    for _ in range(args.repeat):
        dt_n, out_n = _time_one(native_fn, features, None, args.ndigits)
        native_times.append(dt_n)
        if native_issues is None:
            native_issues = len((out_n or {}).get("issues") or [])

        dt_p, out_p = _time_one(python_fn, features, None, args.ndigits)
        py_times.append(dt_p)
        if py_issues is None:
            issues, _stats = out_p
            py_issues = len(issues)

    def fmt(times: List[float]) -> str:
        mean = statistics.mean(times)
        p50 = statistics.median(times)
        pmin = min(times)
        pmax = max(times)
        return f"mean={mean:.4f}s  median={p50:.4f}s  min={pmin:.4f}s  max={pmax:.4f}s"

    print("\n结果（越小越好）：")
    print(f"- C++  : {fmt(native_times)}  issues={native_issues}")
    print(f"- Python: {fmt(py_times)}  issues={py_issues}")

    mean_cpp = statistics.mean(native_times)
    mean_py = statistics.mean(py_times)
    speed_py_over_cpp = mean_py / max(1e-12, mean_cpp)
    speed_cpp_over_py = mean_cpp / max(1e-12, mean_py)
    print(f"\n加速比：")
    print(f"- Python/C++：{speed_py_over_cpp:.2f}x（<1 表示 Python 更快）")
    print(f"- C++/Python：{speed_cpp_over_py:.2f}x（<1 表示 C++ 更快）")

    if native_issues != py_issues:
        print("注意：两种实现 issues 数量不同，说明算法细节或去重策略存在差异（但速度对比仍然直观有效）。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
