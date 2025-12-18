#!/usr/bin/env python
"""Repair speed & consistency benchmark (Python vs C++).

Usage:
  python bench_repair.py --n 2000 --repeat 3

What it measures
- Time to run repair_layer (including its internal iterative noding loops).
- Basic consistency signals:
  - deleted/new_lines counts
  - after-repair topology issue counts (dangling/cross/t)

Notes
- C++ repair currently implements noding-only (cross_without_node + endpoint_on_segment).
  Python repair additionally does dangling trim/connect/dangling delete (depending on env vars).
  To make outputs comparable, you can disable those in Python:
    export SPACE_TOPOLOGY_DANGLING_TRIM_M=0
    export SPACE_TOPOLOGY_DANGLING_CONNECT_M=0
    export SPACE_TOPOLOGY_REPAIR_BACKEND=python

- For C++ run:
    export SPACE_TOPOLOGY_REPAIR_BACKEND=cpp

"""

from __future__ import annotations

import argparse
import gc
import json
import os
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


def _time_one(fn, *args) -> Tuple[float, Any]:
    t0 = time.perf_counter()
    out = fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0), out


def _apply_repair(features: List[Dict[str, Any]], rep: Dict[str, Any]) -> List[Dict[str, Any]]:
    deleted = set(rep.get("deleted") or [])
    keep = [f for f in features if f.get("id") not in deleted]
    new_lines = rep.get("new_lines") or []
    return keep + list(new_lines)


def _issue_counts(issues) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for it in issues:
        k = getattr(it, "kind", None)
        if not k:
            continue
        d[k] = d.get(k, 0) + 1
    return d


def _round_half_away(v: float, ndigits: int) -> float:
    scale = 10.0 ** float(ndigits)
    x = float(v) * scale
    if x >= 0.0:
        r = int(x + 0.5)
    else:
        r = int(x - 0.5)
    return float(r) / scale


def _geom_sig(geom: Dict[str, Any], ndigits: int) -> Any:
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if gtype == "LineString" and isinstance(coords, list):
        out = []
        for pt in coords:
            if not (isinstance(pt, (list, tuple)) and len(pt) >= 2):
                continue
            out.append((_round_half_away(float(pt[0]), ndigits), _round_half_away(float(pt[1]), ndigits)))
        return ("LineString", tuple(out))
    if gtype == "MultiLineString" and isinstance(coords, list):
        parts = []
        for line in coords:
            if not isinstance(line, list):
                continue
            out = []
            for pt in line:
                if not (isinstance(pt, (list, tuple)) and len(pt) >= 2):
                    continue
                out.append((_round_half_away(float(pt[0]), ndigits), _round_half_away(float(pt[1]), ndigits)))
            parts.append(tuple(out))
        return ("MultiLineString", tuple(parts))
    return (str(gtype), None)


def _feature_sig(f: Dict[str, Any], ndigits: int) -> Any:
    props = f.get("properties") or {}
    if not isinstance(props, dict):
        props = {}
    note = props.get("note")
    # 对于 repaired_split 等内部产物，不同后端的 id/命名策略可能不同，
    # 为避免 benchmark 产生“假差异”，只在 dangling_connect 上比较 source/target。
    note_s = str(note)
    if note_s == "dangling_connect":
        source_id = props.get("source_id")
        target_id = props.get("target_id")
    else:
        source_id = None
        target_id = None
    geom = f.get("geometry") or {}
    if not isinstance(geom, dict):
        geom = {}
    return (note_s, str(source_id), str(target_id), _geom_sig(geom, ndigits))


def _sig_set_from_features(features: List[Dict[str, Any]], ndigits: int) -> set:
    out = set()
    for f in features or []:
        if not isinstance(f, dict):
            continue
        try:
            out.add(_feature_sig(f, ndigits))
        except Exception:
            continue
    return out


def _print_set_diff(title: str, a: set, b: set, limit: int = 20) -> None:
    only_a = sorted(list(a - b))[:limit]
    only_b = sorted(list(b - a))[:limit]
    if not only_a and not only_b:
        return
    print(f"\n差异定位：{title}")
    if only_a:
        print(f"- only_in_a({len(a-b)}): {only_a}")
    if only_b:
        print(f"- only_in_b({len(b-a)}): {only_b}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        type=str,
        default="space_app/shenzhen/road_sz.geojson",
        help="GeoJSON 路径（默认：space_app/shenzhen/road_sz.geojson）",
    )
    ap.add_argument("--n", type=int, default=2000, help="使用前 N 个要素（0 表示全部）")
    ap.add_argument("--repeat", type=int, default=3, help="重复次数（取平均）")
    ap.add_argument("--warmup", type=int, default=1, help="预热次数（不计入统计）")
    ap.add_argument("--ndigits", type=int, default=6)
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"找不到文件：{path}")

    features = _load_features(path, args.n)

    from space_app.algorithms import topology

    if not getattr(topology, "_NATIVE_AVAILABLE", False) or getattr(topology, "_topology_cpp", None) is None:
        raise SystemExit("native 扩展不可用：请先编译 space_app/cpp_model/build 并确保 topology_cpp 可加载")

    print(f"数据：{path}  features={len(features)}  repeat={args.repeat}  warmup={args.warmup}")

    def run_python(feats: List[Dict[str, Any]]):
        os.environ["SPACE_TOPOLOGY_REPAIR_BACKEND"] = "python"
        issues, _stats = topology.check_topology_layer(feats, None, args.ndigits)
        return topology.repair_layer(feats, issues, 100.0, args.ndigits)

    def run_cpp(feats: List[Dict[str, Any]]):
        os.environ["SPACE_TOPOLOGY_REPAIR_BACKEND"] = "cpp"
        issues, _stats = topology.check_topology_layer(feats, None, args.ndigits)
        return topology.repair_layer(feats, issues, 100.0, args.ndigits)

    # Warmup
    for _ in range(args.warmup):
        _time_one(run_cpp, features)
        _time_one(run_python, features)

    gc.collect()

    t_cpp: List[float] = []
    t_py: List[float] = []

    rep_cpp = None
    rep_py = None

    for _ in range(args.repeat):
        dt, out = _time_one(run_cpp, features)
        t_cpp.append(dt)
        rep_cpp = out

        dt, out = _time_one(run_python, features)
        t_py.append(dt)
        rep_py = out

    def fmt(times: List[float]) -> str:
        mean = statistics.mean(times)
        p50 = statistics.median(times)
        pmin = min(times)
        pmax = max(times)
        return f"mean={mean:.4f}s  median={p50:.4f}s  min={pmin:.4f}s  max={pmax:.4f}s"

    print("\n速度（越小越好）：")
    print(f"- C++   : {fmt(t_cpp)}")
    print(f"- Python: {fmt(t_py)}")

    mean_cpp = statistics.mean(t_cpp)
    mean_py = statistics.mean(t_py)
    print("\n加速比：")
    print(f"- Python/C++：{(mean_py / max(1e-12, mean_cpp)):.2f}x")
    print(f"- C++/Python：{(mean_cpp / max(1e-12, mean_py)):.2f}x")

    if rep_cpp is not None and rep_py is not None:
        print("\n结果对比（粗一致性）：")
        print(
            f"- C++   deleted={len(rep_cpp.get('deleted') or [])} new_lines={len(rep_cpp.get('new_lines') or [])} counts={rep_cpp.get('counts')}"
        )
        print(
            f"- Python deleted={len(rep_py.get('deleted') or [])} new_lines={len(rep_py.get('new_lines') or [])} counts={rep_py.get('counts')}"
        )

        del_cpp = set(map(str, rep_cpp.get("deleted") or []))
        del_py = set(map(str, rep_py.get("deleted") or []))
        new_cpp = _sig_set_from_features(rep_cpp.get("new_lines") or [], args.ndigits)
        new_py = _sig_set_from_features(rep_py.get("new_lines") or [], args.ndigits)
        _print_set_diff("deleted", del_cpp, del_py)
        _print_set_diff("new_lines(sig)", new_cpp, new_py)

        feats_cpp = _apply_repair(features, rep_cpp)
        feats_py = _apply_repair(features, rep_py)

        issues_cpp, stats_cpp = topology.check_topology_layer(feats_cpp, None, args.ndigits)
        issues_py, stats_py = topology.check_topology_layer(feats_py, None, args.ndigits)

        print("\n修复后拓扑问题数（越少越好）：")
        print(f"- C++   total={len(issues_cpp)} by_kind={stats_cpp.get('issues_by_kind')}")
        print(f"- Python total={len(issues_py)} by_kind={stats_py.get('issues_by_kind')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
