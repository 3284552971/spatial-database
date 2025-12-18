#!/usr/bin/env python3
"""Repair quality check focused on non-dangling topology issues.

Goal
- Compare topology issue counts before/after repair.
- Focus on non-dangling kinds (exclude dangling_endpoint).

Usage
  python3 bench_repair_quality.py --path space_app/table_data/roads_sz_repaired_demo/rtree_source.geojson --ndigits 6

Notes
- Forces C++ backend for both check and repair (falls back automatically if native not available).
- Disables dangling-specific operations during repair (trim/connect/delete) so evaluation focuses on
  endpoint_on_segment / cross_without_node quality.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_features(path: Path, n: int) -> List[Dict[str, Any]]:
    fc = json.loads(path.read_text(encoding="utf-8"))
    feats = fc.get("features") or []
    if not isinstance(feats, list):
        raise ValueError("GeoJSON FeatureCollection.features 不是 list")
    if n > 0:
        feats = feats[:n]
    return feats


def _feature_id_str(f: Dict[str, Any]) -> str:
    fid = f.get("id")
    if fid is None:
        fid = (f.get("properties") or {}).get("id")
    return str(fid)


def _apply_repair(features: List[Dict[str, Any]], rep: Dict[str, Any]) -> List[Dict[str, Any]]:
    deleted_raw = rep.get("deleted") or []
    deleted = {str(x) for x in deleted_raw}
    keep = [f for f in features if _feature_id_str(f) not in deleted]
    new_lines = rep.get("new_lines") or []
    return keep + list(new_lines)


def _count_by_kind(issues: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in issues:
        k = getattr(it, "kind", None)
        if not k:
            continue
        out[str(k)] = out.get(str(k), 0) + 1
    return out


def _sum_without_dangling(by_kind: Dict[str, int]) -> int:
    return sum(v for k, v in by_kind.items() if k != "dangling_endpoint")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        type=str,
        default="space_app/table_data/roads_sz_repaired_demo/rtree_source.geojson",
        help="GeoJSON 路径（默认：roads_sz_repaired_demo/rtree_source.geojson）",
    )
    ap.add_argument("--n", type=int, default=0, help="使用前 N 个要素（0 表示全部）")
    ap.add_argument("--ndigits", type=int, default=6)
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"找不到文件：{path}")

    # Force C++ for check/repair; disable dangling-specific operations.
    os.environ["SPACE_TOPOLOGY_BACKEND"] = "cpp"
    os.environ["SPACE_TOPOLOGY_REPAIR_BACKEND"] = "cpp"
    os.environ["SPACE_TOPOLOGY_DANGLING_TRIM_M"] = "0"
    os.environ["SPACE_TOPOLOGY_DANGLING_CONNECT_M"] = "0"

    features = _load_features(path, args.n)

    from space_app.algorithms import topology

    issues0, stats0 = topology.check_topology_layer(features, None, args.ndigits)
    by0 = _count_by_kind(issues0)

    rep = topology.repair_layer(features, issues0, dangling_delete_threshold_m=0.0, ndigits=args.ndigits)
    features2 = _apply_repair(features, rep)

    issues1, stats1 = topology.check_topology_layer(features2, None, args.ndigits)
    by1 = _count_by_kind(issues1)

    print(f"数据：{path}  features={len(features)}  ndigits={args.ndigits}")
    print("\n修复前：")
    print(f"- total={len(issues0)}  non_dangling={_sum_without_dangling(by0)}")
    print(f"- by_kind={stats0.get('issues_by_kind') if isinstance(stats0, dict) else by0}")

    print("\n修复动作：")
    print(
        f"- deleted={len(rep.get('deleted') or [])}  new_lines={len(rep.get('new_lines') or [])}  counts={rep.get('counts')}"
    )

    print("\n修复后：")
    print(f"- total={len(issues1)}  non_dangling={_sum_without_dangling(by1)}")
    print(f"- by_kind={stats1.get('issues_by_kind') if isinstance(stats1, dict) else by1}")

    # Highlight key kinds
    for k in ("cross_without_node", "endpoint_on_segment"):
        print(f"\n{k}：{by0.get(k, 0)} -> {by1.get(k, 0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
