"""test_06_topology_repair_cross_coverage.py

目的
- 复现并防止回归：当“同一条线”存在大量交叉点时，修复逻辑不应只修前 50 个交点。

构造
- 一条长水平线 h：[(0,0) -> (121,0)]
- 120 条竖线 v_i：[(i,-1) -> (i,1)]  (i=1..120)
- 每条竖线与水平线在 (i,0) 相交，但交点不是端点 => cross_without_node

断言
- 修复后 new_lines 数量应明显大于“只修 50 个交点”时的规模。
  期望：水平线应被切成 121 段（120 个交点），竖线各切成 2 段。

运行
- 在项目根目录执行：
  python space_app/test_cases/test_06_topology_repair_cross_coverage.py
"""

from __future__ import annotations

import os


def _make_features(n: int = 120):
    feats = []
    feats.append(
        {
            "type": "Feature",
            "id": "h",
            "properties": {},
            "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [float(n + 1), 0.0]]},
        }
    )
    for i in range(1, n + 1):
        feats.append(
            {
                "type": "Feature",
                "id": f"v_{i}",
                "properties": {},
                "geometry": {"type": "LineString", "coordinates": [[float(i), -1.0], [float(i), 1.0]]},
            }
        )
    return feats


def main() -> None:
    # 强制 Python，保证在未编译 native 时也可跑
    os.environ["SPACE_TOPOLOGY_BACKEND"] = "python"
    from space_app.algorithms.topology import check_topology_layer, repair_layer

    feats = _make_features(120)
    issues, _stats = check_topology_layer(feats, None, 6)

    cross_n = sum(1 for it in issues if getattr(it, "kind", None) == "cross_without_node")
    assert cross_n >= 100, f"cross issues 过少：{cross_n}"

    rep = repair_layer(feats, issues, ndigits=6)
    new_lines = rep.get("new_lines") or []

    # 旧逻辑（每条线最多 50 交点）下：new_lines 约 291；改进后应约 361。
    assert len(new_lines) >= 330, f"new_lines 过少，疑似仍在截断交点：{len(new_lines)}"

    deleted = set(rep.get("deleted") or [])
    assert "h" in deleted, "水平线 h 未被删除并替换为切分线段"

    # 再跑一次 check：交叉问题应被消除（端点触碰会被视为已有节点）
    repaired_feats = [f for f in feats if f.get("id") not in deleted] + list(new_lines)
    issues2, _stats2 = check_topology_layer(repaired_feats, None, 6)
    cross2 = sum(1 for it in issues2 if getattr(it, "kind", None) == "cross_without_node")
    assert cross2 == 0, f"修复后仍存在 cross_without_node：{cross2}"

    print(f"[OK] cross={cross_n} new_lines={len(new_lines)} deleted={len(deleted)}")


if __name__ == "__main__":
    main()
