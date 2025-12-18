"""test_08_t_junction_endpoint_on_segment_fix.py

目的
- 覆盖“端点落在别的线段内部”（T 形连接缺节点）场景。
- 验证检查能识别 endpoint_on_segment，并且 repair_layer 会把“宿主线”在该点切分，
  从而消除 endpoint_on_segment，同时降低对应端点的 dangling。

运行
- 在项目根目录执行：
  python3 space_app/test_cases/test_08_t_junction_endpoint_on_segment_fix.py

说明
- 默认 SPACE_TOPOLOGY_BACKEND=auto：优先 C++；C++ 不可用或报错会回退 Python。
"""

from __future__ import annotations

import os


def _make_features():
    # 一条水平线 h: (0,0)->(5,0)->(10,0)（(5,0) 是内部顶点）
    # 一条竖线 v: 端点 (5,0) 落在 h 的内部顶点上
    return [
        {
            "type": "Feature",
            "id": "h",
            "properties": {},
            "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]},
        },
        {
            "type": "Feature",
            "id": "v",
            "properties": {},
            "geometry": {"type": "LineString", "coordinates": [[5.0, 0.0], [5.0, 5.0]]},
        },
    ]


def main() -> None:
    os.environ.setdefault("SPACE_TOPOLOGY_BACKEND", "auto")
    os.environ.setdefault("SPACE_TOPOLOGY_MAX_CROSS_PER_LINE", "0")

    from space_app.algorithms.topology import check_topology_layer, repair_layer

    feats = _make_features()

    issues1, stats1 = check_topology_layer(feats, None, 6)
    kinds1 = {}
    for it in issues1:
        kinds1[it.kind] = kinds1.get(it.kind, 0) + 1

    assert kinds1.get("endpoint_on_segment", 0) >= 1, f"未识别 endpoint_on_segment: {kinds1} stats={stats1}"

    dangling_before = kinds1.get("dangling_endpoint", 0)

    rep = repair_layer(feats, issues1, ndigits=6)
    deleted = set(rep.get("deleted") or [])
    new_lines = rep.get("new_lines") or []

    assert "h" in deleted, f"宿主线 h 应被替换切分，但 deleted={deleted}"
    assert len(new_lines) >= 2, f"切分后新线段数量异常：{len(new_lines)}"

    repaired = [f for f in feats if str(f.get("id")) not in deleted] + list(new_lines)

    issues2, stats2 = check_topology_layer(repaired, None, 6)
    kinds2 = {}
    for it in issues2:
        kinds2[it.kind] = kinds2.get(it.kind, 0) + 1

    assert kinds2.get("endpoint_on_segment", 0) == 0, f"修复后仍存在 endpoint_on_segment: {kinds2} stats={stats2}"

    dangling_after = kinds2.get("dangling_endpoint", 0)
    assert dangling_after == max(0, dangling_before - 1), (
        f"dangling 数量未按预期下降 1：before={dangling_before} after={dangling_after} kinds2={kinds2}"
    )

    print(
        "[OK] "
        f"endpoint_on_segment_before={kinds1.get('endpoint_on_segment',0)} "
        f"dangling_before={dangling_before} -> dangling_after={dangling_after} "
        f"deleted={len(deleted)} new_lines={len(new_lines)}"
    )


if __name__ == "__main__":
    main()
