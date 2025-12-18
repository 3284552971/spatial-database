"""test_07_cross_only_fix_verify.py

目的
- 只检查交叉拓扑（cross_without_node），并验证修复后交叉问题是否被正确消除（即交点变成端点/节点）。

运行
- 在项目根目录执行：
  python3 space_app/test_cases/test_07_cross_only_fix_verify.py

可选
- 默认使用合成数据集（稳定、可复现）。
- 如需在真实表上验证，可自行扩展：从表导出 features 后调用同样的函数。
"""

from __future__ import annotations

import os


def _make_features(n: int = 120):
    # 一条水平线 + n 条竖线，产生 n 个非端点交叉
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
    # 尽可能全修，不限制交点数
    os.environ["SPACE_TOPOLOGY_MAX_CROSS_PER_LINE"] = "0"

    # 默认优先 C++（如果可用）；C++ 报错会自动回退 Python
    os.environ.setdefault("SPACE_TOPOLOGY_BACKEND", "auto")

    from space_app.algorithms.topology import check_cross_topology_layer, repair_cross_only

    feats = _make_features(120)

    cross_issues, stats = check_cross_topology_layer(feats, None, 6)
    assert len(cross_issues) >= 100, f"交叉问题过少：{len(cross_issues)} stats={stats}"

    rep = repair_cross_only(feats, cross_issues, ndigits=6)
    replaced = set(rep.get("replaced") or [])
    new_lines = rep.get("new_lines") or []
    assert replaced, "没有替换任何线，修复未生效"
    assert len(new_lines) >= 300, f"新线段数量异常：{len(new_lines)}"

    repaired = [f for f in feats if str(f.get("id")) not in replaced] + list(new_lines)

    cross2, _stats2 = check_cross_topology_layer(repaired, None, 6)
    assert len(cross2) == 0, f"修复后仍存在交叉无节点：{len(cross2)}"

    print(
        f"[OK] cross_before={len(cross_issues)} replaced={len(replaced)} new_lines={len(new_lines)} cross_after={len(cross2)}"
    )


if __name__ == "__main__":
    main()
