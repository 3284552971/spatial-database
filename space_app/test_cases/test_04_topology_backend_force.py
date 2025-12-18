"""test_04_topology_backend_force.py

目的
- 验证拓扑检查在 Python/C++ 两种后端下都可运行。
- 验证可通过环境变量强制后端选择：SPACE_TOPOLOGY_BACKEND=python|cpp。

运行
- 在项目根目录执行：
  python space_app/test_cases/test_04_topology_backend_force.py

断言
- Python 后端：能够检测到 cross_without_node 与 dangling_endpoint（至少各 1 个）。
- C++ 后端：若 native 可用，同样能检测到上述两类问题。

备注
- 该脚本不依赖 Django，不需要启动服务。
"""

from __future__ import annotations

import os


def _make_features():
    # 两条线在 (1,1) 相交，但交点不是端点 => cross_without_node
    # 再加一条孤立短线 => dangling_endpoint
    return [
        {
            "type": "Feature",
            "id": "a",
            "properties": {},
            "geometry": {"type": "LineString", "coordinates": [[0, 0], [2, 2]]},
        },
        {
            "type": "Feature",
            "id": "b",
            "properties": {},
            "geometry": {"type": "LineString", "coordinates": [[0, 2], [2, 0]]},
        },
        {
            "type": "Feature",
            "id": "c",
            "properties": {},
            "geometry": {"type": "LineString", "coordinates": [[10, 10], [11, 10]]},
        },
    ]


def _has_kind(issues, kind: str) -> bool:
    return any(getattr(it, "kind", None) == kind for it in issues)


def main() -> None:
    from space_app.algorithms import topology

    features = _make_features()

    # 1) 强制 Python
    os.environ["SPACE_TOPOLOGY_BACKEND"] = "python"
    issues_py, stats_py = topology.check_topology_layer(features, None, 6)
    assert isinstance(stats_py, dict)
    assert _has_kind(issues_py, "cross_without_node"), "Python 后端未检测到 cross_without_node"
    assert _has_kind(issues_py, "dangling_endpoint"), "Python 后端未检测到 dangling_endpoint"
    print(f"[OK] Python backend: issues={len(issues_py)}")

    # 2) 强制 C++（若可用）
    if getattr(topology, "_NATIVE_AVAILABLE", False) and getattr(topology, "_topology_cpp", None) is not None:
        os.environ["SPACE_TOPOLOGY_BACKEND"] = "cpp"
        issues_cpp, stats_cpp = topology.check_topology_layer(features, None, 6)
        assert isinstance(stats_cpp, dict)
        assert _has_kind(issues_cpp, "cross_without_node"), "C++ 后端未检测到 cross_without_node"
        assert _has_kind(issues_cpp, "dangling_endpoint"), "C++ 后端未检测到 dangling_endpoint"
        print(f"[OK] C++ backend: issues={len(issues_cpp)}")
    else:
        print("[SKIP] C++ backend: native 扩展不可用（未编译或无法加载 topology_cpp）")


if __name__ == "__main__":
    main()
