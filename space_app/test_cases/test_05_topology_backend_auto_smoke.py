"""test_05_topology_backend_auto_smoke.py

目的
- 验证默认 auto 模式不会报错（native 可用则优先 C++）。
- 验证当 C++ 执行报错时，会自动回退到 Python（不需要跑两遍）。

运行
- 在项目根目录执行：
  python space_app/test_cases/test_05_topology_backend_auto_smoke.py

断言
- SPACE_TOPOLOGY_BACKEND=auto 时：调用不报错。
- 若 native 可用：通过 monkeypatch 让 C++ 路径抛错，检查能回退到 Python 并得到预期 kinds。

备注
- 该脚本不依赖 Django，不需要启动服务。
"""

from __future__ import annotations

import os


def _make_features():
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


def main() -> None:
    from space_app.algorithms import topology

    os.environ["SPACE_TOPOLOGY_BACKEND"] = "auto"

    issues, stats = topology.check_topology_layer(_make_features(), None, 6)
    assert isinstance(stats, dict)
    assert isinstance(issues, list)

    print(f"[OK] auto backend smoke: issues={len(issues)}")

    # 验证 fallback：让 C++ 分支抛错，应该回退到 Python
    if getattr(topology, "_NATIVE_AVAILABLE", False) and getattr(topology, "_topology_cpp", None) is not None:
        orig_cpp = topology._check_topology_layer_cpp
        try:
            def _boom(*_args, **_kwargs):
                raise RuntimeError("boom")

            topology._check_topology_layer_cpp = _boom  # type: ignore[assignment]
            issues2, _stats2 = topology.check_topology_layer(_make_features(), None, 6)

            kinds = {getattr(it, "kind", None) for it in issues2}
            assert "cross_without_node" in kinds, "fallback 后未检测到 cross_without_node"
            assert "dangling_endpoint" in kinds, "fallback 后未检测到 dangling_endpoint"
            print(f"[OK] cpp failed -> python fallback: issues={len(issues2)}")
        finally:
            topology._check_topology_layer_cpp = orig_cpp  # type: ignore[assignment]


if __name__ == "__main__":
    main()
