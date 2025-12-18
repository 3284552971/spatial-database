"""线要素拓扑检查/修复（独立于数据库）。

范围（最小可用版）
- 输入：某一线图层（LineString/MultiLineString）的 GeoJSON FeatureCollection（或从表行转成的 feature 列表）。
- 输出：
    - 问题点（红色标注）：
        - 线与线相交但未在端点处相交（"cross_without_node"）
        - 悬挂端点（"dangling_endpoint"）
    - 修复建议：
        - 交叉无节点：对涉及交叉的线在交叉点处插入节点并打断（返回绿色新线段）
        - 悬挂线：若折线整体长度小于 100m 且两端都是悬挂端点，则建议直接删除该线

重要说明
- 该模块不做落盘、不修改数据库；只返回“应如何标注/如何修复”的几何结果。
- 修复策略在课程项目里往往需要取舍：本实现偏向安全（短悬挂线直接删除），避免生成不可信连接。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import importlib.util
import logging
from math import atan2, cos, radians, sin, sqrt
import os
from pathlib import Path
import sys
import sysconfig
from typing import Any, Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)

def _try_load_native() -> Any | None:
    # Prefer in-package import (dev override), otherwise load from cpp_model/build.
    try:
        from . import topology_cpp as m  # type: ignore

        return m
    except Exception:
        pass

    build_dir = Path(__file__).resolve().parents[1] / "cpp_model" / "build"
    if not build_dir.exists():
        return None

    ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    candidates = [build_dir / f"topology_cpp{ext}"]
    if not candidates[0].exists():
        candidates = sorted(build_dir.glob("topology_cpp*.so"))
    if not candidates:
        return None

    so_path = candidates[0]
    modname = "space_app.algorithms.topology_cpp"
    spec = importlib.util.spec_from_file_location(modname, so_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


_topology_cpp = _try_load_native()
_NATIVE_AVAILABLE = _topology_cpp is not None

# 后端选择：
# - 默认 auto：若 native 可用，优先走 C++；否则走 Python。
# - 可通过环境变量强制：SPACE_TOPOLOGY_BACKEND=cpp|python|auto


Coord = Tuple[float, float]
BBox = Tuple[float, float, float, float]


def _iter_xy(coords: Any) -> Iterable[Coord]:
    if isinstance(coords, (list, tuple)):
        if len(coords) >= 2 and isinstance(coords[0], (int, float)) and isinstance(coords[1], (int, float)):
            yield float(coords[0]), float(coords[1])
        else:
            for item in coords:
                yield from _iter_xy(item)


def geom_bbox(coords: Any) -> Optional[BBox]:
    xs: List[float] = []
    ys: List[float] = []
    for x, y in _iter_xy(coords):
        xs.append(x)
        ys.append(y)
        if len(xs) >= 5000:
            break
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def bbox_intersects(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _round_pt(pt: Coord, ndigits: int = 6) -> Coord:
    # 为保证 C++/Python 后端一致性，这里使用“half-away-from-zero”而不是 Python 内建 round()
    # （Python 默认是 bankers rounding ties-to-even，在 *.5 边界会与 C++ llround 分叉）。
    scale = 10.0 ** float(ndigits)

    def _half_away(v: float) -> float:
        x = float(v) * scale
        if x >= 0.0:
            r = math.floor(x + 0.5)
        else:
            r = math.ceil(x - 0.5)
        return float(r) / scale

    return (_half_away(pt[0]), _half_away(pt[1]))


def haversine_m(a: Coord, b: Coord) -> float:
    """近似测地距离（米），用于 100m 这种阈值判断。"""
    lon1, lat1 = a
    lon2, lat2 = b
    r = 6371000.0
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    h = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dl / 2) ** 2
    return 2 * r * atan2(sqrt(h), sqrt(1 - h))


def polyline_length_m(coords: List[Coord]) -> float:
    total = 0.0
    for i in range(1, len(coords)):
        total += haversine_m(coords[i - 1], coords[i])
    return total


def _segments(coords: List[Coord]) -> Iterable[Tuple[Coord, Coord]]:
    for i in range(1, len(coords)):
        yield coords[i - 1], coords[i]


def _orientation(a: Coord, b: Coord, c: Coord) -> float:
    # cross((b-a),(c-a))
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Coord, b: Coord, p: Coord, eps: float = 1e-12) -> bool:
    if abs(_orientation(a, b, p)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def segment_intersection(a1: Coord, a2: Coord, b1: Coord, b2: Coord, eps: float = 1e-12) -> Optional[Coord]:
    """线段相交点（不处理重叠共线的复杂情况）。"""
    # 快速排除
    minax, maxax = sorted([a1[0], a2[0]])
    minay, maxay = sorted([a1[1], a2[1]])
    minbx, maxbx = sorted([b1[0], b2[0]])
    minby, maxby = sorted([b1[1], b2[1]])
    if maxax < minbx - eps or maxbx < minax - eps or maxay < minby - eps or maxby < minay - eps:
        return None

    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)

    # 一般相交
    if (o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps):
        if (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps):
            # 求交点（直线交点）
            x1, y1 = a1
            x2, y2 = a2
            x3, y3 = b1
            x4, y4 = b2
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(den) < eps:
                return None
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
            p = (px, py)
            if _on_segment(a1, a2, p) and _on_segment(b1, b2, p):
                return p
            return None

    # 端点相触
    for p in (b1, b2):
        if _on_segment(a1, a2, p, eps=eps):
            return p
    for p in (a1, a2):
        if _on_segment(b1, b2, p, eps=eps):
            return p

    return None


@dataclass
class TopologyIssue:
    kind: str
    point: Coord
    message: str


def extract_lines_from_features(features: List[Dict[str, Any]]) -> List[Tuple[Any, List[List[Coord]]]]:
    """提取线要素。

    Returns:
        List[(id, parts)] 其中 parts 为若干条 polyline，每条是 coords 列表。
    """
    out: List[Tuple[Any, List[List[Coord]]]] = []
    for feat in features:
        geom = feat.get("geometry") or {}
        gtype = (geom.get("type") or "")
        coords = geom.get("coordinates")
        fid = feat.get("id")
        if fid is None:
            fid = (feat.get("properties") or {}).get("id")

        if gtype == "LineString" and isinstance(coords, list):
            line = [(float(x), float(y)) for x, y in _iter_xy(coords)]
            if len(line) >= 2:
                out.append((fid, [line]))
        elif gtype == "MultiLineString" and isinstance(coords, list):
            parts: List[List[Coord]] = []
            for seg in coords:
                if isinstance(seg, list):
                    line = [(float(x), float(y)) for x, y in _iter_xy(seg)]
                    if len(line) >= 2:
                        parts.append(line)
            if parts:
                out.append((fid, parts))
    return out


def _endpoint_degree(lines: List[Tuple[Any, List[List[Coord]]]], ndigits: int = 6):
    deg = {}
    for _fid, parts in lines:
        for pts in parts:
            a = _round_pt(pts[0], ndigits)
            b = _round_pt(pts[-1], ndigits)
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
    return deg


def _layer_bbox(lines: List[Tuple[Any, List[List[Coord]]]]) -> Optional[BBox]:
    xs: List[float] = []
    ys: List[float] = []
    for _fid, parts in lines:
        bb = geom_bbox([list(p) for p in parts])
        if not bb:
            continue
        xs.extend([bb[0], bb[2]])
        ys.extend([bb[1], bb[3]])
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _grid_cell_size(layer_bb: BBox) -> float:
    # 简单自适应：把范围划成约 50 格，限制在合理区间
    spanx = max(1e-9, layer_bb[2] - layer_bb[0])
    spany = max(1e-9, layer_bb[3] - layer_bb[1])
    size = max(spanx, spany) / 50.0
    return max(0.001, min(0.05, size))


def _bbox_to_cells(bb: BBox, cell: float):
    x0 = int(bb[0] / cell)
    x1 = int(bb[2] / cell)
    y0 = int(bb[1] / cell)
    y1 = int(bb[3] / cell)
    for gx in range(x0, x1 + 1):
        for gy in range(y0, y1 + 1):
            yield gx, gy


def check_topology_layer(
    features: List[Dict[str, Any]],
    search_bbox: Optional[BBox] = None,
    ndigits: int = 6,
) -> Tuple[List[TopologyIssue], Dict[str, Any]]:
    """整层拓扑检查：返回问题点列表 + 一些统计信息。

    后端：
    - 默认优先走 C++（native）以获得更好的性能。
    - 若 C++ 执行报错，则自动回退到 Python 实现（保证可用性）。
    - 可用环境变量强制：SPACE_TOPOLOGY_BACKEND=cpp|python|auto
    """

    backend = _choose_backend()
    if backend == "cpp":
        try:
            return _check_topology_layer_cpp(features, search_bbox=search_bbox, ndigits=ndigits)
        except Exception as e:
            logger.warning("topology native failed; fallback to python: %r", e)
            return _check_topology_layer_python(features, search_bbox=search_bbox, ndigits=ndigits)
    return _check_topology_layer_python(features, search_bbox=search_bbox, ndigits=ndigits)


def check_cross_topology_layer(
    features: List[Dict[str, Any]],
    search_bbox: Optional[BBox] = None,
    ndigits: int = 6,
) -> Tuple[List[TopologyIssue], Dict[str, Any]]:
    """仅检查“交叉无节点”(cross_without_node)。

    说明：当前实现复用 check_topology_layer 的结果并过滤。
    好处：自动沿用默认 C++ 优先/失败回退的后端策略；坏处：内部仍会计算 dangling。
    如需进一步性能优化，可单独实现 cross-only 的核心循环。
    """

    issues, stats = check_topology_layer(features, search_bbox=search_bbox, ndigits=ndigits)
    cross = [it for it in issues if it.kind == "cross_without_node"]
    stats2 = dict(stats or {})
    stats2["cross_only"] = True
    stats2["cross_count"] = len(cross)
    return cross, stats2


def _choose_backend() -> str:
    """选择拓扑后端。

    规则：
    - SPACE_TOPOLOGY_BACKEND=python 强制走 Python。
    - SPACE_TOPOLOGY_BACKEND=cpp 强制优先走 C++（不可用则回退 Python；运行报错也会回退）。
    - SPACE_TOPOLOGY_BACKEND=auto（默认）：native 可用则走 C++，否则走 Python。
    """

    forced = (os.getenv("SPACE_TOPOLOGY_BACKEND") or "auto").strip().lower()
    if forced in {"python", "py"}:
        return "python"
    if forced in {"cpp", "c++", "native"}:
        return "cpp" if (_NATIVE_AVAILABLE and _topology_cpp is not None) else "python"

    # auto
    return "cpp" if (_NATIVE_AVAILABLE and _topology_cpp is not None) else "python"


def _check_topology_layer_cpp(
    features: List[Dict[str, Any]],
    search_bbox: Optional[BBox] = None,
    ndigits: int = 6,
) -> Tuple[List[TopologyIssue], Dict[str, Any]]:
    if not _NATIVE_AVAILABLE or _topology_cpp is None:
        raise RuntimeError("topology native 扩展不可用")

    bbox_obj = None
    if search_bbox is not None:
        bbox_obj = {
            "minx": float(search_bbox[0]),
            "miny": float(search_bbox[1]),
            "maxx": float(search_bbox[2]),
            "maxy": float(search_bbox[3]),
        }
    out = _topology_cpp.check_topology_layer(features, bbox_obj, int(ndigits))
    raw_issues = out.get("issues") or []
    stats = out.get("stats") or {}
    issues: List[TopologyIssue] = []
    for it in raw_issues:
        try:
            kind = it.get("kind")
            pt = it.get("point")
            msg = it.get("message")
            if pt is None:
                continue
            issues.append(TopologyIssue(str(kind), (float(pt[0]), float(pt[1])), str(msg)))
        except Exception:
            continue

    # 补充按类型计数，便于前端解释“问题点很多但修复线段较少”的原因。
    by_kind: Dict[str, int] = {}
    for it in issues:
        by_kind[it.kind] = by_kind.get(it.kind, 0) + 1
    try:
        stats = dict(stats)
        stats["issues_by_kind"] = by_kind
        stats["issues_total"] = len(issues)
    except Exception:
        pass
    return issues, stats


def _check_topology_layer_python(
    features: List[Dict[str, Any]],
    search_bbox: Optional[BBox] = None,
    ndigits: int = 6,
) -> Tuple[List[TopologyIssue], Dict[str, Any]]:
    issues: List[TopologyIssue] = []
    lines = extract_lines_from_features(features)

    # 仅保留 bbox 内候选（若传入）；否则为“整层”
    if search_bbox is not None:
        filtered = []
        for fid, parts in lines:
            bb = geom_bbox([list(p) for p in parts])
            if bb and bbox_intersects(bb, search_bbox):
                filtered.append((fid, parts))
        lines = filtered

    deg = _endpoint_degree(lines, ndigits=ndigits)

    parts_map: Dict[Any, List[List[Coord]]] = {fid: parts for fid, parts in lines}

    # 1) 悬挂端点（整层）
    for fid, parts in lines:
        for pts in parts:
            a = _round_pt(pts[0], ndigits)
            b = _round_pt(pts[-1], ndigits)
            if deg.get(a, 0) <= 1:
                issues.append(TopologyIssue("dangling_endpoint", a, f"悬挂端点：fid={fid}"))
            if deg.get(b, 0) <= 1:
                issues.append(TopologyIssue("dangling_endpoint", b, f"悬挂端点：fid={fid}"))

    # 2) 交叉无节点（整层）
    # 用 bbox 网格做候选对过滤，避免 O(n^2) 全对全爆炸
    layer_bb = _layer_bbox(lines)
    if not layer_bb:
        return issues, {"lines": 0}
    cell = _grid_cell_size(layer_bb)

    line_bbs: Dict[Any, BBox] = {}
    endpoints: Dict[Any, set] = {}
    grid: Dict[Tuple[int, int], List[Any]] = {}
    for fid, parts in lines:
        bb = geom_bbox([list(p) for p in parts])
        if not bb:
            continue
        line_bbs[fid] = bb
        ep = set()
        for pts in parts:
            ep.add(_round_pt(pts[0], ndigits))
            ep.add(_round_pt(pts[-1], ndigits))
        endpoints[fid] = ep
        for cxy in _bbox_to_cells(bb, cell):
            grid.setdefault(cxy, []).append(fid)

    seen_pairs = set()
    seen_points = set()
    seen_t_points = set()
    # 逐格取候选对
    for _cell_key, ids in grid.items():
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a_id = ids[i]
                b_id = ids[j]
                if a_id == b_id:
                    continue
                pair = (a_id, b_id) if str(a_id) <= str(b_id) else (b_id, a_id)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                abb = line_bbs.get(a_id)
                bbb = line_bbs.get(b_id)
                if not abb or not bbb or not bbox_intersects(abb, bbb):
                    continue

                a_parts = parts_map.get(a_id)
                b_parts = parts_map.get(b_id)
                if not a_parts or not b_parts:
                    continue

                a_ep = endpoints.get(a_id, set())
                b_ep = endpoints.get(b_id, set())

                for apts in a_parts:
                    for bpts in b_parts:
                        for a1, a2 in _segments(apts):
                            for b1, b2 in _segments(bpts):
                                p = segment_intersection(a1, a2, b1, b2)
                                if p is None:
                                    continue
                                pr = _round_pt(p, ndigits)
                                a_is_ep = pr in a_ep
                                b_is_ep = pr in b_ep

                                # 两边都是端点：共享端点，视为已有节点
                                if a_is_ep and b_is_ep:
                                    continue

                                # 仅一边是端点：端点落在对方线段内部（T 形连接缺节点）
                                if a_is_ep ^ b_is_ep:
                                    end_id = a_id if a_is_ep else b_id
                                    seg_id = b_id if a_is_ep else a_id
                                    key = (pr, str(end_id), str(seg_id))
                                    if key in seen_t_points:
                                        continue
                                    seen_t_points.add(key)
                                    issues.append(
                                        TopologyIssue(
                                            "endpoint_on_segment",
                                            pr,
                                            f"端点落在线段内部：end={end_id} on {seg_id}",
                                        )
                                    )
                                    continue

                                key = (pr, str(a_id), str(b_id))
                                if key in seen_points:
                                    continue
                                seen_points.add(key)
                                issues.append(TopologyIssue(
                                    "cross_without_node",
                                    pr,
                                    f"线相交无节点：{a_id} x {b_id}",
                                ))

    by_kind: Dict[str, int] = {}
    for it in issues:
        by_kind[it.kind] = by_kind.get(it.kind, 0) + 1

    stats = {"lines": len(lines), "cell": cell, "issues_by_kind": by_kind, "issues_total": len(issues)}
    return issues, stats


def repair_layer(
    features: List[Dict[str, Any]],
    issues: List[TopologyIssue],
    dangling_delete_threshold_m: float = 100.0,
    ndigits: int = 6,
) -> Dict[str, Any]:
    """整层修复建议（不落盘）：返回新线段（绿）与删除列表。

    后端：
    - 默认 auto：native 可用则优先走 C++（更快）；否则走 Python。
    - 若 C++ 执行报错，会自动回退到 Python。
    - 可用环境变量强制：SPACE_TOPOLOGY_REPAIR_BACKEND=cpp|python|auto
    """

    forced = (os.getenv("SPACE_TOPOLOGY_REPAIR_BACKEND") or "auto").strip().lower()
    want_cpp = forced in {"auto", "cpp", "c++", "native"}
    if forced in {"python", "py"}:
        want_cpp = False

    if want_cpp and _NATIVE_AVAILABLE and _topology_cpp is not None and hasattr(_topology_cpp, "repair_layer"):
        try:
            # 允许 issues 是 TopologyIssue dataclass；C++ 侧会用 getattr 读取
            return _topology_cpp.repair_layer(features, issues, float(dangling_delete_threshold_m), int(ndigits))
        except Exception as e:
            logger.warning("topology native repair failed; fallback to python: %r", e)

    # 迭代次数：单轮切分往往会把“原本隐藏的连接缺节点”暴露出来，需要闭包式迭代才能收敛。
    try:
        max_iters = int(os.getenv("SPACE_TOPOLOGY_REPAIR_MAX_ITERS") or "100")
    except Exception:
        max_iters = 100
    # 默认 100；仍做硬上限避免在异常数据上卡死
    max_iters = max(1, min(100, max_iters))

    # 每条线最多处理多少个节点点（防止极端数据过慢）；0=不限制
    try:
        max_cross_per_line = int(os.getenv("SPACE_TOPOLOGY_MAX_CROSS_PER_LINE") or "0")
    except Exception:
        max_cross_per_line = 0
    max_cross_per_line = max(0, max_cross_per_line)

    eps = max(1e-12, (10.0 ** (-float(ndigits))) * 10.0)
    eps2 = eps * eps

    def _dist2(p: Coord, a: Coord, b: Coord) -> float:
        px, py = p
        ax, ay = a
        bx, by = b
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        c1 = vx * wx + vy * wy
        if c1 <= 0:
            return (px - ax) ** 2 + (py - ay) ** 2
        c2 = vx * vx + vy * vy
        if c2 <= 0:
            return (px - ax) ** 2 + (py - ay) ** 2
        t = c1 / c2
        if t >= 1:
            return (px - bx) ** 2 + (py - by) ** 2
        proj = (ax + t * vx, ay + t * vy)
        return (px - proj[0]) ** 2 + (py - proj[1]) ** 2

    original_lines = extract_lines_from_features(features)
    if not original_lines:
        return {"action": "noop", "message": "图层中没有线要素", "deleted": [], "new_lines": []}
    original_line_ids = {fid for fid, _parts in original_lines}

    current_features = list(features)
    current_issues = list(issues)

    def _noding_once(cur_features: List[Dict[str, Any]], cur_issues: List[TopologyIssue], iter_idx: int):
        cur_lines = extract_lines_from_features(cur_features)
        if not cur_lines:
            return set(), []

        # 记录节点点：key=round 坐标；value=代表性坐标（尽量用检查输出的原始浮点，避免不必要的“全量吸附”）
        node_repr: Dict[Coord, Coord] = {}
        for it in cur_issues:
            if it.kind in ("cross_without_node", "endpoint_on_segment"):
                pr = _round_pt(it.point, ndigits)
                node_repr.setdefault(pr, it.point)

        if not node_repr:
            return set(), []

        # bbox 网格：node 点 -> 候选线集合
        layer_bb = _layer_bbox(cur_lines)
        cell = _grid_cell_size(layer_bb) if layer_bb else 0.01

        parts_map: Dict[Any, List[List[Coord]]] = {fid: parts for fid, parts in cur_lines}
        line_bbs: Dict[Any, BBox] = {}
        endpoints: Dict[Any, set] = {}
        grid: Dict[Tuple[int, int], List[Any]] = {}

        for fid, parts in cur_lines:
            bb = geom_bbox([list(p) for p in parts])
            if not bb:
                continue
            line_bbs[fid] = bb
            ep = set()
            for pts in parts:
                if len(pts) >= 1:
                    ep.add(_round_pt(pts[0], ndigits))
                    ep.add(_round_pt(pts[-1], ndigits))
            endpoints[fid] = ep
            for cxy in _bbox_to_cells(bb, cell):
                grid.setdefault(cxy, []).append(fid)

        # 给每条线分配需要切分的节点点（按 fid）
        line_points: Dict[Any, List[Coord]] = {}
        for pr in node_repr.keys():
            cxy = (int(pr[0] / cell), int(pr[1] / cell))
            cand = grid.get(cxy) or []
            for fid in cand:
                bb = line_bbs.get(fid)
                if not bb:
                    continue
                if pr[0] < bb[0] - eps or pr[0] > bb[2] + eps or pr[1] < bb[1] - eps or pr[1] > bb[3] + eps:
                    continue
                if pr in (endpoints.get(fid) or set()):
                    continue

                hit = False
                for pts in parts_map.get(fid, []):
                    if len(pts) < 2:
                        continue
                    # 命中内部顶点
                    for vi in range(1, len(pts) - 1):
                        if _round_pt(pts[vi], ndigits) == pr:
                            hit = True
                            break
                    if hit:
                        break
                    # 命中某条线段（用 round 后点做距离判断，稳定）
                    rpts: List[Coord] = []
                    for p0 in pts:
                        pr0 = _round_pt(p0, ndigits)
                        if not rpts or rpts[-1] != pr0:
                            rpts.append(pr0)
                    if len(rpts) < 2:
                        continue
                    for a1, a2 in _segments(rpts):
                        if _dist2(pr, a1, a2) <= eps2:
                            hit = True
                            break
                    if hit:
                        break

                if hit:
                    line_points.setdefault(fid, []).append(pr)

        replaced_iter: set[Any] = set()
        new_iter: List[Dict[str, Any]] = []
        split_idx = 0

        for fid, parts in cur_lines:
            pts_all_raw = line_points.get(fid)
            if not pts_all_raw:
                continue

            uniq: Dict[Coord, None] = {}
            for pr in pts_all_raw:
                uniq[pr] = None
            pts_all = list(uniq.keys())
            if max_cross_per_line > 0 and len(pts_all) > max_cross_per_line:
                pts_all = pts_all[:max_cross_per_line]

            produced_any = False
            for part_idx, part in enumerate(parts):
                pts = part[:]
                if len(pts) < 2:
                    continue

                # 内部顶点集合（round 后）
                vertex_set = set()
                if len(pts) > 2:
                    for vi in range(1, len(pts) - 1):
                        vertex_set.add(_round_pt(pts[vi], ndigits))

                seg_points: List[List[Tuple[float, Coord]]] = [[] for _ in range(max(0, len(pts) - 1))]
                cut_set: set[Coord] = set()

                for pr in pts_all:
                    if pr == _round_pt(pts[0], ndigits) or pr == _round_pt(pts[-1], ndigits):
                        continue
                    if pr in vertex_set:
                        cut_set.add(pr)
                        continue

                    best_i = None
                    best_d = 1e100
                    best_t = 0.0
                    rep = node_repr.get(pr, pr)
                    for i in range(1, len(pts)):
                        a = _round_pt(pts[i - 1], ndigits)
                        b = _round_pt(pts[i], ndigits)
                        ax, ay = a
                        bx, by = b
                        vx, vy = bx - ax, by - ay
                        wx, wy = rep[0] - ax, rep[1] - ay
                        c2 = vx * vx + vy * vy
                        if c2 <= 0:
                            continue
                        t = (vx * wx + vy * wy) / c2
                        if t <= 0.0 or t >= 1.0:
                            continue
                        proj = (ax + t * vx, ay + t * vy)
                        d2 = (rep[0] - proj[0]) ** 2 + (rep[1] - proj[1]) ** 2
                        if d2 < best_d:
                            best_d = d2
                            best_i = i
                            best_t = t

                    if best_i is None or best_d > eps2:
                        continue
                    seg_points[best_i - 1].append((best_t, pr))
                    cut_set.add(pr)

                if not cut_set:
                    continue

                # 插入点序列（用代表性坐标插入，但用 round 后坐标去重）
                new_pts: List[Coord] = [pts[0]]
                for seg_i in range(1, len(pts)):
                    inserts = seg_points[seg_i - 1]
                    if inserts:
                        seen_ins = set()
                        for _t, pr in sorted(inserts, key=lambda x: x[0]):
                            if pr in seen_ins:
                                continue
                            seen_ins.add(pr)
                            ip = node_repr.get(pr, pr)
                            if _round_pt(new_pts[-1], ndigits) != pr:
                                new_pts.append((float(ip[0]), float(ip[1])))
                    if _round_pt(new_pts[-1], ndigits) != _round_pt(pts[seg_i], ndigits):
                        new_pts.append(pts[seg_i])

                # 按 cut_set 切分
                segments: List[List[Coord]] = []
                current: List[Coord] = []
                for p in new_pts:
                    if not current:
                        current.append(p)
                        continue
                    current.append(p)
                    if _round_pt(p, ndigits) in cut_set and len(current) >= 2:
                        segments.append(current)
                        current = [p]
                if len(current) >= 2:
                    segments.append(current)

                for idx, seg in enumerate(segments):
                    split_idx += 1
                    produced_any = True
                    new_iter.append(
                        {
                            "type": "Feature",
                            "id": f"{fid}__split__it{iter_idx}__{part_idx}__{idx}",
                            "properties": {"source_id": fid, "note": "repaired_split", "cost_factor": 10.0},
                            "geometry": {"type": "LineString", "coordinates": [[float(x), float(y)] for x, y in seg]},
                        }
                    )

            if produced_any:
                replaced_iter.add(fid)

        return replaced_iter, new_iter

    total_new: List[Dict[str, Any]] = []
    replaced_original: set[Any] = set()
    iters_done = 0

    for it in range(max_iters):
        iters_done = it + 1
        replaced_iter, new_iter = _noding_once(current_features, current_issues, it)
        if not replaced_iter:
            break

        # 只记录“原始线”被替换的情况，用于返回 deleted（避免中间产物污染）
        for fid in replaced_iter:
            if fid in original_line_ids:
                replaced_original.add(fid)

        # 更新当前特征集合：删除本轮被替换线，并加入新线段
        current_features = [f for f in current_features if f.get("id") not in replaced_iter] + list(new_iter)
        total_new = [f for f in total_new if f.get("id") not in replaced_iter] + list(new_iter)

        # 复检，得到下一轮需要处理的节点
        current_issues, _stats = check_topology_layer(current_features, search_bbox=None, ndigits=ndigits)
        if not any(it2.kind in ("cross_without_node", "endpoint_on_segment") for it2 in current_issues):
            break

    # --- 悬挂点修复（在交叉/端点落线段已尽量闭包后执行）---
    # 1) 删除“超出一点点”的短悬挂碎段（通常是 noding 后产生的尾巴）
    # 2) 对靠近其它线但未连接的悬挂端点，补一条连接线段（<=50m）

    try:
        trim_m = float(os.getenv("SPACE_TOPOLOGY_DANGLING_TRIM_M") or "50")
    except Exception:
        trim_m = 50.0
    trim_m = max(0.0, trim_m)

    try:
        connect_m = float(os.getenv("SPACE_TOPOLOGY_DANGLING_CONNECT_M") or "50")
    except Exception:
        connect_m = 50.0
    connect_m = max(0.0, connect_m)

    import math

    R = 6371000.0

    def _proj_xy(p: Coord, lat0_rad: float) -> Tuple[float, float]:
        # 简单等距圆柱近似：在小范围（几十米）内足够稳定
        lon, lat = p
        x = math.radians(lon) * R * math.cos(lat0_rad)
        y = math.radians(lat) * R
        return x, y

    def _unproj_xy(x: float, y: float, lat0_rad: float) -> Coord:
        lon = math.degrees(x / (R * max(1e-12, math.cos(lat0_rad))))
        lat = math.degrees(y / R)
        return lon, lat

    def _nearest_point_on_polyline(p: Coord, parts: List[List[Coord]]) -> Tuple[float, Optional[Coord]]:
        # 返回 (距离米, 最近点 lon/lat)
        lat0 = math.radians(p[1])
        px, py = _proj_xy(p, lat0)
        best_d2 = 1e100
        best_xy = None
        for pts in parts:
            if len(pts) < 2:
                continue
            rpts: List[Coord] = []
            for p0 in pts:
                pr0 = _round_pt(p0, ndigits)
                if not rpts or rpts[-1] != pr0:
                    rpts.append(pr0)
            if len(rpts) < 2:
                continue
            for a, b in _segments(rpts):
                ax, ay = _proj_xy(a, lat0)
                bx, by = _proj_xy(b, lat0)
                vx, vy = bx - ax, by - ay
                wx, wy = px - ax, py - ay
                c2 = vx * vx + vy * vy
                if c2 <= 0:
                    continue
                t = (vx * wx + vy * wy) / c2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                qx = ax + t * vx
                qy = ay + t * vy
                d2 = (px - qx) ** 2 + (py - qy) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_xy = (qx, qy)
        if best_xy is None:
            return 1e100, None
        return math.sqrt(best_d2), _unproj_xy(best_xy[0], best_xy[1], lat0)

    trimmed_spurs = 0
    if trim_m > 0:
        final_lines = extract_lines_from_features(current_features)
        deg2 = _endpoint_degree(final_lines, ndigits=ndigits)
        spur_ids: set[Any] = set()
        for fid, parts in final_lines:
            # 仅对“非原始线段”（noding 生成的 split 线段）做自动剪尾
            if fid in original_line_ids:
                continue
            for pts in parts:
                if len(pts) < 2:
                    continue
                a = _round_pt(pts[0], ndigits)
                b = _round_pt(pts[-1], ndigits)
                da = deg2.get(a, 0)
                db = deg2.get(b, 0)
                if not ((da <= 1 and db >= 2) or (db <= 1 and da >= 2)):
                    continue
                if polyline_length_m(pts) <= trim_m:
                    spur_ids.add(fid)
                    break
        if spur_ids:
            trimmed_spurs = len(spur_ids)
            current_features = [f for f in current_features if f.get("id") not in spur_ids]

    # 连接：对悬挂端点寻找 50m 内最近线，补一条连接线段
    connectors_added = 0
    if connect_m > 0:
        final_lines = extract_lines_from_features(current_features)
        deg2 = _endpoint_degree(final_lines, ndigits=ndigits)

        # 线 bbox 网格，用于快速筛候选
        layer_bb = _layer_bbox(final_lines)
        cell = _grid_cell_size(layer_bb) if layer_bb else 0.01
        line_bbs: Dict[Any, BBox] = {}
        parts_map: Dict[Any, List[List[Coord]]] = {fid: parts for fid, parts in final_lines}
        grid: Dict[Tuple[int, int], List[Any]] = {}
        for fid, parts in final_lines:
            bb = geom_bbox([list(p) for p in parts])
            if not bb:
                continue
            line_bbs[fid] = bb
            for cxy in _bbox_to_cells(bb, cell):
                grid.setdefault(cxy, []).append(fid)

        # 遍历悬挂端点
        for fid, parts in final_lines:
            for pts in parts:
                if len(pts) < 2:
                    continue
                ends = [pts[0], pts[-1]]
                for e_idx, ep0 in enumerate(ends):
                    ep = _round_pt(ep0, ndigits)
                    if deg2.get(ep, 0) > 1:
                        continue

                    # 找最近线（排除自身 fid）
                    gx = int(ep[0] / cell)
                    gy = int(ep[1] / cell)
                    cand = []
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            cand.extend(grid.get((gx + dx, gy + dy)) or [])
                    best_d = 1e100
                    best_q = None
                    best_target = None
                    for tfid in cand:
                        if tfid == fid:
                            continue
                        tbb = line_bbs.get(tfid)
                        if not tbb:
                            continue
                        # bbox 粗筛：用 degrees 近似，扩大一点点
                        if ep[0] < tbb[0] - 0.001 or ep[0] > tbb[2] + 0.001 or ep[1] < tbb[1] - 0.001 or ep[1] > tbb[3] + 0.001:
                            continue
                        d, q = _nearest_point_on_polyline(ep, parts_map.get(tfid, []))
                        if q is None:
                            continue
                        if d < best_d:
                            best_d = d
                            best_q = q
                            best_target = tfid

                    if best_q is None or best_d > connect_m:
                        continue

                    # 生成连接线段（用 round 坐标，避免微小缝隙）
                    p1 = ep
                    p2 = _round_pt(best_q, ndigits)
                    if p1 == p2:
                        continue

                    connectors_added += 1
                    current_features.append(
                        {
                            "type": "Feature",
                            "id": f"connect__{fid}__{connectors_added}",
                            "properties": {
                                "source_id": fid,
                                "target_id": best_target,
                                "note": "dangling_connect",
                                "cost_factor": 10.0,
                            },
                            "geometry": {"type": "LineString", "coordinates": [[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]]},
                        }
                    )

        # 连接后再做一次 noding 闭包，确保连接点成为节点
        if connectors_added:
            current_issues, _stats = check_topology_layer(current_features, search_bbox=None, ndigits=ndigits)
            for it in range(max_iters):
                replaced_iter, new_iter = _noding_once(current_features, current_issues, iters_done + it)
                if not replaced_iter:
                    break
                for fid in replaced_iter:
                    if fid in original_line_ids:
                        replaced_original.add(fid)
                current_features = [f for f in current_features if f.get("id") not in replaced_iter] + list(new_iter)
                current_issues, _stats = check_topology_layer(current_features, search_bbox=None, ndigits=ndigits)
                if not any(it2.kind in ("cross_without_node", "endpoint_on_segment") for it2 in current_issues):
                    break

            # 连接后可能又生成短尾巴，再剪一次
            if trim_m > 0:
                final_lines2 = extract_lines_from_features(current_features)
                deg3 = _endpoint_degree(final_lines2, ndigits=ndigits)
                spur_ids: set[Any] = set()
                for fid, parts in final_lines2:
                    if fid in original_line_ids:
                        continue
                    for pts in parts:
                        if len(pts) < 2:
                            continue
                        a = _round_pt(pts[0], ndigits)
                        b = _round_pt(pts[-1], ndigits)
                        da = deg3.get(a, 0)
                        db = deg3.get(b, 0)
                        if not ((da <= 1 and db >= 2) or (db <= 1 and da >= 2)):
                            continue
                        if polyline_length_m(pts) <= trim_m:
                            spur_ids.add(fid)
                            break
                if spur_ids:
                    trimmed_spurs += len(spur_ids)
                    current_features = [f for f in current_features if f.get("id") not in spur_ids]

    # 最后一轮基于“最终结果”做短悬挂删除（只针对原始线 id，避免删到中间产物）
    dangling_deleted: set[Any] = set()
    if dangling_delete_threshold_m and dangling_delete_threshold_m > 0:
        final_lines = extract_lines_from_features(current_features)
        deg = _endpoint_degree(final_lines, ndigits=ndigits)
        for fid, parts in final_lines:
            if fid not in original_line_ids:
                continue
            for pts in parts:
                if len(pts) < 2:
                    continue
                a = _round_pt(pts[0], ndigits)
                b = _round_pt(pts[-1], ndigits)
                if deg.get(a, 0) <= 1 and deg.get(b, 0) <= 1:
                    length = polyline_length_m(pts)
                    if length < dangling_delete_threshold_m:
                        dangling_deleted.add(fid)
                        break

    deleted = set(replaced_original) | set(dangling_deleted)

    # 返回的 new_lines：仅包含最终仍然存在、且不在原始线集合中的线段
    final_new_lines: List[Dict[str, Any]] = []
    for f in current_features:
        try:
            geom = (f.get("geometry") or {})
            gtype = geom.get("type")
            if gtype not in ("LineString", "MultiLineString"):
                continue
            fid = f.get("id")
            if fid in original_line_ids:
                continue
            final_new_lines.append(f)
        except Exception:
            continue

    msg_parts = []
    if deleted:
        msg_parts.append(
            f"建议删除线要素 {len(deleted)} 条（短悬挂={len(dangling_deleted)}；交叉被替换={len(replaced_original)}）"
        )
    if final_new_lines:
        msg_parts.append(f"生成绿色修复线段 {len(final_new_lines)} 条（cost_factor=10.0）")
    if iters_done > 1:
        msg_parts.append(f"迭代修复轮次 {iters_done}/{max_iters}")
    if trimmed_spurs or connectors_added:
        msg_parts.append(f"悬挂处理：剪尾={trimmed_spurs}；补连接={connectors_added}")
    if not msg_parts:
        msg_parts.append("未发现可自动修复的问题")

    return {
        "action": "layer_repair",
        "message": "；".join(msg_parts),
        "deleted": list(deleted),
        "deleted_dangling": list(dangling_deleted),
        "deleted_cross_replaced": list(replaced_original),
        "new_lines": final_new_lines,
        "counts": {
            "dangling_deleted": len(dangling_deleted),
            "cross_replaced": len(replaced_original),
            "new_lines": len(final_new_lines),
            "iters": iters_done,
            "trimmed_spurs": trimmed_spurs,
            "connectors_added": connectors_added,
        },
    }


def repair_cross_only(
    features: List[Dict[str, Any]],
    cross_issues: List[TopologyIssue],
    ndigits: int = 6,
) -> Dict[str, Any]:
    """只修复交叉无节点：对参与交叉的线插入交点并切分，生成新线段。

    - 不做悬挂线删除。
    - 输出包含需要被替换的原线 id 列表（replaced），以及新线段（new_lines）。

    可选环境变量：
    - SPACE_TOPOLOGY_MAX_CROSS_PER_LINE：每条线最多处理的交点数量；默认 0=不限制。
    """

    lines = extract_lines_from_features(features)
    if not lines:
        return {"action": "cross_repair", "message": "图层中没有线要素", "replaced": [], "new_lines": []}

    # 收集每条线的交点（以 fid 字符串为 key）
    cross_map: Dict[str, List[Coord]] = {}
    for it in cross_issues:
        if it.kind != "cross_without_node":
            continue
        msg = it.message
        if "：" in msg and " x " in msg:
            try:
                pair = msg.split("：", 1)[1]
                a_str, b_str = pair.split(" x ")
                a_str = a_str.strip()
                b_str = b_str.strip()
                cross_map.setdefault(a_str, []).append(it.point)
                cross_map.setdefault(b_str, []).append(it.point)
            except Exception:
                continue

    if not cross_map:
        return {"action": "cross_repair", "message": "未发现可自动修复的交叉问题", "replaced": [], "new_lines": []}

    try:
        max_cross_per_line = int(os.getenv("SPACE_TOPOLOGY_MAX_CROSS_PER_LINE") or "0")
    except Exception:
        max_cross_per_line = 0
    max_cross_per_line = max(0, max_cross_per_line)

    eps = max(1e-12, (10.0 ** (-float(ndigits))) * 10.0)
    eps2 = eps * eps

    new_feats: List[Dict[str, Any]] = []
    replaced: set[str] = set()
    split_count = 0

    for fid, parts in lines:
        fid_str = str(fid)
        pts_all_raw = cross_map.get(fid_str)
        if not pts_all_raw:
            continue

        # 去重 + 可选截断
        uniq: Dict[Coord, None] = {}
        for p in pts_all_raw:
            uniq[_round_pt(p, ndigits)] = None
        pts_all = list(uniq.keys())
        if max_cross_per_line > 0 and len(pts_all) > max_cross_per_line:
            pts_all = pts_all[:max_cross_per_line]

        produced_any = False
        for part_idx, part in enumerate(parts):
            pts: List[Coord] = []
            for p0 in part:
                pr0 = _round_pt(p0, ndigits)
                if not pts or pts[-1] != pr0:
                    pts.append(pr0)
            if len(pts) < 2:
                continue

            vertex_set = set()
            if len(pts) > 2:
                for vi in range(1, len(pts) - 1):
                    vertex_set.add(_round_pt(pts[vi], ndigits))

            seg_points: List[List[Tuple[float, Coord]]] = [[] for _ in range(max(0, len(pts) - 1))]
            cut_set = set()
            for p in pts_all:
                pr = _round_pt(p, ndigits)
                if pr == _round_pt(pts[0], ndigits) or pr == _round_pt(pts[-1], ndigits):
                    continue
                if pr in vertex_set:
                    cut_set.add(pr)
                    continue
                best_i = None
                best_d = 1e100
                best_t = 0.0
                for i in range(1, len(pts)):
                    a = pts[i - 1]
                    b = pts[i]
                    ax, ay = a
                    bx, by = b
                    vx, vy = bx - ax, by - ay
                    wx, wy = pr[0] - ax, pr[1] - ay
                    c2 = vx * vx + vy * vy
                    if c2 <= 0:
                        continue
                    t = (vx * wx + vy * wy) / c2
                    if t <= 0.0 or t >= 1.0:
                        continue
                    proj = (ax + t * vx, ay + t * vy)
                    d2 = (pr[0] - proj[0]) ** 2 + (pr[1] - proj[1]) ** 2
                    if d2 < best_d:
                        best_d = d2
                        best_i = i
                        best_t = t

                if best_i is None or best_d > eps2:
                    continue
                seg_points[best_i - 1].append((best_t, pr))
                cut_set.add(pr)

            # 构造插入后的点序列
            new_pts: List[Coord] = [pts[0]]
            for seg_i in range(1, len(pts)):
                inserts = seg_points[seg_i - 1]
                if inserts:
                    seen_ins = set()
                    for _t, ip in sorted(inserts, key=lambda x: x[0]):
                        if ip in seen_ins:
                            continue
                        seen_ins.add(ip)
                        if new_pts[-1] != ip:
                            new_pts.append(ip)
                if new_pts[-1] != pts[seg_i]:
                    new_pts.append(pts[seg_i])

            if not cut_set:
                continue

            segments: List[List[Coord]] = []
            current: List[Coord] = []
            for p in new_pts:
                if not current:
                    current.append(p)
                    continue
                current.append(p)
                if _round_pt(p, ndigits) in cut_set and len(current) >= 2:
                    segments.append(current)
                    current = [p]
            if len(current) >= 2:
                segments.append(current)

            for idx, seg in enumerate(segments):
                split_count += 1
                produced_any = True
                new_feats.append(
                    {
                        "type": "Feature",
                        "id": f"{fid_str}__split__{part_idx}__{idx}",
                        "properties": {"source_id": fid, "note": "cross_only_split", "cost_factor": 10.0},
                        "geometry": {"type": "LineString", "coordinates": [[x, y] for x, y in seg]},
                    }
                )

        if produced_any:
            replaced.add(fid_str)

    msg = f"交叉修复：替换 {len(replaced)} 条线；生成新线段 {split_count} 条"
    return {"action": "cross_repair", "message": msg, "replaced": sorted(replaced), "new_lines": new_feats}
