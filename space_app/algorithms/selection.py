from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_xy_from_coords(coords: Any) -> Iterable[Tuple[float, float]]:
    if isinstance(coords, list):
        if len(coords) >= 2 and isinstance(coords[0], (int, float)) and isinstance(coords[1], (int, float)):
            yield float(coords[0]), float(coords[1])
        else:
            for item in coords:
                yield from _iter_xy_from_coords(item)
    elif isinstance(coords, str):
        parts = [p.strip() for p in coords.split(',')]
        if len(parts) >= 2:
            try:
                yield float(parts[0]), float(parts[1])
            except Exception:
                return


def geom_bbox(geom_type: str, coords: Any, max_samples: int = 5000) -> Optional[Tuple[float, float, float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    for x, y in _iter_xy_from_coords(coords):
        xs.append(x)
        ys.append(y)
        if len(xs) >= max_samples:
            break
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def bbox_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def meters_to_radius_deg(lat: float, radius_m: float) -> float:
    # 粗略换算：1°纬度约 111.32km；经度按 cos(lat) 缩放。
    # 取两者中更大的角度，确保“半径覆盖”不会变小（偏保守）。
    deg_lat = radius_m / 111_320.0
    coslat = max(1e-6, abs(math.cos(math.radians(lat))))
    deg_lon = radius_m / (111_320.0 * coslat)
    return max(deg_lat, deg_lon)


def meters_to_deg_components(lat: float, radius_m: float) -> Tuple[float, float]:
    """将米半径换算为经纬度角度半径。

    返回 (deg_lon, deg_lat)。用于构造“包围圆”的查询 bbox。
    """
    deg_lat = radius_m / 111_320.0
    coslat = max(1e-6, abs(math.cos(math.radians(lat))))
    deg_lon = radius_m / (111_320.0 * coslat)
    return deg_lon, deg_lat


def _bbox_intersects_circle_m(
    cx: float,
    cy: float,
    bb: Tuple[float, float, float, float],
    radius_m: float,
) -> bool:
    """判断 bbox 是否与以 (cx,cy) 为圆心、半径 radius_m 的圆相交。

    采用局部平面近似（经度按 cos(lat) 缩放）。
    - 对点要素：bbox 很小/退化，等价于点在圆内
    - 对线/面：bbox 与圆相交时返回 True（比“中心点距离”更合理）
    """
    minx, miny, maxx, maxy = bb
    if radius_m <= 0:
        return False

    # 最近点到矩形的 dx/dy（角度）
    if cx < minx:
        dx_deg = minx - cx
    elif cx > maxx:
        dx_deg = cx - maxx
    else:
        dx_deg = 0.0

    if cy < miny:
        dy_deg = miny - cy
    elif cy > maxy:
        dy_deg = cy - maxy
    else:
        dy_deg = 0.0

    m_per_deg_lat = 111_320.0
    coslat = max(1e-6, abs(math.cos(math.radians(cy))))
    m_per_deg_lon = 111_320.0 * coslat

    dx_m = dx_deg * m_per_deg_lon
    dy_m = dy_deg * m_per_deg_lat
    return (dx_m * dx_m + dy_m * dy_m) <= (radius_m * radius_m)


def _coerce_value(text: str) -> Any:
    t = (text or "").strip()
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        return t[1:-1]
    low = t.lower()
    if low in {"null", "none"}:
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if t and all(c in "-0123456789" for c in t):
            return int(t)
        if any(c == '.' for c in t) and all(c in "-0123456789." for c in t):
            return float(t)
    except Exception:
        pass
    return t


def _get_field_value(row: Dict[str, Any], field: str) -> Any:
    # 支持：
    # - 顶层列名：id/lon/lat/geom_type 等
    # - properties.xxx 或直接 xxx（从 properties 中取）
    token = (field or "").strip()
    if not token:
        return None

    # 直接列
    for k in row.keys():
        if k.lower() == token.lower():
            return row.get(k)

    props = row.get("properties")
    if isinstance(props, dict):
        if token.lower().startswith("properties."):
            sub = token.split('.', 1)[1]
            for k in props.keys():
                if str(k).lower() == sub.lower():
                    return props.get(k)
        for k in props.keys():
            if str(k).lower() == token.lower():
                return props.get(k)
    return None


def _match_condition(left: Any, op: str, right_raw: str) -> bool:
    right = _coerce_value(right_raw)
    if op in {"=", "==", "eq"}:
        return left == right
    if op == "!=":
        return left != right
    if op == "contains":
        if left is None:
            return False
        return str(right) in str(left)
    if op in {"not_contains", "notcontains"}:
        if left is None:
            return False
        return str(right) not in str(left)
    if op in {"wildcard", "like"}:
        if left is None:
            return False
        return _wildcard_match(str(left), str(right))

    # 数值比较尝试
    try:
        left_num = float(left)
        right_num = float(right)
    except Exception:
        return False

    if op == ">":
        return left_num > right_num
    if op == "<":
        return left_num < right_num
    if op == ">=":
        return left_num >= right_num
    if op == "<=":
        return left_num <= right_num
    return False


def _wildcard_match(text: str, pattern: str) -> bool:
    """无第三方依赖的通配符匹配（glob）。

    支持：
    - 任意串：`*` 或 `%`
    - 任意单字符：`?` 或 `_`
    - 转义：`\\`（例如 `\\*` 表示字面量 `*`）
    """
    t = text if text is not None else ""
    p = pattern if pattern is not None else ""

    tokens: list[tuple[str, str] | str] = []
    i = 0
    while i < len(p):
        ch = p[i]
        if ch == "\\":
            if i + 1 < len(p):
                tokens.append(("L", p[i + 1]))
                i += 2
            else:
                tokens.append(("L", "\\"))
                i += 1
            continue
        if ch in {"*", "%"}:
            tokens.append("*")
            i += 1
            continue
        if ch in {"?", "_"}:
            tokens.append("?")
            i += 1
            continue
        tokens.append(("L", ch))
        i += 1

    # glob backtracking match
    ti = 0
    pi = 0
    star_pi = -1
    star_ti = 0

    while ti < len(t):
        if pi < len(tokens):
            token = tokens[pi]
            if token == "?":
                ti += 1
                pi += 1
                continue
            if token == "*":
                star_pi = pi
                star_ti = ti
                pi += 1
                continue
            if isinstance(token, tuple) and token[0] == "L" and token[1] == t[ti]:
                ti += 1
                pi += 1
                continue

        if star_pi != -1:
            pi = star_pi + 1
            star_ti += 1
            ti = star_ti
            continue

        return False

    while pi < len(tokens) and tokens[pi] == "*":
        pi += 1
    return pi == len(tokens)


def select_by_attribute(dt, conditions: List[Dict[str, str]]) -> Tuple[List[int], Dict[int, Tuple[float, float, float, float]]]:
    attrs = dt.attributes
    ids: List[int] = []
    bboxes: Dict[int, Tuple[float, float, float, float]] = {}

    conds = conditions or []
    for row in dt.data:
        row_dict = {col: row[idx] for col, idx in attrs.items()}
        if not conds:
            ok = True
        else:
            ok: Optional[bool] = None
            for idx, c in enumerate(conds):
                field = str(c.get("field") or "").strip()
                op = str(c.get("op") or "=").strip().lower()
                value = str(c.get("value") or "").strip()
                left = _get_field_value(row_dict, field)
                cur = _match_condition(left, op, value)

                if ok is None:
                    ok = cur
                else:
                    join = str(c.get("join") or c.get("logic") or c.get("rel") or "AND").strip().upper()
                    if join == "OR":
                        ok = ok or cur
                    else:
                        ok = ok and cur
            ok = bool(ok)

        if not ok:
            continue

        try:
            fid = int(row_dict.get("id"))
        except Exception:
            continue
        ids.append(fid)

        gtype = row_dict.get("geom_type") or ""
        coords = row_dict.get("geometry")
        bb = geom_bbox(str(gtype), coords)
        if bb is not None:
            bboxes[fid] = bb

    return ids, bboxes


def _fallback_query_circle(dt, cx: float, cy: float, radius_deg: float) -> Tuple[List[int], Dict[int, Tuple[float, float, float, float]]]:
    attrs = dt.attributes
    ids: List[int] = []
    bboxes: Dict[int, Tuple[float, float, float, float]] = {}

    # 用 bbox 扩张做粗略过滤
    for row in dt.data:
        row_dict = {col: row[idx] for col, idx in attrs.items()}
        try:
            fid = int(row_dict.get("id"))
        except Exception:
            continue
        gtype = row_dict.get("geom_type") or ""
        coords = row_dict.get("geometry")
        bb = geom_bbox(str(gtype), coords)
        if bb is None:
            continue

        expanded = (bb[0] - radius_deg, bb[1] - radius_deg, bb[2] + radius_deg, bb[3] + radius_deg)
        if expanded[0] <= cx <= expanded[2] and expanded[1] <= cy <= expanded[3]:
            ids.append(fid)
            bboxes[fid] = bb

    return ids, bboxes


def _fallback_query_box(dt, query_bbox: Tuple[float, float, float, float]) -> Tuple[List[int], Dict[int, Tuple[float, float, float, float]]]:
    attrs = dt.attributes
    ids: List[int] = []
    bboxes: Dict[int, Tuple[float, float, float, float]] = {}

    for row in dt.data:
        row_dict = {col: row[idx] for col, idx in attrs.items()}
        try:
            fid = int(row_dict.get("id"))
        except Exception:
            continue
        gtype = row_dict.get("geom_type") or ""
        coords = row_dict.get("geometry")
        bb = geom_bbox(str(gtype), coords)
        if bb is None:
            continue
        if bbox_intersects(bb, query_bbox):
            ids.append(fid)
            bboxes[fid] = bb

    return ids, bboxes


def select_by_location_rtree(
    rtree,
    mode: str,
    *,
    point: Optional[Tuple[float, float]] = None,
    radius_deg: Optional[float] = None,
    radius_m: Optional[float] = None,
    ref_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
) -> Tuple[List[int], Dict[int, Tuple[float, float, float, float]]]:
    ids_set = set()
    bboxes: Dict[int, Tuple[float, float, float, float]] = {}

    def _hit_table_id(h: Dict[str, Any]) -> int:
        attrs = h.get("attributes")
        if isinstance(attrs, dict):
            # 优先使用 ingest 时注入的表主键
            if "__table_id" in attrs:
                return int(attrs.get("__table_id"))
            # 兼容：如果数据本身 properties.id 就是主键，也可用
            if "id" in attrs:
                return int(attrs.get("id"))
        return int(h.get("id"))

    if mode == "point":
        if point is None or (radius_m is None and radius_deg is None):
            raise ValueError("point 模式需要 point/radius")
        cx, cy = point

        # 使用“包围圆”的 bbox 查询候选，再用 bbox-圆距离过滤，避免角度半径导致的误选。
        if radius_m is not None:
            deg_lon, deg_lat = meters_to_deg_components(cy, float(radius_m))
            hits = rtree.query_box(cx - deg_lon, cy - deg_lat, cx + deg_lon, cy + deg_lat, "")
            for h in hits:
                fid = _hit_table_id(h)
                hb = h.get("bbox") or {}
                bb = (float(hb.get("minx")), float(hb.get("miny")), float(hb.get("maxx")), float(hb.get("maxy")))
                if _bbox_intersects_circle_m(cx, cy, bb, float(radius_m)):
                    bboxes[fid] = bb
                    ids_set.add(fid)
            return sorted(ids_set), bboxes

        # 兼容旧逻辑：radius_deg
        hits = rtree.query_circle(cx, cy, float(radius_deg), "")
        for h in hits:
            fid = _hit_table_id(h)
            bb = h.get("bbox") or {}
            bboxes[fid] = (float(bb.get("minx")), float(bb.get("miny")), float(bb.get("maxx")), float(bb.get("maxy")))
            ids_set.add(fid)
        return sorted(ids_set), bboxes

    if mode == "bbox_intersects":
        if not ref_bboxes:
            return [], {}
        for bb in ref_bboxes:
            hits = rtree.query_box(bb[0], bb[1], bb[2], bb[3], "")
            for h in hits:
                fid = int(h.get("id"))
                hb = h.get("bbox") or {}
                bboxes[fid] = (float(hb.get("minx")), float(hb.get("miny")), float(hb.get("maxx")), float(hb.get("maxy")))
                ids_set.add(fid)
        return sorted(ids_set), bboxes

    if mode == "circle_from_selected":
        if not ref_bboxes or (radius_m is None and radius_deg is None):
            return [], {}
        for bb in ref_bboxes:
            cx = (bb[0] + bb[2]) / 2.0
            cy = (bb[1] + bb[3]) / 2.0

            if radius_m is not None:
                deg_lon, deg_lat = meters_to_deg_components(cy, float(radius_m))
                hits = rtree.query_box(cx - deg_lon, cy - deg_lat, cx + deg_lon, cy + deg_lat, "")
                for h in hits:
                    fid = _hit_table_id(h)
                    hb = h.get("bbox") or {}
                    hb2 = (float(hb.get("minx")), float(hb.get("miny")), float(hb.get("maxx")), float(hb.get("maxy")))
                    if _bbox_intersects_circle_m(cx, cy, hb2, float(radius_m)):
                        bboxes[fid] = hb2
                        ids_set.add(fid)
            else:
                hits = rtree.query_circle(cx, cy, float(radius_deg), "")
                for h in hits:
                    fid = _hit_table_id(h)
                    hb = h.get("bbox") or {}
                    bboxes[fid] = (float(hb.get("minx")), float(hb.get("miny")), float(hb.get("maxx")), float(hb.get("maxy")))
                    ids_set.add(fid)
        return sorted(ids_set), bboxes

    raise ValueError(f"未知 mode: {mode}")
