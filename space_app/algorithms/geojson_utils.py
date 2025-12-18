"""GeoJSON 规范化与校验工具（独立于数据库）。

目标
- 对输入 JSON/GeoJSON 做最小限度的格式校验与规范化：
  - FeatureCollection -> 原样返回
  - Feature -> 包装成 FeatureCollection
  - Geometry(含 type/coordinates) -> 包装成单要素 FeatureCollection
  - Esri JSON（geometryType=esriGeometryPoint/Polyline/Polygon/Multipoint）-> 转换为 FeatureCollection

注意
- 若无法识别/转换，会抛出 ValueError，供上层接口返回弹窗提示。
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _is_geometry(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("type"), str) and "coordinates" in obj


def _wrap_featurecollection(features):
    return {"type": "FeatureCollection", "features": list(features)}


def _esri_feature_id(attrs: dict, fallback: int) -> Any:
    for k in ("OBJECTID", "ObjectId", "objectid", "FID", "fid", "ID", "id"):
        if k in attrs:
            return attrs.get(k)
    return fallback


def _close_ring_if_needed(ring: list) -> list:
    if not isinstance(ring, list) or len(ring) < 3:
        return ring
    first = ring[0]
    last = ring[-1]
    if isinstance(first, list) and isinstance(last, list) and len(first) >= 2 and len(last) >= 2:
        if first[0] == last[0] and first[1] == last[1]:
            return ring
        return ring + [first]
    return ring


def esri_to_featurecollection(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """将 Esri JSON 规范化为 FeatureCollection。

    支持 geometryType：
    - esriGeometryPoint -> Point
    - esriGeometryMultipoint -> MultiPoint
    - esriGeometryPolyline -> LineString / MultiLineString
    - esriGeometryPolygon -> Polygon
    """
    gtype = doc.get("geometryType")
    if gtype not in {"esriGeometryPoint", "esriGeometryPolyline", "esriGeometryPolygon", "esriGeometryMultipoint"}:
        return None
    feats = []
    for idx, feat in enumerate(doc.get("features", []) or []):
        attrs = feat.get("attributes", {}) or {}
        geom = feat.get("geometry", {}) or {}
        if not isinstance(attrs, dict):
            attrs = {}

        gj_geom: Optional[Dict[str, Any]] = None
        if gtype == "esriGeometryPoint":
            x = geom.get("x")
            y = geom.get("y")
            if x is None or y is None:
                continue
            gj_geom = {"type": "Point", "coordinates": [x, y]}
        elif gtype == "esriGeometryMultipoint":
            pts = geom.get("points")
            if not isinstance(pts, list) or not pts:
                continue
            gj_geom = {"type": "MultiPoint", "coordinates": pts}
        elif gtype == "esriGeometryPolyline":
            paths = geom.get("paths")
            if not isinstance(paths, list) or not paths:
                continue
            if len(paths) == 1:
                gj_geom = {"type": "LineString", "coordinates": paths[0]}
            else:
                gj_geom = {"type": "MultiLineString", "coordinates": paths}
        elif gtype == "esriGeometryPolygon":
            rings = geom.get("rings")
            if not isinstance(rings, list) or not rings:
                continue
            # Esri Polygon 的 rings 语义可同时表达外环/洞；GeoJSON Polygon 允许多 ring
            gj_geom = {"type": "Polygon", "coordinates": [_close_ring_if_needed(r) for r in rings]}

        if gj_geom is None:
            continue
        feats.append({
            "type": "Feature",
            "id": _esri_feature_id(attrs, idx),
            "geometry": gj_geom,
            "properties": attrs,
        })
    return {"type": "FeatureCollection", "features": feats}


def normalize_geojson(doc: Dict[str, Any]) -> Dict[str, Any]:
    """把输入 JSON 规范化成 FeatureCollection。

    Raises:
        ValueError: 无法识别或数据结构不合法
    """
    if not isinstance(doc, dict):
        raise ValueError("数据内容错误：JSON 顶层必须是对象")

    # 1) 标准 FeatureCollection
    if doc.get("type") == "FeatureCollection":
        feats = doc.get("features")
        if not isinstance(feats, list):
            raise ValueError("数据内容错误：FeatureCollection.features 必须是数组")
        return doc

    # 2) Feature
    if doc.get("type") == "Feature":
        geom = doc.get("geometry")
        if geom is None or not isinstance(geom, dict):
            raise ValueError("数据内容错误：Feature.geometry 缺失")
        return _wrap_featurecollection([doc])

    # 3) Geometry object
    if _is_geometry(doc):
        feat = {"type": "Feature", "geometry": doc, "properties": {}}
        return _wrap_featurecollection([feat])

    # 4) Esri 点 JSON
    converted = esri_to_featurecollection(doc)
    if converted is not None:
        return converted

    raise ValueError("仅支持 FeatureCollection/Feature/Geometry 或 Esri JSON（Point/Polyline/Polygon/Multipoint）")
