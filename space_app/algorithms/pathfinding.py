from __future__ import annotations

import heapq
import importlib.util
import math
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .topology import extract_lines_from_features, haversine_m

Coord = Tuple[float, float]  # (x=lon, y=lat)
Edge = Tuple[int, int, float]  # (u, v, weight_m)


def _round_pt_half_away(pt: Coord, ndigits: int = 6) -> Coord:
    scale = 10.0 ** float(ndigits)

    def _half_away(v: float) -> float:
        x = float(v) * scale
        if x >= 0.0:
            r = math.floor(x + 0.5)
        else:
            r = math.ceil(x - 0.5)
        return float(r) / scale

    return (_half_away(pt[0]), _half_away(pt[1]))


@dataclass
class RoadGraphPy:
    nodes: List[Coord]
    adj: List[List[Tuple[int, float]]]
    weight: Dict[Tuple[int, int], float]


def build_road_graph(features: List[Dict[str, Any]], ndigits: int = 6) -> Tuple[RoadGraphPy, List[Edge]]:
    lines = extract_lines_from_features(features)
    if not lines:
        raise ValueError("未从图层中提取到线要素（LineString/MultiLineString）")

    nodes: List[Coord] = []
    node_index: Dict[Coord, int] = {}
    edges: List[Edge] = []

    def _get_node(pt: Coord) -> int:
        key = _round_pt_half_away(pt, ndigits=ndigits)
        idx = node_index.get(key)
        if idx is not None:
            return idx
        idx = len(nodes)
        node_index[key] = idx
        nodes.append(key)
        return idx

    for _fid, parts in lines:
        for coords in parts:
            if not coords or len(coords) < 2:
                continue
            for i in range(1, len(coords)):
                a = coords[i - 1]
                b = coords[i]
                u = _get_node(a)
                v = _get_node(b)
                if u == v:
                    continue
                w = float(haversine_m(a, b))
                if w <= 0:
                    continue
                edges.append((u, v, w))

    if not edges:
        raise ValueError("路网中未产生任何有效边（可能坐标重复或数据异常）")

    adj: List[List[Tuple[int, float]]] = [[] for _ in range(len(nodes))]
    weight: Dict[Tuple[int, int], float] = {}
    for u, v, w in edges:
        # 保留最短的一条（避免重复边导致寻路不稳定）
        prev_uv = weight.get((u, v))
        if prev_uv is None or w < prev_uv:
            weight[(u, v)] = w
            weight[(v, u)] = w

    for (u, v), w in weight.items():
        adj[u].append((v, w))

    return RoadGraphPy(nodes=nodes, adj=adj, weight=weight), edges


def nearest_nodes(nodes: Sequence[Coord], point: Coord, k: int = 2) -> List[Tuple[int, float]]:
    if k <= 0:
        return []
    out: List[Tuple[float, int]] = []
    px, py = float(point[0]), float(point[1])
    for idx, (x, y) in enumerate(nodes):
        d = float(haversine_m((px, py), (float(x), float(y))))
        out.append((d, idx))
    out.sort(key=lambda t: t[0])
    return [(idx, d) for d, idx in out[: min(k, len(out))]]


def _reconstruct_path(parent: Dict[int, int], start: int, goal: int) -> List[int]:
    cur = goal
    path = [cur]
    while cur != start:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def _shortest_path_py(
    g: RoadGraphPy,
    start: int,
    goal: int,
    algo: str,
    banned_nodes: Optional[Set[int]] = None,
    banned_edges: Optional[Set[Tuple[int, int]]] = None,
) -> Optional[Tuple[float, List[int]]]:
    algo = (algo or "dijkstra").strip().lower()
    if algo == "floyd":
        algo = "dijkstra"

    banned_nodes = banned_nodes or set()
    banned_edges = banned_edges or set()

    if start in banned_nodes or goal in banned_nodes:
        return None

    dist: Dict[int, float] = {start: 0.0}
    parent: Dict[int, int] = {}
    pq: List[Tuple[float, int]] = []

    def h(n: int) -> float:
        if algo != "astar":
            return 0.0
        a = g.nodes[n]
        b = g.nodes[goal]
        return float(haversine_m(a, b))

    heapq.heappush(pq, (h(start), start))
    visited: Set[int] = set()

    while pq:
        prio, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == goal:
            return float(dist.get(goal, 0.0)), _reconstruct_path(parent, start, goal)
        du = dist.get(u)
        if du is None:
            continue
        # A* 里 pq 里存的是 du+h；这里用 du 做松弛即可
        for v, w in g.adj[u]:
            if v in banned_nodes:
                continue
            if (u, v) in banned_edges:
                continue
            nd = du + float(w)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd + h(v), v))

    return None


def k_shortest_paths_py(g: RoadGraphPy, start: int, goal: int, k: int = 2, algo: str = "dijkstra") -> List[Dict[str, Any]]:
    if k <= 0:
        return []

    first = _shortest_path_py(g, start, goal, algo=algo)
    if first is None:
        return []
    a_dist, a_path = first
    a: List[Tuple[float, List[int]]] = [(a_dist, a_path)]

    if k == 1:
        return [{"nodes": a_path, "distance_m": a_dist}]

    # Yen's algorithm (K=2 is enough for current需求)
    candidates: List[Tuple[float, Tuple[int, ...]]] = []
    cand_seen: Set[Tuple[int, ...]] = set()

    base_dist, base_path = a[0]
    for i in range(0, len(base_path) - 1):
        spur_node = base_path[i]
        root_path = base_path[: i + 1]

        banned_nodes: Set[int] = set(root_path[:-1])
        banned_edges: Set[Tuple[int, int]] = set()

        # remove the edge that would recreate the same root prefix
        for _dist_p, p in a:
            if len(p) > i + 1 and p[: i + 1] == root_path:
                banned_edges.add((p[i], p[i + 1]))

        spur = _shortest_path_py(g, spur_node, goal, algo=algo, banned_nodes=banned_nodes, banned_edges=banned_edges)
        if spur is None:
            continue
        spur_dist, spur_path = spur
        total_nodes = root_path[:-1] + spur_path
        t_nodes = tuple(total_nodes)
        if t_nodes in cand_seen:
            continue
        cand_seen.add(t_nodes)

        # compute total cost
        root_cost = 0.0
        ok = True
        for j in range(1, len(root_path)):
            u = root_path[j - 1]
            v = root_path[j]
            w = g.weight.get((u, v))
            if w is None:
                ok = False
                break
            root_cost += float(w)
        if not ok:
            continue

        total_cost = root_cost + float(spur_dist)
        heapq.heappush(candidates, (total_cost, t_nodes))

    if candidates:
        second_cost, second_nodes = heapq.heappop(candidates)
        a.append((float(second_cost), list(second_nodes)))

    return [{"nodes": p, "distance_m": float(d)} for d, p in a[:k]]


def _try_load_native() -> Any | None:
    try:
        from . import pathfinding_cpp as m  # type: ignore

        return m
    except Exception:
        pass

    build_dir = Path(__file__).resolve().parents[1] / "cpp_model" / "build"
    if not build_dir.exists():
        return None

    ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    candidates = [build_dir / f"pathfinding_cpp{ext}"]
    if not candidates[0].exists():
        candidates = sorted(build_dir.glob("pathfinding_cpp*.so"))
    if not candidates:
        return None

    so_path = candidates[0]
    modname = "space_app.algorithms.pathfinding_cpp"
    spec = importlib.util.spec_from_file_location(modname, so_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_pathfinding_cpp = _try_load_native()


def native_available() -> bool:
    return _pathfinding_cpp is not None


def build_native_graph(nodes: Sequence[Coord], edges: Sequence[Edge]):
    if _pathfinding_cpp is None:
        return None
    if hasattr(_pathfinding_cpp, "RoadGraph"):
        return _pathfinding_cpp.RoadGraph(list(nodes), list(edges))
    return None


def k_shortest_paths_native(native_graph, start: int, goal: int, k: int, algo: str) -> List[Dict[str, Any]]:
    if native_graph is None:
        return []
    if not hasattr(native_graph, "k_shortest_paths"):
        return []
    out = native_graph.k_shortest_paths(int(start), int(goal), int(k), str(algo))
    if not isinstance(out, dict):
        return []
    paths = out.get("paths")
    dists = out.get("distances")
    if not isinstance(paths, list) or not isinstance(dists, list):
        return []
    res: List[Dict[str, Any]] = []
    for p, d in zip(paths, dists):
        if not isinstance(p, list):
            continue
        try:
            nodes_list = [int(x) for x in p]
            dist_m = float(d)
        except Exception:
            continue
        res.append({"nodes": nodes_list, "distance_m": dist_m})
    return res

