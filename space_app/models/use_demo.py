"""综合回归示例：导入线/点底图，属性查询、更新、空间查询，R 树序列化验证。"""

from pathlib import Path
from manager import MANAGER

ROOT = Path(__file__).resolve().parent.parent

# 1) 导入道路（FeatureCollection LineString），自动建表 + 索引 + R 树
road_geo = ROOT / 'shenzhen' / 'road_sz.geojson'
MANAGER.ingest_geojson('roads_sz', road_geo)

# 2) 属性查询：住宅道路数量
roads_res = MANAGER.query('roads_sz', predicate=lambda r: r['properties'].get('fclass') == 'residential')
print('roads_sz residential count:', len(roads_res))

# 3) 属性更新：给 id=0 的道路打标记
MANAGER.update_rows(
    'roads_sz',
    predicate=lambda r: r['id'] == 0,
    updater=lambda r: {'properties': {**r['properties'], 'tag': 'reviewed'}},
)

# 4) 导入 POI 点（Esri 点 JSON 自动转换为 FeatureCollection）
pois_geo = ROOT / 'shenzhen' / 'pois_sz.geojson'
MANAGER.ingest_geojson('pois_sz', pois_geo)

# 5) 导入交通设施点（同上）
transport_geo = ROOT / 'shenzhen' / 'transport_sz.geojson'
MANAGER.ingest_geojson('transport_sz', transport_geo)

# 6) 空间查询（道路与 POI）
rtree_roads = MANAGER.load_rtree('roads_sz')
rtree_pois = MANAGER.load_rtree('pois_sz')
if rtree_roads:
    hits = rtree_roads.query_box(113.8, 22.4, 114.1, 22.7, 'polyline')
    print('roads_sz box hits:', len(hits))
if rtree_pois:
    hits = rtree_pois.query_box(113.8, 22.4, 114.1, 22.7, 'point')
    print('pois_sz box hits:', len(hits))

# 7) R 树序列化（C++ 端 JSON 序列化）验证：保存后重载并再次查询
if rtree_pois:
    ser_path = ROOT / 'table_data' / 'pois_sz' / 'rtree_serialized.json'
    ser_path.parent.mkdir(parents=True, exist_ok=True)
    if rtree_pois.save_serialized(str(ser_path)):
        try:
            import rtree_module
            fresh = rtree_module.RTreeIndex()
            if fresh.load_serialized(str(ser_path)):
                hits2 = fresh.query_box(113.8, 22.4, 114.1, 22.7, 'point')
                print('pois_sz box hits after reload:', len(hits2))
        except Exception as exc:  # noqa: BLE001
            print('R-tree serialized reload failed:', exc)

print('demo done')
