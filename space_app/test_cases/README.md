# Test cases

我们在本目录放置“可直接运行”的轻量测试脚本（不依赖 pytest）。

## 运行方式
在项目根目录（有 `manage.py` 的目录）执行：

```bash
python space_app/test_cases/test_01_table_lonlat_upgrade.py
python space_app/test_cases/test_02_ingest_geojson_lonlat.py
python space_app/test_cases/test_03_geojson_endpoint.py
python space_app/test_cases/test_04_topology_backend_force.py
python space_app/test_cases/test_05_topology_backend_auto_smoke.py
python space_app/test_cases/test_06_topology_repair_cross_coverage.py
python space_app/test_cases/test_07_cross_only_fix_verify.py
python space_app/test_cases/test_08_t_junction_endpoint_on_segment_fix.py
```

## 覆盖点
- `test_01_table_lonlat_upgrade.py`
  - 验证现有落盘表（`space_app/table_data/*/data_table.pkl`）是否具备 `lon/lat` 字段。
  - 抽样检查若干行的 `lon/lat` 是否可解析为 float（允许为 None）。

- `test_02_ingest_geojson_lonlat.py`
  - 对任意一个 GeoJSON（例如 `space_app/shenzhen/road_sz.geojson`）进行重新入库到临时表名。
  - 验证新表 schema 包含 `lon/lat`。
  - 测试结束会清理临时表目录，避免污染 `table_data/`。

- `test_03_geojson_endpoint.py`
  - 使用 Django `Client` 模拟浏览器：先登录 `/space_app/login/`，再请求 `/space_app/geojson/<table>/`。
  - 验证响应状态码为 200 且返回合法 GeoJSON `FeatureCollection`。

- `test_04_topology_backend_force.py`
  - 强制 Python / 强制 C++（若 native 可用）两条路径可运行。
  - 断言能识别 `cross_without_node` 与 `dangling_endpoint`。

- `test_05_topology_backend_auto_smoke.py`
  - 默认 auto 模式：native 可用则优先走 C++，调用不报错。
  - 当 C++ 执行报错时应自动回退到 Python。

- `test_06_topology_repair_cross_coverage.py`
  - 构造“一条线有大量交点”的场景。
  - 防止交叉修复只处理少量交点（例如早期的每条线最多 50 个交点）。

- `test_07_cross_only_fix_verify.py`
  - 只检查交叉拓扑（`cross_without_node`）。
  - 只修复交叉拓扑（插入交点并切分生成新线段），并验证修复后 `cross_without_node` 为 0。

- `test_08_t_junction_endpoint_on_segment_fix.py`
  - 覆盖“端点落在别的线段内部”（`endpoint_on_segment` / T 形连接缺节点）。
  - 验证修复会切分宿主线，使该点成为节点，并消除 `endpoint_on_segment`。

> 说明：这些脚本是“回归检查”而不是单元测试框架；优点是依赖少、跑起来快。
