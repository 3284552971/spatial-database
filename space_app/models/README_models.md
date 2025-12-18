# 模型与存储（仿真数据库）

我们在本目录实现了一个轻量“仿真数据库”：
- 表数据落盘为 `data_table.pkl`（pickle）
- 索引（B+ / Hash / R-tree）可选落盘或缓存
- 与 Django ORM 无关（Django 的 `db.sqlite3` 仅用于 session）

## 多用户目录结构
所有表按用户隔离存储：
```
space_app/table_data/<username>/<table>/
  data_table.pkl
  rtree_source.geojson
  rtree_serialized.json
  rtree_version.txt
  index_*.pkl
```

`admin` 用户可访问所有用户目录；普通用户只能访问自己的目录。

## Manager（注册表与持久化）
核心入口：`space_app/models/manager.py`

启动行为：
- 先尝试从 `space_app/models/manager_registry.pkl` / `manager_registry.json` 恢复注册表
- 无论是否恢复成功，都会扫描 `space_app/table_data/` 补全（避免 registry 为空）
- 若发现旧结构 `table_data/<table>/...`，会自动迁移到 `table_data/admin/<table>/...`

写入行为：
- 创建表 / 导入 GeoJSON / 创建索引后会自动 `save_registry()`（json+pkl 原子替换）

## Table（表与 GeoJSON 入库）
核心：`space_app/models/table.py`

GeoJSON 入库 `ingest_geojson()` 会生成字段：
- `id / geom_type / geometry / properties / lon / lat`

说明：
- `geom_type + geometry` 用于保持原始要素类型（Point/LineString/Polygon/Multi*）
- `properties` 保留原始属性（dict）
- `lon/lat` 仅用于地图展示/辅助筛选：优先从 `properties` 推导，否则从 `geometry.coordinates` 提取首个点

## SQL 支持（路由解析）
SQL 解析器在 `space_app/models/sql_router.py`，由 `space_app/views.py::sql_view()` 执行。

支持：
- `SELECT / INSERT / UPDATE / DELETE FROM / CREATE TABLE / DROP TABLE`

表名引用支持：
- `table`（普通用户默认 owner=自己）
- `owner:table`（admin 可用）
- `owner.table`（兼容）

删表：
```sql
DROP TABLE admin:trajectory;
```
该操作会：
- 删除 `space_app/table_data/<owner>/<table>/` 整个目录
- 清理注册表（`MANAGER.table/index/rtree_cache` 等）并落盘

## R-tree（可选 C++ 扩展）
R-tree 绑定模块：`space_app/cpp_model/rtree_bindings.cpp` → `rtree_module*.so`

表入库时会保存：
- `rtree_source.geojson`：可重建源
- `rtree_serialized.json`：C++ 端序列化（版本匹配时优先快速加载）

注意：R-tree 命中结果通过属性 `__table_id` 映射回 hookup 表主键，避免“选中变色错位”。
