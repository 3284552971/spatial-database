# 算法说明（索引 / 空间查询 / 拓扑 / 寻路 / 轨迹）

本文档总结我们在项目里使用的“仿真数据库 + GIS 工具链”的核心算法与数据结构，并标注对应的代码位置，便于我们维护和扩展。

> 我们约定：本项目的“表”不使用 Django ORM，而是自研落盘结构（pickle）；空间算法既有 Python 版本，也有 C++17 + pybind11 加速版本。

## 1. 仿真数据库与数据模型

### 1.0 用户管理与密码存储（PBKDF2）
代码：`space_app/user_store.py`（数据文件：`space_app/users.json`）

我们不在数据库里存明文密码，而是存“盐 + PBKDF2 派生密钥（digest）+ 轮数”：
- `salt`：16 bytes 真随机（`secrets.token_bytes(16)`），Base64 编码后写入 JSON
- `digest`：`PBKDF2-HMAC-SHA256(password, salt, rounds=200000)` 的输出（bytes），Base64 编码写入 JSON
- `rounds`：轮数（默认 `200000`），便于后续升级成本（可按用户记录分别存）

创建用户（`create_user`）算法：
1) `validate_username()`：用户名长度 3~32，字符集限制为 `[A-Za-z0-9_-]`，并禁止 `: / \\`（避免目录穿越与 table_id 分隔冲突）
2) 生成随机 `salt`（16 bytes）
3) 用 `hashlib.pbkdf2_hmac("sha256", password_utf8, salt, rounds)` 计算 `digest`
4) 把 `salt/digest` 用 Base64 编码为 ASCII 字符串，连同 `rounds` 一起写入 `users.json`

校验密码（`verify_user`）算法：
1) 从 `users.json` 读取 `salt/digest/rounds`
2) Base64 解码得到 `salt_bytes`、`digest_bytes`
3) 重新计算 `calc = PBKDF2-HMAC-SHA256(password, salt_bytes, rounds)`
4) 使用 `hmac.compare_digest(calc, digest_bytes)` 做常量时间比较（避免 timing side-channel）

落盘（`save_users`）策略：
- 采用原子写：先写 `users.json.tmp`，再 `os.replace()` 覆盖（避免写到一半导致文件损坏）

### 1.1 表结构（GeoJSON 入库）
代码：`space_app/models/table.py`

GeoJSON 导入后，表默认包含字段：
- `id`：要素主键（来自 GeoJSON feature.id；缺失时使用自增）
- `geom_type`：几何类型（Point/LineString/Polygon/Multi*）
- `geometry`：GeoJSON coordinates（保持原类型）
- `properties`：属性字典（dict）
- `lon` / `lat`：辅助字段（优先从 properties 推导，否则从 geometry 抽取首个点坐标）

落盘目录（多用户隔离）：
```
space_app/table_data/<username>/<table>/data_table.pkl
```

### 1.1.1 GeoJSON/JSON 输入规范化（含 Esri JSON）
代码：
- `space_app/algorithms/geojson_utils.py`
- `space_app/views.py::import_geojson_view`
- `space_app/models/table.py::ingest_geojson`

我们把“各种输入 JSON”统一规范化为 GeoJSON `FeatureCollection`，后续入库只处理一种格式，避免分支逻辑散落各处。

支持的输入类型与转换策略：
- `{"type":"FeatureCollection","features":[...]}`：校验 `features` 为数组后原样返回
- `{"type":"Feature", ...}`：包装成单要素的 FeatureCollection
- Geometry object：形如 `{"type":"Point|LineString|Polygon|...","coordinates":...}`，包装成 `Feature(properties={})`
- Esri JSON：顶层含 `geometryType`，并有 `features[i].geometry` 与 `features[i].attributes`

Esri -> GeoJSON 细节（`esri_to_featurecollection`）：
- `esriGeometryPoint`：`{x,y}` -> `Point([x,y])`
- `esriGeometryMultipoint`：`{points:[[x,y],...]}` -> `MultiPoint(points)`
- `esriGeometryPolyline`：`{paths:[[[x,y],...], ...]}` -> `LineString`（单 path）或 `MultiLineString`（多 path）
- `esriGeometryPolygon`：`{rings:[[[x,y],...], ...]}` -> `Polygon(rings)`
  - 我们会对 ring 做闭合：若 ring 首尾不相同，自动补一个首点到末尾（提高兼容性）

### 1.2 Manager 注册表（table/index/缓存）
代码：`space_app/models/manager.py`

Manager 维护：
- `MANAGER.table[table_id] -> data_table.pkl 路径`
- `MANAGER.index[table_id][(colA,colB,...)] -> index_xxx.pkl 路径`
- `MANAGER.rtree_cache[table_id] -> RTreeIndex 实例（内存缓存）`

注册表持久化：
- `space_app/models/manager_registry.json`
- `space_app/models/manager_registry.pkl`

并且会扫描 `space_app/table_data/` 兜底恢复，避免 registry 丢失导致“找不到表”。

## 2. 索引结构

### 2.1 哈希索引（HashIndex）
代码：`space_app/models/index.py`（`class HashIndex`）

用途：
- 主要用于主键索引（`type='primary'`）

核心思路：
- 使用哈希表数组 + 拉链法（链表 Node）
- key 通常是（单列或多列）组成的 tuple
- value 存储为“行号”（row index），用于定位 `DATA_table.data` 中的行

基本操作：
- `build_index(index_data)`：批量插入（key,value）
- `insert(key, value)`：插入单条
- `search(key)`：返回 value 或抛 KeyError
- `delete(key)`：删除 key

注意：
- 这里的 value 是“行号”，如果删除行或发生重排，索引可能需要重建（本项目多采用“插入后维护、其他情况允许重建”的策略）。

### 2.2 B+ 树索引（BPlusTreeindex）
代码：`space_app/models/index.py`（`class BPlusTreeindex`、`BtreeNode` 等）

用途：
- 普通索引（`type != 'primary'`）默认构建 B+ 树

核心结构：
- 内部节点：只存 key，不存记录；children 指向子节点
- 叶子节点：存 key 与记录指针（本项目中一般是行号），并用 `next` 指针串成有序链表，支持范围扫描

关键过程：
- `find_leaf(root, key)`：沿着内部节点路由到叶子
- `insert(key, value)`：
  - 叶子节点插入并保持 key 有序
  - 如果叶子溢出：`split_leaf()`
  - 内部节点溢出：`split_internal_node()`
  - 分裂后调用 `insert_toparent()` 把中间 key 上推

范围查询（典型 B+ 树能力）：
- 叶子链表 + 有序 key 使得 `>=` / `BETWEEN` 等查询可用“从起点叶子扫描”实现
- 目前项目 SQL 层主要实现了全表扫描的 WHERE；若要提速，我们可以把条件解析后走索引。

### 2.3 R-tree 空间索引（RTreeIndex）
代码：
- C++：`space_app/cpp_model/R_tree.cpp`、`space_app/cpp_model/rtree_bindings.cpp`
- Python 侧加载与落盘：`space_app/models/table.py`

用途：
- 空间查询加速（按位置选择、bbox/circle 查询等）

存储对象：
- Point / Polyline / Polygon 三类几何都会转换为 bbox 写入 R-tree
- attributes 会把 properties（以及项目注入的 `__table_id`）一起写入，便于命中后映射回表主键

关键点：
- **Linear Split** 分裂策略：`R_tree::split()`（构建速度与查询质量的折中）
- 序列化：
  - `rtree_serialized.json`：C++ 端序列化（加载快）
  - `rtree_source.geojson`：源数据副本（当版本不匹配或序列化失效时可重建）
- 版本机制：`rtree_version.txt`
  - 当属性结构/语义变化时递增版本，触发自动重建，避免“命中 id 对不上导致选中错位”

## 3. 空间选择（Selection）

### 3.1 按属性选择
代码：`space_app/algorithms/selection.py` + API `space_app/views.py::select_by_attribute_view`

思路：
- 对每行构造 `row_dict`，支持字段引用：
  - 顶层列：`id / lon / lat / geom_type ...`
  - 属性子字段：`properties.xxx` 或直接 `xxx`（从 properties 中取）
- 运算符支持：
  - 数值/通用：`= != > < >= <=`
  - 字符串：`contains`（包含子串）、`not_contains`（不包含子串）
  - 通配符：`wildcard`（也兼容别名 `like`）
- 返回：
  - `ids`：命中要素 id 列表
  - `bboxes`：每个命中要素的 bbox（前端用于画框/圆等标注）

多约束（AND/OR）：
- API 入参 `conditions` 支持多条约束，按“从左到右”组合：
  - 第 1 条：只有 `{field, op, value}`
  - 从第 2 条开始：增加 `join`（或兼容字段 `logic/rel`），取值 `AND/OR`
- 组合语义是左结合：`c1 (join2) c2 (join3) c3 ...`

示例：
```json
{
  "table": "admin:traj",
  "conditions": [
    { "field": "properties.线路", "op": "contains", "value": "E17" },
    { "join": "AND", "field": "id", "op": ">", "value": "1000" },
    { "join": "OR", "field": "properties.车牌号", "op": "wildcard", "value": "*A?B*" }
  ]
}
```

#### 字符串模糊 / 通配符（不引入第三方库）
实现：`selection.py::_wildcard_match(text, pattern)`（纯 Python）

规则：
- `contains`：判断 `value` 是否为字段值的子串（当前为大小写敏感，按 `str(value) in str(field_value)`）
- `not_contains`：`contains` 的取反
- `wildcard` / `like`：glob 风格匹配
  - `*` 或 `%`：匹配任意长度字符串（含空串）
  - `?` 或 `_`：匹配任意单字符
  - `\\`：转义符（例如 `\\*` 表示字面量 `*`，`\\\\` 表示字面量 `\\`）

算法（回溯指针）：
- 先把 `pattern` 编译成 token 序列：字面量、`*`、`?`
- 双指针扫描 `text` 与 token：
  - 遇到字面量/`?` 尝试推进
  - 遇到 `*` 记录“星号位置”和“当时的文本位置”，先贪婪推进 pattern
  - 后续匹配失败时，如果存在记录的 `*`，就回溯到 `*` 后一个 token，并让文本位置前进 1，继续尝试
- 文本扫完后，允许 pattern 末尾剩余若干 `*`，最终 token 也消耗完则匹配成功

### 3.2 按位置选择
代码：`space_app/views.py::select_by_location_view` + `space_app/algorithms/selection.py`

模式：
- `point`：给定圆心 + 半径（米）
- `bbox_intersects`：与参照要素 bbox 相交
- `circle_from_selected`：以参照要素 bbox 中心为圆心，做圆形范围查询

加速策略：
- 优先使用 R-tree（如果 `rtree_module` 可用且该表已构建/可重建索引）
- 否则回退到 Python 扫描：先 bbox 粗过滤，再做圆-矩形相交测试

## 4. 拓扑检查与修复（线图层）
代码：
- Python：`space_app/algorithms/topology.py`
- C++：`space_app/cpp_model/topology_bindings.cpp`（可选加速）
- API：`space_app/views.py::topology_check_view/topology_repair_view/topology_save_view`

### 4.1 拓扑检查（常见问题类型）
针对线要素（LineString/MultiLineString），典型检测：
- `cross_without_node`：两条线相交但交点不是节点（需要打断并插入节点）
- `endpoint_on_segment`：一条线端点落在另一条线的内部（T 形连接缺节点）
- `dangling_endpoint`：悬挂端点（端点没有与其他线连接）

输出：
- issue 列表（点位置 + kind + message）
- stats（计数等）

### 4.2 拓扑修复（当前策略）
修复不会直接修改源表目录，流程是：
1) 把源表目录复制到项目内临时目录 `.tmp/topology/...`
2) 临时将 `MANAGER.table[table_id]` 指向临时副本
3) 在临时副本上计算修复结果，生成“新线段/删除建议”
4) 结束后恢复指针并删除临时目录（除非调试 keep_tmp）

修复输出一般包含：
- `new_lines`：建议新增/替换的线段集合（用于前端渲染“绿色修复线”）
- `deleted`：建议删除的原线 id 集合（例如短悬挂线）

持久化：
- `topology_save` 会把修复后的 FeatureCollection 导入为新表（落盘 + 可再加载验证）

## 5. 寻路（最短路 / K 最短路）
代码：
- Python：`space_app/algorithms/pathfinding.py`
- C++：`space_app/cpp_model/pathfinding_bindings.cpp`
- API：`space_app/views.py::shortest_path_view`

### 5.1 路网图构建
从线要素提取线段：
- 把坐标点按精度（默认 ndigits=6）做“半舍五入”，去噪并合并近似节点
- 每条线的相邻点形成边（双向），权重为 haversine 距离（米）

### 5.2 起终点候选
起点/终点是地图点选的任意点（不一定落在节点上）：
- 在路网节点中选最近的候选点集合（当前：起点 2 个、终点 2 个）

### 5.3 算法
支持选择：
- Dijkstra
- A*（启发式为 haversine 距离）
- Floyd（前端提供选项，但后端会回退到 Dijkstra；避免大图 O(n^3)）

输出：
- 对每对 (startCandidate, endCandidate) 计算 `K=2` 条候选路径（Yen’s algorithm）
- 最终最多 2*2*2=8 条 polyline，作为临时图层输出到前端

## 6. 轨迹校正（HMM map matching + Kalman）
代码：
- C++：`space_app/cpp_model/trajectory_bindings.cpp`（手写实现，无第三方算法库）
- Python loader：`space_app/algorithms/trajectory.py`
- API：`space_app/views.py::trajectory_correct_view`

输入：
- 轨迹点图层（Point）
- 时间字段（支持 ISO `2025-06-25T01:06:31`/`2025-07-01 09:30:01` 或 Unix 秒/毫秒时间戳；不能为空）
- 路网线图层（构建路网图时复用寻路的缓存）

时间顺序策略（后端容错）：
- 我们会先按时间排序
- 如果存在重复/逆序时间戳，我们会对后续点做 **1ms 级别的微小抖动**，强制变为严格递增（Kalman 需要 `dt>0`）

### 6.1 候选生成（candidate generation）
为了速度与工程实现简洁，本项目当前以“路网节点”为候选状态：
- 把路网节点投影到局部平面（以首点纬度做 equirectangular 近似）
- 建一个简单网格索引（grid hash），快速找每个观测点最近的 K 个节点（默认 K=2）

### 6.2 HMM（Viterbi）
状态：每个时刻的候选节点集合。

代价函数：
- 发射代价 emission：观测点到候选节点的距离（米）的高斯负对数（`d^2 / (2σ_e^2)`）
- 转移代价 transition：用“候选节点间直线距离”近似路网距离，与相邻观测点位移对比：
  - `((d_candidate - d_obs)^2) / (2σ_t^2)`

求解：
- Viterbi 动态规划得到全局最小代价路径（每个时刻选一个候选节点）

输出：
- 匹配后的经纬度（取候选节点坐标）
- `matched_node_index`：对应节点索引（用于调试/复现）

### 6.3 Kalman（速度/航向估计）
模型：匀速（CV）状态空间，状态为 `[x, y, vx, vy]`。

流程：
- 观测：匹配后的 (x,y)
- 预测：`x+=vx*dt`, `y+=vy*dt`
- 更新：使用测量噪声 `R` 与过程噪声 `Q`（加速度噪声）修正

输出字段（写入结果点图层 properties）：
- `est_speed_mps`：`sqrt(vx^2+vy^2)`（m/s）
- `est_heading_deg`：`atan2(vy,vx)` 转角度，正东为 0°，范围 [0,360)

## 7. SQL 执行与删表（DROP TABLE）
代码：
- SQL 解析：`space_app/models/sql_router.py`
- SQL 执行：`space_app/views.py::sql_view`

支持语句：
- `SELECT / INSERT / UPDATE / DELETE FROM / CREATE TABLE / DROP TABLE`

`DROP TABLE` 的行为：
- 删除 `space_app/table_data/<owner>/<table>/` 整个目录
- 清理 `MANAGER.table/index/rtree_cache` 等缓存并持久化 registry

多用户权限：
- 普通用户：只能操作自己的表
- `admin`：可操作所有用户表（用 `owner:table` 或 `owner.table` 指定）
