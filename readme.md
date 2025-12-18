# Spatial：仿真空间数据库 + 地图可视化 + C++ 算法扩展

我们做了一个“课程级”的空间数据平台：后端用 Django 提供登录、多用户隔离、SQL 操作与算法 API；前端用 Leaflet 做地图与交互；核心算法（R-tree、拓扑、最短路、轨迹校正）在 Python 与 C++（pybind11）之间组合实现。

在线文档页：`/space_app/docs/`（地图/SQL/登录页面右上角也有“文档”入口）。

## 快速导航
- 快速启动：见「快速启动」
- 算法说明：`docs/ALGORITHMS.md`
- 架构图：`docs/ARCHITECTURE.svg`

## 技术栈
- 后端：Python 3.8+（当前环境常见为 conda 3.8），Django 4.2.x
- 前端：原生 HTML/JavaScript + Leaflet
- 空间索引/算法：
  - Python：拓扑检查/修复（回退实现）、属性/位置选择等
  - C++17 + pybind11：`rtree_module`、`topology_cpp`、`pathfinding_cpp`、`trajectory_cpp`
- 数据存储：
  - Django 自带 SQLite：`db.sqlite3`（仅用于 session 等）
  - “仿真数据库”（自研）：`space_app/table_data/<username>/<table>/data_table.pkl` + 索引/rtree 文件

## 快速启动
1) 安装 Python 依赖
```bash
pip install -r requirements.txt
```

> 依赖提示：
> - 如果使用 Python >= 3.9，标准库已包含 `zoneinfo`，通常不需要 `backports.zoneinfo`；若安装报错，可把 `requirements.txt` 中的 `backports.zoneinfo` 改成带环境标记的写法（例如 `backports.zoneinfo==0.2.1; python_version<'3.9'`）或直接移除后重装。

2) 启动服务
```bash
python manage.py runserver 127.0.0.1:8000
```

3) 访问入口
- 登录：`http://127.0.0.1:8000/space_app/login/`
- 地图：`http://127.0.0.1:8000/space_app/map/`
- SQL：`http://127.0.0.1:8000/space_app/sql/`
- 文档：`http://127.0.0.1:8000/space_app/docs/`

## 账号与多用户模式
- 用户存储文件：`space_app/users.json`（PBKDF2 哈希，不明文）
- 系统会在启动时确保存在 `admin` 用户（默认密码：`zblnb666`）
- 数据隔离：
  - 普通用户：只能看到/操作 `space_app/table_data/<自己的用户名>/...`
  - `admin`：可访问所有用户的表（支持 `owner:table`）

## 仿真数据库：表结构与落盘
### 目录结构
```
space_app/table_data/
  admin/
    roads_sz/
      data_table.pkl
      rtree_source.geojson
      rtree_serialized.json
      rtree_version.txt
      index_*.pkl
```

### 入库表字段（GeoJSON）
GeoJSON 导入后，表默认包含：
- `id`：主键
- `geom_type`：Point/LineString/Polygon/Multi*
- `geometry`：coordinates（保留原始类型）
- `properties`：原始属性（dict）
- `lon/lat`：用于点选/展示的辅助字段（从 properties 或 geometry 推导）

## 前端功能概览（map 页）
- 加载表为图层（按当前地图视野 bbox 请求，减少一次性渲染量）
- 左侧图层视图：图层显隐、移除、清除选择、右键菜单（属性表/导出/全选等）
- 工具栏：
  - GeoJSON 导入（入库 + R 树）
  - 拓扑检查/修复（可选“持久化保存为新表”）
  - 按属性选择 / 按位置选择
  - 导出要素（已选择要素 → 新表 / GeoJSON 下载）
  - 最短路径（路网图层 + 起终点点选 + 算法选择）
  - 轨迹校正（轨迹点图层 + 时间字段 + 路网图层 → HMM 匹配 + Kalman 速度/航向）

## 文档与架构
- 文档页面：`/space_app/docs/`
- 算法说明：`docs/ALGORITHMS.md`
- 架构图（SVG）：`docs/ARCHITECTURE.svg`

## SQL 界面（支持与限制）
SQL 页面：`/space_app/sql/`

支持语句：
- `SELECT ... FROM <table> ...`
- `INSERT INTO <table> ...`
- `UPDATE <table> SET ... WHERE ...`
- `DELETE FROM <table> WHERE ...`（删行）
- `CREATE TABLE <table> (...)`
- `DROP TABLE <table>`（删表：会删除磁盘目录与注册表）

表引用写法：
- `table`（普通用户默认指自己）
- `owner:table`（admin 可跨用户）
- `owner.table`（兼容写法）

示例：
```sql
DROP TABLE admin:trajectory;
```

## C++ 扩展构建（推荐）
本项目用 CMake 构建 pybind11 模块，产物输出到 `space_app/cpp_model/build/`。

依赖（Ubuntu/Debian，apt）：
```bash
sudo apt update
sudo apt install -y build-essential cmake pybind11-dev nlohmann-json3-dev libyaml-cpp-dev
```

构建（在项目根目录执行，指向我们当前的虚拟环境 Python）：
```bash
rm -rf space_app/cpp_model/build
cmake -S space_app/cpp_model -B space_app/cpp_model/build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=/path/to/python
cmake --build space_app/cpp_model/build -j 2
```

验证（可选）：
```bash
python -c "import rtree_module, topology_cpp, pathfinding_cpp, trajectory_cpp; print('cpp modules ok')"
```

模块：
- `rtree_module`：空间索引（用于按位置选择/空间查询加速）
- `topology_cpp`：拓扑检查/修复加速
- `pathfinding_cpp`：Dijkstra/A*/K shortest paths
- `trajectory_cpp`：轨迹 HMM 匹配 + Kalman 估计（速度/航向）

## 常见问题
### 1) /favicon.ico 404 是什么
浏览器会自动请求网站图标 `favicon.ico`，本项目未提供该文件所以日志会 404，不影响功能。

### 2) 底图加载失败是不是“梯子”原因
OSM 瓦片是在前端浏览器直接向 `tile.openstreetmap.org` 请求的，不经过我们的后端分发；网络受限会导致底图加载失败。当前 map 默认不自动加载底图，需要手动点“加载底图”。

## License
见 `LICENSE`。

## 署名
- 作者：
  - 六舅：357801@whut.edu.cn
  - 鸿少：3087592085@qq.com
  - 伟哥：3563447571@qq.com
  - 春春：3128438988@qq.com
  - 小白：3367781424@qq.com
- 通讯：
  - 357801@whut.edu.cn
