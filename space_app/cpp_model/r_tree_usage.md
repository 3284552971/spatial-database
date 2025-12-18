# C++ 扩展使用说明（R-tree / 拓扑 / 寻路 / 轨迹）

我们在本目录通过 CMake 构建多个 pybind11 模块，统一输出到 `space_app/cpp_model/build/`。

## 依赖与构建（apt）
Ubuntu/Debian：
```bash
sudo apt update
sudo apt install -y build-essential cmake pybind11-dev nlohmann-json3-dev libyaml-cpp-dev
```

构建（项目根目录执行，指定当前虚拟环境 python）：
```bash
rm -rf space_app/cpp_model/build
cmake -S space_app/cpp_model -B space_app/cpp_model/build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=/path/to/python
cmake --build space_app/cpp_model/build -j 2
```

产物示例：
- `rtree_module.cpython-38-*.so`
- `topology_cpp.cpython-38-*.so`
- `pathfinding_cpp.cpython-38-*.so`
- `trajectory_cpp.cpython-38-*.so`

## 模块简介
- `rtree_module`：空间索引（被“按位置选择”等功能使用）
- `topology_cpp`：拓扑检查/修复加速
- `pathfinding_cpp`：Dijkstra/A*/K shortest paths
- `trajectory_cpp`：轨迹 HMM 匹配 + Kalman（速度/航向）

## 示例数据
- 目录：`space_app/shenzhen/` 包含 `road_sz.geojson`, `transport_sz.geojson`, `pois_sz.geojson`
- 示例使用 `road_sz.geojson`（线要素丰富）

## Python 加载注意事项（强烈建议）

由于 `space_app/cpp_model/build/` 里可能残留不同 Python 版本编译出来的 `.so`，
我们**不要用通配符随便挑一个 `.so`**；应该按当前解释器的 `EXT_SUFFIX` 精确加载。

推荐方式 1（最简单）：把 build 目录加入 `sys.path` 然后 `import rtree_module`。

推荐方式 2（最稳健）：按 `EXT_SUFFIX` 精确加载：

```python
import importlib.util
import sys
import sysconfig
from pathlib import Path

build_dir = Path('space_app/cpp_model/build').resolve()
ext = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
so_path = build_dir / f'rtree_module{ext}'

spec = importlib.util.spec_from_file_location('rtree_module', so_path)
module = importlib.util.module_from_spec(spec)
sys.modules['rtree_module'] = module
spec.loader.exec_module(module)

rtree_module = module
```

## rtree_module：Python 调用（含序列化）
返回值结构（列表元素）：
```python
{
    "id": int,                 # 内部递增ID
    "type": "point"|"polyline"|"polygon",
    "bbox": {"minx": float, "miny": float, "maxx": float, "maxy": float},
    "attributes": {str: str}   # GeoJSON properties 转成字符串
}
```

示例代码（线类型过滤查询）：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('space_app/cpp_model/build').resolve()))
import rtree_module

idx = rtree_module.RTreeIndex()
idx.load_from_file(str(Path('space_app/shenzhen/road_sz.geojson').resolve()))

# 取文件中一段道路的包围盒进行查询（仅线类型）
minx, miny, maxx, maxy = 114.1080567, 22.5393866, 114.108805, 22.5415858
results = idx.query_box(minx, miny, maxx, maxy, 'polyline')
print(f"hits={len(results)}")
if results:
    f0 = results[0]
    print(f0['bbox'], f0['attributes'])
```
期望输出（示例）：
```
hits=4
{'minx': 114.1080567, 'miny': 22.5393866, 'maxx': 114.108805, 'maxy': 22.5415858} {'FID': '62125.000000', 'bridge': 'F', 'oneway': 'B', ...}
```

## 直接加载并查询三类数据
- 道路：`idx.load_from_file('road_sz.geojson')`，查询线要素。
- 交通设施：`transport_sz.geojson`，要素类型根据 `geometry.type` 自动识别。
- POI 点：`pois_sz.geojson`，查询点要素时使用 `type_filter='point'`。

## 序列化 / 反序列化（每表一份 JSON）
```python
import rtree_module
idx = rtree_module.RTreeIndex()
idx.load_from_file('space_app/shenzhen/road_sz.geojson')
idx.save_serialized('table_data/roads_sz/rtree_serialized.json')

idx2 = rtree_module.RTreeIndex()
idx2.load_serialized('table_data/roads_sz/rtree_serialized.json')
print(len(idx2.query_box(113.8, 22.4, 114.1, 22.7, 'polyline')))
```

## 删除示例
```python
idx.delete_by_attribute('FID', '62125.000000')  # 属性值保证唯一时可直接删除
```

## 实现细节：分裂策略（Linear Split，性价比最高）

当前 `R_tree::split()` 使用 **Linear Split**（线性分裂）作为默认策略，原因是它在
“分裂质量（重叠更小）”与“构建速度（更快）”之间的性价比最好，适合课程项目/工程快速迭代。

关键点：

- **触发条件**：插入后，叶节点 `features.size() > max_features_per_node` 时触发分裂。
    - 当前默认 `max_features_per_node = 4`，因此叶子达到 5 个要素会触发分裂。
- **选种子（O(n)）**：分别在 X/Y 轴做一次扫描，计算归一化分离度并选择更优轴。
    - 对某一轴：
        - `min_low = min(b.minx)`、`max_high = max(b.maxx)`
        - `max_low = max(b.minx)`（对应索引 `idx_max_low`）
        - `min_high = min(b.maxx)`（对应索引 `idx_min_high`）
        - 分离度：$sep = \max(0, (max\_low - min\_high) / (max\_high - min\_low))$
    - 选择 X/Y 中 `sep` 更大的轴，种子为 `idx_max_low` 与 `idx_min_high`。
    - 退化处理：若种子碰巧相同，会退回到“选中心最远的另一项”。
- **分配策略**：对剩余项逐个分配到 left/right：
    1) 优先让 **面积扩张（enlargement）最小** 的那边接收；
    2) 扩张相同则选当前 bbox **面积更小** 的；
    3) 再平手则选当前 **元素更少** 的（避免极端偏斜）。
- **最小填充（min-fill）**：分裂时强制每个子节点至少拿到 $\lceil M/2 \rceil$ 个项。
    - 其中 $M = max\_features\_per\_node + 1$（也就是本次溢出后的总项数）。

注：分裂/插入全程直接使用 `Feature.bbox`，避免重复从几何重算 bbox。

## pathfinding_cpp / trajectory_cpp 的加载
- `space_app/algorithms/pathfinding.py` / `space_app/algorithms/trajectory.py` 会优先 `import xxx_cpp`，失败则按 `EXT_SUFFIX` 去 `cpp_model/build/` 精确加载对应 `.so`。
