from __future__ import annotations

import os
import sys
import json
import importlib.util
import sysconfig
from typing import Any, List, Dict, Callable, Iterable, Optional, Tuple
from pathlib import Path
import pickle as pkl
from .index import Index

file_path = Path(__file__)
data_path = file_path.parent.parent / 'table_data'
data_path.mkdir(parents=True, exist_ok=True)
cpp_build_path = file_path.parent.parent / 'cpp_model' / 'build'

# R-tree 持久化版本号：当 R-tree 属性结构/语义变化时递增，触发自动重建。
# v2: 在属性中注入 __table_id，用于把 R-tree 命中映射回表主键 id。
RTREE_VERSION = "2_table_id_attrs"

class ROW:
    def __init__(self, data:List[Any]):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, index:int) -> Any:
        if isinstance(index, int):
            if index < 0 or index >= len(self.data):
                raise IndexError("索引超出范围")
            return self.data[index]
        elif isinstance(index, str):
            if index in self.columns:
                for idx, col in enumerate(self.columns):
                    if col == index:
                        return self.data[idx]
            else:
                raise ValueError(f"Column {index} does not exist")
        else:
            raise TypeError("Column index must be an integer or string")

    def __len__(self) -> int:
        return len(self.data)

class DATA_table:
    def __init__(self, attributes:Dict[str,int], primary_key:tuple, data:Optional[List[ROW]]=None):
        self.attributes = attributes
        self.primary_key = primary_key
        self.data = data if data is not None else []

    def add_row(self, row:ROW):
        if not isinstance(row, ROW):
            raise TypeError("Only ROW instances can be added")
        self.data.append(row)

    def column_views(self, column_name:list[str]) -> List[Any]:
        for col in column_name:
            if col not in self.attributes:
                raise ValueError(f"Column {col} does not exist")
        view = {} # column_name: List[Any]
        for col in column_name:
            col_index = self.attributes[col]
            view[col] = [row[col_index] for row in self.data]
        return view
    def __getitem__(self, key:int) -> ROW:
        if isinstance(key, int):
            if key < 0 or key >= len(self.data):
                raise IndexError("Row index out of range")
            return self.data[key]
        else:
            raise TypeError("Row index must be an integer")
        
    def __getattr__(self, name:str) -> List[Any]:
        if 'attributes' not in self.__dict__:
            raise AttributeError(name)
        if not isinstance(name, str):
            raise TypeError("Attribute name must be a string")
        if name in self.attributes:
            col_index = self.attributes[name]
            return [row[col_index] for row in self.data]
        raise AttributeError(f"'DATA_table' object has no attribute '{name}'")
        
    def __len__(self) -> int:
        return len(self.data)

class Table:
    """
    表类
    args:
        table_name: string 表名
        columns: list[string] 列名列表
        primary_key: list[string] 主键列名
        must------------------------------------------select
        data: list[list] 数据列表
        indexes: list[string] 索引列名列表
        foreign_keys: dict[local_column: [referenced_table, referenced_column]] 外键字典
        checks: dict{column: [def,···]} 检查约束列表
    """

    def _ensure_rtree_cache(self) -> None:
        if not hasattr(self, 'rtree_cache'):
            self.rtree_cache = {}

    def _table_dir_for_name(self, table_name: str, table_dir: Optional[Path] = None) -> Path:
        """根据 table_name + Manager 注册信息推导真实落盘目录。"""
        if table_dir is not None:
            return Path(str(table_dir))
        try:
            table_path = getattr(self, "table", {}).get(table_name)
            if table_path:
                return Path(str(table_path)).parent
        except Exception:
            pass
        return data_path / table_name

    def _load_rtree_module(self):
        """按需加载 rtree_module，缺少构建时返回 None。"""
        if '_rtree_module_cached' in self.__dict__:
            return self._rtree_module_cached
        # Prefer normal import (e.g. if user installed/put it on PYTHONPATH),
        # otherwise load the exact build artifact for the current interpreter.
        try:
            import rtree_module  # type: ignore

            self._rtree_module_cached = rtree_module
            return self._rtree_module_cached
        except Exception:
            pass

        try:
            if not cpp_build_path.exists():
                self._rtree_module_cached = None
                return self._rtree_module_cached

            ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
            so_path = cpp_build_path / f"rtree_module{ext}"
            if not so_path.exists():
                self._rtree_module_cached = None
                return self._rtree_module_cached

            spec = importlib.util.spec_from_file_location("rtree_module", so_path)
            if spec is None or spec.loader is None:
                self._rtree_module_cached = None
                return self._rtree_module_cached
            module = importlib.util.module_from_spec(spec)
            sys.modules["rtree_module"] = module
            spec.loader.exec_module(module)
            self._rtree_module_cached = module
        except Exception as exc:  # pylint: disable=broad-except
            print(f"rtree_module not available, skip spatial index build: {exc}")
            self._rtree_module_cached = None
        return self._rtree_module_cached

    @staticmethod
    def _extract_geom_parts(feature: Dict[str, Any]) -> List[Tuple[str, List[List[float]]]]:
        """将 GeoJSON 要素拆成 R 树需要的 (type, coords) 列表，处理 Multi*。"""
        geom = feature.get("geometry") or {}
        gtype = (geom.get("type") or "").lower()
        coords = geom.get("coordinates")
        parts: List[Tuple[str, List[List[float]]]] = []

        if gtype == "point" and isinstance(coords, list):
            if coords and isinstance(coords[0], (int, float)):
                parts.append(("point", [coords[:2]]))
        elif gtype == "linestring" and isinstance(coords, list):
            polyline = [c[:2] for c in coords if isinstance(c, list) and len(c) >= 2]
            if polyline:
                parts.append(("polyline", polyline))
        elif gtype == "polygon" and isinstance(coords, list):
            ring = coords[0] if coords else []
            polygon = [c[:2] for c in ring if isinstance(c, list) and len(c) >= 2]
            if polygon:
                parts.append(("polygon", polygon))
        elif gtype == "multilinestring" and isinstance(coords, list):
            for segment in coords:
                if isinstance(segment, list):
                    polyline = [c[:2] for c in segment if isinstance(c, list) and len(c) >= 2]
                    if polyline:
                        parts.append(("polyline", polyline))
        elif gtype == "multipolygon" and isinstance(coords, list):
            for poly in coords:
                if isinstance(poly, list) and poly:
                    ring = poly[0]
                    if isinstance(ring, list):
                        polygon = [c[:2] for c in ring if isinstance(c, list) and len(c) >= 2]
                        if polygon:
                            parts.append(("polygon", polygon))
        return parts

    @staticmethod
    def _persist_rtree_source(table_name: str, doc: Dict[str, Any], table_dir: Optional[Path] = None) -> Path:
        """保存一份可重建 R 树的 GeoJSON 副本。"""
        base = Path(str(table_dir)) if table_dir is not None else (data_path / table_name)
        target = base / 'rtree_source.geojson'
        os.makedirs(target.parent, exist_ok=True)
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(doc, f)
        return target

    def _populate_rtree(self, table_name: str, features: List[Dict[str, Any]]) -> int:
        rtree_mod = self._load_rtree_module()
        if rtree_mod is None:
            return 0
        self._ensure_rtree_cache()
        index = rtree_mod.RTreeIndex()
        inserted = 0
        for idx, feat in enumerate(features):
            attrs = feat.get("properties") or {}
            if not isinstance(attrs, dict):
                attrs = {}

            # 关键：把“表主键 id”显式写入属性，保证 query 返回的 id 可映射回表要素。
            # 表的主键写入逻辑：feat.get('id', idx)
            # 这里必须与 ingest_geojson 中入库的 id 一致，否则会出现“选中/变色错位”。
            attrs = dict(attrs)
            attrs["__table_id"] = feat.get("id", idx)

            for geom_type, coords in self._extract_geom_parts(feat):
                if not coords:
                    continue
                try:
                    index.insert(geom_type, coords, attrs)
                    inserted += 1
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"rtree insert skipped for table {table_name}: {exc}")
        self.rtree_cache[table_name] = index
        # 使用 C++ 端 JSON 序列化，便于后续快速重载
        try:
            target = self._table_dir_for_name(table_name) / 'rtree_serialized.json'
            os.makedirs(target.parent, exist_ok=True)
            index.save_serialized(str(target))

            # 写入版本文件，用于后续判定序列化是否可用
            vpath = self._table_dir_for_name(table_name) / 'rtree_version.txt'
            with open(vpath, 'w', encoding='utf-8') as f:
                f.write(RTREE_VERSION)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"rtree serialized persist skipped for table {table_name}: {exc}")
        return inserted

    @staticmethod
    def _try_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except Exception:
                return None
        return None

    @staticmethod
    def _first_xy(coords: Any) -> Optional[Tuple[float, float]]:
        """从任意嵌套 coordinates 中提取第一个 (x, y) 点。"""
        if isinstance(coords, list) and len(coords) >= 2:
            if isinstance(coords[0], (int, float)) and isinstance(coords[1], (int, float)):
                return float(coords[0]), float(coords[1])
            for item in coords:
                found = Table._first_xy(item)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _lon_lat_from_feature(feat: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """优先从 properties 中取经纬度，其次从 geometry.coordinates 推导。"""
        props = feat.get("properties") or {}
        if isinstance(props, dict):
            lower_map = {str(k).lower(): k for k in props.keys()}
            lon_key = None
            lat_key = None
            for k in ("lon", "lng", "longitude", "x"):
                if k in lower_map:
                    lon_key = lower_map[k]
                    break
            for k in ("lat", "latitude", "y"):
                if k in lower_map:
                    lat_key = lower_map[k]
                    break
            lon_val = Table._try_float(props.get(lon_key)) if lon_key is not None else None
            lat_val = Table._try_float(props.get(lat_key)) if lat_key is not None else None
            if lon_val is not None and lat_val is not None:
                return lon_val, lat_val

        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        xy = Table._first_xy(coords)
        if xy is None:
            return None, None
        return xy[0], xy[1]

    def load_rtree(self, table_name: str):
        """从缓存或持久化的 rtree_source.geojson 重建 R 树索引。"""
        self._ensure_rtree_cache()
        if table_name in self.rtree_cache:
            return self.rtree_cache[table_name]
        rtree_mod = self._load_rtree_module()
        if rtree_mod is None:
            return None
        tdir = self._table_dir_for_name(table_name)
        ser_path = tdir / 'rtree_serialized.json'
        vpath = tdir / 'rtree_version.txt'

        # 只有版本匹配才允许复用序列化索引；否则强制从 geojson 重建
        version_ok = False
        try:
            if vpath.exists():
                version_ok = (vpath.read_text(encoding='utf-8').strip() == RTREE_VERSION)
        except Exception:
            version_ok = False

        # 优先尝试加载 C++ JSON 序列化索引（仅当版本一致）
        if version_ok and ser_path.exists():
            try:
                idx = rtree_mod.RTreeIndex()
                if idx.load_serialized(str(ser_path)):
                    self.rtree_cache[table_name] = idx
                    return idx
            except Exception as exc:  # pylint: disable=broad-except
                print(f"rtree serialized load failed, fallback to geojson: {exc}")
        source = tdir / 'rtree_source.geojson'
        if not source.exists():
            return None

        # 不直接走 C++ load_from_file：它无法保证 feature.id 与表主键一致，
        # 容易导致后端返回 ids 与前端要素 id 无法对齐（表现为“变色错位”）。
        try:
            with open(source, 'r', encoding='utf-8') as f:
                doc = json.load(f)
            feats = doc.get('features', []) if isinstance(doc, dict) else []
            if not isinstance(feats, list):
                feats = []
            self._populate_rtree(table_name, feats)
            return self.rtree_cache.get(table_name)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"rtree geojson rebuild failed: {exc}")
            return None

    def create(self, **kwargs):
        # 检查必要参数
        self.check_require_parameters(**kwargs)
        # 检查可选参数
        self.check_optional_parameters(**kwargs)
        attributes = {i: idx for idx, i in enumerate(kwargs['columns'])}
        
        # 创建主键索引
        self.classes['Index'].create_index(type='primary', index=kwargs['primary_key'], check=False, **kwargs)
        # 创建其他索引
        if 'indexes' in kwargs:
            for index in kwargs['indexes']:
                self.classes['Index'].create_index(type='normal', index=index, check=False, **kwargs)
        # 存储表数据
        self.storage_table_data(table_name=kwargs['table_name'], attributes=attributes, data=kwargs.get('data', []), table_dir=kwargs.get("table_dir"))

    def test(self):
        row = ROW([1, 'Alice', 25])
        print(row[0])  # 通过索引访问
        print(row[1])  # 通过索引访问
        row.__class__.__getattr__ = lambda class_self, x: class_self[1]
        print(row.w)

    def check_require_parameters(self, **kwargs):
        required_params = ['table_name', 'columns', 'primary_key']
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"缺少必要参数: {param}")
        
        # table_name检查
        table_name = kwargs['table_name']
        table_dir = kwargs.get("table_dir")
        if table_dir is not None and not isinstance(table_dir, Path):
            table_dir = Path(str(table_dir))
            kwargs["table_dir"] = table_dir
        base_dir = Path(str(table_dir)) if table_dir is not None else (data_path / table_name)
        if isinstance(table_name, str) is False:
            raise TypeError("表名必须为字符串类型")
        # 兼容：既检查 Manager 注册表，也检查磁盘（避免 registry 丢失导致重复创建）
        if table_name in getattr(self, "table", {}):
            raise ValueError("表名已存在")
        if (base_dir / "data_table.pkl").exists():
            raise ValueError("表名已存在")
        
        # columns检查
        columns = kwargs['columns']
        if isinstance(columns, list) is False:
            raise TypeError("列名必须为列表类型")
        for col in columns:
            if isinstance(col, str) is False:
                raise TypeError("列名必须为字符串类型")
        if len(set(columns)) != len(columns):
            raise ValueError("列名不能重复")
        
        # primary_key检查
        primary_key = kwargs['primary_key']
        if isinstance(primary_key, list) is False:
            raise TypeError("主键必须以列表形式传入")
        for key in primary_key:
            if isinstance(key, str) is False:
                raise TypeError("主键必须为字符串类型")
            if key not in columns:
                raise ValueError(f"主键 {key} 不在列中")
        if len(set(primary_key)) != len(primary_key):
            raise ValueError("主键不能重复")
        if len(primary_key) > len(columns):
            raise ValueError("主键数量不能超过列名数量")
        self.table[table_name] = base_dir / 'data_table.pkl'
        self.index[table_name] = {}
        self.index[table_name][tuple(sorted(primary_key))] = base_dir / f"index_{'-'.join(sorted(primary_key))}.pkl"
        
    def check_optional_parameters(self, **kwargs):
        potional_params = ['data', 'indexes', 'foreign_keys', 'checks']
        columns = kwargs['columns']
        # tycheck : checks
        if 'checks' in kwargs:
            checks = kwargs['checks']
            if isinstance(checks, dict) is False:
                raise TypeError("""检查约束必须以字典形式传入
                                例如: {'column': [def,···]}
                                {列名: 类型:非空\}""")
            for column, constraint in checks.items():
                if isinstance(column, str) is False:
                    raise TypeError("列名必须为字符串类型")
                if column not in columns:
                    raise ValueError(f"列 {column} 不在列中")
                if isinstance(constraint, list) is False:
                    raise TypeError("检查约束必须为列表类型，其中可以包含多个约束定义，例如类型、非空等")
                for c in constraint:
                    if callable(c) is False:
                        raise TypeError("检查约束中的每个定义必须为可调用函数")
            self.column_constraint[kwargs['table_name']] = checks

                    
        # foreign_keys检查
        if 'foreign_keys' in kwargs:
            foreign_keys = kwargs['foreign_keys']
            if isinstance(foreign_keys, dict) is False:
                raise TypeError("""外键必须以字典形式传入
                                例如: {'local_column': ['referenced_table', 'referenced_column']\}
                                {本地列: [外键表, 列名]\}
                                外键不支持聚簇索引""")
            for local_column, (ref_table, ref_column) in foreign_keys.items():
                if isinstance(local_column, str) is False:
                    raise TypeError("本地列名必须为字符串类型")
                if local_column not in columns:
                    raise ValueError(f"本地列 {local_column} 不在列中")
                if isinstance(ref_table, str) is False or isinstance(ref_column, str) is False:
                    raise TypeError("外键表名和列名必须为字符串类型")
        
        # data检查
        if 'data' in kwargs:
            data = kwargs['data']
            if isinstance(data, list) is False:
                raise TypeError("数据必须以列表形式传入")
            column_data = [[None]*len(data) for _ in range(len(columns))]
            for row in data:
                if isinstance(row, ROW):
                    values = row.data
                elif isinstance(row, (list, tuple)):
                    values = list(row)
                else:
                    raise TypeError("每一行数据必须以列表或 ROW 形式传入")
                if len(values) != len(columns):
                    raise ValueError("每一行数据的长度必须与列名数量相等")
                for idx, value in enumerate(values):
                    column_data[idx].append(value)
            if 'checks' in kwargs:
                checks = kwargs['checks']
                for col_idx, col_name in enumerate(columns):
                    constraints = checks[col_name]
                    for constraint in constraints:
                        if callable(constraint):
                            col_values = column_data[col_idx]
                            if not constraint(col_values):
                                raise ValueError(f"列 {col_name} 不满足检查约束")
            
        # indexes检查
        if 'indexes' in kwargs:
            indexes = kwargs['indexes']
            if isinstance(indexes, list) is False:
                raise TypeError("索引必须以列表形式传入")
            primary_key = tuple(sorted(kwargs['primary_key']))
            table_name = kwargs['table_name']
            for index in indexes:
                if isinstance(index, list):
                    index = tuple(sorted(index))
                    for key in index:
                        if key not in columns:
                            raise ValueError(f"索引 {key} 不在列中")
                    if index == primary_key:
                        raise ValueError(f"索引 {'-'.join(index)} 重复主键")
                    if self.index.get(table_name, {}).get(index, None) is not None:
                        raise ValueError(f"索引 {'-'.join(index)} 已经存在或者重复")
                    if self.index.get(table_name, None) is None:
                        self.index[table_name] = {}
                    self.index[table_name][index] = data_path / table_name / f"index_{'_'.join(index)}.pkl"
                else:
                    raise TypeError("索引必须以列表形式传入")
                
    def storage_table_data(self, table_name:str, attributes:Dict[str,int], data:List[ROW], table_dir: Optional[Path] = None):
        """
        存储表数据
        args:
            table_name: 表名
            attributes: 列名字典
            data: 数据列表
        """
        table_path = self._table_dir_for_name(table_name, table_dir=table_dir) / 'data_table.pkl'
        data_table = DATA_table(attributes=attributes, primary_key=tuple(self.index[table_name].keys()))
        for row in data:
            if isinstance(row, ROW):
                data_table.add_row(row)
            elif isinstance(row, (list, tuple)):
                data_table.add_row(ROW(list(row)))
            else:
                raise TypeError("数据行必须是 ROW 或 list/tuple")
        os.makedirs(str(table_path.parent), exist_ok=True)
        with open(str(table_path), 'wb') as f:
            pkl.dump(data_table, f)
        
    def insert_row(self, table_name:str, rows:List[ROW]):
        """
        插入行数据
        args:
            table_name: 表名
            rows: 行数据
        """
        # 输入数据检查
        table_path = self.table.get(table_name, None)
        if table_path is None:
            raise ValueError(f"表 {table_name} 不存在")
        if not table_path.exists():
            raise ValueError(f"表 {table_name} 不存在")
        if not isinstance(rows, list):
            raise TypeError("插入的数据必须以列表形式传入")
        for value in rows:
            if not isinstance(value, ROW):
                raise TypeError("插入的数据必须为 ROW 实例")
        with open(str(table_path), 'rb') as f:
            data_table = pkl.load(f)
        # 空表（data_table.data 为空）时，data_table[0] 会抛 Row index out of range；
        # 这里应当使用 schema（attributes）判断列数。
        expected_cols = len(getattr(data_table, "attributes", {}) or {})
        if expected_cols and len(rows[0]) != expected_cols:
            raise ValueError("插入的行数据长度与表列数不匹配")
        
        # 列约束检查
        if table_name in self.column_constraint:
            column_views = data_table.column_views(list(self.column_constraint[table_name].keys()))
            for col_name, column_data in column_views.items():
                now_value = [r[data_table.attributes[col_name]] for r in rows]
                constraint = self.column_constraint[table_name].get(col_name)
                if constraint and callable(constraint):
                    if not constraint(column_data + now_value):
                        raise ValueError(f"列 {col_name} 不满足列约束")
        
        # 插入数据
        for row in rows:
            data_table.add_row(row)
            self.classes['Index'].update_indexes_after_insert(data_table=data_table, table_name=table_name, new_row=row)
        with open(str(table_path), 'wb') as f:
            pkl.dump(data_table, f)

    def load(self, table_name: str) -> DATA_table:
        table_path = self.table.get(table_name)
        if not table_path or not table_path.exists():
            raise ValueError(f"表 {table_name} 不存在")
        with open(table_path, 'rb') as f:
            data_table: DATA_table = pkl.load(f)

        # 旧“空间要素表”自动升级：补齐 lon/lat 两列（避免历史 GeoJSON 入库表缺经纬度字段）。
        # 说明：用户在 SQL 界面创建的普通表不应被强行注入 lon/lat。
        attrs = data_table.attributes or {}
        is_spatial_table = (
            isinstance(attrs, dict)
            and "geom_type" in attrs
            and "geometry" in attrs
            and "properties" in attrs
        )
        if is_spatial_table and ("lon" not in attrs or "lat" not in attrs):
            if "lon" not in data_table.attributes:
                data_table.attributes["lon"] = len(data_table.attributes)
            if "lat" not in data_table.attributes:
                data_table.attributes["lat"] = len(data_table.attributes)

            geom_type_idx = data_table.attributes.get("geom_type")
            geom_idx = data_table.attributes.get("geometry")
            props_idx = data_table.attributes.get("properties")

            for row in data_table.data:
                values = row.data if isinstance(row, ROW) else list(row)
                gtype = values[geom_type_idx] if geom_type_idx is not None and geom_type_idx < len(values) else ""
                coords = values[geom_idx] if geom_idx is not None and geom_idx < len(values) else None
                props = values[props_idx] if props_idx is not None and props_idx < len(values) else {}

                feat = {
                    "type": "Feature",
                    "geometry": {"type": gtype, "coordinates": coords},
                    "properties": props if isinstance(props, dict) else {},
                }
                lon_val, lat_val = self._lon_lat_from_feature(feat)

                # 追加到行尾，保持 attributes 索引一致
                if isinstance(row, ROW):
                    row.data.append(lon_val)
                    row.data.append(lat_val)
                else:
                    # 理论上不会走到这里，但保持兼容
                    new_row = ROW(values + [lon_val, lat_val])
                    row = new_row

            # 写回升级结果，确保后续加载不重复计算
            with open(table_path, 'wb') as f:
                pkl.dump(data_table, f)

        return data_table

    def query_rows(self, table_name: str, predicate: Optional[Callable[[Dict[str, Any]], bool]] = None, columns: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        predicate = predicate or (lambda _: True)
        data_table = self.load(table_name)
        col_names = list(data_table.attributes.keys())
        col_filter = set(columns) if columns else None
        results: List[Dict[str, Any]] = []
        for row in data_table.data:
            row_dict = {col: row[idx] for col, idx in data_table.attributes.items()}
            if predicate(row_dict):
                if col_filter:
                    row_dict = {k: v for k, v in row_dict.items() if k in col_filter}
                results.append(row_dict)
        return results

    def update_rows(self, table_name: str, predicate: Callable[[Dict[str, Any]], bool], updater: Callable[[Dict[str, Any]], Dict[str, Any]]) -> int:
        data_table = self.load(table_name)
        updated = 0
        for i, row in enumerate(data_table.data):
            row_dict = {col: row[idx] for col, idx in data_table.attributes.items()}
            if predicate(row_dict):
                new_values = updater(row_dict)
                for col, val in new_values.items():
                    if col not in data_table.attributes:
                        raise ValueError(f"列 {col} 不存在")
                    row.data[data_table.attributes[col]] = val
                updated += 1
        table_path = self.table.get(table_name)
        if not table_path:
            raise ValueError(f"表 {table_name} 不存在")
        with open(table_path, 'wb') as f:
            pkl.dump(data_table, f)
        return updated

    def delete_rows(self, table_name: str, predicate: Optional[Callable[[Dict[str, Any]], bool]] = None) -> int:
        predicate = predicate or (lambda _: True)
        data_table = self.load(table_name)
        before = len(data_table.data)
        kept: List[ROW] = []
        for row in data_table.data:
            row_dict = {col: row[idx] for col, idx in data_table.attributes.items()}
            if not predicate(row_dict):
                kept.append(row)
        data_table.data = kept
        deleted = before - len(kept)

        table_path = self.table.get(table_name)
        if not table_path:
            raise ValueError(f"表 {table_name} 不存在")
        with open(table_path, 'wb') as f:
            pkl.dump(data_table, f)
        return deleted

    def ingest_geojson(self, table_name: str, geojson_path: Path, table_dir: Optional[Path] = None):
        """将 GeoJSON FeatureCollection 静态存储为表，列包含 id、type、geometry、properties、lon、lat。"""
        with open(geojson_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        try:
            from space_app.algorithms.geojson_utils import normalize_geojson
            doc = normalize_geojson(doc)
        except Exception as exc:
            raise ValueError(f"仅支持 FeatureCollection/Feature/Geometry 或 Esri JSON（Point/Polyline/Polygon/Multipoint）：{exc}") from exc
        features = doc.get("features", [])
        columns = ["id", "geom_type", "geometry", "properties", "lon", "lat"]
        rows: List[ROW] = []
        for idx, feat in enumerate(features):
            geom = feat.get("geometry", {})
            lon_val, lat_val = self._lon_lat_from_feature(feat)
            rows.append(ROW([
                feat.get("id", idx),
                geom.get("type", ""),
                geom.get("coordinates", None),
                feat.get("properties", {}),
                lon_val,
                lat_val,
            ]))
        self.create(table_name=table_name, columns=columns, primary_key=["id"], data=rows, table_dir=table_dir)
        self._persist_rtree_source(table_name, doc, table_dir=self._table_dir_for_name(table_name, table_dir=table_dir))
        self._populate_rtree(table_name, features)
