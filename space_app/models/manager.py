from __future__ import annotations

import os
import json
import pickle as pkl
from pathlib import Path
from typing import Any, List, Dict, Callable, Optional, Iterable, Tuple
from .table import Table
from .index import Index
from .foreign_constraint import Foreign_Constraint
from .column_constraint import Column_Constraint


file_path = Path(__file__)
data_path = file_path.parent.parent / 'table_data'
data_path.mkdir(parents=True, exist_ok=True)
registry_json_path = file_path.parent / 'manager_registry.json'
registry_pkl_path = file_path.parent / 'manager_registry.pkl'

ADMIN_USER = "admin"


def user_data_dir(username: str) -> Path:
    return data_path / str(username)


def table_dir_for(owner: str, table: str) -> Path:
    return user_data_dir(owner) / str(table)


def make_table_id(owner: str, table: str) -> str:
    return f"{owner}:{table}"


def _migrate_flat_tables_to_admin() -> None:
    """把旧结构 table_data/<table> 迁移为 table_data/admin/<table>（幂等）。"""
    try:
        admin_dir = user_data_dir(ADMIN_USER)
        admin_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    try:
        for p in data_path.iterdir():
            if not p.is_dir():
                continue
            # 已经是用户目录则跳过（包含 admin 或其他用户）
            if (p / "data_table.pkl").exists():
                # 旧格式：这是一个表目录
                src = p
                dst = admin_dir / src.name
                if dst.exists():
                    # 冲突：追加后缀
                    i = 1
                    while True:
                        cand = admin_dir / f"{src.name}_{i}"
                        if not cand.exists():
                            dst = cand
                            break
                        i += 1
                try:
                    src.rename(dst)
                except Exception:
                    try:
                        import shutil

                        shutil.move(str(src), str(dst))
                    except Exception:
                        pass
    except Exception:
        return

class Manager:
    def __init__(self):
        self.__manager_name = {
            'Table':self.__mana_table,
            'Index':self.__mana_index,
            'Foreign_Constraint': self.__mana_foreign_constraint,
            'Column_Constraint': self.__mana_column_constraint,
        }
        self.classes = {}
        self.table = {} # table_name: table_path
        self.index = {} # table_name: index_name: index_path
        self.foreign_key = {} # (table_name, foreign_key_name, local_column, foreign_column): bool
        self.column_constraint = {} # table_name: column_name: [constraint_value]
        self.rtree_cache = {} # table_name: RTreeIndex instance
        self.test = "manager class test"

    def _log(self, msg: str):
        if os.environ.get("SPATIAL_MANAGER_VERBOSE") == "1":
            print(msg)

    def register(self, cls):
        self._log(f"注册类: {cls.__name__}")
        if cls.__name__ in self.__manager_name.keys():
            self.classes[cls.__name__] = cls()
            return self.__manager_name[cls.__name__]()
        else:
            raise ValueError("Class not supported for registration")

    def register_defaults(self):
        self.register(Table)
        self.register(Index)
        self.register(Foreign_Constraint)
        self.register(Column_Constraint)
    
    def __mana_table(self):
        """"表管理器，在这里实现表的创建、删除、修改等功能"""
        instance = self.classes['Table']
        # 将本地属性传递给类属性
        instance.__class__.__getattr__ = lambda obj, name: self.__derive_attr(name)

    def __derive_attr(self, attr_name):
        # __getattr__ 约定：缺失属性必须抛 AttributeError；否则会干扰 pickle 等反射逻辑
        try:
            return getattr(self, attr_name)
        except AttributeError as exc:
            raise AttributeError(attr_name) from exc

    def __mana_index(self):
        instance = self.classes['Index']
        instance.__class__.__getattr__ = lambda obj, name: self.__derive_attr(name)

    def __mana_foreign_constraint(self):
        instance = self.classes['Foreign_Constraint']
        instance.__class__.__getattr__ = lambda obj, name: self.__derive_attr(name)

    def __mana_column_constraint(self):
        instance = self.classes['Column_Constraint']
        instance.__class__.__getattr__ = lambda obj, name: self.__derive_attr(name)
    
    def create_table(self, **kwargs):
        try:
            table_dir = kwargs.get("table_dir")
            if table_dir is not None and not isinstance(table_dir, Path):
                table_dir = Path(str(table_dir))
                kwargs["table_dir"] = table_dir
            self.classes['Table'].create(**kwargs)
            # 统一使用 data_table.pkl 路径，避免后续 DML 打开目录失败
            if table_dir is None:
                table_dir = data_path / kwargs['table_name']
            self.table[kwargs['table_name']] = table_dir / 'data_table.pkl'
            self.save_registry()
        except Exception as e:
            # 删除已创建的表数据
            table_dir = kwargs.get("table_dir")
            if table_dir is None:
                table_dir = data_path / kwargs['table_name']
            table_path = Path(str(table_dir))
            if table_path.exists():
                import shutil
                shutil.rmtree(table_path)
            
            # 删除注册表中的表信息
            if kwargs['table_name'] in self.table:
                del self.table[kwargs['table_name']]
            if kwargs['table_name'] in self.index:
                del self.index[kwargs['table_name']]
            for key in list(self.foreign_key.keys()):
                if key[0] == kwargs['table_name']:
                    del self.foreign_key[key]
            self.save_registry()
            raise e
    
    def create_index(self, **kwargs):
        out = self.classes['Index'].create_index(**kwargs)
        self.save_registry()
        return out

    # --- 数据操作接口 ---
    def ingest_geojson(self, table_name: str, geojson_path: Path, table_dir: Optional[Path] = None):
        """读取 GeoJSON 文件并静态存储到表中。"""
        out = self.classes['Table'].ingest_geojson(table_name=table_name, geojson_path=geojson_path, table_dir=table_dir)
        self.save_registry()
        return out

    def query(self, table_name: str, predicate: Optional[Callable[[Dict[str, Any]], bool]] = None, columns: Optional[List[str]] = None):
        return self.classes['Table'].query_rows(table_name, predicate, columns)

    def insert_rows(self, table_name: str, rows: List[Any]):
        return self.classes['Table'].insert_row(table_name, rows)

    def update_rows(self, table_name: str, predicate: Callable[[Dict[str, Any]], bool], updater: Callable[[Dict[str, Any]], Dict[str, Any]]):
        return self.classes['Table'].update_rows(table_name, predicate, updater)

    def delete_rows(self, table_name: str, predicate: Optional[Callable[[Dict[str, Any]], bool]] = None) -> int:
        return self.classes['Table'].delete_rows(table_name, predicate)

    def load_rtree(self, table_name: str):
        return self.classes['Table'].load_rtree(table_name)

    def dump_registry_state(self) -> Dict[str, Any]:
        """导出可序列化的注册表状态（只包含纯数据）。"""
        idx_out: Dict[str, List[Dict[str, Any]]] = {}
        for tname, mapping in (self.index or {}).items():
            items: List[Dict[str, Any]] = []
            if isinstance(mapping, dict):
                for cols, path in mapping.items():
                    try:
                        cols_list = list(cols) if isinstance(cols, tuple) else list(cols)
                    except Exception:
                        cols_list = [str(cols)]
                    items.append({
                        "columns": [str(c) for c in cols_list],
                        "path": str(path) if path is not None else "",
                    })
            idx_out[str(tname)] = items

        fk_out: List[Dict[str, Any]] = []
        for k, enabled in (self.foreign_key or {}).items():
            if not isinstance(k, tuple) or len(k) < 4:
                continue
            fk_out.append({
                "table": str(k[0]),
                "name": str(k[1]),
                "local_column": str(k[2]),
                "foreign_column": str(k[3]),
                "enabled": bool(enabled),
            })

        # column_constraint 里可能包含 callable（无法可靠序列化），只保留可 JSON 化的部分
        cc_out: Dict[str, Dict[str, Any]] = {}
        for tname, cols in (self.column_constraint or {}).items():
            if not isinstance(cols, dict):
                continue
            keep: Dict[str, Any] = {}
            for col, v in cols.items():
                try:
                    json.dumps(v)
                except Exception:
                    continue
                keep[str(col)] = v
            if keep:
                cc_out[str(tname)] = keep

        return {
            "version": 1,
            "table": {str(k): str(v) for k, v in (self.table or {}).items()},
            "index": idx_out,
            "foreign_key": fk_out,
            "column_constraint": cc_out,
        }

    def load_registry_state(self, state: Dict[str, Any]) -> None:
        """从 dump_registry_state() 的输出恢复注册表状态。"""
        if not isinstance(state, dict):
            raise TypeError("registry state must be dict")

        # 兼容旧 schema（没有 version）
        table_in = state.get("table", {})
        index_in = state.get("index", {})
        fk_in = state.get("foreign_key", [])
        cc_in = state.get("column_constraint", {})

        self.table = {}
        if isinstance(table_in, dict):
            for k, v in table_in.items():
                if not k:
                    continue
                if v:
                    self.table[str(k)] = Path(str(v))

        self.index = {}
        if isinstance(index_in, dict):
            for tname, items in index_in.items():
                if not tname:
                    continue
                mapping: Dict[Tuple[str, ...], Path] = {}
                if isinstance(items, list):
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        cols = it.get("columns")
                        path = it.get("path")
                        if not cols or not path:
                            continue
                        if not isinstance(cols, list):
                            continue
                        mapping[tuple(str(c) for c in cols)] = Path(str(path))
                elif isinstance(items, dict):
                    # 极简/旧格式：{"('id',)": "path"}
                    for cols, path in items.items():
                        if not path:
                            continue
                        if isinstance(cols, (list, tuple)):
                            mapping[tuple(str(c) for c in cols)] = Path(str(path))
                        else:
                            mapping[(str(cols),)] = Path(str(path))
                if mapping:
                    self.index[str(tname)] = mapping

        self.foreign_key = {}
        if isinstance(fk_in, list):
            for it in fk_in:
                if not isinstance(it, dict):
                    continue
                try:
                    key = (str(it["table"]), str(it["name"]), str(it["local_column"]), str(it["foreign_column"]))
                    self.foreign_key[key] = bool(it.get("enabled", True))
                except Exception:
                    continue

        # 仅加载可 JSON 化的列约束（函数类约束无法恢复）
        self.column_constraint = {}
        if isinstance(cc_in, dict):
            for tname, cols in cc_in.items():
                if isinstance(cols, dict):
                    self.column_constraint[str(tname)] = cols

    def _scan_table_data_dir(self) -> None:
        """兜底：从 space_app/table_data 扫描恢复 table/index 映射（支持 user/table 结构）。"""
        if not data_path.exists():
            return

        # 兼容：若仍存在旧结构，先迁移到 admin
        _migrate_flat_tables_to_admin()

        for udir in data_path.iterdir():
            if not udir.is_dir():
                continue
            owner = udir.name

            # 兼容极端情况：用户目录里直接放了 data_table.pkl（旧结构误放）
            if (udir / "data_table.pkl").exists():
                tname = udir.name
                tid = make_table_id(ADMIN_USER, tname)
                dt = udir / "data_table.pkl"
                self.table[tid] = dt
                continue

            for tdir in udir.iterdir():
                if not tdir.is_dir():
                    continue
                dt = tdir / "data_table.pkl"
                if not dt.exists():
                    continue
                tname = tdir.name
                tid = make_table_id(owner, tname)
                self.table[tid] = dt

                for idx_file in tdir.glob("index_*.pkl"):
                    stem = idx_file.stem  # index_xxx
                    suffix = stem[len("index_"):] if stem.startswith("index_") else stem
                    if not suffix:
                        continue
                    cols = (
                        tuple([c for c in suffix.split("-") if c])
                        if "-" in suffix
                        else tuple([c for c in suffix.split("_") if c])
                        if "_" in suffix
                        else (suffix,)
                    )
                    self.index.setdefault(tid, {})
                    self.index[tid][tuple(sorted(cols))] = idx_file

    def save_registry(self) -> None:
        """把注册表状态写入 json + pkl（原子替换）。"""
        state = self.dump_registry_state()
        try:
            tmp = registry_json_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, registry_json_path)
        except Exception as exc:
            self._log(f"[manager] save json failed: {exc}")

        try:
            tmp = registry_pkl_path.with_suffix(".pkl.tmp")
            tmp.write_bytes(pkl.dumps(state))
            os.replace(tmp, registry_pkl_path)
        except Exception as exc:
            self._log(f"[manager] save pkl failed: {exc}")

    def try_load_registry(self) -> bool:
        """启动时调用：优先 pkl，其次 json；失败则返回 False。"""
        state = None
        if registry_pkl_path.exists():
            try:
                state = pkl.loads(registry_pkl_path.read_bytes())
            except Exception as exc:
                self._log(f"[manager] load pkl failed: {exc}")
                state = None
        if state is None and registry_json_path.exists():
            try:
                state = json.loads(registry_json_path.read_text(encoding="utf-8") or "{}")
            except Exception as exc:
                self._log(f"[manager] load json failed: {exc}")
                state = None

        if not isinstance(state, dict):
            return False
        try:
            self.load_registry_state(state)
            return True
        except Exception as exc:
            self._log(f"[manager] apply registry state failed: {exc}")
            return False


def _build_manager() -> Manager:
    m = Manager()
    m.register_defaults()
    return m


def _load_or_create_manager() -> Manager:
    """闭环启动逻辑：优先从 pkl/json 恢复注册表，否则重新注册并扫描 table_data。"""
    m = _build_manager()
    _migrate_flat_tables_to_admin()
    m.try_load_registry()
    # 无论是否加载成功，都以磁盘 table_data 扫描结果做兜底补全（避免 registry 丢失/为空）
    m._scan_table_data_dir()
    # 如果还没有 pkl，则在启动阶段生成一次，避免 registry 永远为空/缺失
    if not registry_pkl_path.exists():
        try:
            m.save_registry()
        except Exception:
            pass
    if not hasattr(m, "rtree_cache"):
        m.rtree_cache = {}
    return m


MANAGER = _load_or_create_manager()
