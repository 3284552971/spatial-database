from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import html

from django.shortcuts import render, redirect
from django.http import FileResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.safestring import mark_safe

from .models.manager import (
    MANAGER,
    ADMIN_USER,
    data_path,
    make_table_id,
    table_dir_for,
    user_data_dir,
)
from .models.select import SQL_parser, Condition_where
from .models.sql_router import parse as parse_sql
from .models.table import ROW
from .algorithms.geojson_utils import normalize_geojson
from .algorithms.topology import check_topology_layer, repair_layer, haversine_m
from .algorithms.selection import (
    select_by_attribute as algo_select_by_attribute,
    geom_bbox as algo_geom_bbox,
    bbox_intersects as algo_bbox_intersects,
    select_by_location_rtree,
    meters_to_deg_components,
    _bbox_intersects_circle_m,
)
from .algorithms.pathfinding import (
    build_road_graph,
    build_native_graph,
    k_shortest_paths_native,
    k_shortest_paths_py,
    nearest_nodes as algo_nearest_nodes,
)
from .algorithms.trajectory import (
    hmm_match_and_kalman as trajectory_hmm_match_and_kalman,
    native_available as trajectory_native_available,
)

from .user_store import create_user, ensure_admin, validate_username, verify_user

ensure_admin(password="zblnb666")

# 最短路径：按表缓存路网图（避免每次请求都重新建图）
_ROUTE_GRAPH_CACHE: dict[str, dict[str, Any]] = {}


def _current_user(request) -> str:
    return str(request.session.get("user") or "").strip()


def _is_admin(user: str) -> bool:
    return (user or "").strip() == ADMIN_USER


def _resolve_owner(owner_raw: str) -> str:
    """将 owner 解析为磁盘真实用户名目录名（大小写不敏感）。"""
    name = (owner_raw or "").strip()
    if not name:
        return name
    if user_data_dir(name).exists():
        return name
    low = name.lower()
    try:
        for p in data_path.iterdir():
            if p.is_dir() and p.name.lower() == low:
                return p.name
    except Exception:
        pass
    return name


def _resolve_table_name_in_owner(owner: str, table_raw: str) -> str:
    """在某个 owner 目录下做表名大小写不敏感匹配。"""
    name = (table_raw or "").strip()
    if not name:
        return name
    tdir = table_dir_for(owner, name)
    if (tdir / "data_table.pkl").exists():
        return name
    low = name.lower()
    udir = user_data_dir(owner)
    try:
        if udir.exists():
            for p in udir.iterdir():
                if p.is_dir() and (p / "data_table.pkl").exists() and p.name.lower() == low:
                    return p.name
    except Exception:
        pass
    return name


def _parse_table_ref(table_ref: str, default_owner: str) -> tuple[str, str]:
    raw = (table_ref or "").strip()
    if not raw:
        raise ValueError("表名不能为空")
    if ":" in raw:
        owner_raw, table_raw = raw.split(":", 1)
        owner = (owner_raw or "").strip()
        table = (table_raw or "").strip()
        if not owner or not table:
            raise ValueError("表名格式错误，应为 table 或 owner:table")
        return owner, table
    # 兼容 admin.table 这种写法：仅当点号分隔且 owner 目录存在时才当作 owner.table
    if "." in raw and raw.count(".") == 1:
        owner_raw, table_raw = raw.split(".", 1)
        owner = (owner_raw or "").strip()
        table = (table_raw or "").strip()
        if owner and table and user_data_dir(owner).exists():
            return owner, table
    return default_owner, raw


def _canonical_table(request_user: str, table_ref: str, must_exist: bool = True) -> tuple[str, str, str, Path]:
    """返回 (owner, table, table_id, table_dir)。"""
    owner_raw, table_raw = _parse_table_ref(table_ref, default_owner=request_user)
    owner = _resolve_owner(owner_raw)
    if not owner:
        raise ValueError("owner 不能为空")
    if (not _is_admin(request_user)) and owner != request_user:
        raise ValueError("无权限访问其他用户的表")

    table = _resolve_table_name_in_owner(owner, table_raw)
    tid = make_table_id(owner, table)
    tdir = table_dir_for(owner, table)
    if must_exist and not (tdir / "data_table.pkl").exists():
        raise ValueError(f"表 {table_ref} 不存在")
    return owner, table, tid, tdir


def _ensure_table_registered_id(table_id: str, table_dir: Path) -> None:
    if table_id in MANAGER.table and Path(str(MANAGER.table.get(table_id))).exists():
        return
    data_file = Path(str(table_dir)) / "data_table.pkl"
    if not data_file.exists():
        raise ValueError(f"表 {table_id} 不存在")
    MANAGER.table[table_id] = data_file
    MANAGER.index.setdefault(table_id, {})
    for idx_file in Path(str(table_dir)).glob("index_*.pkl"):
        stem = idx_file.stem
        suffix = stem[len("index_"):] if stem.startswith("index_") else stem
        if not suffix:
            continue
        cols = tuple([c for c in suffix.split("-") if c]) if "-" in suffix else tuple([c for c in suffix.split("_") if c]) if "_" in suffix else (suffix,)
        MANAGER.index[table_id][tuple(sorted(cols))] = idx_file


def _list_tables_for_user(user: str) -> list[str]:
    """返回可见表名列表：普通用户返回短表名；admin 返回 owner:table。"""
    u = (user or "").strip()
    if not u:
        return []
    out: list[str] = []
    if _is_admin(u):
        try:
            for udir in data_path.iterdir():
                if not udir.is_dir():
                    continue
                owner = udir.name
                for tdir in udir.iterdir():
                    if tdir.is_dir() and (tdir / "data_table.pkl").exists():
                        out.append(make_table_id(owner, tdir.name))
        except Exception:
            pass
        return sorted(out)

    udir = user_data_dir(u)
    try:
        if udir.exists():
            for tdir in udir.iterdir():
                if tdir.is_dir() and (tdir / "data_table.pkl").exists():
                    out.append(tdir.name)
    except Exception:
        pass
    return sorted(out)

def index(request):
    # /space_app/ 默认入口：未登录先去登录页（避免先跳 map 再重定向一次）
    if not request.session.get("user"):
        return redirect("login")
    return redirect("map")

def login_view(request):
    error = None
    message = None
    if request.method == "POST":
        action = (request.POST.get("action") or "login").strip().lower()
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        if action == "register":
            confirm = request.POST.get("confirm_password", "")
            try:
                validate_username(username)
                if password != confirm:
                    raise ValueError("两次密码不一致")
                create_user(username, password)
                # 为新用户创建数据目录
                try:
                    user_data_dir(username).mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                message = "创建成功，请登录"
            except Exception as exc:
                error = str(exc)
        else:
            if verify_user(username, password):
                request.session["user"] = username
                return redirect("map")
            error = "用户名或密码错误"

    return render(request, "login.html", {"error": error, "message": message})


def logout_view(request):
    request.session.flush()
    return redirect("login")


# --- Helpers ---
def _match_key(row: dict, token: str):
    for k in row.keys():
        if k.lower() == token.lower():
            return k
    return None


def _coerce_value(val: str):
    text = (val or "").strip()
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        return text[1:-1]

    lowered = text.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        if re.fullmatch(r"-?\d*\.\d+", text):
            return float(text)
    except Exception:
        pass

    return text


def _coerce_for_compare(value):
    if isinstance(value, (int, float)) or value is None:
        return value
    if isinstance(value, str):
        return _coerce_value(value)
    return value


def _eval_condition(cond, row: dict) -> bool:
    if cond is None:
        return True

    if isinstance(cond, Condition_where) and cond.operate in {"AND", "OR"}:
        left_ok = _eval_condition(cond.left, row)
        right_ok = _eval_condition(cond.right, row)
        return (left_ok and right_ok) if cond.operate == "AND" else (left_ok or right_ok)

    if not isinstance(cond, Condition_where):
        return True

    left_token = cond.left
    right_token = cond.right
    if isinstance(left_token, Condition_where) or isinstance(right_token, Condition_where):
        return False

    key = _match_key(row, str(left_token))
    if key is None:
        return False

    left_val = row.get(key)
    right_val = _coerce_value(str(right_token))

    op = cond.operate
    try:
        if op == "=":
            return left_val == right_val
        if op == "!=":
            return left_val != right_val

        left_cmp = _coerce_for_compare(left_val)
        right_cmp = _coerce_for_compare(right_val)

        if op == ">":
            return left_cmp > right_cmp
        if op == "<":
            return left_cmp < right_cmp
        if op == ">=":
            return left_cmp >= right_cmp
        if op == "<=":
            return left_cmp <= right_cmp
    except Exception:
        return False

    return False


def _execute_select(node, request_user: str):
    _owner, _t, table_id, tdir = _canonical_table(request_user, node.from_table or "", must_exist=True)
    _ensure_table_registered_id(table_id, tdir)

    def predicate(row):
        return _eval_condition(node.where, row) if node.where else True

    selected_cols = None
    if node.columns and node.columns.column_name and '*' not in node.columns.column_name:
        selected_cols = node.columns.column_name

    rows = MANAGER.query(table_id, predicate=predicate, columns=None)

    # Reorder/filter columns if needed
    if selected_cols:
        filtered = []
        for r in rows:
            mapped = {}
            for col in selected_cols:
                key = _match_key(r, col)
                if key:
                    mapped[key] = r[key]
            filtered.append(mapped)
        rows = filtered

    if node.order_by:
        order_key = node.order_by.column
        key_in_row = _match_key(rows[0], order_key) if rows else None
        if key_in_row:
            reverse = (node.order_by.direction or '').upper() == 'DESC'
            rows = sorted(rows, key=lambda x: x.get(key_in_row), reverse=reverse)

    if node.limit:
        try:
            limit_val = int(node.limit.limit)
            rows = rows[:limit_val]
        except Exception:
            pass
    return rows


def _match_column(attributes: dict, token: str):
    for k in attributes.keys():
        if k.lower() == token.lower():
            return k
    return None


def _execute_insert(stmt: dict, request_user: str) -> int:
    _owner, _t, table_id, tdir = _canonical_table(request_user, stmt["table"], must_exist=True)
    _ensure_table_registered_id(table_id, tdir)
    data_table = MANAGER.classes['Table'].load(table_id)
    columns_order = list(data_table.attributes.keys())

    values_raw = stmt.get("values") or []
    values = [_coerce_value(str(v)) for v in values_raw]
    cols = stmt.get("columns")

    if cols is None:
        if len(values) != len(columns_order):
            raise ValueError(f"VALUES 数量({len(values)})与列数({len(columns_order)})不匹配")
        row_values = values
    else:
        row_values = [None] * len(columns_order)
        for col_token, val in zip(cols, values):
            real = _match_column(data_table.attributes, str(col_token))
            if real is None:
                raise ValueError(f"列 {col_token} 不存在")
            row_values[data_table.attributes[real]] = val

    MANAGER.insert_rows(table_id, [ROW(row_values)])
    return 1


def _execute_update(stmt: dict, request_user: str) -> int:
    _owner, _t, table_id, tdir = _canonical_table(request_user, stmt["table"], must_exist=True)
    _ensure_table_registered_id(table_id, tdir)
    where = stmt.get("where")
    set_map = stmt.get("set") or {}

    def predicate(row):
        return _eval_condition(where, row) if where else True

    def updater(row):
        updated: dict = {}
        for k, v in set_map.items():
            real = _match_key(row, str(k))
            if real is None:
                raise ValueError(f"列 {k} 不存在")
            updated[real] = _coerce_value(str(v))
        return updated

    return MANAGER.update_rows(table_id, predicate=predicate, updater=updater)


def _execute_delete(stmt: dict, request_user: str) -> int:
    _owner, _t, table_id, tdir = _canonical_table(request_user, stmt["table"], must_exist=True)
    _ensure_table_registered_id(table_id, tdir)
    where = stmt.get("where")

    def predicate(row):
        return _eval_condition(where, row) if where else True

    return MANAGER.delete_rows(table_id, predicate=predicate)


def _execute_create(stmt: dict, request_user: str) -> None:
    owner, table, table_id, tdir = _canonical_table(request_user, stmt["table"], must_exist=False)
    if (tdir / "data_table.pkl").exists():
        raise ValueError(f"表 {table} 已存在")
    columns = stmt.get("columns") or []
    primary_key = stmt.get("primary_key") or []
    try:
        user_data_dir(owner).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    MANAGER.create_table(table_name=table_id, columns=columns, primary_key=primary_key, data=[], table_dir=tdir)

def _execute_drop(stmt: dict, request_user: str) -> None:
    table_ref = str(stmt.get("table") or "").strip()
    if not table_ref:
        raise ValueError("DROP TABLE 缺少表名")
    if_exists = bool(stmt.get("if_exists", False))

    try:
        owner, table, table_id, tdir = _canonical_table(request_user, table_ref, must_exist=True)
    except Exception as exc:
        if if_exists:
            return
        raise exc

    # 删除磁盘目录（该目录仅包含该表的数据/索引/rtree 等）
    try:
        if tdir.exists():
            shutil.rmtree(tdir)
    except Exception as exc:
        raise ValueError(f"删除表目录失败：{tdir}（{exc}）") from exc

    # 清理内存注册表/缓存
    try:
        MANAGER.table.pop(table_id, None)
    except Exception:
        pass
    try:
        MANAGER.index.pop(table_id, None)
    except Exception:
        pass
    try:
        for k in list((MANAGER.foreign_key or {}).keys()):
            if isinstance(k, tuple) and k and str(k[0]) == str(table_id):
                MANAGER.foreign_key.pop(k, None)
    except Exception:
        pass
    try:
        MANAGER.column_constraint.pop(table_id, None)
    except Exception:
        pass
    try:
        if hasattr(MANAGER, "rtree_cache"):
            MANAGER.rtree_cache.pop(table_id, None)
    except Exception:
        pass
    try:
        _ROUTE_GRAPH_CACHE.pop(table_id, None)
    except Exception:
        pass

    try:
        MANAGER.save_registry()
    except Exception:
        pass


def sql_view(request):
    if not request.session.get("user"):
        return redirect("login")

    user = _current_user(request)
    context = {"user": user, "tables": _list_tables_for_user(user)}
    if request.method == "POST":
        sql_text = request.POST.get("sql", "").strip()
        if sql_text:
            try:
                ast = parse_sql(sql_text)
                if isinstance(ast, dict):
                    stmt_type = ast.get("type")
                    if stmt_type == "INSERT":
                        n = _execute_insert(ast, user)
                        context.update({"sql": sql_text, "rows": [], "columns": [], "message": f"INSERT 成功：插入 {n} 行"})
                    elif stmt_type == "UPDATE":
                        n = _execute_update(ast, user)
                        context.update({"sql": sql_text, "rows": [], "columns": [], "message": f"UPDATE 完成：影响 {n} 行"})
                    elif stmt_type == "DELETE":
                        n = _execute_delete(ast, user)
                        context.update({"sql": sql_text, "rows": [], "columns": [], "message": f"DELETE 完成：删除 {n} 行"})
                    elif stmt_type == "CREATE":
                        _execute_create(ast, user)
                        context.update({"sql": sql_text, "rows": [], "columns": [], "message": "CREATE TABLE 成功"})
                    elif stmt_type == "DROP":
                        _execute_drop(ast, user)
                        context.update({"sql": sql_text, "rows": [], "columns": [], "message": "DROP TABLE 成功"})
                    else:
                        raise ValueError(f"未知语句类型: {stmt_type}")
                    # 执行写操作后刷新“可用表”列表
                    context["tables"] = _list_tables_for_user(user)
                else:
                    rows = _execute_select(ast, user)
                    context.update({
                        "sql": sql_text,
                        "rows": rows,
                        "columns": list(rows[0].keys()) if rows else [],
                        "message": f"共返回 {len(rows)} 行",
                    })
            except Exception as exc:  # pylint: disable=broad-except
                context.update({"sql": sql_text, "error": str(exc)})
    return render(request, "sql.html", context)

@csrf_exempt
def sql_select_api_view(request):
    """Map 工具栏 SQL 查询：仅支持 SELECT，并可返回 ids/bboxes 用于前端高亮选择。"""
    if not request.session.get("user"):
        return JsonResponse({"ok": False, "error": "未登录"}, status=401)
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "仅支持 POST"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    sql_text = str(payload.get("sql") or "").strip()
    if not sql_text:
        return JsonResponse({"ok": False, "error": "缺少 sql"}, status=400)

    max_rows = payload.get("max_rows")
    try:
        max_rows_i = int(max_rows) if max_rows is not None else 50
    except Exception:
        max_rows_i = 50
    max_rows_i = max(1, min(500, max_rows_i))

    user = _current_user(request)
    try:
        ast = parse_sql(sql_text)
    except Exception as exc:  # pylint: disable=broad-except
        return JsonResponse({"ok": False, "error": f"SQL 解析失败：{exc}"}, status=400)

    if isinstance(ast, dict):
        return JsonResponse({"ok": False, "error": "此工具仅支持 SELECT（不支持写操作）"}, status=400)

    node = ast

    try:
        owner, table, table_id, tdir = _canonical_table(user, node.from_table or "", must_exist=True)
        _ensure_table_registered_id(table_id, tdir)

        def predicate(row):
            return _eval_condition(node.where, row) if node.where else True

        # 先拿全列：用于后续生成 ids/bboxes（避免 SELECT 部分列导致无法联动选择）
        rows_full = MANAGER.query(table_id, predicate=predicate, columns=None)

        if node.order_by and rows_full:
            order_key = node.order_by.column
            key_in_row = _match_key(rows_full[0], order_key) if rows_full else None
            if key_in_row:
                reverse = (node.order_by.direction or "").upper() == "DESC"
                rows_full = sorted(rows_full, key=lambda x: x.get(key_in_row), reverse=reverse)

        # 先应用 SQL 自带 LIMIT
        if node.limit:
            try:
                limit_val = int(node.limit.limit)
                rows_full = rows_full[: max(0, limit_val)]
            except Exception:
                pass

        # 再应用 API 层 cap，避免一次返回过大导致浏览器内存飙升
        rows_full = rows_full[:max_rows_i]

        # ids/bboxes：用于前端把该表在地图中的要素高亮为“已选择”
        ids: list[str] = []
        bboxes: dict[str, list[float]] = {}
        for r in rows_full:
            id_key = _match_key(r, "id")
            if not id_key:
                continue
            fid = r.get(id_key)
            if fid is None:
                continue
            fid_s = str(fid)
            if not fid_s:
                continue

            ids.append(fid_s)

            gtype_key = _match_key(r, "geom_type")
            geom_key = _match_key(r, "geometry")
            gtype = r.get(gtype_key) if gtype_key else ""
            coords = r.get(geom_key) if geom_key else None
            if isinstance(coords, str):
                s = coords.strip()
                if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                    try:
                        coords = json.loads(s)
                    except Exception:
                        pass
            bb = algo_geom_bbox(str(gtype), coords)
            if bb is not None:
                bboxes[fid_s] = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]

        # 展示列：按 SELECT 列表裁剪（但不影响 ids/bboxes）
        display_rows = rows_full
        selected_cols = None
        if node.columns and getattr(node.columns, "column_name", None) and "*" not in node.columns.column_name:
            selected_cols = node.columns.column_name
        if selected_cols:
            filtered = []
            for r in rows_full:
                mapped = {}
                for col in selected_cols:
                    key = _match_key(r, str(col))
                    if key:
                        mapped[key] = r.get(key)
                filtered.append(mapped)
            display_rows = filtered

        columns = list(display_rows[0].keys()) if display_rows else []
        display_table = make_table_id(owner, table) if _is_admin(user) else table
        return JsonResponse(
            {
                "ok": True,
                "table": display_table,
                "columns": columns,
                "rows": display_rows,
                "row_count": len(display_rows),
                "ids": ids,
                "bboxes": bboxes,
                "selected_count": len(ids),
                "max_rows": max_rows_i,
            }
        )
    except Exception as exc:  # pylint: disable=broad-except
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)


def map_view(request):
    if not request.session.get("user"):
        return redirect("login")
    tables_info = []
    user = _current_user(request)
    for name in _list_tables_for_user(user):
        try:
            _owner, _t, table_id, tdir = _canonical_table(user, name, must_exist=True)
            _ensure_table_registered_id(table_id, tdir)
            data_table = MANAGER.classes['Table'].load(table_id)
            columns = list(data_table.attributes.keys())
            tables_info.append({"name": name, "columns": columns})
        except Exception:
            continue
    return render(request, "map.html", {"user": user, "tables": tables_info})

def _render_markdown_basic(md: str) -> str:
    """极简 Markdown 渲染（无第三方依赖），用于项目内文档页。

    支持：
    - 标题：#..######
    - 列表：- / * / 1.
    - 代码块：``` fenced
    - 行内 code：`...`
    - 链接：[text](url)
    """
    text = md or ""
    lines = text.splitlines()

    def esc(s: str) -> str:
        return html.escape(s, quote=True)

    def inline(s: str) -> str:
        # s is escaped, so we can safely inject tags by replacing escaped delimiters
        out = s
        # inline code: `...`
        out = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", out)
        # links: [t](u)
        out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: f'<a href="{m.group(2)}" target="_blank" rel="noopener noreferrer">{m.group(1)}</a>', out)
        return out

    html_out: list[str] = []
    in_code = False
    in_ul = False
    in_ol = False
    para: list[str] = []

    def flush_para():
        nonlocal para
        if para:
            html_out.append(f"<p>{'<br/>'.join(para)}</p>")
            para = []

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            html_out.append("</ul>")
            in_ul = False
        if in_ol:
            html_out.append("</ol>")
            in_ol = False

    for raw in lines:
        line = raw.rstrip("\n")
        if line.strip().startswith("```"):
            flush_para()
            close_lists()
            if not in_code:
                in_code = True
                html_out.append("<pre><code>")
            else:
                in_code = False
                html_out.append("</code></pre>")
            continue

        if in_code:
            html_out.append(esc(line))
            continue

        if not line.strip():
            flush_para()
            close_lists()
            continue

        # headings
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush_para()
            close_lists()
            level = len(m.group(1))
            title = inline(esc(m.group(2).strip()))
            html_out.append(f"<h{level}>{title}</h{level}>")
            continue

        # unordered list
        m = re.match(r"^\s*[-*]\s+(.*)$", line)
        if m:
            flush_para()
            if in_ol:
                html_out.append("</ol>")
                in_ol = False
            if not in_ul:
                html_out.append("<ul>")
                in_ul = True
            html_out.append(f"<li>{inline(esc(m.group(1).strip()))}</li>")
            continue

        # ordered list
        m = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if m:
            flush_para()
            if in_ul:
                html_out.append("</ul>")
                in_ul = False
            if not in_ol:
                html_out.append("<ol>")
                in_ol = True
            html_out.append(f"<li>{inline(esc(m.group(1).strip()))}</li>")
            continue

        # paragraph line
        para.append(inline(esc(line.strip())))

    flush_para()
    close_lists()
    if in_code:
        html_out.append("</code></pre>")
    return "\n".join(html_out)


def docs_view(request):
    """项目文档页：展示仓库内的 Markdown（无需外网/第三方库）。"""
    user = _current_user(request)
    logged_in = bool(user)

    project_root = Path(__file__).resolve().parents[1]
    docs_map = {
        "root": ("项目说明", project_root / "readme.md"),
        "algorithms": ("算法说明", project_root / "docs" / "ALGORITHMS.md"),
        "models": ("模型与存储", project_root / "space_app" / "models" / "README_models.md"),
        "cpp": ("C++ 扩展", project_root / "space_app" / "cpp_model" / "r_tree_usage.md"),
        "tests": ("测试脚本", project_root / "space_app" / "test_cases" / "README.md"),
        "report": ("工作报告", project_root / "report_2025-12-16.md"),
    }
    doc_key = str(request.GET.get("doc") or "root").strip()
    if doc_key not in docs_map:
        doc_key = "root"
    title, path = docs_map[doc_key]

    md = ""
    if path.exists():
        try:
            md = path.read_text(encoding="utf-8")
        except Exception:
            md = path.read_text(errors="ignore")
    else:
        md = f"# Not Found\n\n文档文件不存在：`{path}`"

    html_body = mark_safe(_render_markdown_basic(md))  # noqa: S308
    return render(request, "docs.html", {
        "user": user,
        "logged_in": logged_in,
        "doc_key": doc_key,
        "docs": [{"key": k, "title": t, "path": str(p)} for k, (t, p) in docs_map.items()],
        "doc_path": str(path),
        "doc_title": title,
        "doc_html": html_body,
    })


def geojson_view(request, table_name: str):
    if not request.session.get("user"):
        return redirect("login")
    user = _current_user(request)
    lon_field = request.GET.get("lon")
    lat_field = request.GET.get("lat")
    limit_raw = request.GET.get("limit")
    limit = None
    if limit_raw:
        try:
            limit = max(0, int(limit_raw))
        except Exception:
            limit = None
    _owner, _t, table_id, tdir = _canonical_table(user, table_name, must_exist=True)
    _ensure_table_registered_id(table_id, tdir)
    data_table = MANAGER.classes['Table'].load(table_id)
    attrs = data_table.attributes
    rows = data_table.data

    # 可选 bbox 过滤（用于按地图视野加载，避免一次性返回全量要素导致前端卡顿）
    def _parse_float(name: str):
        raw = request.GET.get(name)
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except Exception:
            return None

    minx = _parse_float("minx")
    miny = _parse_float("miny")
    maxx = _parse_float("maxx")
    maxy = _parse_float("maxy")
    bbox = (minx, miny, maxx, maxy) if None not in (minx, miny, maxx, maxy) else None

    def _iter_xy_from_coords(coords):
        if isinstance(coords, list):
            if len(coords) >= 2 and isinstance(coords[0], (int, float)) and isinstance(coords[1], (int, float)):
                yield float(coords[0]), float(coords[1])
            else:
                for item in coords:
                    yield from _iter_xy_from_coords(item)
        elif isinstance(coords, str):
            # 兼容极少数把点坐标存成 "lon,lat" 的情况
            parts = [p.strip() for p in coords.split(',')]
            if len(parts) >= 2:
                try:
                    yield float(parts[0]), float(parts[1])
                except Exception:
                    return

    def _geom_bbox(geom_type: str, coords):
        xs = []
        ys = []
        for x, y in _iter_xy_from_coords(coords):
            xs.append(x)
            ys.append(y)
            # 采样到一定数量就够了（防止极大几何导致这里过慢）
            if len(xs) >= 5000:
                break
        if not xs:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    def _bbox_intersects(a, b):
        # a,b: (minx,miny,maxx,maxy)
        return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

    features = []
    for row in rows:
        row_dict = {col: row[idx] for col, idx in attrs.items()}
        geom_type = (row_dict.get("geom_type") or "")
        geom = row_dict.get("geometry")

        # 默认优先使用原始 geometry，保留 Point/LineString/Polygon 等要素类型
        geometry = {"type": geom_type, "coordinates": geom} if geom is not None else None

        # 仅当“原始几何缺失”或“本身就是点”时，才允许使用 lon/lat 强制构造 Point。
        # 否则（线/面等）如果传了 lon/lat，会导致整张线图被错误渲染为点。
        lon_key = _match_key(row_dict, lon_field) if lon_field else None
        lat_key = _match_key(row_dict, lat_field) if lat_field else None
        if lon_key and lat_key and (geometry is None or str(geom_type).lower() == "point"):
            try:
                lon_val = float(row_dict[lon_key])
                lat_val = float(row_dict[lat_key])
            except Exception:
                lon_val = None
                lat_val = None
            if lon_val is not None and lat_val is not None:
                geometry = {"type": "Point", "coordinates": [lon_val, lat_val]}

        if not geometry:
            continue

        if bbox is not None:
            gb = _geom_bbox(geometry.get("type") or "", geometry.get("coordinates"))
            if gb is None or not _bbox_intersects(gb, bbox):
                continue

        base_props = {
            "id": row_dict.get("id"),
            "geom_type": geom_type,
            "lon": row_dict.get("lon"),
            "lat": row_dict.get("lat"),
        }
        orig_props = row_dict.get("properties")
        if not isinstance(orig_props, dict):
            orig_props = {}
        features.append({
            "type": "Feature",
            # 避免把 geometry 坐标重复塞到 properties 里（线/面会非常大，导致响应膨胀/前端卡顿）
            "properties": {**base_props, "properties": orig_props},
            "geometry": geometry,
        })

        if limit is not None and limit > 0 and len(features) >= limit:
            break
    return JsonResponse({"type": "FeatureCollection", "features": features})


def _tmp_export_dir() -> Path:
    # 与 .gitignore 的 .tmp/ 保持一致
    base = Path(__file__).resolve().parent.parent
    out = base / ".tmp" / "exports"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _selected_featurecollection_from_table(table_id: str, selected_ids: set[str]):
    """从仿真表构造“选中要素”的 FeatureCollection，用于导出/再入库。"""
    dt = MANAGER.classes["Table"].load(table_id)
    attrs = dt.attributes

    feats = []
    for row in dt.data:
        row_dict = {col: row[idx] for col, idx in attrs.items()}
        fid = row_dict.get("id")
        fid_s = "" if fid is None else str(fid)
        if fid_s not in selected_ids:
            continue

        geom_type = row_dict.get("geom_type") or ""
        coords = row_dict.get("geometry")
        if coords is None:
            continue

        props = row_dict.get("properties")
        if not isinstance(props, dict):
            props = {}
        # 保底把 id/geom_type 放进 properties（方便外部查看）
        if "id" not in props:
            props["id"] = fid
        if "geom_type" not in props:
            props["geom_type"] = geom_type

        feats.append({
            "type": "Feature",
            "id": fid,
            "properties": props,
            "geometry": {"type": geom_type, "coordinates": coords},
        })
    return {"type": "FeatureCollection", "features": feats}


def _feature_key_from_geojson_feature(feature: dict) -> Optional[str]:
    try:
        props = feature.get("properties") if isinstance(feature, dict) else None
        if isinstance(props, dict) and props.get("__client_id"):
            return str(props.get("__client_id"))
        if isinstance(props, dict):
            for k in ("id", "ID", "fid", "FID", "objectid", "OBJECTID"):
                if k in props and props.get(k) is not None:
                    return str(props.get(k))
        fid = feature.get("id") if isinstance(feature, dict) else None
        if fid is not None and not isinstance(fid, (dict, list)):
            return str(fid)
        return None
    except Exception:
        return None


def _as_features_from_table(table_name: str):
    """从仿真表（data_table.pkl）构造 GeoJSON features 列表（用于算法，不对外输出）。"""
    dt = MANAGER.classes['Table'].load(table_name)
    attrs = dt.attributes
    features = []
    for row in dt.data:
        row_dict = {col: row[idx] for col, idx in attrs.items()}
        geom_type = row_dict.get("geom_type") or ""
        geom = row_dict.get("geometry")
        if geom is None:
            continue
        features.append({
            "type": "Feature",
            "id": row_dict.get("id"),
            "properties": {"id": row_dict.get("id"), "geom_type": geom_type},
            "geometry": {"type": geom_type, "coordinates": geom},
        })
    return features


def _parse_time_value(value: Any) -> float:
    """解析轨迹时间字段。

    支持：
    - ISO: 2025-06-25T01:06:31（允许带毫秒/时区；也允许空格分隔：2025-07-01 09:30:01）
    - Unix timestamp: int/float/数字字符串（秒或毫秒）
    """
    if value is None:
        raise ValueError("时间字段为空")
    if isinstance(value, (int, float)):
        v = float(value)
        # 毫秒时间戳（13 位左右）自动转秒
        if abs(v) > 1e11:
            v = v / 1000.0
        return v
    s = str(value).strip()
    if not s:
        raise ValueError("时间字段为空字符串")
    # 纯数字 => unix seconds
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
        try:
            v = float(s)
            if abs(v) > 1e11:
                v = v / 1000.0
            return v
        except Exception as exc:
            raise ValueError(f"时间字段不是有效数字：{s}") from exc
    # ISO time
    try:
        # Python 3.8 支持 fromisoformat（含毫秒/时区）；不支持 Z，做一次兼容替换
        ss = s[:-1] + "+00:00" if s.endswith("Z") else s
        dt = datetime.fromisoformat(ss)
        return float(dt.timestamp())
    except Exception:
        pass
    # 常见格式兜底（允许空格、斜杠、毫秒）
    fmts = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%dT%H:%M:%S.%f",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return float(dt.timestamp())
        except Exception:
            continue
    raise ValueError(f"时间字段格式不合法：{s}")


def _get_row_field_value(row: dict, field: str) -> Any:
    token = (field or "").strip()
    if not token:
        return None
    # 支持 properties.xxx
    if token.lower().startswith("properties."):
        sub = token.split(".", 1)[1]
        props = row.get("properties")
        if isinstance(props, dict):
            for k in props.keys():
                if str(k) == sub or str(k).lower() == sub.lower():
                    return props.get(k)
        return None

    # 顶层列优先（大小写不敏感）
    for k in row.keys():
        if str(k).lower() == token.lower():
            return row.get(k)

    # 其次从 properties 中取（允许直接写子字段名）
    props = row.get("properties")
    if isinstance(props, dict):
        for k in props.keys():
            if str(k).lower() == token.lower():
                return props.get(k)
    return None


@csrf_exempt
def import_geojson_view(request):
    """从上传的 .json/.geojson 导入数据到仿真数据库，并记录到 R 树。

    - 如果不是 FeatureCollection，会尝试规范化转换。
    - 转换失败或文件后缀不合法则返回错误信息（前端弹窗）。
    """
    if not request.session.get("user"):
        return JsonResponse({"ok": False, "error": "未登录"}, status=401)
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "仅支持 POST"}, status=405)

    table_name = (request.POST.get("table_name") or "").strip()
    if not table_name:
        return JsonResponse({"ok": False, "error": "缺少 table_name"}, status=400)
    user = _current_user(request)
    try:
        owner, table, table_id, tdir = _canonical_table(user, table_name, must_exist=False)
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)

    if (tdir / 'data_table.pkl').exists():
        return JsonResponse({"ok": False, "error": f"表 {table_name} 已存在"}, status=400)

    up = request.FILES.get("file")
    if up is None:
        return JsonResponse({"ok": False, "error": "未选择文件"}, status=400)

    filename = up.name or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".geojson", ".json"}:
        return JsonResponse({
            "ok": False,
            "error": f"文件后缀错误：{suffix}（仅支持 .geojson 或 .json）",
        }, status=400)

    try:
        raw = up.read().decode("utf-8")
        doc = json.loads(raw)
    except Exception as exc:
        return JsonResponse({"ok": False, "error": f"数据内容出错：无法解析 JSON（{exc}）"}, status=400)

    try:
        normalized = normalize_geojson(doc)
    except Exception as exc:
        return JsonResponse({"ok": False, "error": f"数据内容出错：{exc}"}, status=400)

    # 写入临时文件后调用 ingest_geojson（由 Table 负责入库 + rtree_source + rtree_serialized）
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / f"{table_name}.geojson"
        tmp_path.write_text(json.dumps(normalized, ensure_ascii=False), encoding="utf-8")
        try:
            try:
                user_data_dir(owner).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            MANAGER.ingest_geojson(table_id, tmp_path, table_dir=tdir)
        except Exception as exc:
            # 若入库失败，清理可能创建的目录
            try:
                shutil.rmtree(tdir)
            except Exception:
                pass
            return JsonResponse({"ok": False, "error": f"入库失败：{exc}"}, status=400)

    # 返回新表列信息，前端可立刻加入下拉
    _ensure_table_registered_id(table_id, tdir)
    dt = MANAGER.classes['Table'].load(table_id)
    display = make_table_id(owner, table) if _is_admin(user) else table
    return JsonResponse({
        "ok": True,
        "table": display,
        "columns": list(dt.attributes.keys()),
        "message": "导入成功（已规范化并写入表 + R 树）",
    })


@csrf_exempt
def table_fields_view(request):
    """返回某张表可用于“字段选择”的候选列表（列名 + properties.*）。"""
    if not request.session.get("user"):
        return JsonResponse({"ok": False, "error": "未登录"}, status=401)
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "仅支持 POST"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    table_name = (payload.get("table") or payload.get("table_name") or "").strip()
    if not table_name:
        return JsonResponse({"ok": False, "error": "缺少 table"}, status=400)

    try:
        user = _current_user(request)
        _owner, _t, table_id, tdir = _canonical_table(user, table_name, must_exist=True)
        _ensure_table_registered_id(table_id, tdir)
        dt = MANAGER.classes["Table"].load(table_id)
    except Exception as exc:  # pylint: disable=broad-except
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)

    attrs = getattr(dt, "attributes", {}) or {}
    columns = list(attrs.keys()) if isinstance(attrs, dict) else []

    prop_keys: set[str] = set()
    truncated = False
    max_unique = 5000

    props_idx = attrs.get("properties") if isinstance(attrs, dict) else None
    if isinstance(props_idx, int) and props_idx >= 0:
        try:
            for row in getattr(dt, "data", []) or []:
                values = row.data if hasattr(row, "data") else row
                if not isinstance(values, (list, tuple)) or props_idx >= len(values):
                    continue
                props = values[props_idx]
                if isinstance(props, str):
                    s = props.strip()
                    if s.startswith("{") and s.endswith("}"):
                        try:
                            props = json.loads(s)
                        except Exception:
                            props = None
                if not isinstance(props, dict):
                    continue

                # 兼容历史/其他导入：properties 里可能又包一层 properties
                nested = props.get("properties") if isinstance(props.get("properties"), dict) else None
                for k in list(props.keys()):
                    if str(k).lower() == "properties":
                        continue
                    prop_keys.add(str(k))
                    if len(prop_keys) >= max_unique:
                        truncated = True
                        break
                if (not truncated) and isinstance(nested, dict):
                    for k in list(nested.keys()):
                        prop_keys.add(str(k))
                        if len(prop_keys) >= max_unique:
                            truncated = True
                            break
                if truncated:
                    break
        except Exception:
            pass

    prop_keys_sorted = sorted(prop_keys, key=lambda x: (x.lower(), x))
    options = list(columns)
    options.extend([f"properties.{k}" for k in prop_keys_sorted])

    return JsonResponse(
        {
            "ok": True,
            "table": table_name,
            "columns": columns,
            "prop_keys": prop_keys_sorted,
            "options": options,
            "truncated": truncated,
        }
    )


@csrf_exempt
def select_by_attribute_view(request):
    if not request.session.get("user"):
        return JsonResponse({"ok": False, "error": "未登录"}, status=401)
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "仅支持 POST"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    table_name = (payload.get("table") or "").strip()
    if not table_name:
        return JsonResponse({"ok": False, "error": "缺少 table"}, status=400)

    conditions = payload.get("conditions")
    if not isinstance(conditions, list):
        conditions = []

    try:
        user = _current_user(request)
        _owner, _t, table_id, tdir = _canonical_table(user, table_name, must_exist=True)
        _ensure_table_registered_id(table_id, tdir)
        dt = MANAGER.classes['Table'].load(table_id)
        ids, bboxes = algo_select_by_attribute(dt, conditions=conditions)
    except Exception as exc:  # pylint: disable=broad-except
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)

    bbox_out = {str(k): [v[0], v[1], v[2], v[3]] for k, v in bboxes.items()}
    return JsonResponse({
        "ok": True,
        "table": table_name,
        "ids": ids,
        "bboxes": bbox_out,
        "count": len(ids),
    })


def _coerce_bbox_list(raw) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        try:
            if isinstance(item, dict):
                out.append((float(item["minx"]), float(item["miny"]), float(item["maxx"]), float(item["maxy"])))
            elif isinstance(item, (list, tuple)) and len(item) >= 4:
                out.append((float(item[0]), float(item[1]), float(item[2]), float(item[3])))
        except Exception:
            continue
    return out


def _fallback_query_circle(dt, cx: float, cy: float, radius_m: float):
    attrs = dt.attributes
    ids: list[int] = []
    bboxes: dict[int, tuple[float, float, float, float]] = {}
    for row in dt.data:
        row_dict = {col: row[idx] for col, idx in attrs.items()}
        try:
            fid = int(row_dict.get("id"))
        except Exception:
            continue
        gtype = row_dict.get("geom_type") or ""
        coords = row_dict.get("geometry")
        bb = algo_geom_bbox(str(gtype), coords)
        if bb is None:
            continue
        # 先用“包围圆”的 bbox 做候选，再用 bbox-圆距离过滤
        deg_lon, deg_lat = meters_to_deg_components(cy, float(radius_m))
        query_box = (cx - deg_lon, cy - deg_lat, cx + deg_lon, cy + deg_lat)
        if algo_bbox_intersects(bb, query_box) and _bbox_intersects_circle_m(cx, cy, bb, float(radius_m)):
            ids.append(fid)
            bboxes[fid] = bb
    return ids, bboxes


@csrf_exempt
def export_selected_view(request):
    """导出“已选择要素”：

    - 导出到新表（调用 MANAGER.ingest_geojson，自动维护 R 树）
    - 导出到临时 GeoJSON 文件（返回下载链接）
    """
    if not request.session.get("user"):
        return JsonResponse({"ok": False, "error": "未登录"}, status=401)
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "仅支持 POST"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    table_name = (payload.get("table") or "").strip()
    if not table_name:
        return JsonResponse({"ok": False, "error": "缺少 table"}, status=400)
    user = _current_user(request)

    ids_raw = payload.get("ids")
    if not isinstance(ids_raw, list):
        ids_raw = []
    selected_ids = {str(x) for x in ids_raw if x is not None and str(x).strip() != ""}
    if not selected_ids:
        return JsonResponse({"ok": False, "error": "目标图层没有已选择要素"}, status=400)

    to_table = bool(payload.get("to_table", False))
    to_geojson = bool(payload.get("to_geojson", False))
    if not to_table and not to_geojson:
        return JsonResponse({"ok": False, "error": "请至少选择一种导出方式（导出到表 / 导出到 GeoJSON）"}, status=400)

    out_table = (payload.get("out_table") or "").strip()
    if to_table and not out_table:
        return JsonResponse({"ok": False, "error": "导出到表需要 out_table（新表名）"}, status=400)

    # 1) 生成 FeatureCollection（支持两种来源：后端表 / 前端传入的 FeatureCollection）
    fc_in = payload.get("feature_collection")
    if isinstance(fc_in, dict) and fc_in.get("type") == "FeatureCollection":
        feats_in = fc_in.get("features")
        if not isinstance(feats_in, list):
            feats_in = []
        # 仅导出“已选择要素”
        feats = []
        for f in feats_in:
            if not isinstance(f, dict):
                continue
            key = _feature_key_from_geojson_feature(f)
            if key is None:
                continue
            if key in selected_ids:
                feats.append(f)
        fc = {"type": "FeatureCollection", "features": feats}
    else:
        try:
            _owner, _t, table_id, tdir = _canonical_table(user, table_name, must_exist=True)
            _ensure_table_registered_id(table_id, tdir)
            fc = _selected_featurecollection_from_table(table_id, selected_ids)
        except Exception as exc:
            return JsonResponse({"ok": False, "error": f"构造导出数据失败：{exc}"}, status=400)

    feats = fc.get("features") if isinstance(fc, dict) else None
    if not isinstance(feats, list) or not feats:
        return JsonResponse({"ok": False, "error": "未找到可导出的要素（可能 geometry 缺失）"}, status=400)

    res: dict[str, Any] = {
        "ok": True,
        "source_table": table_name,
        "count": len(feats),
        "exported": {},
    }

    # 2) 导出到 GeoJSON：写入 .tmp/exports 并返回下载链接
    if to_geojson:
        token = uuid.uuid4().hex
        path = _tmp_export_dir() / f"{token}.geojson"
        try:
            path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            return JsonResponse({"ok": False, "error": f"写入 GeoJSON 失败：{exc}"}, status=500)
        res["exported"]["geojson"] = {
            "token": token,
            "download_url": f"/space_app/api/export/download/{token}/",
            "filename": f"{table_name}_selected_{len(feats)}.geojson",
        }

    # 3) 导出到新表：走 ingest_geojson（会维护 rtree_source + rtree_cache）
    if to_table:
        try:
            out_owner, out_table_name, out_table_id, out_tdir = _canonical_table(user, out_table, must_exist=False)
        except Exception as exc:
            return JsonResponse({"ok": False, "error": str(exc)}, status=400)
        if (out_tdir / "data_table.pkl").exists():
            return JsonResponse({"ok": False, "error": f"目标表已存在：{out_table}"}, status=400)
        tmp_name = f"export_{uuid.uuid4().hex}.geojson"
        tmp_path = _tmp_export_dir() / tmp_name
        try:
            tmp_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
            try:
                user_data_dir(out_owner).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            MANAGER.ingest_geojson(out_table_id, tmp_path, table_dir=out_tdir)
            _ensure_table_registered_id(out_table_id, out_tdir)
            dt = MANAGER.classes["Table"].load(out_table_id)
            display = make_table_id(out_owner, out_table_name) if _is_admin(user) else out_table_name
            res["exported"]["table"] = {"table": display, "columns": list(dt.attributes.keys())}
        except Exception as exc:
            try:
                if (out_tdir).exists():
                    shutil.rmtree(out_tdir)
            except Exception:
                pass
            return JsonResponse({"ok": False, "error": f"导出到表失败：{exc}"}, status=400)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    return JsonResponse(res)


def export_download_view(request, token: str):
    if not request.session.get("user"):
        return JsonResponse({"ok": False, "error": "未登录"}, status=401)
    safe = (token or "").strip()
    if not safe or any(c for c in safe if c not in "0123456789abcdef"):
        return JsonResponse({"ok": False, "error": "invalid token"}, status=400)
    path = _tmp_export_dir() / f"{safe}.geojson"
    if not path.exists():
        return JsonResponse({"ok": False, "error": "file not found"}, status=404)
    return FileResponse(open(path, "rb"), as_attachment=True, filename=f"{safe}.geojson", content_type="application/geo+json")


def _parse_point_xy(raw) -> tuple[float, float]:
    """支持 dict/list 两种格式，统一解析为 (x=lon, y=lat)。"""
    if isinstance(raw, dict):
        x = raw.get("x")
        if x is None:
            x = raw.get("lng")
        if x is None:
            x = raw.get("lon")
        if x is None:
            x = raw.get("longitude")

        y = raw.get("y")
        if y is None:
            y = raw.get("lat")
        if y is None:
            y = raw.get("latitude")

        if x is None or y is None:
            raise ValueError("点坐标缺少 x/y（或 lng/lat）")
        return float(x), float(y)

    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return float(raw[0]), float(raw[1])

    raise ValueError("点坐标格式错误（应为 {x,y} 或 [x,y]）")


def _get_route_graph(table_name: str) -> dict[str, Any]:
    cached = _ROUTE_GRAPH_CACHE.get(table_name)
    if cached:
        return cached

    feats = _as_features_from_table(table_name)
    graph_py, edges = build_road_graph(feats, ndigits=6)
    native_graph = build_native_graph(graph_py.nodes, edges)

    item = {
        "graph_py": graph_py,
        "edges": edges,
        "native_graph": native_graph,
        "built_at": time.time(),
    }
    _ROUTE_GRAPH_CACHE[table_name] = item
    return item


@csrf_exempt
def shortest_path_view(request):
    if not request.session.get("user"):
        return JsonResponse({"ok": False, "error": "未登录"}, status=401)
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "仅支持 POST"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    table_name = (payload.get("table") or "").strip()
    if not table_name:
        return JsonResponse({"ok": False, "error": "缺少 table（路网表名）"}, status=400)
    user = _current_user(request)
    try:
        owner, table, table_id, tdir = _canonical_table(user, table_name, must_exist=True)
        _ensure_table_registered_id(table_id, tdir)
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)

    try:
        start = _parse_point_xy(payload.get("start"))
        end = _parse_point_xy(payload.get("end"))
    except Exception as exc:
        return JsonResponse({"ok": False, "error": f"起终点参数错误：{exc}"}, status=400)

    algo = (payload.get("algo") or "dijkstra").strip().lower()
    if algo not in {"dijkstra", "astar", "floyd"}:
        return JsonResponse({"ok": False, "error": f"未知算法：{algo}（支持 floyd/astar/dijkstra）"}, status=400)

    # 固定参数：起点集=2，终点集=2，K=2
    cand_k = 2
    k_paths = 2

    try:
        gitem = _get_route_graph(table_id)
        gpy = gitem["graph_py"]
        native_graph = gitem.get("native_graph")
    except Exception as exc:
        return JsonResponse({"ok": False, "error": f"构建路网失败：{exc}"}, status=400)

    if not gpy.nodes:
        return JsonResponse({"ok": False, "error": "路网为空"}, status=400)

    s_cands = algo_nearest_nodes(gpy.nodes, start, k=cand_k)
    e_cands = algo_nearest_nodes(gpy.nodes, end, k=cand_k)
    if len(s_cands) < 1 or len(e_cands) < 1:
        return JsonResponse({"ok": False, "error": "路网节点不足，无法选取候选点"}, status=400)

    def _pt(xy):
        return [float(xy[0]), float(xy[1])]

    features_out = []
    req_id = uuid.uuid4().hex[:10]

    # 说明：Floyd 在大图上不可用，这里先做兼容：回退到 Dijkstra/A*
    algo_used = algo
    algo_internal = algo
    if algo == "floyd":
        algo_internal = "dijkstra"
        algo_used = "dijkstra(floyd_fallback)"

    for si, (s_idx, s_dist) in enumerate(s_cands[:2]):
        for ei, (e_idx, e_dist) in enumerate(e_cands[:2]):
            # per (start,end) pair: compute K shortest paths
            paths = []
            if native_graph is not None:
                try:
                    paths = k_shortest_paths_native(native_graph, s_idx, e_idx, k=k_paths, algo=algo_internal)
                except Exception:
                    paths = []
            if not paths:
                paths = k_shortest_paths_py(gpy, s_idx, e_idx, k=k_paths, algo=algo_internal)

            for rank, p in enumerate(paths):
                nodes = p.get("nodes")
                if not isinstance(nodes, list) or len(nodes) < 2:
                    continue
                try:
                    route = [gpy.nodes[int(n)] for n in nodes]
                except Exception:
                    continue

                # 拼接：点击点 -> 起点候选节点 -> 路网 -> 终点候选节点 -> 点击点
                coords: List[Coord] = []

                def _push(ptxy: Coord):
                    if coords and (abs(coords[-1][0] - ptxy[0]) < 1e-12) and (abs(coords[-1][1] - ptxy[1]) < 1e-12):
                        return
                    coords.append(ptxy)

                _push(start)
                for ptxy in route:
                    _push(ptxy)
                _push(end)

                network_len = float(p.get("distance_m") or 0.0)
                extra_len = float(haversine_m(start, gpy.nodes[s_idx])) + float(haversine_m(gpy.nodes[e_idx], end))
                total_len = network_len + extra_len

                props = {
                    "id": f"route_{req_id}_{si}{ei}_{rank}",
                    "__client_id": f"route_{req_id}_{si}{ei}_{rank}",
                    "source_table": table_name,
                    "algo": algo,
                    "algo_used": algo_used,
                    "pair": [si, ei],
                    "rank": rank,
                    "start_click": _pt(start),
                    "end_click": _pt(end),
                    "start_candidate": _pt(gpy.nodes[s_idx]),
                    "end_candidate": _pt(gpy.nodes[e_idx]),
                    "start_candidate_dist_m": float(s_dist),
                    "end_candidate_dist_m": float(e_dist),
                    "network_length_m": network_len,
                    "total_length_m": total_len,
                }

                features_out.append({
                    "type": "Feature",
                    "id": props["id"],
                    "properties": props,
                    "geometry": {"type": "LineString", "coordinates": [_pt(xy) for xy in coords]},
                })

    if not features_out:
        return JsonResponse({"ok": False, "error": "未找到可用路线（候选点之间无连通路径）"}, status=400)

    return JsonResponse({
        "ok": True,
        "table": make_table_id(owner, table) if _is_admin(user) else table,
        "algo": algo,
        "algo_used": algo_used,
        "candidates": {
            "start": [{"idx": int(i), "coord": _pt(gpy.nodes[i]), "dist_m": float(d)} for i, d in s_cands[:2]],
            "end": [{"idx": int(i), "coord": _pt(gpy.nodes[i]), "dist_m": float(d)} for i, d in e_cands[:2]],
        },
        "count": len(features_out),
        "feature_collection": {"type": "FeatureCollection", "features": features_out},
    })


@csrf_exempt
def trajectory_correct_view(request):
    """轨迹校正：HMM map matching + Kalman（C++ 侧实现）。

    输入：
    - trajectory_table: 点图层（表）
    - road_table: 路网线图层（表）
    - time_field: 时间字段名（支持 properties.xxx 或直接字段名）
    - bbox(可选): {minx,miny,maxx,maxy}，用于只处理当前视野
    - max_points(可选): 最大处理点数（防止一次性处理超大轨迹）
    """
    if not request.session.get("user"):
        return JsonResponse({"ok": False, "error": "未登录"}, status=401)
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "仅支持 POST"}, status=405)
    if not trajectory_native_available():
        return JsonResponse({"ok": False, "error": "trajectory_cpp 未构建：请先在 cpp_model 目录 cmake 构建"}, status=500)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    traj_table = (payload.get("trajectory_table") or payload.get("traj_table") or "").strip()
    road_table = (payload.get("road_table") or "").strip()
    time_field = (payload.get("time_field") or "").strip()
    if not traj_table:
        return JsonResponse({"ok": False, "error": "缺少 trajectory_table"}, status=400)
    if not road_table:
        return JsonResponse({"ok": False, "error": "缺少 road_table"}, status=400)
    if not time_field:
        return JsonResponse({"ok": False, "error": "缺少 time_field（时间字段）"}, status=400)

    bbox_raw = payload.get("bbox")
    bbox = None
    if isinstance(bbox_raw, dict):
        try:
            bbox = (
                float(bbox_raw["minx"]),
                float(bbox_raw["miny"]),
                float(bbox_raw["maxx"]),
                float(bbox_raw["maxy"]),
            )
        except Exception:
            bbox = None

    max_points = payload.get("max_points")
    try:
        max_points_i = int(max_points) if max_points is not None else 5000
    except Exception:
        max_points_i = 5000
    max_points_i = max(200, min(100000, max_points_i))

    user = _current_user(request)
    try:
        _to, _tt, traj_id, traj_dir = _canonical_table(user, traj_table, must_exist=True)
        _ensure_table_registered_id(traj_id, traj_dir)
        _ro, _rt, road_id, road_dir = _canonical_table(user, road_table, must_exist=True)
        _ensure_table_registered_id(road_id, road_dir)
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)

    # 1) 路网节点：复用最短路径的建图缓存
    try:
        gitem = _get_route_graph(road_id)
        gpy = gitem["graph_py"]
        nodes = gpy.nodes
    except Exception as exc:
        return JsonResponse({"ok": False, "error": f"构建路网失败：{exc}"}, status=400)
    if not nodes:
        return JsonResponse({"ok": False, "error": "路网为空"}, status=400)

    nodes_lon = [float(x) for x, _y in nodes]
    nodes_lat = [float(y) for _x, y in nodes]

    # 2) 读取轨迹点（点图层），检查时间字段
    dt = MANAGER.classes["Table"].load(traj_id)
    attrs = dt.attributes

    obs = []  # [(t_unix, lon, lat, time_raw, props)]
    missing_time = 0
    invalid_geom = 0

    for row in dt.data:
        row_dict = {col: row[idx] for col, idx in attrs.items()}
        gtype = str(row_dict.get("geom_type") or "")
        if gtype.lower() != "point":
            invalid_geom += 1
            continue
        coords = row_dict.get("geometry")
        if not (isinstance(coords, (list, tuple)) and len(coords) >= 2):
            continue
        lon = coords[0]
        lat = coords[1]
        try:
            lon_f = float(lon)
            lat_f = float(lat)
        except Exception:
            continue

        if bbox is not None:
            if not (bbox[0] <= lon_f <= bbox[2] and bbox[1] <= lat_f <= bbox[3]):
                continue

        props = row_dict.get("properties")
        if not isinstance(props, dict):
            props = {}
        time_raw_val = _get_row_field_value({**row_dict, "properties": props}, time_field)
        if time_raw_val is None:
            missing_time += 1
            continue
        try:
            t_unix = _parse_time_value(time_raw_val)
        except Exception:
            missing_time += 1
            continue

        obs.append((t_unix, lon_f, lat_f, time_raw_val, dict(props)))
        if len(obs) >= max_points_i * 3:
            # 先做一个软上限：避免“bbox 很大 + 超大表”导致一次性扫太久
            break

    if invalid_geom > 0 and not obs:
        return JsonResponse({"ok": False, "error": "轨迹图层必须为点图层（Point）"}, status=400)
    if not obs:
        return JsonResponse({"ok": False, "error": f"未提取到可用轨迹点（时间字段缺失/格式错误：{missing_time} 条）"}, status=400)

    # 按时间排序并检查严格递增
    obs.sort(key=lambda t: float(t[0]))
    if len(obs) < 2:
        return JsonResponse({"ok": False, "error": "轨迹点数量不足（至少 2 个点）"}, status=400)

    times = [float(x[0]) for x in obs]
    # 宽容处理：允许逆序/重复。我们排序后，若存在重复/非递增时间，做一个很小的抖动，保证严格递增
    #（Kalman 需要 dt>0；否则会出现除零或 NaN）。
    adjusted_time = 0
    eps = 1e-3
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            times[i] = times[i - 1] + eps
            adjusted_time += 1

    # 真正参与计算的点数限制
    if len(obs) > max_points_i:
        obs = obs[:max_points_i]
        times = times[:max_points_i]

    obs_lon = [float(x[1]) for x in obs]
    obs_lat = [float(x[2]) for x in obs]

    # 3) 调用 C++：HMM map matching + Kalman
    try:
        result = trajectory_hmm_match_and_kalman(
            obs_lon=obs_lon,
            obs_lat=obs_lat,
            obs_t=times,
            nodes_lon=nodes_lon,
            nodes_lat=nodes_lat,
            k_candidates=2,
        )
    except Exception as exc:
        return JsonResponse({"ok": False, "error": f"轨迹校正失败：{exc}"}, status=400)

    matched_lon = result.get("matched_lon") or []
    matched_lat = result.get("matched_lat") or []
    est_speed = result.get("est_speed_mps") or []
    est_heading = result.get("est_heading_deg") or []
    node_idx = result.get("matched_node_index") or []

    n = min(len(obs), len(matched_lon), len(matched_lat), len(est_speed), len(est_heading))
    if n < 2:
        return JsonResponse({"ok": False, "error": "轨迹校正结果为空"}, status=400)

    req_id = uuid.uuid4().hex[:10]
    out_name = (payload.get("out_name") or "").strip() or f"traj_corrected_{req_id}"

    # time_field 输出：保留用户输入原字段名（支持中文），以及 unix seconds
    feats = []
    for i in range(n):
        t_unix, raw_lon, raw_lat, time_raw_val, props = obs[i]
        props = dict(props) if isinstance(props, dict) else {}
        props["__client_id"] = f"{out_name}_{i}"
        props["id"] = i
        props["source_trajectory"] = traj_table
        props["road_table"] = road_table
        props["time_field"] = time_field
        props["time_unix_s"] = float(t_unix)
        props["time_raw"] = time_raw_val
        props["raw_lon"] = float(raw_lon)
        props["raw_lat"] = float(raw_lat)
        props["matched_lon"] = float(matched_lon[i])
        props["matched_lat"] = float(matched_lat[i])
        try:
            props["matched_node_index"] = int(node_idx[i]) if i < len(node_idx) else None
        except Exception:
            props["matched_node_index"] = None
        props["est_speed_mps"] = float(est_speed[i])
        props["est_heading_deg"] = float(est_heading[i]) % 360.0

        feats.append({
            "type": "Feature",
            "id": props["id"],
            "properties": props,
            "geometry": {"type": "Point", "coordinates": [float(matched_lon[i]), float(matched_lat[i])]},
        })

    return JsonResponse({
        "ok": True,
        "out_name": out_name,
        "source": {"trajectory_table": traj_table, "road_table": road_table},
        "count": len(feats),
        "feature_collection": {"type": "FeatureCollection", "features": feats},
        "note": (
            f"已处理 {len(feats)} 个点（max_points={max_points_i}；bbox={'on' if bbox else 'off'}；"
            f"time_sorted=on；time_adjusted={adjusted_time}）"
        ),
    })


def _fallback_query_box(dt, query_bbox: tuple[float, float, float, float]):
    attrs = dt.attributes
    ids: list[int] = []
    bboxes: dict[int, tuple[float, float, float, float]] = {}
    for row in dt.data:
        row_dict = {col: row[idx] for col, idx in attrs.items()}
        try:
            fid = int(row_dict.get("id"))
        except Exception:
            continue
        gtype = row_dict.get("geom_type") or ""
        coords = row_dict.get("geometry")
        bb = algo_geom_bbox(str(gtype), coords)
        if bb is None:
            continue
        if algo_bbox_intersects(bb, query_bbox):
            ids.append(fid)
            bboxes[fid] = bb
    return ids, bboxes


@csrf_exempt
def select_by_location_view(request):
    if not request.session.get("user"):
        return JsonResponse({"ok": False, "error": "未登录"}, status=401)
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "仅支持 POST"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    target_table = (payload.get("target_table") or "").strip()
    if not target_table:
        return JsonResponse({"ok": False, "error": "缺少 target_table"}, status=400)

    mode = (payload.get("mode") or "").strip()
    if mode not in {"point", "bbox_intersects", "circle_from_selected"}:
        return JsonResponse({"ok": False, "error": f"未知 mode: {mode}"}, status=400)

    user = _current_user(request)
    try:
        _owner, _t, target_table_id, target_tdir = _canonical_table(user, target_table, must_exist=True)
        _ensure_table_registered_id(target_table_id, target_tdir)
        dt = MANAGER.classes['Table'].load(target_table_id)
    except Exception as exc:  # pylint: disable=broad-except
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)

    # 参照要素 bbox（由前端传入：来自参照图层的当前选择）
    ref_bboxes = _coerce_bbox_list(payload.get("ref_bboxes"))

    ids: list[int] = []
    bboxes: dict[int, tuple[float, float, float, float]] = {}

    debug = bool(payload.get("debug"))
    debug_force_rebuild = bool(payload.get("debug_force_rebuild_rtree"))
    debug_info = {
        "mode": mode,
        "target_table": target_table,
        "used_rtree": False,
        "force_rebuild": debug_force_rebuild,
    } if debug else None

    def _dist_m(cx: float, cy: float, x: float, y: float) -> float:
        # 局部平面近似：经度按 cos(lat) 缩放
        m_per_deg_lat = 111_320.0
        coslat = max(1e-6, abs(math.cos(math.radians(cy))))
        m_per_deg_lon = 111_320.0 * coslat
        dx = (x - cx) * m_per_deg_lon
        dy = (y - cy) * m_per_deg_lat
        return float((dx * dx + dy * dy) ** 0.5)

    # 优先用 C++ R-tree 加速；不可用则 fallback 纯 Python 扫描
    rtree = None
    try:
        if debug_force_rebuild:
            # 清理缓存 + 删除序列化/版本文件，强制重建（用于排查 ids 错位等问题）
            try:
                MANAGER.rtree_cache.pop(target_table_id, None)
            except Exception:
                pass
            try:
                ser = Path(str(target_tdir)) / 'rtree_serialized.json'
                ver = Path(str(target_tdir)) / 'rtree_version.txt'
                if ser.exists():
                    ser.unlink()
                if ver.exists():
                    ver.unlink()
            except Exception:
                pass
        rtree = MANAGER.load_rtree(target_table_id)
    except Exception:
        rtree = None

    try:
        if mode == "point":
            pt = payload.get("point") or {}
            cx = float(pt.get("x"))
            cy = float(pt.get("y"))
            radius_m = float(payload.get("radius_m") or 0.0)
            if radius_m <= 0:
                return JsonResponse({"ok": False, "error": "radius_m 必须大于 0"}, status=400)
            if rtree is not None:
                if debug_info is not None:
                    debug_info["used_rtree"] = True
                ids, bboxes = select_by_location_rtree(rtree, "point", point=(cx, cy), radius_m=radius_m)
            else:
                ids, bboxes = _fallback_query_circle(dt, cx, cy, radius_m)

            if debug_info is not None:
                debug_info.update({"cx": cx, "cy": cy, "radius_m": radius_m, "returned": len(ids)})
                # 返回少量样本：用于人工核对“返回点是否在圆内”
                samples = []
                for k, bb in list(bboxes.items())[:10]:
                    try:
                        x0 = (bb[0] + bb[2]) / 2.0
                        y0 = (bb[1] + bb[3]) / 2.0
                        samples.append({"id": int(k), "x": x0, "y": y0, "dist_m": _dist_m(cx, cy, x0, y0)})
                    except Exception:
                        continue
                debug_info["sample_hits"] = samples
                # 断言：返回的每个 bbox 都应与米圆相交（点要素等价于点在圆内）
                for k, bb in list(bboxes.items())[:2000]:
                    assert _bbox_intersects_circle_m(cx, cy, bb, radius_m), f"bbox outside circle: id={k} bb={bb}"

                # 进一步断言：返回的 id 在表中对应的几何 bbox 也应与圆相交
                sample_ids = set(ids[:200])
                found = 0
                mismatches = []
                attrs = dt.attributes
                for row in dt.data:
                    row_id = row[attrs.get("id")] if "id" in attrs else None
                    try:
                        row_id_int = int(row_id)
                    except Exception:
                        continue
                    if row_id_int not in sample_ids:
                        continue
                    found += 1
                    gtype = row[attrs.get("geom_type")] if "geom_type" in attrs else ""
                    coords = row[attrs.get("geometry")] if "geometry" in attrs else None
                    tbb = algo_geom_bbox(str(gtype), coords)
                    if tbb is None:
                        mismatches.append({"id": row_id_int, "reason": "table_bbox_none"})
                        continue
                    if not _bbox_intersects_circle_m(cx, cy, tbb, radius_m):
                        mismatches.append({"id": row_id_int, "table_bbox": list(tbb)})
                    if len(mismatches) >= 10:
                        break
                debug_info["table_bbox_checked"] = found
                debug_info["table_bbox_mismatch_sample"] = mismatches
                assert not mismatches, f"table bbox mismatch sample: {mismatches[:3]}"

        elif mode == "bbox_intersects":
            if not ref_bboxes:
                return JsonResponse({"ok": True, "target_table": target_table, "ids": [], "bboxes": {}, "count": 0})
            if rtree is not None:
                if debug_info is not None:
                    debug_info["used_rtree"] = True
                ids, bboxes = select_by_location_rtree(rtree, "bbox_intersects", ref_bboxes=ref_bboxes)
            else:
                ids_set = set()
                for bb in ref_bboxes:
                    i2, b2 = _fallback_query_box(dt, bb)
                    for fid in i2:
                        ids_set.add(fid)
                    bboxes.update(b2)
                ids = sorted(ids_set)

            if debug_info is not None:
                debug_info.update({"ref_bboxes": len(ref_bboxes), "returned": len(ids)})

        elif mode == "circle_from_selected":
            radius_m = float(payload.get("radius_m") or 0.0)
            if radius_m <= 0:
                return JsonResponse({"ok": False, "error": "radius_m 必须大于 0"}, status=400)
            if not ref_bboxes:
                return JsonResponse({"ok": True, "target_table": target_table, "ids": [], "bboxes": {}, "count": 0})
            # 用参照要素 bbox 中心作为圆心
            ids_set = set()
            for bb in ref_bboxes:
                cx = (bb[0] + bb[2]) / 2.0
                cy = (bb[1] + bb[3]) / 2.0
                if rtree is not None:
                    if debug_info is not None:
                        debug_info["used_rtree"] = True
                    i2, b2 = select_by_location_rtree(rtree, "point", point=(cx, cy), radius_m=radius_m)
                else:
                    i2, b2 = _fallback_query_circle(dt, cx, cy, radius_m)
                for fid in i2:
                    ids_set.add(fid)
                bboxes.update(b2)
            ids = sorted(ids_set)

            if debug_info is not None:
                debug_info.update({"ref_bboxes": len(ref_bboxes), "radius_m": radius_m, "returned": len(ids)})
                # 对每一个参照圆心，采样校验前 N 个 bbox 至少与某个圆相交
                centers = [((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0) for bb in ref_bboxes]
                for k, bb in list(bboxes.items())[:2000]:
                    ok_any = False
                    for (cx, cy) in centers[:50]:
                        if _bbox_intersects_circle_m(cx, cy, bb, radius_m):
                            ok_any = True
                            break
                    assert ok_any, f"bbox outside all circles (sampled): id={k} bb={bb}"

                # 进一步断言：返回的 id 在表中对应的 bbox 应至少与某个圆相交
                sample_ids = set(ids[:200])
                found = 0
                mismatches = []
                attrs = dt.attributes
                for row in dt.data:
                    row_id = row[attrs.get("id")] if "id" in attrs else None
                    try:
                        row_id_int = int(row_id)
                    except Exception:
                        continue
                    if row_id_int not in sample_ids:
                        continue
                    found += 1
                    gtype = row[attrs.get("geom_type")] if "geom_type" in attrs else ""
                    coords = row[attrs.get("geometry")] if "geometry" in attrs else None
                    tbb = algo_geom_bbox(str(gtype), coords)
                    if tbb is None:
                        mismatches.append({"id": row_id_int, "reason": "table_bbox_none"})
                        continue
                    ok_any = False
                    for (cx, cy) in centers[:50]:
                        if _bbox_intersects_circle_m(cx, cy, tbb, radius_m):
                            ok_any = True
                            break
                    if not ok_any:
                        mismatches.append({"id": row_id_int, "table_bbox": list(tbb)})
                    if len(mismatches) >= 10:
                        break
                debug_info["table_bbox_checked"] = found
                debug_info["table_bbox_mismatch_sample"] = mismatches
                assert not mismatches, f"table bbox mismatch sample: {mismatches[:3]}"

    except Exception as exc:  # pylint: disable=broad-except
        if debug_info is not None:
            debug_info["exception"] = str(exc)
        return JsonResponse({"ok": False, "error": str(exc), "debug": debug_info}, status=400)

    bbox_out = {str(k): [v[0], v[1], v[2], v[3]] for k, v in bboxes.items()}
    resp = {
        "ok": True,
        "target_table": target_table,
        "ids": ids,
        "bboxes": bbox_out,
        "count": len(ids),
    }
    if debug_info is not None:
        resp["debug"] = debug_info
    return JsonResponse(resp)


def _with_temp_backup(table_id: str, table_dir: Path):
    """为拓扑操作创建临时备份，确保不会污染原始表目录。

    约定：
    - 读取/计算全部基于临时副本
    - 完成后恢复 MANAGER.table 指向原始 pkl
    """
    src_dir = Path(str(table_dir))
    if not src_dir.exists():
        raise ValueError(f"表目录不存在：{table_id}")

    # 项目内临时目录（避免使用系统级临时目录）
    project_root = Path(__file__).resolve().parents[1]
    tmp_base = project_root / ".tmp" / "topology"
    tmp_base.mkdir(parents=True, exist_ok=True)
    safe = str(table_id).replace(":", "_")
    tmp_root = tempfile.mkdtemp(prefix=f"topology_{safe}_", dir=str(tmp_base))
    backup_dir = Path(tmp_root) / "table"
    shutil.copytree(src_dir, backup_dir)

    orig = MANAGER.table.get(table_id)
    MANAGER.table[table_id] = backup_dir / 'data_table.pkl'

    return tmp_root, orig


def _project_tmp_dir(*parts: str) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    base = project_root / ".tmp"
    if parts:
        base = base.joinpath(*parts)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _build_repaired_featurecollection(source_table: str) -> dict:
    """从源表构造“修复后路网”的 FeatureCollection（不写回原表）。"""
    features = _as_features_from_table(source_table)
    issues, _stats = check_topology_layer(features, search_bbox=None)
    rep = repair_layer(features, issues=issues)

    deleted = set(rep.get("deleted") or [])
    keep = []
    for f in features:
        fid = f.get("id")
        if fid in deleted:
            continue
        keep.append(f)
    new_lines = rep.get("new_lines") or []

    return {
        "type": "FeatureCollection",
        "features": keep + list(new_lines),
    }


@csrf_exempt
def topology_check_view(request):
    if not request.session.get("user"):
        return JsonResponse({"ok": False, "error": "未登录"}, status=401)
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "仅支持 POST"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    keep_tmp = bool(payload.get("keep_tmp"))

    table_ref = (payload.get("table") or "").strip()
    if not table_ref:
        return JsonResponse({"ok": False, "error": "缺少 table"}, status=400)
    user = _current_user(request)
    try:
        _owner, _t, table_id, tdir = _canonical_table(user, table_ref, must_exist=True)
        _ensure_table_registered_id(table_id, tdir)
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)

    # 整层检查：默认不限制 bbox（如果你担心性能，可在前端传 bbox 做局部检查）
    bbox = None
    try:
        b = payload.get("bbox")
        if isinstance(b, dict):
            bbox = (float(b["minx"]), float(b["miny"]), float(b["maxx"]), float(b["maxy"]))
    except Exception:
        bbox = None

    tmp_root = None
    orig = None
    try:
        tmp_root, orig = _with_temp_backup(table_id, tdir)
        features = _as_features_from_table(table_id)
        issues, stats = check_topology_layer(features, search_bbox=bbox)
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)
    finally:
        if orig is not None:
            MANAGER.table[table_id] = orig

    # 返回红点（问题点）
    feat_out = []
    for i in issues:
        feat_out.append({
            "type": "Feature",
            "properties": {"kind": i.kind, "message": i.message},
            "geometry": {"type": "Point", "coordinates": [i.point[0], i.point[1]]},
        })

    # 两阶段流程：发放一次性检查 token，修复必须携带该 token
    check_token = uuid.uuid4().hex
    tokens = request.session.get("topology_check_tokens")
    if not isinstance(tokens, dict):
        tokens = {}

    # 清理过期 token（避免 session 膨胀）
    try:
        now_ts = time.time()
        expired = []
        for k, v in tokens.items():
            ts = (v or {}).get("ts")
            if ts is None:
                continue
            if (now_ts - float(ts)) > 1800.0:  # 30 minutes
                expired.append(k)
        for k in expired:
            tokens.pop(k, None)
    except Exception:
        pass

    tokens[check_token] = {
        "table_id": table_id,
        "table_ref": table_ref,
        "bbox": list(bbox) if bbox is not None else None,
        "ts": time.time(),
    }
    request.session["topology_check_tokens"] = tokens
    request.session.modified = True

    return JsonResponse({
        "ok": True,
        "issues": {"type": "FeatureCollection", "features": feat_out},
        "count": len(feat_out),
        "stats": stats,
        "check_token": check_token,
        "tmp_dir": tmp_root if keep_tmp else None,
    })


@csrf_exempt
def topology_repair_view(request):
    if not request.session.get("user"):
        return JsonResponse({"ok": False, "error": "未登录"}, status=401)
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "仅支持 POST"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    keep_tmp = bool(payload.get("keep_tmp"))

    table_ref = (payload.get("table") or "").strip()
    if not table_ref:
        return JsonResponse({"ok": False, "error": "缺少 table"}, status=400)
    user = _current_user(request)
    try:
        _owner, _t, table_id, tdir = _canonical_table(user, table_ref, must_exist=True)
        _ensure_table_registered_id(table_id, tdir)
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)

    check_token = (payload.get("check_token") or "").strip()
    if not check_token:
        return JsonResponse({"ok": False, "error": "请先点击“检查拓扑”，再执行修复"}, status=400)

    bbox = None
    try:
        b = payload.get("bbox")
        if isinstance(b, dict):
            bbox = (float(b["minx"]), float(b["miny"]), float(b["maxx"]), float(b["maxy"]))
    except Exception:
        bbox = None

    # 校验 token 与参数一致，保证“两阶段：先检查后修复”
    tokens = request.session.get("topology_check_tokens")
    if not isinstance(tokens, dict) or check_token not in tokens:
        return JsonResponse({"ok": False, "error": "拓扑检查状态已失效：请重新点击“检查拓扑”"}, status=400)
    meta = tokens.get(check_token) or {}
    if str(meta.get("table_id")) != table_id:
        return JsonResponse({"ok": False, "error": "检查/修复的图层不一致：请重新检查拓扑"}, status=400)

    meta_bbox = meta.get("bbox")
    if (bbox is None and meta_bbox is not None) or (bbox is not None and meta_bbox is None):
        return JsonResponse({"ok": False, "error": "检查/修复的 bbox 不一致：请重新检查拓扑"}, status=400)
    if bbox is not None and meta_bbox is not None:
        try:
            if list(bbox) != list(meta_bbox):
                return JsonResponse({"ok": False, "error": "检查/修复的 bbox 不一致：请重新检查拓扑"}, status=400)
        except Exception:
            return JsonResponse({"ok": False, "error": "检查状态异常：请重新检查拓扑"}, status=400)

    tmp_root = None
    orig = None
    try:
        tmp_root, orig = _with_temp_backup(table_id, tdir)
        features = _as_features_from_table(table_id)
        issues, _stats = check_topology_layer(features, search_bbox=bbox)
        rep = repair_layer(features, issues=issues)

        # 修复成功：一次性 token 作废，强制下一次必须重新检查
        try:
            del tokens[check_token]
            request.session["topology_check_tokens"] = tokens
            request.session.modified = True
        except Exception:
            pass
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)
    finally:
        if orig is not None:
            MANAGER.table[table_id] = orig

    return JsonResponse({
        "ok": True,
        "result": rep,
        "tmp_dir": tmp_root if keep_tmp else None,
    })


@csrf_exempt
def topology_save_view(request):
    """将修复后的路网持久化保存为新表，供用户重新加载验证。"""
    if not request.session.get("user"):
        return JsonResponse({"ok": False, "error": "未登录"}, status=401)
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "仅支持 POST"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    source_table = (payload.get("source_table") or payload.get("table") or "").strip()
    output_table = (payload.get("output_table") or "").strip()
    if not source_table:
        return JsonResponse({"ok": False, "error": "缺少 source_table"}, status=400)
    if not output_table:
        return JsonResponse({"ok": False, "error": "缺少 output_table"}, status=400)
    user = _current_user(request)
    try:
        _so, _st, source_id, source_dir = _canonical_table(user, source_table, must_exist=True)
        _ensure_table_registered_id(source_id, source_dir)
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)
    try:
        out_owner, out_table_name, out_id, out_dir = _canonical_table(user, output_table, must_exist=False)
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)
    if (out_dir / "data_table.pkl").exists():
        return JsonResponse({"ok": False, "error": f"表 {output_table} 已存在"}, status=400)

    # 仍然使用临时备份读取源表，确保不会污染原始表目录
    tmp_root = None
    orig = None
    try:
        tmp_root, orig = _with_temp_backup(source_id, source_dir)
        fc = _build_repaired_featurecollection(source_id)
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)
    finally:
        if orig is not None:
            MANAGER.table[source_id] = orig

    # 写入项目内临时文件，然后 ingest 为新表（落盘到 space_app/table_data/output_table）
    tmp_build = _project_tmp_dir("topology", "build")
    tmp_path = tmp_build / f"{out_id.replace(':', '_')}.geojson"
    tmp_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")

    try:
        try:
            user_data_dir(out_owner).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        MANAGER.ingest_geojson(out_id, tmp_path, table_dir=out_dir)
    except Exception as exc:
        try:
            shutil.rmtree(out_dir)
        except Exception:
            pass
        return JsonResponse({"ok": False, "error": f"保存失败：{exc}"}, status=400)

    _ensure_table_registered_id(out_id, out_dir)
    dt = MANAGER.classes['Table'].load(out_id)
    display = make_table_id(out_owner, out_table_name) if _is_admin(user) else out_table_name
    return JsonResponse({
        "ok": True,
        "table": display,
        "columns": list(dt.attributes.keys()),
        "message": f"已保存修复结果为新表：{display}",
    })
