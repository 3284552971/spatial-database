import re
from typing import Any, Dict, List, Optional, Union

from .select import SelectNode, ColumnNode, Condition_where, Order_Node, Limit


TABLE_REF = r"[\w:-]+(?:\.[\w:-]+)?"


def _upper_outside_quotes(sql: str) -> str:
	out: List[str] = []
	in_single = False
	in_double = False
	for ch in sql:
		if ch == "'" and not in_double:
			in_single = not in_single
			out.append(ch)
			continue
		if ch == '"' and not in_single:
			in_double = not in_double
			out.append(ch)
			continue
		out.append(ch if (in_single or in_double) else ch.upper())
	return "".join(out)


def _split_commas_outside(s: str) -> List[str]:
	parts: List[str] = []
	buf: List[str] = []
	depth = 0
	in_single = False
	in_double = False
	for ch in s:
		if ch == "'" and not in_double:
			in_single = not in_single
		elif ch == '"' and not in_single:
			in_double = not in_double
		elif not in_single and not in_double:
			if ch == '(':
				depth += 1
			elif ch == ')':
				depth = max(0, depth - 1)
			elif ch == ',' and depth == 0:
				part = "".join(buf).strip()
				if part:
					parts.append(part)
				buf = []
				continue
		buf.append(ch)
	tail = "".join(buf).strip()
	if tail:
		parts.append(tail)
	return parts


def _parse_single_condition(expr: str) -> Condition_where:
	expr = expr.strip()
	for op in [">=", "<=", "!=", "=", ">", "<"]:
		if op in expr:
			left, right = expr.split(op, 1)
			return Condition_where(left.strip(), right.strip(), op)
	raise ValueError(f"无法解析 WHERE 条件: {expr}")


def _parse_where_clause(where_text: str) -> Condition_where:
	text = where_text.strip()
	m = re.search(r"\s+(AND|OR)\s+", text, flags=re.IGNORECASE)
	if m:
		op = str(m.group(1)).upper()
		left = text[: m.start()].strip()
		right = text[m.end() :].strip()
		return Condition_where(_parse_single_condition(left), _parse_single_condition(right), op)
	return _parse_single_condition(text)


def parse_select(sql: str) -> SelectNode:
	# 不要把整句 SQL upper()：否则表名/列名会被强制大写，导致磁盘表目录大小写混乱。
	# 依赖 re.IGNORECASE 匹配关键字即可。
	text = sql.strip().rstrip(";")
	pattern = rf"SELECT\s+(.+?)\s+FROM\s+({TABLE_REF})(?:\s+WHERE\s+(.+?))?(?:\s+ORDER\s+BY\s+(.+?))?(?:\s+LIMIT\s+(\d+))?$"
	match = re.match(pattern, text, re.IGNORECASE)
	if not match:
		raise ValueError("无效的 SELECT 语句")
	columns_name, table_name, where_clause, order_by, limit = match.groups()

	node = SelectNode()
	node.from_table = table_name
	node.columns = ColumnNode([c.strip() for c in columns_name.split(",")])
	if where_clause:
		node.where = _parse_where_clause(where_clause)
	if order_by:
		ob = order_by.strip().split()
		col = ob[0]
		direction = ob[1] if len(ob) > 1 else "ASC"
		node.order_by = Order_Node(col, direction)
	if limit:
		node.limit = Limit(limit)
	return node


def parse_insert(sql: str) -> Dict[str, Any]:
	text = sql.strip().rstrip(";")
	m1 = re.match(rf"INSERT\s+INTO\s+({TABLE_REF})\s+VALUES\s*\((.+)\)$", text, re.IGNORECASE)
	m2 = re.match(rf"INSERT\s+INTO\s+({TABLE_REF})\s*\((.+)\)\s+VALUES\s*\((.+)\)$", text, re.IGNORECASE)
	if m2:
		table_name, cols_str, vals_str = m2.groups()
		cols = [c.strip() for c in _split_commas_outside(cols_str)]
		vals = [v.strip() for v in _split_commas_outside(vals_str)]
		return {"type": "INSERT", "table": table_name, "columns": cols, "values": vals}
	if m1:
		table_name, vals_str = m1.groups()
		vals = [v.strip() for v in _split_commas_outside(vals_str)]
		return {"type": "INSERT", "table": table_name, "columns": None, "values": vals}
	raise ValueError("无效的 INSERT 语句")


def parse_update(sql: str) -> Dict[str, Any]:
	text = sql.strip().rstrip(";")
	m = re.match(rf"UPDATE\s+({TABLE_REF})\s+SET\s+(.+?)(?:\s+WHERE\s+(.+))?$", text, re.IGNORECASE)
	if not m:
		raise ValueError("无效的 UPDATE 语句")
	table_name, set_clause, where_clause = m.groups()

	set_pairs = _split_commas_outside(set_clause)
	set_map: Dict[str, str] = {}
	for pair in set_pairs:
		if "=" not in pair:
			continue
		k, v = pair.split("=", 1)
		set_map[k.strip()] = v.strip()

	where_ast = _parse_where_clause(where_clause) if where_clause else None
	return {"type": "UPDATE", "table": table_name, "set": set_map, "where": where_ast}


def parse_delete(sql: str) -> Dict[str, Any]:
	text = sql.strip().rstrip(";")
	# 兼容：DELETE TABLE owner.table / owner:table（等价 DROP TABLE）
	m_drop = re.match(rf"DELETE\s+TABLE\s+({TABLE_REF})(?:\s+IF\s+EXISTS)?$", text, re.IGNORECASE)
	if m_drop:
		return {"type": "DROP", "table": m_drop.group(1), "if_exists": bool(re.search(r"\bIF\s+EXISTS\b", text, re.IGNORECASE))}

	m = re.match(rf"DELETE\s+FROM\s+({TABLE_REF})(?:\s+WHERE\s+(.+))?$", text, re.IGNORECASE)
	if not m:
		raise ValueError("无效的 DELETE 语句")
	table_name, where_clause = m.groups()
	if where_clause:
		return {"type": "DELETE", "table": table_name, "where": _parse_where_clause(where_clause)}
	return {"type": "DELETE", "table": table_name, "where": None}


def parse_create(sql: str) -> Dict[str, Any]:
	text = sql.strip().rstrip(";")
	m = re.match(rf"CREATE\s+TABLE\s+({TABLE_REF})\s*\((.*)\)$", text, re.IGNORECASE)
	if not m:
		raise ValueError("无效的 CREATE TABLE 语句")

	table_name, body = m.groups()
	items = _split_commas_outside(body)

	columns: List[str] = []
	primary_key: List[str] = []

	for item in items:
		t = item.strip()
		if not t:
			continue
		if t.upper().startswith("PRIMARY KEY"):
			m_pk = re.search(r"PRIMARY\s+KEY\s*\((.+)\)", t, re.IGNORECASE)
			if m_pk:
				pk_cols = [c.strip() for c in _split_commas_outside(m_pk.group(1))]
				primary_key.extend(pk_cols)
			continue

		parts = t.split()
		if not parts:
			continue
		col_name = parts[0].strip()
		columns.append(col_name)
		if re.search(r"\bPRIMARY\s+KEY\b", t, re.IGNORECASE):
			primary_key.append(col_name)

	if not primary_key and columns:
		primary_key = [columns[0]]

	return {"type": "CREATE", "table": table_name, "columns": columns, "primary_key": primary_key}

def parse_drop(sql: str) -> Dict[str, Any]:
	text = sql.strip().rstrip(";")
	m = re.match(rf"DROP\s+TABLE\s+(IF\s+EXISTS\s+)?({TABLE_REF})$", text, re.IGNORECASE)
	if not m:
		raise ValueError("无效的 DROP TABLE 语句")
	if_exists = bool(m.group(1))
	table_name = m.group(2)
	return {"type": "DROP", "table": table_name, "if_exists": if_exists}


def parse(sql: str) -> Union[SelectNode, Dict[str, Any]]:
	if not sql or not sql.strip():
		raise ValueError("SQL 不能为空")
	head = sql.strip().split(None, 1)[0].upper()
	if head == "SELECT":
		return parse_select(sql)
	if head == "INSERT":
		return parse_insert(sql)
	if head == "UPDATE":
		return parse_update(sql)
	if head == "DELETE":
		return parse_delete(sql)
	if head == "CREATE":
		return parse_create(sql)
	if head == "DROP":
		return parse_drop(sql)
	raise ValueError(f"暂不支持的 SQL 类型: {head}")
