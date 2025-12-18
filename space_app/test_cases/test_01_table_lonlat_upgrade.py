"""test_01_table_lonlat_upgrade.py

目的
- 验证“空间要素表”（包含 `geom_type/geometry/properties` 的表）是否都包含 `lon/lat` 字段。
- 这是你提出的核心需求："重新载入数据，让代表数据表的pkl文件拥有经纬度字段" 的回归检查。

运行
- 在项目根目录执行：
  python space_app/test_cases/test_01_table_lonlat_upgrade.py

断言
- 对空间要素表：attributes 中必须出现 `lon` 与 `lat`。
- 抽样若干行：
  - `lon/lat` 可以是 None（例如线/面要素或源数据缺失时）；
  - 但如果非 None，必须能转成 float。

备注
- 本脚本不启动 Django 服务；只直接调用仿真数据库的 Table.load()。
"""

from __future__ import annotations

from typing import Any

from space_app.models.manager import MANAGER, data_path


def _as_float_or_none(value: Any) -> float | None:
    """把值尽量转成 float；None 保持 None；否则抛错。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return float(value)
    raise TypeError(f"Unexpected lon/lat type: {type(value)}")


def main() -> None:
    # 1) 扫描 data_path 下所有已落盘的表
    table_dirs = [p for p in data_path.iterdir() if p.is_dir() and (p / "data_table.pkl").exists()]
    assert table_dirs, f"未找到任何落盘表：{data_path}"

    # 2) 逐表加载并检查 schema（仅空间要素表）
    for table_dir in sorted(table_dirs, key=lambda p: p.name):
        table_name = table_dir.name
        MANAGER.table[table_name] = table_dir / "data_table.pkl"

        dt = MANAGER.classes["Table"].load(table_name)
        cols = list(dt.attributes.keys())

        # SQL 创建的普通表不应强制有 lon/lat；仅检查空间要素表
        if not all(k in dt.attributes for k in ("geom_type", "geometry", "properties")):
            print(f"[SKIP] {table_name}: non-spatial table cols={cols}")
            continue

        assert "lon" in dt.attributes, f"表 {table_name} 缺少 lon 字段，当前列：{cols}"
        assert "lat" in dt.attributes, f"表 {table_name} 缺少 lat 字段，当前列：{cols}"

        # 3) 抽样检查若干行
        sample_n = min(10, len(dt))
        for i in range(sample_n):
            row = dt[i]
            row_dict = {col: row[idx] for col, idx in dt.attributes.items()}

            lon = _as_float_or_none(row_dict.get("lon"))
            lat = _as_float_or_none(row_dict.get("lat"))

            # 允许 None；但如果两个都不是 None，则必须是“有限”数值
            if lon is not None and lat is not None:
                assert abs(lon) <= 180.0, f"表 {table_name} 第 {i} 行 lon 异常：{lon}"
                assert abs(lat) <= 90.0, f"表 {table_name} 第 {i} 行 lat 异常：{lat}"

        print(f"[OK] {table_name}: cols={cols} rows={len(dt)}")


if __name__ == "__main__":
    main()
