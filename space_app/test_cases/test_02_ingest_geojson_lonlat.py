"""test_02_ingest_geojson_lonlat.py

目的
- 验证“重新入库（ingest_geojson）”时，新建的表 schema 会包含 `lon/lat`。
- 同时抽样检查新表中 `lon/lat` 的值类型合理。

运行
- 在项目根目录执行：
  python space_app/test_cases/test_02_ingest_geojson_lonlat.py

注意
- 该脚本会创建一个临时表（默认名：`__tmp_ingest_lonlat__`），
  并在结束后删除其目录，避免污染你的 `space_app/table_data/`。
- 选择的 GeoJSON 输入文件：优先使用 `space_app/shenzhen/road_sz.geojson`，
  如果你换了数据集，可按需修改 `GEOJSON_CANDIDATES`。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from space_app.models.manager import MANAGER, data_path


TMP_TABLE = "__tmp_ingest_lonlat__"

# 你可以按需追加候选数据源。
GEOJSON_CANDIDATES = [
    Path(__file__).resolve().parents[1] / "shenzhen" / "road_sz.geojson",
    Path(__file__).resolve().parents[1] / "shenzhen" / "transport_sz.geojson",
    Path(__file__).resolve().parents[1] / "shenzhen" / "pois_sz.geojson",
]


def _pick_geojson() -> Path:
    for p in GEOJSON_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到可用 GeoJSON：{GEOJSON_CANDIDATES}")


def _as_float_or_none(value: Any) -> float | None:
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
    geojson_path = _pick_geojson()
    print(f"Using geojson: {geojson_path}")

    # 1) 清理可能存在的旧临时表
    tmp_dir = data_path / TMP_TABLE
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    # 2) 入库
    MANAGER.ingest_geojson(TMP_TABLE, geojson_path)

    # 3) 验证新表 schema
    MANAGER.table[TMP_TABLE] = tmp_dir / "data_table.pkl"
    dt = MANAGER.classes["Table"].load(TMP_TABLE)

    cols = list(dt.attributes.keys())
    assert "lon" in dt.attributes and "lat" in dt.attributes, f"入库后仍缺少 lon/lat，当前列：{cols}"

    # 4) 抽样检查
    sample_n = min(20, len(dt))
    non_null = 0
    for i in range(sample_n):
        row = dt[i]
        row_dict = {col: row[idx] for col, idx in dt.attributes.items()}
        lon = _as_float_or_none(row_dict.get("lon"))
        lat = _as_float_or_none(row_dict.get("lat"))
        if lon is not None and lat is not None:
            non_null += 1
            assert abs(lon) <= 180.0, f"第 {i} 行 lon 异常：{lon}"
            assert abs(lat) <= 90.0, f"第 {i} 行 lat 异常：{lat}"

    print(f"[OK] ingest schema contains lon/lat. sample_non_null={non_null}/{sample_n} cols={cols}")

    # 5) 清理临时表（很重要，避免污染你的真实数据）
    shutil.rmtree(tmp_dir)
    print(f"[CLEANED] removed {tmp_dir}")


if __name__ == "__main__":
    main()
