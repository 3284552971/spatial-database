"""test_03_geojson_endpoint.py

目的
- 验证 Django 路由前缀与接口是否正确：
  - 登录：/space_app/login/
  - GeoJSON：/space_app/geojson/<table>/

为什么需要这个测试
- 之前出现过前端请求 `/geojson/...` 导致 404 的问题。
- 现在前端已改为用 Django `{% url 'geojson' %}` 生成带前缀的 URL，本脚本用 Django Test Client 做回归。

运行
- 在项目根目录执行：
  python space_app/test_cases/test_03_geojson_endpoint.py

断言
- 登录成功后，请求 GeoJSON 接口返回 200。
- 响应 JSON 是 FeatureCollection，且 features 为 list。

备注
- 该脚本不会启动 runserver；它直接加载 Django settings 并走测试 client。
- session 已改为 signed cookies，不依赖 django_session 表。
"""

from __future__ import annotations

import os

import django
from django.test import Client


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")
    django.setup()

    c = Client()

    # 1) 登录
    resp = c.post("/space_app/login/", {"username": "admin", "password": "admin123"})

    # 登录成功通常是 302 重定向到 map；这里允许 302。
    assert resp.status_code in (200, 302), f"登录返回异常：{resp.status_code}"

    # 2) 请求 GeoJSON
    #    选择一个你肯定存在的表名（已落盘的三表之一）
    resp2 = c.get("/space_app/geojson/roads_sz/")
    assert resp2.status_code == 200, f"GeoJSON 接口异常：{resp2.status_code} content={resp2.content[:200]}"

    data = resp2.json()
    assert data.get("type") == "FeatureCollection", f"不是 FeatureCollection：{data.get('type')}"
    assert isinstance(data.get("features"), list), "features 不是 list"

    # 3) 再测带 lon/lat 参数的请求（即使几何不是点，也应该返回合法 GeoJSON）
    resp3 = c.get("/space_app/geojson/roads_sz/?lon=lon&lat=lat")
    assert resp3.status_code == 200, f"带参数的 GeoJSON 接口异常：{resp3.status_code}"
    data3 = resp3.json()
    assert data3.get("type") == "FeatureCollection"

    print("[OK] geojson endpoint works with /space_app prefix")


if __name__ == "__main__":
    main()
