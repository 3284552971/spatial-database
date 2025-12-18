from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("sql/", views.sql_view, name="sql"),
    path("docs/", views.docs_view, name="docs"),
    path("map/", views.map_view, name="map"),
    path("geojson/<str:table_name>/", views.geojson_view, name="geojson"),

    # 工具栏 API
    path("api/import_geojson/", views.import_geojson_view, name="import_geojson"),
    path("api/topology/check/", views.topology_check_view, name="topology_check"),
    path("api/topology/repair/", views.topology_repair_view, name="topology_repair"),
    path("api/topology/save/", views.topology_save_view, name="topology_save"),

    # 选择工具 API
    path("api/table/fields/", views.table_fields_view, name="table_fields"),
    path("api/select/attribute/", views.select_by_attribute_view, name="select_attribute"),
    path("api/select/location/", views.select_by_location_view, name="select_location"),

    # SQL（仅 SELECT，用于 map 工具栏）
    path("api/sql/select/", views.sql_select_api_view, name="sql_select_api"),

    # 导出工具 API
    path("api/export/", views.export_selected_view, name="export_selected"),
    path("api/export/download/<str:token>/", views.export_download_view, name="export_download"),

    # 最短路径
    path("api/route/shortest/", views.shortest_path_view, name="shortest_path"),

    # 轨迹校正
    path("api/trajectory/correct/", views.trajectory_correct_view, name="trajectory_correct"),
]
