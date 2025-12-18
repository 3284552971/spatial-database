#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>

// 使用别名简化书写
using json = nlohmann::json;

int main() {
    std::string json_path = "/Users/shuaige/日常/空间数据库原理/space-database/space_app/traj_val/boston-seaport/11012.geojson";

    // ✅ 1. 读取文件内容到字符串
    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::cerr << "❌ 错误：无法打开文件 '" << json_path << "'\n";
        return 1;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();

    // ✅ 2. 解析 JSON 字符串为 json 对象
    try {
        json j = json::parse(content);

        // ✅ 3. 验证是否为合法 GeoJSON（检查 type 字段）
        if (!j.contains("type") || j["type"] != "FeatureCollection") {
            std::cerr << "⚠️ 警告：这不是标准 FeatureCollection GeoJSON（type = " 
                      << j.value("type", "unknown") << "）\n";
        }

        // ✅ 4. 获取 features 数组（GeoJSON 核心：所有地理要素在此）
        if (j.contains("features") && j["features"].is_array()) {
            size_t feature_count = j["features"].size();
            std::cout << "✅ 成功加载 " << feature_count << " 个要素（features）\n";

            // 🔍 示例：遍历前 2 个 feature，打印其类型和部分属性
            for (size_t i = 0; i < std::min(feature_count, size_t(2)); ++i) {
                const auto& feat = j["features"][i];

                // 获取 geometry.type（如 "Point", "LineString"）
                std::string geom_type = feat.value("geometry.type", "unknown");

                // 获取 properties（自定义属性，常见于轨迹数据）
                if (feat.contains("properties") && feat["properties"].is_object()) {
                    const auto& props = feat["properties"];
                    std::string id = props.value("id", "N/A");
                    std::string name = props.value("name", "N/A");
                    std::cout << "  [Feature " << i+1 << "] id=" << id 
                              << ", name=" << name << ", geom=" << geom_type << "\n";
                } else {
                    std::cout << "  [Feature " << i+1 << "] (无 properties)\n";
                }

                // 🧭 示例：读取 LineString 坐标（假设是轨迹）
                if (geom_type == "LineString" && feat.contains("geometry") &&
                    feat["geometry"].contains("coordinates") &&
                    feat["geometry"]["coordinates"].is_array()) {

                    const auto& coords = feat["geometry"]["coordinates"];
                    std::cout << "    → 坐标点数: " << coords.size() << "\n";
                    if (coords.size() > 0) {
                        // GeoJSON 坐标格式：[lon, lat, alt?] → 注意是 [x,y] 即 [经度, 纬度]
                        double lon = coords[0][0].get<double>();
                        double lat = coords[0][1].get<double>();
                        std::cout << "    → 首点: [" << lon << ", " << lat << "]\n";
                    }
                }
            }
        } else {
            std::cerr << "❌ 错误：GeoJSON 中缺少 'features' 数组或格式错误\n";
            return 1;
        }

    }

    return 0;
}