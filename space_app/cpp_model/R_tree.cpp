#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <variant>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <functional>
#include <nlohmann/json.hpp>

#include "geom_common.h"

int ID = 0;
struct point {
    std::string type = "point";
    double x;
    double y;
    int id = ID++;
};
struct line {
    std::string type = "polyline";
    std::vector<point> points;
    int id = ID++;
};
struct polygon {
    std::string type = "polygon";
    std::vector<line> lines;
    int id = ID++;
};
struct layer{
    std::string type;
    std::string name;
    std::unordered_map<std::string, std::variant<int, double, std::string>> attributes; // variant 顺序和前面定义的一致
    int id = ID++;
};

using BBox = geom_common::BBox;

struct Feature {
    std::variant<point, line, polygon> geometry;
    std::unordered_map<std::string, std::string> attributes;
    int id = ID++;
    BBox bbox;
    std::string geometry_type() const {
        return std::visit(
            [](auto&& g) -> std::string {
                using T = std::decay_t<decltype(g)>;
                if constexpr (std::is_same_v<T, point>) return "point";
                if constexpr (std::is_same_v<T, line>) return "polyline";
                return "polygon";
            },
            geometry);
    }
};

struct R_tree_node{
    R_tree_node* parent = nullptr;
    R_tree_node* left = nullptr;
    R_tree_node* right = nullptr;
    double minx = std::numeric_limits<double>::infinity();
    double miny = std::numeric_limits<double>::infinity();
    double maxx = -std::numeric_limits<double>::infinity();
    double maxy = -std::numeric_limits<double>::infinity();
    std::vector<Feature> features;
};

class R_tree{
    public:
        inline R_tree() = default;
        int max_features_per_node = 4;
        R_tree_node root = R_tree_node();
        void insert_from_coordinates(const std::string& type, const std::vector<std::vector<double>>& coords, const std::unordered_map<std::string, std::string>& attributes = {});
        // Fast path for topology: insert a 2-point segment and assign a stable integer id.
        void insert_segment(double ax, double ay, double bx, double by, int sid);
        // 分裂
        void split(R_tree_node& node);
        void build_tree();
        void query_box(double minx, double miny, double maxx, double maxy, std::vector<Feature>& results, const std::string& type_filter = "");
        // Fast path for topology: query only ids (avoids copying Features).
        void query_box_ids(double minx, double miny, double maxx, double maxy, std::vector<int>& out_ids);
        void query_circle(double center_x, double center_y, double radius, std::vector<Feature>& results, const std::string& type_filter = "");
        bool delete_by_attribute(const std::string& key, const std::string& value);
        bool load_from_file(const std::string& path);
        bool save_serialized(const std::string& path) const;
        bool load_serialized(const std::string& path);
        static Feature make_feature(std::variant<point, line, polygon> geom, std::unordered_map<std::string, std::string> attributes);
    private:
        void insert(Feature feature);
        void add_point_feature(const std::vector<double>& coord, const std::unordered_map<std::string, std::string>& attributes);
        void add_linestring_feature(const std::vector<std::vector<double>>& coords, const std::unordered_map<std::string, std::string>& attributes);
        void add_polygon_feature(const std::vector<std::vector<std::vector<double>>>& rings, const std::unordered_map<std::string, std::string>& attributes);
        static void recompute_bbox(R_tree_node* node);
        static bool bbox_intersects_box(const R_tree_node* node, double minx, double miny, double maxx, double maxy);
        static bool feature_intersects_box(const Feature& f, double minx, double miny, double maxx, double maxy);
        static bool bbox_intersects_circle(const R_tree_node* node, double cx, double cy, double radius);
        static bool feature_intersects_circle(const Feature& f, double cx, double cy, double radius);
        
};

// ----------------- Helper utilities for bounding boxes -----------------
static void reset_bbox(R_tree_node& n) {
    n.minx = std::numeric_limits<double>::infinity();
    n.miny = std::numeric_limits<double>::infinity();
    n.maxx = -std::numeric_limits<double>::infinity();
    n.maxy = -std::numeric_limits<double>::infinity();
}

static double bbox_area(const BBox& b) {
    if (b.minx == std::numeric_limits<double>::infinity()) return 0.0;
    const double w = std::max(0.0, b.maxx - b.minx);
    const double h = std::max(0.0, b.maxy - b.miny);
    return w * h;
}

static BBox feature_bbox(const Feature& f) {
    BBox b = geom_common::bbox_empty();
    std::visit(
        [&](auto&& geom) {
            using T = std::decay_t<decltype(geom)>;
            if constexpr (std::is_same_v<T, point>) {
                geom_common::bbox_expand(b, geom_common::Pt{geom.x, geom.y});
            } else if constexpr (std::is_same_v<T, line>) {
                for (const auto& p : geom.points) {
                    geom_common::bbox_expand(b, geom_common::Pt{p.x, p.y});
                }
            } else if constexpr (std::is_same_v<T, polygon>) {
                for (const auto& ln : geom.lines) {
                    for (const auto& p : ln.points) {
                        geom_common::bbox_expand(b, geom_common::Pt{p.x, p.y});
                    }
                }
            }
        }, f.geometry);
    return b;
}

static double enlargement_needed(const BBox& box, const BBox& add) {
    BBox merged = box;
    geom_common::bbox_merge(merged, add);
    return bbox_area(merged) - bbox_area(box);
}

static nlohmann::json feature_to_json(const Feature& f) {
    nlohmann::json j;
    j["type"] = f.geometry_type();
    j["attributes"] = f.attributes;

    nlohmann::json coords;
    std::visit(
        [&](auto&& geom) {
            using T = std::decay_t<decltype(geom)>;
            if constexpr (std::is_same_v<T, point>) {
                coords = nlohmann::json::array();
                coords.push_back(geom.x);
                coords.push_back(geom.y);
            } else if constexpr (std::is_same_v<T, line>) {
                coords = nlohmann::json::array();
                for (const auto& p : geom.points) {
                    coords.push_back(nlohmann::json::array({p.x, p.y}));
                }
            } else if constexpr (std::is_same_v<T, polygon>) {
                // 还原一条环：取每条边的首点，最后补上最后一条边的末点
                coords = nlohmann::json::array();
                nlohmann::json ring = nlohmann::json::array();
                for (const auto& ln : geom.lines) {
                    if (!ln.points.empty()) {
                        ring.push_back(nlohmann::json::array({ln.points.front().x, ln.points.front().y}));
                    }
                }
                if (!geom.lines.empty() && !geom.lines.back().points.empty()) {
                    const auto& last_pt = geom.lines.back().points.back();
                    ring.push_back(nlohmann::json::array({last_pt.x, last_pt.y}));
                }
                if (!ring.empty()) {
                    coords.push_back(ring);
                }
            }
        },
        f.geometry);
    j["coordinates"] = coords;
    return j;
}

static Feature feature_from_json(const nlohmann::json& j) {
    std::string gtype = j.value("type", "");
    auto attrs = j.value("attributes", std::unordered_map<std::string, std::string>{});
    const auto& coords = j.at("coordinates");

    if (gtype == "point") {
        std::vector<double> pt = coords.get<std::vector<double>>();
        point p{.x = pt.size() > 0 ? pt[0] : 0.0, .y = pt.size() > 1 ? pt[1] : 0.0};
        return R_tree::make_feature(p, attrs);
    }
    if (gtype == "polyline") {
        std::vector<std::vector<double>> line_coords = coords.get<std::vector<std::vector<double>>>();
        line ln;
        for (const auto& c : line_coords) {
            if (c.size() < 2) continue;
            point p{.x = c[0], .y = c[1]};
            ln.points.push_back(p);
        }
        return R_tree::make_feature(ln, attrs);
    }
    if (gtype == "polygon") {
        std::vector<std::vector<std::vector<double>>> rings = coords.get<std::vector<std::vector<std::vector<double>>>>();
        polygon poly;
        if (!rings.empty()) {
            const auto& ring = rings.front();
            for (size_t i = 0; i + 1 < ring.size(); ++i) {
                if (ring[i].size() < 2 || ring[i + 1].size() < 2) continue;
                line seg;
                point a{.x = ring[i][0], .y = ring[i][1]};
                point b{.x = ring[i + 1][0], .y = ring[i + 1][1]};
                seg.points.push_back(a);
                seg.points.push_back(b);
                poly.lines.push_back(seg);
            }
            // 闭合边
            if (ring.size() >= 2) {
                line closing;
                point a{.x = ring.back()[0], .y = ring.back()[1]};
                point b{.x = ring.front()[0], .y = ring.front()[1]};
                closing.points.push_back(a);
                closing.points.push_back(b);
                poly.lines.push_back(closing);
            }
        }
        return R_tree::make_feature(poly, attrs);
    }
    // 兜底：返回空点
    point p{.x = 0.0, .y = 0.0};
    return R_tree::make_feature(p, attrs);
}
Feature R_tree::make_feature(std::variant<point, line, polygon> geom, std::unordered_map<std::string, std::string> attributes) {
    Feature f;
    f.geometry = std::move(geom);
    f.attributes = std::move(attributes);
    f.bbox = feature_bbox(f);
    return f;
}

void R_tree::insert_from_coordinates(const std::string& type, const std::vector<std::vector<double>>& coords, const std::unordered_map<std::string, std::string>& attributes){
    double current_minx = std::numeric_limits<double>::infinity();
    double current_miny = std::numeric_limits<double>::infinity();
    double current_maxx = -std::numeric_limits<double>::infinity();
    double current_maxy = -std::numeric_limits<double>::infinity();
    if(type == "point"){
        for(const auto& coord : coords){
            if(coord.size() >= 2){
                point p;
                p.x = coord[0];
                p.y = coord[1];
                insert(make_feature(p, attributes));
            break;
            }
        }
    } else if(type == "polyline"){
        line l;
        for(const auto& coord : coords){
            if(coord.size() >= 2){
                point p;
                p.x = coord[0];
                p.y = coord[1];
                l.points.push_back(p);
            }
        }
        insert(make_feature(l, attributes));
    } else if(type == "polygon"){
        polygon polyg;
        for(size_t i = 0; i < coords.size()-1; ++i){
            line l;
            if(coords[i].size() >= 2){
                point p;
                p.x = coords[i][0];
                p.y = coords[i][1];
                l.points.push_back(p);
                point p_next;
                p_next.x = coords[i+1][0];
                p_next.y = coords[i+1][1];
                l.points.push_back(p_next);
                polyg.lines.push_back(l);
            }
        }
        // 闭合多边形
        if(coords.size() >= 2){
            line closing_line;
            point first_point;
            first_point.x = coords[0][0];
            first_point.y = coords[0][1];
            point last_point;
            last_point.x = coords[coords.size()-1][0];
            last_point.y = coords[coords.size()-1][1];
            closing_line.points.push_back(last_point);
            closing_line.points.push_back(first_point);
            polyg.lines.push_back(closing_line);
        }
        insert(make_feature(polyg, attributes));
    }
}

void R_tree::insert_segment(double ax, double ay, double bx, double by, int sid) {
    line l;
    point pa;
    pa.x = ax;
    pa.y = ay;
    point pb;
    pb.x = bx;
    pb.y = by;
    l.points.push_back(pa);
    l.points.push_back(pb);

    Feature f = make_feature(l, {});
    f.id = sid;
    insert(std::move(f));
}

void R_tree::add_point_feature(const std::vector<double>& coord, const std::unordered_map<std::string, std::string>& attributes) {
    if (coord.size() < 2) return;
    point p;
    p.x = coord[0];
    p.y = coord[1];
    insert(make_feature(p, attributes));
}

void R_tree::add_linestring_feature(const std::vector<std::vector<double>>& coords, const std::unordered_map<std::string, std::string>& attributes) {
    if (coords.empty()) return;
    line l;
    for (const auto& c : coords) {
        if (c.size() < 2) continue;
        point p;
        p.x = c[0];
        p.y = c[1];
        l.points.push_back(p);
    }
    if (!l.points.empty()) {
        insert(make_feature(l, attributes));
    }
}

void R_tree::add_polygon_feature(const std::vector<std::vector<std::vector<double>>>& rings, const std::unordered_map<std::string, std::string>& attributes) {
    if (rings.empty()) return;
    polygon poly;
    for (const auto& ring : rings) {
        if (ring.size() < 2) continue;
        line edge_chain;
        for (size_t i = 0; i + 1 < ring.size(); ++i) {
            if (ring[i].size() < 2 || ring[i + 1].size() < 2) continue;
            line seg;
            point a{.x = ring[i][0], .y = ring[i][1]};
            point b{.x = ring[i + 1][0], .y = ring[i + 1][1]};
            seg.points.push_back(a);
            seg.points.push_back(b);
            poly.lines.push_back(seg);
        }
        // 闭合环
        if (ring.front().size() >= 2 && ring.back().size() >= 2 &&
            (ring.front()[0] != ring.back()[0] || ring.front()[1] != ring.back()[1])) {
            line closing;
            point a{.x = ring.back()[0], .y = ring.back()[1]};
            point b{.x = ring.front()[0], .y = ring.front()[1]};
            closing.points.push_back(a);
            closing.points.push_back(b);
            poly.lines.push_back(closing);
        }
    }
    if (!poly.lines.empty()) {
        insert(make_feature(poly, attributes));
    }
}

void R_tree::split(R_tree_node& node) {
    const std::size_t n = node.features.size();
    if (n <= 1) {
        return;
    }

    // R-tree Linear Split（性价比最高）：
    // - O(n) 选种子：分别在 X/Y 轴计算“归一化分离度”，取更大的轴；
    // - 分配：最小面积扩张优先，平手按面积/数量；
    // - 最小填充：避免一边过空。

    struct SeedPick {
        int a = -1;
        int b = -1;
        double sep = 0.0;
    };

    const auto pick_seeds_axis = [&](bool use_x) -> SeedPick {
        double min_low = std::numeric_limits<double>::infinity();
        double max_high = -std::numeric_limits<double>::infinity();
        double max_low = -std::numeric_limits<double>::infinity();
        double min_high = std::numeric_limits<double>::infinity();
        int idx_max_low = -1;
        int idx_min_high = -1;

        for (std::size_t i = 0; i < n; ++i) {
            const auto& b = node.features[i].bbox;
            const double low = use_x ? b.minx : b.miny;
            const double high = use_x ? b.maxx : b.maxy;
            min_low = std::min(min_low, low);
            max_high = std::max(max_high, high);
            if (low > max_low) {
                max_low = low;
                idx_max_low = static_cast<int>(i);
            }
            if (high < min_high) {
                min_high = high;
                idx_min_high = static_cast<int>(i);
            }
        }

        SeedPick out;
        out.a = idx_max_low;
        out.b = idx_min_high;
        const double width = std::max(1e-12, max_high - min_low);
        out.sep = (max_low - min_high) / width;
        if (out.sep < 0.0) {
            out.sep = 0.0;
        }
        return out;
    };

    SeedPick sx = pick_seeds_axis(true);
    SeedPick sy = pick_seeds_axis(false);
    SeedPick s = (sy.sep > sx.sep) ? sy : sx;

    int seed_a = s.a;
    int seed_b = s.b;
    if (seed_a < 0 || seed_b < 0) {
        return;
    }
    if (seed_a == seed_b) {
        // 退化情况：选与 seed_a 中心最远的作为 seed_b。
        const auto& ba = node.features[static_cast<std::size_t>(seed_a)].bbox;
        const double acx = (ba.minx + ba.maxx) * 0.5;
        const double acy = (ba.miny + ba.maxy) * 0.5;
        double best_d2 = -1.0;
        int best_i = seed_a;
        for (std::size_t i = 0; i < n; ++i) {
            if (static_cast<int>(i) == seed_a) continue;
            const auto& bb = node.features[i].bbox;
            const double bcx = (bb.minx + bb.maxx) * 0.5;
            const double bcy = (bb.miny + bb.maxy) * 0.5;
            const double dx = bcx - acx;
            const double dy = bcy - acy;
            const double d2 = dx * dx + dy * dy;
            if (d2 > best_d2) {
                best_d2 = d2;
                best_i = static_cast<int>(i);
            }
        }
        seed_b = best_i;
        if (seed_a == seed_b) {
            return;
        }
    }

    auto* left = new R_tree_node();
    auto* right = new R_tree_node();
    left->parent = &node;
    right->parent = &node;
    reset_bbox(*left);
    reset_bbox(*right);

    auto left_bb = geom_common::bbox_empty();
    auto right_bb = geom_common::bbox_empty();

    const auto push_left = [&](const Feature& f) {
        left->features.push_back(f);
        geom_common::bbox_merge(left_bb, f.bbox);
        left->minx = left_bb.minx;
        left->miny = left_bb.miny;
        left->maxx = left_bb.maxx;
        left->maxy = left_bb.maxy;
    };
    const auto push_right = [&](const Feature& f) {
        right->features.push_back(f);
        geom_common::bbox_merge(right_bb, f.bbox);
        right->minx = right_bb.minx;
        right->miny = right_bb.miny;
        right->maxx = right_bb.maxx;
        right->maxy = right_bb.maxy;
    };

    push_left(node.features[static_cast<std::size_t>(seed_a)]);
    push_right(node.features[static_cast<std::size_t>(seed_b)]);

    const std::size_t M = static_cast<std::size_t>(max_features_per_node) + 1;
    const std::size_t min_fill = (M + 1) / 2; // ceil(M/2)

    // 分配剩余项（按原顺序扫描；n 很小，简单策略够用）
    for (std::size_t k = 0; k < n; ++k) {
        if (static_cast<int>(k) == seed_a || static_cast<int>(k) == seed_b) {
            continue;
        }
        const auto& f = node.features[k];

        const std::size_t remaining = (n - k); // 粗略上界（用于 min_fill 约束）

        if (left->features.size() + remaining <= min_fill) {
            push_left(f);
            continue;
        }
        if (right->features.size() + remaining <= min_fill) {
            push_right(f);
            continue;
        }

        BBox lbox{left_bb.minx, left_bb.miny, left_bb.maxx, left_bb.maxy};
        BBox rbox{right_bb.minx, right_bb.miny, right_bb.maxx, right_bb.maxy};
        const double el = enlargement_needed(lbox, f.bbox);
        const double er = enlargement_needed(rbox, f.bbox);

        if (el < er) {
            push_left(f);
        } else if (er < el) {
            push_right(f);
        } else {
            const double al = bbox_area(lbox);
            const double ar = bbox_area(rbox);
            if (al < ar) {
                push_left(f);
            } else if (ar < al) {
                push_right(f);
            } else {
                // 最后平手：放到更少的一边，避免极端偏斜
                if (left->features.size() <= right->features.size()) {
                    push_left(f);
                } else {
                    push_right(f);
                }
            }
        }
    }

    node.left = left;
    node.right = right;
    node.features.clear();
    node.minx = std::min(left->minx, right->minx);
    node.miny = std::min(left->miny, right->miny);
    node.maxx = std::max(left->maxx, right->maxx);
    node.maxy = std::max(left->maxy, right->maxy);
}

void R_tree::insert(Feature feature) {
    // 插入新要素到根节点（简化处理）
    R_tree_node* current_node = &root;
    while (current_node->left != nullptr && current_node->right != nullptr) {
        // 选择子节点进行插入
        BBox fb = feature.bbox;
        BBox left_box{current_node->left->minx, current_node->left->miny, current_node->left->maxx, current_node->left->maxy};
        BBox right_box{current_node->right->minx, current_node->right->miny, current_node->right->maxx, current_node->right->maxy};
        double enlarge_left = enlargement_needed(left_box, fb);
        double enlarge_right = enlargement_needed(right_box, fb);

        if (enlarge_left < enlarge_right) {
            current_node = current_node->left;
        } else if (enlarge_right < enlarge_left) {
            current_node = current_node->right;
        } else {
            // 相等时，选择当前面积更小的一边。
            if (bbox_area(left_box) <= bbox_area(right_box)) {
                current_node = current_node->left;
            } else {
                current_node = current_node->right;
            }
        }
    }
    current_node->features.push_back(feature);
    BBox fb = feature.bbox;
    current_node->minx = std::min(current_node->minx, fb.minx);
    current_node->miny = std::min(current_node->miny, fb.miny);
    current_node->maxx = std::max(current_node->maxx, fb.maxx);
    current_node->maxy = std::max(current_node->maxy, fb.maxy);

    if (current_node->features.size() > max_features_per_node) {
        split(*current_node);
    }

    // Important: keep ancestor bounding boxes correct for incremental inserts.
    // Without this, queries can miss almost everything once the tree has split.
    R_tree_node* n = current_node;
    while (n) {
        recompute_bbox(n);
        n = n->parent;
    }
}

bool R_tree::load_from_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }
    nlohmann::json doc;
    try {
        ifs >> doc;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse JSON: " << e.what() << std::endl;
        return false;
    }

    if (doc.is_object() && doc.value("type", "") == "FeatureCollection") {
        const auto& features = doc.at("features");
        if (!features.is_array()) {
            std::cerr << "GeoJSON features must be array" << std::endl;
            return false;
        }
        for (const auto& feature : features) {
            if (!feature.is_object()) continue;
            const auto& geom = feature.value("geometry", nlohmann::json::object());
            const std::string gtype = geom.value("type", "");
            std::unordered_map<std::string, std::string> attrs;
            if (feature.contains("properties") && feature.at("properties").is_object()) {
                for (auto it = feature.at("properties").begin(); it != feature.at("properties").end(); ++it) {
                    if (it.value().is_string()) {
                        attrs[it.key()] = it.value().get<std::string>();
                    } else if (it.value().is_number()) {
                        attrs[it.key()] = std::to_string(it.value().get<double>());
                    } else if (it.value().is_boolean()) {
                        attrs[it.key()] = it.value().get<bool>() ? "true" : "false";
                    }
                }
            }
            if (gtype == "Point") {
                add_point_feature(geom.value("coordinates", std::vector<double>{}), attrs);
            } else if (gtype == "LineString") {
                add_linestring_feature(geom.value("coordinates", std::vector<std::vector<double>>{}), attrs);
            } else if (gtype == "Polygon") {
                add_polygon_feature(geom.value("coordinates", std::vector<std::vector<std::vector<double>>>{}), attrs);
            }
        }
        return true;
    }

    // 简单 JSON 层格式：{"type":"point|polyline|polygon", "coordinates": ... }
    if (doc.is_object()) {
        const std::string type = doc.value("type", "");
        std::unordered_map<std::string, std::string> attrs;
        if (doc.contains("properties") && doc.at("properties").is_object()) {
            for (auto it = doc.at("properties").begin(); it != doc.at("properties").end(); ++it) {
                if (it.value().is_string()) {
                    attrs[it.key()] = it.value().get<std::string>();
                } else if (it.value().is_number()) {
                    attrs[it.key()] = std::to_string(it.value().get<double>());
                } else if (it.value().is_boolean()) {
                    attrs[it.key()] = it.value().get<bool>() ? "true" : "false";
                }
            }
        }
        if (type == "point") {
            add_point_feature(doc.value("coordinates", std::vector<double>{}), attrs);
            return true;
        } else if (type == "polyline") {
            add_linestring_feature(doc.value("coordinates", std::vector<std::vector<double>>{}), attrs);
            return true;
        } else if (type == "polygon") {
            add_polygon_feature(doc.value("coordinates", std::vector<std::vector<std::vector<double>>>{}), attrs);
            return true;
        }
    }

    std::cerr << "Unsupported JSON/GeoJSON format in: " << path << std::endl;
    return false;
}

bool R_tree::save_serialized(const std::string& path) const {
    nlohmann::json doc;
    doc["type"] = "RTreeIndex";
    doc["features"] = nlohmann::json::array();

    std::vector<const R_tree_node*> stack;
    stack.push_back(&root);
    while (!stack.empty()) {
        const auto* node = stack.back();
        stack.pop_back();
        if (!node) continue;
        if (node->left || node->right) {
            stack.push_back(node->left);
            stack.push_back(node->right);
        } else {
            for (const auto& f : node->features) {
                doc["features"].push_back(feature_to_json(f));
            }
        }
    }

    std::ofstream ofs(path);
    if (!ofs) {
        return false;
    }
    ofs << doc.dump();
    return true;
}

bool R_tree::load_serialized(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    nlohmann::json doc;
    try {
        ifs >> doc;
    } catch (...) {
        return false;
    }
    if (!doc.is_object() || doc.value("type", "") != "RTreeIndex") return false;
    const auto& feats = doc.at("features");
    if (!feats.is_array()) return false;

    // 清空树
    root = R_tree_node();

    for (const auto& jf : feats) {
        try {
            Feature f = feature_from_json(jf);
            insert(f);
        } catch (...) {
            continue;
        }
    }
    return true;
}

bool R_tree::bbox_intersects_box(const R_tree_node* node, double minx, double miny, double maxx, double maxy) {
    if (!node) return false;
    return !(node->maxx < minx || node->maxy < miny || node->minx > maxx || node->miny > maxy);
}

bool R_tree::feature_intersects_box(const Feature& f, double minx, double miny, double maxx, double maxy) {
    const auto& b = f.bbox;
    return !(b.maxx < minx || b.maxy < miny || b.minx > maxx || b.miny > maxy);
}

bool R_tree::bbox_intersects_circle(const R_tree_node* node, double cx, double cy, double radius) {
    if (!node) return false;
    double nearest_x = std::max(node->minx, std::min(cx, node->maxx));
    double nearest_y = std::max(node->miny, std::min(cy, node->maxy));
    double dx = cx - nearest_x;
    double dy = cy - nearest_y;
    return (dx * dx + dy * dy) <= radius * radius;
}

bool R_tree::feature_intersects_circle(const Feature& f, double cx, double cy, double radius) {
    const auto& b = f.bbox;
    double nearest_x = std::max(b.minx, std::min(cx, b.maxx));
    double nearest_y = std::max(b.miny, std::min(cy, b.maxy));
    double dx = cx - nearest_x;
    double dy = cy - nearest_y;
    return (dx * dx + dy * dy) <= radius * radius;
}

void R_tree::recompute_bbox(R_tree_node* node) {
    if (!node) return;
    reset_bbox(*node);
    if (node->left || node->right) {
        if (node->left) {
            node->minx = std::min(node->minx, node->left->minx);
            node->miny = std::min(node->miny, node->left->miny);
            node->maxx = std::max(node->maxx, node->left->maxx);
            node->maxy = std::max(node->maxy, node->left->maxy);
        }
        if (node->right) {
            node->minx = std::min(node->minx, node->right->minx);
            node->miny = std::min(node->miny, node->right->miny);
            node->maxx = std::max(node->maxx, node->right->maxx);
            node->maxy = std::max(node->maxy, node->right->maxy);
        }
    } else {
        for (const auto& f : node->features) {
            const auto& b = f.bbox;
            node->minx = std::min(node->minx, b.minx);
            node->miny = std::min(node->miny, b.miny);
            node->maxx = std::max(node->maxx, b.maxx);
            node->maxy = std::max(node->maxy, b.maxy);
        }
    }
}

void R_tree::query_box(double minx, double miny, double maxx, double maxy, std::vector<Feature>& results, const std::string& type_filter) {
    std::function<void(R_tree_node*)> dfs = [&](R_tree_node* node) {
        if (!node || !bbox_intersects_box(node, minx, miny, maxx, maxy)) return;
        if (node->left == nullptr && node->right == nullptr) {
            for (const auto& f : node->features) {
                if (!type_filter.empty() && f.geometry_type() != type_filter) continue;
                if (feature_intersects_box(f, minx, miny, maxx, maxy)) {
                    results.push_back(f);
                }
            }
        } else {
            dfs(node->left);
            dfs(node->right);
        }
    };
    dfs(&root);
}

void R_tree::query_box_ids(double minx, double miny, double maxx, double maxy, std::vector<int>& out_ids) {
    std::function<void(R_tree_node*)> dfs = [&](R_tree_node* node) {
        if (!node || !bbox_intersects_box(node, minx, miny, maxx, maxy)) return;
        if (node->left == nullptr && node->right == nullptr) {
            for (const auto& f : node->features) {
                if (feature_intersects_box(f, minx, miny, maxx, maxy)) {
                    out_ids.push_back(f.id);
                }
            }
        } else {
            dfs(node->left);
            dfs(node->right);
        }
    };
    dfs(&root);
}

void R_tree::query_circle(double center_x, double center_y, double radius, std::vector<Feature>& results, const std::string& type_filter) {
    std::function<void(R_tree_node*)> dfs = [&](R_tree_node* node) {
        if (!node || !bbox_intersects_circle(node, center_x, center_y, radius)) return;
        if (node->left == nullptr && node->right == nullptr) {
            for (const auto& f : node->features) {
                if (!type_filter.empty() && f.geometry_type() != type_filter) continue;
                if (feature_intersects_circle(f, center_x, center_y, radius)) {
                    results.push_back(f);
                }
            }
        } else {
            dfs(node->left);
            dfs(node->right);
        }
    };
    dfs(&root);
}

bool R_tree::delete_by_attribute(const std::string& key, const std::string& value) {
    std::function<bool(R_tree_node*)> erase_dfs = [&](R_tree_node* node) -> bool {
        if (!node) return false;
        if (node->left == nullptr && node->right == nullptr) {
            auto it = std::find_if(node->features.begin(), node->features.end(), [&](const Feature& f) {
                auto attr_it = f.attributes.find(key);
                return attr_it != f.attributes.end() && attr_it->second == value;
            });
            if (it != node->features.end()) {
                node->features.erase(it);
                recompute_bbox(node);
                return true;
            }
            return false;
        }
        if (erase_dfs(node->left)) {
            recompute_bbox(node);
            return true;
        }
        if (erase_dfs(node->right)) {
            recompute_bbox(node);
            return true;
        }
        return false;
    };
    return erase_dfs(&root);
}

void R_tree::build_tree() {
    // 当前实现为增量插入，未使用批量构建；保留占位以满足链接。
}
