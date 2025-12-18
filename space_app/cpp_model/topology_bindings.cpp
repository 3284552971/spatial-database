#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "src/topology_check.h"

namespace py = pybind11;

namespace {

using geom_common::BBox;
using geom_common::Pt;

static bool parse_bbox(const py::handle &bbox_obj, BBox &out) {
  if (bbox_obj.is_none()) {
    return false;
  }
  if (!py::isinstance<py::dict>(bbox_obj)) {
    return false;
  }
  auto d = bbox_obj.cast<py::dict>();
  if (!d.contains("minx") || !d.contains("miny") || !d.contains("maxx") || !d.contains("maxy")) {
    return false;
  }
  out.minx = py::float_(d["minx"]);
  out.miny = py::float_(d["miny"]);
  out.maxx = py::float_(d["maxx"]);
  out.maxy = py::float_(d["maxy"]);
  return true;
}

static std::vector<topology::FeatureLines> extract_lines(const py::list &features, const py::object &bbox_opt) {
  BBox clip{};
  const bool has_clip = parse_bbox(bbox_opt, clip);

  std::vector<topology::FeatureLines> out;
  out.reserve(features.size());

  for (auto item : features) {
    if (!py::isinstance<py::dict>(item)) {
      continue;
    }
    auto f = item.cast<py::dict>();
    if (!f.contains("geometry")) {
      continue;
    }
    auto geom = f["geometry"].cast<py::dict>();
    if (!geom.contains("type") || !geom.contains("coordinates")) {
      continue;
    }
    const std::string gtype = py::str(geom["type"]);
    if (gtype != "LineString" && gtype != "MultiLineString") {
      continue;
    }

    std::string fid;
    if (f.contains("id") && !f["id"].is_none()) {
      fid = py::str(f["id"]);
    } else if (f.contains("properties")) {
      auto props = f["properties"].cast<py::dict>();
      if (props.contains("id") && !props["id"].is_none()) {
        fid = py::str(props["id"]);
      }
    }
    if (fid.empty()) {
      fid = "unknown";
    }

    topology::FeatureLines fl;
    fl.fid = fid;

    if (gtype == "LineString") {
      std::vector<Pt> pts;
      for (auto xy : geom["coordinates"].cast<py::list>()) {
        auto pair = xy.cast<py::list>();
        if (pair.size() < 2) {
          continue;
        }
        pts.push_back(Pt{py::float_(pair[0]), py::float_(pair[1])});
      }
      if (pts.size() >= 2) {
        fl.parts.push_back(std::move(pts));
      }
    } else {
      // MultiLineString
      for (auto part : geom["coordinates"].cast<py::list>()) {
        std::vector<Pt> pts;
        for (auto xy : part.cast<py::list>()) {
          auto pair = xy.cast<py::list>();
          if (pair.size() < 2) {
            continue;
          }
          pts.push_back(Pt{py::float_(pair[0]), py::float_(pair[1])});
        }
        if (pts.size() >= 2) {
          fl.parts.push_back(std::move(pts));
        }
      }
    }

    if (fl.parts.empty()) {
      continue;
    }

    // bbox of feature
    auto bb = geom_common::bbox_empty();
    for (const auto &pts : fl.parts) {
      for (const auto &p : pts) {
        geom_common::bbox_expand(bb, p);
      }
    }
    fl.bb = bb;

    if (has_clip && !geom_common::bbox_intersects(fl.bb, clip)) {
      continue;
    }

    out.push_back(std::move(fl));
  }

  return out;
}

static double read_env_double(const char *key, double defv) {
  const char *v = std::getenv(key);
  if (!v) {
    return defv;
  }
  try {
    return std::stod(std::string(v));
  } catch (...) {
    return defv;
  }
}

static int read_env_int(const char *key, int defv) {
  const char *v = std::getenv(key);
  if (!v) {
    return defv;
  }
  try {
    return std::stoi(std::string(v));
  } catch (...) {
    return defv;
  }
}

static inline std::int64_t py_round_i64(double v) {
  // Deterministic rounding for keying: half-away-from-zero (matches llround).
  if (!std::isfinite(v)) {
    return 0;
  }
  return static_cast<std::int64_t>(std::llround(v));
}

static inline geom_common::Pt round_pt(const geom_common::Pt &p, double scale) {
  const auto ix = py_round_i64(p.x * scale);
  const auto iy = py_round_i64(p.y * scale);
  return geom_common::Pt{static_cast<double>(ix) / scale, static_cast<double>(iy) / scale};
}

static inline std::uint64_t round_key_py(const geom_common::Pt &p, double scale) {
  const auto ix = py_round_i64(p.x * scale);
  const auto iy = py_round_i64(p.y * scale);
  return geom_common::pack_key(ix, iy);
}

static inline double dist2_point_seg(const geom_common::Pt &p, const geom_common::Pt &a, const geom_common::Pt &b) {
  const double vx = b.x - a.x;
  const double vy = b.y - a.y;
  const double wx = p.x - a.x;
  const double wy = p.y - a.y;
  const double c1 = vx * wx + vy * wy;
  if (c1 <= 0.0) {
    const double dx = p.x - a.x;
    const double dy = p.y - a.y;
    return dx * dx + dy * dy;
  }
  const double c2 = vx * vx + vy * vy;
  if (c2 <= 0.0) {
    const double dx = p.x - a.x;
    const double dy = p.y - a.y;
    return dx * dx + dy * dy;
  }
  double t = c1 / c2;
  if (t >= 1.0) {
    const double dx = p.x - b.x;
    const double dy = p.y - b.y;
    return dx * dx + dy * dy;
  }
  const double px = a.x + t * vx;
  const double py = a.y + t * vy;
  const double dx = p.x - px;
  const double dy = p.y - py;
  return dx * dx + dy * dy;
}

static std::vector<topology::Issue> parse_issues_generic(const py::list &issues) {
  std::vector<topology::Issue> out;
  out.reserve(issues.size());
  for (auto item : issues) {
    topology::Issue iss;
    bool ok = false;
    if (py::isinstance<py::dict>(item)) {
      auto d = item.cast<py::dict>();
      if (d.contains("kind") && d.contains("point")) {
        iss.kind = py::str(d["kind"]);
        auto pt = d["point"].cast<py::tuple>();
        if (pt.size() >= 2) {
          iss.point = geom_common::Pt{py::float_(pt[0]), py::float_(pt[1])};
          iss.message = d.contains("message") ? std::string(py::str(d["message"])) : std::string();
          ok = true;
        }
      }
    } else {
      // TopologyIssue dataclass from Python side
      if (py::hasattr(item, "kind") && py::hasattr(item, "point")) {
        iss.kind = py::str(item.attr("kind"));
        auto pt = item.attr("point");
        if (py::isinstance<py::tuple>(pt) || py::isinstance<py::list>(pt)) {
          auto seq = pt.cast<py::sequence>();
          if (seq.size() >= 2) {
            iss.point = geom_common::Pt{py::float_(seq[0]), py::float_(seq[1])};
            iss.message = py::hasattr(item, "message") ? std::string(py::str(item.attr("message"))) : std::string();
            ok = true;
          }
        }
      }
    }
    if (ok) {
      out.push_back(std::move(iss));
    }
  }
  return out;
}

static py::dict repair_layer_cpp(const py::list &features, const py::list &issues_obj, double dangling_delete_threshold_m,
                                 int ndigits) {
  // 环境变量与 Python 侧保持一致
  int max_iters = read_env_int("SPACE_TOPOLOGY_REPAIR_MAX_ITERS", 100);
  if (max_iters < 1) {
    max_iters = 1;
  }
  if (max_iters > 100) {
    max_iters = 100;
  }
  int max_cross_per_line = read_env_int("SPACE_TOPOLOGY_MAX_CROSS_PER_LINE", 0);
  if (max_cross_per_line < 0) {
    max_cross_per_line = 0;
  }
  const double trim_m = std::max(0.0, read_env_double("SPACE_TOPOLOGY_DANGLING_TRIM_M", 50.0));
  const double connect_m = std::max(0.0, read_env_double("SPACE_TOPOLOGY_DANGLING_CONNECT_M", 50.0));

  std::int64_t scale_i = 1;
  if (ndigits < 0) {
    ndigits = 0;
  }
  if (ndigits > 9) {
    // keep integer scale in a safe range for 32-bit packing
    ndigits = 9;
  }
  for (int i = 0; i < ndigits; i++) {
    scale_i *= 10;
  }
  const double scale = static_cast<double>(scale_i);
  const double eps = std::max(1e-12, std::pow(10.0, -static_cast<double>(ndigits)) * 10.0);
  const double eps2 = eps * eps;

  constexpr double kEarthR = 6371000.0;

  auto haversine_m = [](const geom_common::Pt &a, const geom_common::Pt &b) -> double {
    const double lon1 = a.x;
    const double lat1 = a.y;
    const double lon2 = b.x;
    const double lat2 = b.y;
    const double r = kEarthR;
    const double phi1 = lat1 * M_PI / 180.0;
    const double phi2 = lat2 * M_PI / 180.0;
    const double dphi = (lat2 - lat1) * M_PI / 180.0;
    const double dl = (lon2 - lon1) * M_PI / 180.0;
    const double s1 = std::sin(dphi / 2.0);
    const double s2 = std::sin(dl / 2.0);
    const double h = s1 * s1 + std::cos(phi1) * std::cos(phi2) * s2 * s2;
    return 2.0 * r * std::atan2(std::sqrt(h), std::sqrt(std::max(0.0, 1.0 - h)));
  };

  auto polyline_length_m = [&](const std::vector<geom_common::Pt> &pts) -> double {
    double total = 0.0;
    for (std::size_t i = 1; i < pts.size(); i++) {
      total += haversine_m(pts[i - 1], pts[i]);
    }
    return total;
  };

  auto proj_xy = [&](const geom_common::Pt &p, double lat0_rad) -> geom_common::Pt {
    const double x = (p.x * M_PI / 180.0) * kEarthR * std::cos(lat0_rad);
    const double y = (p.y * M_PI / 180.0) * kEarthR;
    return geom_common::Pt{x, y};
  };

  auto unproj_xy = [&](const geom_common::Pt &p, double lat0_rad) -> geom_common::Pt {
    const double c = std::max(1e-12, std::cos(lat0_rad));
    const double lon = (p.x / (kEarthR * c)) * 180.0 / M_PI;
    const double lat = (p.y / kEarthR) * 180.0 / M_PI;
    return geom_common::Pt{lon, lat};
  };

  auto nearest_point_on_parts_m = [&](const geom_common::Pt &p, const std::vector<std::vector<geom_common::Pt>> &parts,
                                      geom_common::Pt &out_q) -> double {
    const double lat0 = p.y * M_PI / 180.0;
    const auto pp = proj_xy(p, lat0);
    double best_d2 = 1e100;
    bool has = false;
    geom_common::Pt best_xy{};

    for (const auto &pts0 : parts) {
      if (pts0.size() < 2) {
        continue;
      }
      // Dedup consecutive rounded points
      std::vector<geom_common::Pt> pts;
      pts.reserve(pts0.size());
      for (const auto &pt : pts0) {
        const auto pr = round_pt(pt, scale);
        if (pts.empty() || (pts.back().x != pr.x || pts.back().y != pr.y)) {
          pts.push_back(pr);
        }
      }
      if (pts.size() < 2) {
        continue;
      }

      for (std::size_t i = 1; i < pts.size(); i++) {
        const auto a = proj_xy(pts[i - 1], lat0);
        const auto b = proj_xy(pts[i], lat0);
        const double vx = b.x - a.x;
        const double vy = b.y - a.y;
        const double wx = pp.x - a.x;
        const double wy = pp.y - a.y;
        const double c2 = vx * vx + vy * vy;
        if (c2 <= 0.0) {
          continue;
        }
        double t = (vx * wx + vy * wy) / c2;
        if (t < 0.0) {
          t = 0.0;
        } else if (t > 1.0) {
          t = 1.0;
        }
        const double qx = a.x + t * vx;
        const double qy = a.y + t * vy;
        const double dx = pp.x - qx;
        const double dy = pp.y - qy;
        const double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
          best_d2 = d2;
          best_xy = geom_common::Pt{qx, qy};
          has = true;
        }
      }
    }

    if (!has) {
      return 1e100;
    }
    out_q = unproj_xy(best_xy, lat0);
    return std::sqrt(best_d2);
  };

  auto source_from_fid = [](const std::string &fid) -> std::string {
    const auto pos = fid.find("__split__");
    if (pos != std::string::npos) {
      return fid.substr(0, pos);
    }
    // connect__{source}__to__{target}__{k}
    if (fid.rfind("connect__", 0) == 0) {
      const auto pos_to = fid.find("__to__");
      if (pos_to != std::string::npos && pos_to > 9) {
        return fid.substr(9, pos_to - 9);
      }
    }
    return fid;
  };

  // 解析输入 features，并保留 original id 对象用于 deleted 回传
  std::unordered_map<std::string, py::object> fid_to_obj;
  fid_to_obj.reserve(features.size());
  for (auto item : features) {
    if (!py::isinstance<py::dict>(item)) {
      continue;
    }
    auto f = item.cast<py::dict>();
    py::object fid_obj = py::none();
    if (f.contains("id") && !f["id"].is_none()) {
      fid_obj = f["id"].cast<py::object>();
    } else if (f.contains("properties")) {
      auto props = f["properties"].cast<py::dict>();
      if (props.contains("id") && !props["id"].is_none()) {
        fid_obj = props["id"].cast<py::object>();
      }
    }
    std::string fid_str = fid_obj.is_none() ? std::string("unknown") : std::string(py::str(fid_obj));
    if (!fid_to_obj.count(fid_str)) {
      fid_to_obj.emplace(fid_str, fid_obj);
    }
  }

  auto current_lines = extract_lines(features, py::none());
  if (current_lines.empty()) {
    py::dict out;
    out["action"] = "noop";
    out["message"] = "图层中没有线要素";
    out["deleted"] = py::list();
    out["new_lines"] = py::list();
    return out;
  }

  std::unordered_set<std::string> original_fids;
  original_fids.reserve(current_lines.size());
  for (const auto &fl : current_lines) {
    original_fids.insert(fl.fid);
  }

  std::unordered_set<std::string> replaced_original;
  int iters_done = 0;
  int iters_report = 0;

  auto cur_issues = parse_issues_generic(issues_obj);
  if (cur_issues.empty()) {
    cur_issues = topology::check_topology_layer(current_lines, ndigits).issues;
  }

  auto has_cross_like = [](const std::vector<topology::Issue> &iss) {
    for (const auto &it : iss) {
      if (it.kind == "cross_without_node" || it.kind == "endpoint_on_segment") {
        return true;
      }
    }
    return false;
  };

  struct NodeInfo {
    geom_common::Pt repr;
  };

  for (int it = 0; it < max_iters; it++) {
    iters_done = it + 1;

    // node points: key -> repr point
    std::unordered_map<std::uint64_t, NodeInfo> nodes;
    nodes.reserve(cur_issues.size());
    for (const auto &iss : cur_issues) {
      if (iss.kind != "cross_without_node" && iss.kind != "endpoint_on_segment") {
        continue;
      }
      const auto pr = round_pt(iss.point, scale);
      const auto k = round_key_py(pr, scale);
      if (!nodes.count(k)) {
        // Use canonical rounded point for deterministic insertion/splitting.
        nodes.emplace(k, NodeInfo{pr});
      }
    }
    if (nodes.empty()) {
      break;
    }

    // layer bbox & grid over line bboxes
    geom_common::BBox layer_bb = geom_common::bbox_empty();
    bool bb_first = true;
    for (const auto &fl : current_lines) {
      if (bb_first) {
        layer_bb = fl.bb;
        bb_first = false;
      } else {
        geom_common::bbox_merge(layer_bb, fl.bb);
      }
    }
    const double cell = geom_common::grid_cell_size(layer_bb);

    auto cell_key = [](std::int64_t gx, std::int64_t gy) -> std::uint64_t {
      return geom_common::pack_key(gx, gy);
    };

    std::unordered_map<std::uint64_t, std::vector<std::size_t>> grid;
    grid.reserve(current_lines.size() * 2);
    for (std::size_t i = 0; i < current_lines.size(); i++) {
      const auto &bb = current_lines[i].bb;
      const auto x0 = geom_common::grid_i(bb.minx, cell);
      const auto x1 = geom_common::grid_i(bb.maxx, cell);
      const auto y0 = geom_common::grid_i(bb.miny, cell);
      const auto y1 = geom_common::grid_i(bb.maxy, cell);
      for (std::int64_t gx = x0; gx <= x1; gx++) {
        for (std::int64_t gy = y0; gy <= y1; gy++) {
          grid[cell_key(gx, gy)].push_back(i);
        }
      }
    }

    // endpoints per line (rounded)
    std::vector<std::unordered_set<std::uint64_t>> endpoints;
    endpoints.resize(current_lines.size());
    for (std::size_t i = 0; i < current_lines.size(); i++) {
      auto &eps = endpoints[i];
      for (const auto &pts : current_lines[i].parts) {
        if (pts.size() < 2) {
          continue;
        }
        eps.insert(round_key_py(round_pt(pts.front(), scale), scale));
        eps.insert(round_key_py(round_pt(pts.back(), scale), scale));
      }
    }

    // assign nodes to lines
    std::vector<std::vector<std::uint64_t>> line_nodes;
    line_nodes.resize(current_lines.size());

    for (const auto &kv : nodes) {
      const std::uint64_t nkey = kv.first;
      const auto pr = round_pt(kv.second.repr, scale);
      const auto gx = geom_common::grid_i(pr.x, cell);
      const auto gy = geom_common::grid_i(pr.y, cell);

      // Query 3x3 neighbor cells to avoid missing candidates near cell boundaries.
      // This improves robustness for lat/lon data and rounding-based keys.
      std::vector<std::size_t> cand;
      cand.reserve(128);
      for (std::int64_t dx = -1; dx <= 1; dx++) {
        for (std::int64_t dy = -1; dy <= 1; dy++) {
          auto itv = grid.find(cell_key(gx + dx, gy + dy));
          if (itv == grid.end()) {
            continue;
          }
          cand.insert(cand.end(), itv->second.begin(), itv->second.end());
        }
      }
      if (cand.empty()) {
        continue;
      }
      for (const auto li : cand) {
        const auto &fl = current_lines[li];
        if (pr.x < fl.bb.minx - eps || pr.x > fl.bb.maxx + eps || pr.y < fl.bb.miny - eps || pr.y > fl.bb.maxy + eps) {
          continue;
        }
        if (endpoints[li].find(nkey) != endpoints[li].end()) {
          continue;
        }

        bool hit = false;
        for (const auto &pts : fl.parts) {
          if (pts.size() < 2) {
            continue;
          }
          // internal vertex match
          for (std::size_t vi = 1; vi + 1 < pts.size(); vi++) {
            const auto vk = round_key_py(round_pt(pts[vi], scale), scale);
            if (vk == nkey) {
              hit = true;
              break;
            }
          }
          if (hit) {
            break;
          }

          // segment hit
          {
            // Python 会先对 round 后点做“连续去重”，避免出现零长度线段导致误判命中。
            std::vector<geom_common::Pt> rpts;
            rpts.reserve(pts.size());
            std::uint64_t lastk = 0;
            bool has_lastk = false;
            for (const auto &p0 : pts) {
              const auto rp = round_pt(p0, scale);
              const auto rk = round_key_py(rp, scale);
              if (!has_lastk || rk != lastk) {
                rpts.push_back(rp);
                lastk = rk;
                has_lastk = true;
              }
            }
            if (rpts.size() >= 2) {
              for (std::size_t i = 1; i < rpts.size(); i++) {
                const auto &a = rpts[i - 1];
                const auto &b = rpts[i];
                if (dist2_point_seg(pr, a, b) <= eps2) {
                  hit = true;
                  break;
                }
              }
            }
          }
          if (hit) {
            break;
          }
        }

        if (hit) {
          line_nodes[li].push_back(nkey);
        }
      }
    }

    std::unordered_set<std::string> replaced_iter;
    std::vector<topology::FeatureLines> new_iter_lines;
    new_iter_lines.reserve(current_lines.size() * 2);

    int split_idx = 0;

    for (std::size_t li = 0; li < current_lines.size(); li++) {
      auto &nks = line_nodes[li];
      if (nks.empty()) {
        continue;
      }
      // uniq
      std::sort(nks.begin(), nks.end());
      nks.erase(std::unique(nks.begin(), nks.end()), nks.end());
      if (max_cross_per_line > 0 && static_cast<int>(nks.size()) > max_cross_per_line) {
        nks.resize(static_cast<std::size_t>(max_cross_per_line));
      }

      const auto &fl = current_lines[li];
      bool produced_any = false;

      for (std::size_t part_idx = 0; part_idx < fl.parts.size(); part_idx++) {
        const auto &part = fl.parts[part_idx];
        if (part.size() < 2) {
          continue;
        }
        // vertex_set (internal)
        std::unordered_set<std::uint64_t> vertex_set;
        vertex_set.reserve(part.size());
        for (std::size_t vi = 1; vi + 1 < part.size(); vi++) {
          vertex_set.insert(round_key_py(round_pt(part[vi], scale), scale));
        }

        struct Ins {
          double t;
          std::uint64_t key;
        };
        std::vector<std::vector<Ins>> seg_points;
        if (part.size() >= 2) {
          seg_points.resize(part.size() - 1);
        }
        std::unordered_set<std::uint64_t> cut_set;

        const auto akey0 = round_key_py(round_pt(part.front(), scale), scale);
        const auto bkey0 = round_key_py(round_pt(part.back(), scale), scale);

        for (const auto nk : nks) {
          if (nk == akey0 || nk == bkey0) {
            continue;
          }
          if (vertex_set.find(nk) != vertex_set.end()) {
            cut_set.insert(nk);
            continue;
          }
          const auto rep = nodes[nk].repr;

          bool found = false;
          std::size_t best_i = 0;
          double best_d = 1e100;
          double best_t = 0.0;

          for (std::size_t i = 1; i < part.size(); i++) {
            const auto a = round_pt(part[i - 1], scale);
            const auto b = round_pt(part[i], scale);
            const double vx = b.x - a.x;
            const double vy = b.y - a.y;
            const double wx = rep.x - a.x;
            const double wy = rep.y - a.y;
            const double c2 = vx * vx + vy * vy;
            if (c2 <= 0.0) {
              continue;
            }
            const double t = (vx * wx + vy * wy) / c2;
            // Allow a tiny tolerance to avoid missing near-endpoint splits after rounding.
            constexpr double kT_Eps = 1e-12;
            if (t <= kT_Eps || t >= 1.0 - kT_Eps) {
              continue;
            }
            const double px = a.x + t * vx;
            const double py = a.y + t * vy;
            const double d2 = (rep.x - px) * (rep.x - px) + (rep.y - py) * (rep.y - py);
            if (d2 < best_d) {
              best_d = d2;
              best_i = i;
              best_t = t;
              found = true;
            }
          }

          if (!found || best_d > eps2) {
            continue;
          }
          seg_points[best_i - 1].push_back(Ins{best_t, nk});
          cut_set.insert(nk);
        }

        if (cut_set.empty()) {
          continue;
        }

        // build new_pts with inserts
        std::vector<geom_common::Pt> new_pts;
        new_pts.reserve(part.size() + cut_set.size());
        new_pts.push_back(part.front());
        auto last_key = round_key_py(round_pt(new_pts.back(), scale), scale);

        for (std::size_t seg_i = 1; seg_i < part.size(); seg_i++) {
          auto &ins = seg_points[seg_i - 1];
          if (!ins.empty()) {
            std::sort(ins.begin(), ins.end(), [](const Ins &a, const Ins &b) { return a.t < b.t; });
            std::uint64_t prev_k = 0;
            bool has_prev = false;
            for (const auto &it2 : ins) {
              if (has_prev && it2.key == prev_k) {
                continue;
              }
              has_prev = true;
              prev_k = it2.key;
              if (last_key != it2.key) {
                new_pts.push_back(nodes[it2.key].repr);
                last_key = it2.key;
              }
            }
          }
          const auto next_k = round_key_py(round_pt(part[seg_i], scale), scale);
          if (last_key != next_k) {
            new_pts.push_back(part[seg_i]);
            last_key = next_k;
          }
        }

        // split by cut_set
        std::vector<std::vector<geom_common::Pt>> segments;
        std::vector<geom_common::Pt> cur;
        for (const auto &p : new_pts) {
          if (cur.empty()) {
            cur.push_back(p);
            continue;
          }
          cur.push_back(p);
          const auto pk = round_key_py(round_pt(p, scale), scale);
          if (cut_set.find(pk) != cut_set.end() && cur.size() >= 2) {
            segments.push_back(cur);
            cur.clear();
            cur.push_back(p);
          }
        }
        if (cur.size() >= 2) {
          segments.push_back(cur);
        }

        for (std::size_t si = 0; si < segments.size(); si++) {
          split_idx += 1;
          produced_any = true;

          topology::FeatureLines nfl;
          nfl.fid = fl.fid + "__split__it" + std::to_string(it) + "__" + std::to_string(part_idx) + "__" +
                    std::to_string(static_cast<int>(si));
          nfl.parts.push_back(std::move(segments[si]));
          nfl.bb = geom_common::bbox_empty();
          for (const auto &p : nfl.parts[0]) {
            geom_common::bbox_expand(nfl.bb, p);
          }
          new_iter_lines.push_back(std::move(nfl));
        }
      }

      if (produced_any) {
        replaced_iter.insert(fl.fid);
        if (original_fids.find(fl.fid) != original_fids.end()) {
          replaced_original.insert(fl.fid);
        }
      }
    }

    if (replaced_iter.empty()) {
      break;
    }

    // update current_lines
    std::vector<topology::FeatureLines> next;
    next.reserve(current_lines.size() - replaced_iter.size() + new_iter_lines.size());
    for (auto &fl : current_lines) {
      if (replaced_iter.find(fl.fid) != replaced_iter.end()) {
        continue;
      }
      next.push_back(std::move(fl));
    }
    for (auto &nl : new_iter_lines) {
      next.push_back(std::move(nl));
    }
    current_lines = std::move(next);

    // re-check for next iter
    auto chk = topology::check_topology_layer(current_lines, ndigits);
    cur_issues = std::move(chk.issues);
    if (!has_cross_like(cur_issues)) {
      break;
    }
  }

  // Python 侧 counts['iters'] 只统计第一段 noding 闭包迭代次数（连接后再次 noding 不计入 iters）。
  iters_report = iters_done;

  // --- 悬挂点修复（与 Python 侧对齐：在交叉/端点落线段闭包后执行） ---
  int trimmed_spurs = 0;
  int connectors_added = 0;

  auto endpoint_degree = [&](const std::vector<topology::FeatureLines> &lines) {
    std::unordered_map<std::uint64_t, int> deg;
    deg.reserve(lines.size() * 4);
    for (const auto &fl : lines) {
      for (const auto &pts : fl.parts) {
        if (pts.size() < 2) {
          continue;
        }
        const auto akey = round_key_py(round_pt(pts.front(), scale), scale);
        const auto bkey = round_key_py(round_pt(pts.back(), scale), scale);
        deg[akey] += 1;
        deg[bkey] += 1;
      }
    }
    return deg;
  };

  auto rebuild_bbs = [&](std::vector<topology::FeatureLines> &lines) {
    for (auto &fl : lines) {
      auto bb = geom_common::bbox_empty();
      for (const auto &pts : fl.parts) {
        for (const auto &p : pts) {
          geom_common::bbox_expand(bb, p);
        }
      }
      fl.bb = bb;
    }
  };

  // 1) trim spurs (only non-original)
  if (trim_m > 0.0) {
    rebuild_bbs(current_lines);
    const auto deg = endpoint_degree(current_lines);
    std::unordered_set<std::string> spur_ids;
    for (const auto &fl : current_lines) {
      if (original_fids.find(fl.fid) != original_fids.end()) {
        continue;
      }
      for (const auto &pts : fl.parts) {
        if (pts.size() < 2) {
          continue;
        }
        const auto akey = round_key_py(round_pt(pts.front(), scale), scale);
        const auto bkey = round_key_py(round_pt(pts.back(), scale), scale);
        const int da = deg.count(akey) ? deg.at(akey) : 0;
        const int db = deg.count(bkey) ? deg.at(bkey) : 0;
        if (!((da <= 1 && db >= 2) || (db <= 1 && da >= 2))) {
          continue;
        }
        if (polyline_length_m(pts) <= trim_m) {
          spur_ids.insert(fl.fid);
          break;
        }
      }
    }
    if (!spur_ids.empty()) {
      trimmed_spurs += static_cast<int>(spur_ids.size());
      std::vector<topology::FeatureLines> next;
      next.reserve(current_lines.size() - spur_ids.size());
      for (auto &fl : current_lines) {
        if (spur_ids.find(fl.fid) != spur_ids.end()) {
          continue;
        }
        next.push_back(std::move(fl));
      }
      current_lines = std::move(next);
    }
  }

  // 2) connect dangling endpoints (<= connect_m)
  std::unordered_map<std::string, std::pair<std::string, std::string>> connector_meta;
  if (connect_m > 0.0) {
    rebuild_bbs(current_lines);
    const auto deg = endpoint_degree(current_lines);

    // layer bbox
    geom_common::BBox layer_bb = geom_common::bbox_empty();
    bool bb_first = true;
    for (const auto &fl : current_lines) {
      if (bb_first) {
        layer_bb = fl.bb;
        bb_first = false;
      } else {
        geom_common::bbox_merge(layer_bb, fl.bb);
      }
    }
    const double cell = geom_common::grid_cell_size(layer_bb);

    auto cell_key = [](std::int64_t gx, std::int64_t gy) -> std::uint64_t { return geom_common::pack_key(gx, gy); };

    std::unordered_map<std::uint64_t, std::vector<std::size_t>> grid;
    grid.reserve(current_lines.size() * 2);
    for (std::size_t i = 0; i < current_lines.size(); i++) {
      const auto &bb = current_lines[i].bb;
      const auto x0 = geom_common::grid_i(bb.minx, cell);
      const auto x1 = geom_common::grid_i(bb.maxx, cell);
      const auto y0 = geom_common::grid_i(bb.miny, cell);
      const auto y1 = geom_common::grid_i(bb.maxy, cell);
      for (std::int64_t gx = x0; gx <= x1; gx++) {
        for (std::int64_t gy = y0; gy <= y1; gy++) {
          grid[cell_key(gx, gy)].push_back(i);
        }
      }
    }

    std::vector<topology::FeatureLines> connectors;

    for (const auto &fl : current_lines) {
      for (const auto &pts : fl.parts) {
        if (pts.size() < 2) {
          continue;
        }
        const std::vector<geom_common::Pt> ends{pts.front(), pts.back()};
        for (const auto &ep0 : ends) {
          const auto ep = round_pt(ep0, scale);
          const auto epkey = round_key_py(ep, scale);
          const int d = deg.count(epkey) ? deg.at(epkey) : 0;
          if (d > 1) {
            continue;
          }

          // find nearest line (exclude same fid)
          const auto gx = geom_common::grid_i(ep.x, cell);
          const auto gy = geom_common::grid_i(ep.y, cell);
          std::vector<std::size_t> cand;
          cand.reserve(128);
          for (std::int64_t dx = -1; dx <= 1; dx++) {
            for (std::int64_t dy = -1; dy <= 1; dy++) {
              auto itv = grid.find(cell_key(gx + dx, gy + dy));
              if (itv == grid.end()) {
                continue;
              }
              cand.insert(cand.end(), itv->second.begin(), itv->second.end());
            }
          }

          double best_d = 1e100;
          geom_common::Pt best_q{};
          std::string best_target;
          bool has_best = false;

          for (const auto li : cand) {
            const auto &tfl = current_lines[li];
            if (tfl.fid == fl.fid) {
              continue;
            }
            // bbox coarse filter in degrees (~111m when 0.001deg at equator)
            constexpr double deg_expand = 0.001;
            if (ep.x < tfl.bb.minx - deg_expand || ep.x > tfl.bb.maxx + deg_expand || ep.y < tfl.bb.miny - deg_expand ||
                ep.y > tfl.bb.maxy + deg_expand) {
              continue;
            }
            geom_common::Pt q{};
            const double dist_m = nearest_point_on_parts_m(ep, tfl.parts, q);
            if (dist_m < best_d) {
              best_d = dist_m;
              best_q = q;
              best_target = tfl.fid;
              has_best = true;
            }
          }

          if (!has_best || best_d > connect_m) {
            continue;
          }

          const auto p1 = ep;
          const auto p2 = round_pt(best_q, scale);
          if (p1.x == p2.x && p1.y == p2.y) {
            continue;
          }

          connectors_added += 1;
          topology::FeatureLines cfl;
          cfl.fid = "connect__" + fl.fid + "__to__" + best_target + "__" + std::to_string(connectors_added);
          cfl.parts.push_back(std::vector<geom_common::Pt>{p1, p2});
          cfl.bb = geom_common::bbox_empty();
          geom_common::bbox_expand(cfl.bb, p1);
          geom_common::bbox_expand(cfl.bb, p2);
          connector_meta[cfl.fid] = {fl.fid, best_target};
          connectors.push_back(std::move(cfl));
        }
      }
    }

    if (!connectors.empty()) {
      // add connectors
      for (auto &cfl : connectors) {
        current_lines.push_back(std::move(cfl));
      }

      // noding closure again after connectors
      auto chk = topology::check_topology_layer(current_lines, ndigits);
      cur_issues = std::move(chk.issues);
      const int iter_offset = iters_done;
      for (int it2 = 0; it2 < max_iters; it2++) {
        const int iter_idx = iter_offset + it2;

        std::unordered_map<std::uint64_t, NodeInfo> nodes;
        nodes.reserve(cur_issues.size());
        for (const auto &iss : cur_issues) {
          if (iss.kind != "cross_without_node" && iss.kind != "endpoint_on_segment") {
            continue;
          }
          const auto pr = round_pt(iss.point, scale);
          const auto k = round_key_py(pr, scale);
          if (!nodes.count(k)) {
            // Use canonical rounded point for deterministic insertion/splitting.
            nodes.emplace(k, NodeInfo{pr});
          }
        }
        if (nodes.empty()) {
          break;
        }

        // build grid again
        rebuild_bbs(current_lines);
        geom_common::BBox layer_bb2 = geom_common::bbox_empty();
        bool first2 = true;
        for (const auto &fl2 : current_lines) {
          if (first2) {
            layer_bb2 = fl2.bb;
            first2 = false;
          } else {
            geom_common::bbox_merge(layer_bb2, fl2.bb);
          }
        }
        const double cell2 = geom_common::grid_cell_size(layer_bb2);

        std::unordered_map<std::uint64_t, std::vector<std::size_t>> grid2;
        grid2.reserve(current_lines.size() * 2);
        for (std::size_t i = 0; i < current_lines.size(); i++) {
          const auto &bb = current_lines[i].bb;
          const auto x0 = geom_common::grid_i(bb.minx, cell2);
          const auto x1 = geom_common::grid_i(bb.maxx, cell2);
          const auto y0 = geom_common::grid_i(bb.miny, cell2);
          const auto y1 = geom_common::grid_i(bb.maxy, cell2);
          for (std::int64_t gx = x0; gx <= x1; gx++) {
            for (std::int64_t gy = y0; gy <= y1; gy++) {
              grid2[cell_key(gx, gy)].push_back(i);
            }
          }
        }

        std::vector<std::unordered_set<std::uint64_t>> endpoints;
        endpoints.resize(current_lines.size());
        for (std::size_t i = 0; i < current_lines.size(); i++) {
          auto &eps = endpoints[i];
          for (const auto &pts : current_lines[i].parts) {
            if (pts.size() < 2) {
              continue;
            }
            eps.insert(round_key_py(round_pt(pts.front(), scale), scale));
            eps.insert(round_key_py(round_pt(pts.back(), scale), scale));
          }
        }

        std::vector<std::vector<std::uint64_t>> line_nodes;
        line_nodes.resize(current_lines.size());
        for (const auto &kv : nodes) {
          const std::uint64_t nkey = kv.first;
          const auto pr = round_pt(kv.second.repr, scale);
          const auto gx = geom_common::grid_i(pr.x, cell2);
          const auto gy = geom_common::grid_i(pr.y, cell2);

          std::vector<std::size_t> cand;
          cand.reserve(128);
          for (std::int64_t dx = -1; dx <= 1; dx++) {
            for (std::int64_t dy = -1; dy <= 1; dy++) {
              auto itv = grid2.find(cell_key(gx + dx, gy + dy));
              if (itv == grid2.end()) {
                continue;
              }
              cand.insert(cand.end(), itv->second.begin(), itv->second.end());
            }
          }
          if (cand.empty()) {
            continue;
          }

          for (const auto li : cand) {
            const auto &fl2 = current_lines[li];
            if (pr.x < fl2.bb.minx - eps || pr.x > fl2.bb.maxx + eps || pr.y < fl2.bb.miny - eps ||
                pr.y > fl2.bb.maxy + eps) {
              continue;
            }
            if (endpoints[li].find(nkey) != endpoints[li].end()) {
              continue;
            }

            bool hit = false;
            for (const auto &pts : fl2.parts) {
              if (pts.size() < 2) {
                continue;
              }

              // Match Python: use rounded points with consecutive de-dup for stable hit-testing.
              std::vector<geom_common::Pt> rpts;
              rpts.reserve(pts.size());
              for (const auto &p0 : pts) {
                const auto rp0 = round_pt(p0, scale);
                if (rpts.empty() || rpts.back().x != rp0.x || rpts.back().y != rp0.y) {
                  rpts.push_back(rp0);
                }
              }
              if (rpts.size() < 2) {
                continue;
              }

              for (std::size_t vi = 1; vi + 1 < rpts.size(); vi++) {
                const auto vk = round_key_py(rpts[vi], scale);
                if (vk == nkey) {
                  hit = true;
                  break;
                }
              }
              if (hit) {
                break;
              }
              for (std::size_t i = 1; i < rpts.size(); i++) {
                const auto &a = rpts[i - 1];
                const auto &b = rpts[i];
                if (dist2_point_seg(pr, a, b) <= eps2) {
                  hit = true;
                  break;
                }
              }
              if (hit) {
                break;
              }
            }
            if (hit) {
              line_nodes[li].push_back(nkey);
            }
          }
        }

        std::unordered_set<std::string> replaced_iter;
        std::vector<topology::FeatureLines> new_iter_lines;
        for (std::size_t li = 0; li < current_lines.size(); li++) {
          auto &nks = line_nodes[li];
          if (nks.empty()) {
            continue;
          }
          std::sort(nks.begin(), nks.end());
          nks.erase(std::unique(nks.begin(), nks.end()), nks.end());
          if (max_cross_per_line > 0 && static_cast<int>(nks.size()) > max_cross_per_line) {
            nks.resize(static_cast<std::size_t>(max_cross_per_line));
          }

          const auto &fl2 = current_lines[li];
          bool produced_any = false;
          for (std::size_t part_idx = 0; part_idx < fl2.parts.size(); part_idx++) {
            const auto &part = fl2.parts[part_idx];
            if (part.size() < 2) {
              continue;
            }
            std::unordered_set<std::uint64_t> vertex_set;
            for (std::size_t vi = 1; vi + 1 < part.size(); vi++) {
              vertex_set.insert(round_key_py(round_pt(part[vi], scale), scale));
            }
            struct Ins {
              double t;
              std::uint64_t key;
            };
            std::vector<std::vector<Ins>> seg_points;
            seg_points.resize(part.size() - 1);
            std::unordered_set<std::uint64_t> cut_set;

            const auto akey0 = round_key_py(round_pt(part.front(), scale), scale);
            const auto bkey0 = round_key_py(round_pt(part.back(), scale), scale);
            for (const auto nk : nks) {
              if (nk == akey0 || nk == bkey0) {
                continue;
              }
              if (vertex_set.find(nk) != vertex_set.end()) {
                cut_set.insert(nk);
                continue;
              }
              const auto rep = nodes[nk].repr;
              bool found = false;
              std::size_t best_i = 0;
              double best_d = 1e100;
              double best_t = 0.0;
              for (std::size_t i = 1; i < part.size(); i++) {
                const auto a = round_pt(part[i - 1], scale);
                const auto b = round_pt(part[i], scale);
                const double vx = b.x - a.x;
                const double vy = b.y - a.y;
                const double wx = rep.x - a.x;
                const double wy = rep.y - a.y;
                const double c2 = vx * vx + vy * vy;
                if (c2 <= 0.0) {
                  continue;
                }
                const double t = (vx * wx + vy * wy) / c2;
                // Allow a tiny tolerance to avoid missing near-endpoint splits after rounding.
                constexpr double kT_Eps = 1e-12;
                if (t <= kT_Eps || t >= 1.0 - kT_Eps) {
                  continue;
                }
                const double px = a.x + t * vx;
                const double py = a.y + t * vy;
                const double d2 = (rep.x - px) * (rep.x - px) + (rep.y - py) * (rep.y - py);
                if (d2 < best_d) {
                  best_d = d2;
                  best_i = i;
                  best_t = t;
                  found = true;
                }
              }
              if (!found || best_d > eps2) {
                continue;
              }
              seg_points[best_i - 1].push_back(Ins{best_t, nk});
              cut_set.insert(nk);
            }
            if (cut_set.empty()) {
              continue;
            }
            std::vector<geom_common::Pt> new_pts;
            new_pts.reserve(part.size() + cut_set.size());
            new_pts.push_back(part.front());
            auto last_key = round_key_py(round_pt(new_pts.back(), scale), scale);
            for (std::size_t seg_i = 1; seg_i < part.size(); seg_i++) {
              auto &ins = seg_points[seg_i - 1];
              if (!ins.empty()) {
                std::sort(ins.begin(), ins.end(), [](const Ins &a, const Ins &b) { return a.t < b.t; });
                std::uint64_t prev_k = 0;
                bool has_prev = false;
                for (const auto &it3 : ins) {
                  if (has_prev && it3.key == prev_k) {
                    continue;
                  }
                  has_prev = true;
                  prev_k = it3.key;
                  if (last_key != it3.key) {
                    new_pts.push_back(nodes[it3.key].repr);
                    last_key = it3.key;
                  }
                }
              }
              const auto next_k = round_key_py(round_pt(part[seg_i], scale), scale);
              if (last_key != next_k) {
                new_pts.push_back(part[seg_i]);
                last_key = next_k;
              }
            }
            std::vector<std::vector<geom_common::Pt>> segments;
            std::vector<geom_common::Pt> cur;
            for (const auto &p : new_pts) {
              if (cur.empty()) {
                cur.push_back(p);
                continue;
              }
              cur.push_back(p);
              const auto pk = round_key_py(round_pt(p, scale), scale);
              if (cut_set.find(pk) != cut_set.end() && cur.size() >= 2) {
                segments.push_back(cur);
                cur.clear();
                cur.push_back(p);
              }
            }
            if (cur.size() >= 2) {
              segments.push_back(cur);
            }
            for (std::size_t si = 0; si < segments.size(); si++) {
              topology::FeatureLines nfl;
              nfl.fid = fl2.fid + "__split__it" + std::to_string(iter_idx) + "__" + std::to_string(part_idx) + "__" +
                        std::to_string(static_cast<int>(si));
              nfl.parts.push_back(std::move(segments[si]));
              nfl.bb = geom_common::bbox_empty();
              for (const auto &p : nfl.parts[0]) {
                geom_common::bbox_expand(nfl.bb, p);
              }
              new_iter_lines.push_back(std::move(nfl));
              produced_any = true;
            }
          }
          if (produced_any) {
            replaced_iter.insert(fl2.fid);
            if (original_fids.find(fl2.fid) != original_fids.end()) {
              replaced_original.insert(fl2.fid);
            }
          }
        }
        if (replaced_iter.empty()) {
          break;
        }
        std::vector<topology::FeatureLines> next;
        next.reserve(current_lines.size() - replaced_iter.size() + new_iter_lines.size());
        for (auto &fl3 : current_lines) {
          if (replaced_iter.find(fl3.fid) != replaced_iter.end()) {
            continue;
          }
          next.push_back(std::move(fl3));
        }
        for (auto &nl : new_iter_lines) {
          next.push_back(std::move(nl));
        }
        current_lines = std::move(next);
        auto chk2 = topology::check_topology_layer(current_lines, ndigits);
        cur_issues = std::move(chk2.issues);
        if (!has_cross_like(cur_issues)) {
          break;
        }
      }

      // trim again after connectors+noding
      if (trim_m > 0.0) {
        rebuild_bbs(current_lines);
        const auto deg2 = endpoint_degree(current_lines);
        std::unordered_set<std::string> spur_ids;
        for (const auto &flx : current_lines) {
          if (original_fids.find(flx.fid) != original_fids.end()) {
            continue;
          }
          for (const auto &pts : flx.parts) {
            if (pts.size() < 2) {
              continue;
            }
            const auto akey = round_key_py(round_pt(pts.front(), scale), scale);
            const auto bkey = round_key_py(round_pt(pts.back(), scale), scale);
            const int da = deg2.count(akey) ? deg2.at(akey) : 0;
            const int db = deg2.count(bkey) ? deg2.at(bkey) : 0;
            if (!((da <= 1 && db >= 2) || (db <= 1 && da >= 2))) {
              continue;
            }
            if (polyline_length_m(pts) <= trim_m) {
              spur_ids.insert(flx.fid);
              break;
            }
          }
        }
        if (!spur_ids.empty()) {
          trimmed_spurs += static_cast<int>(spur_ids.size());
          std::vector<topology::FeatureLines> next;
          next.reserve(current_lines.size() - spur_ids.size());
          for (auto &flx : current_lines) {
            if (spur_ids.find(flx.fid) != spur_ids.end()) {
              continue;
            }
            next.push_back(std::move(flx));
          }
          current_lines = std::move(next);
        }
      }
    }
  }

  // 3) dangling delete threshold (original lines only, both ends dangling and short)
  std::unordered_set<std::string> dangling_deleted;
  if (dangling_delete_threshold_m > 0.0) {
    rebuild_bbs(current_lines);
    const auto deg = endpoint_degree(current_lines);
    for (const auto &fl : current_lines) {
      if (original_fids.find(fl.fid) == original_fids.end()) {
        continue;
      }
      for (const auto &pts : fl.parts) {
        if (pts.size() < 2) {
          continue;
        }
        const auto akey = round_key_py(round_pt(pts.front(), scale), scale);
        const auto bkey = round_key_py(round_pt(pts.back(), scale), scale);
        const int da = deg.count(akey) ? deg.at(akey) : 0;
        const int db = deg.count(bkey) ? deg.at(bkey) : 0;
        if (da <= 1 && db <= 1) {
          if (polyline_length_m(pts) < dangling_delete_threshold_m) {
            dangling_deleted.insert(fl.fid);
            break;
          }
        }
      }
    }
  }

  // Prepare output: deleted (original ids) + new_lines (all non-original fids)
  py::list deleted;
  py::list deleted_dangling;
  py::list deleted_cross_replaced;
  for (const auto &fid : replaced_original) {
    auto it = fid_to_obj.find(fid);
    if (it != fid_to_obj.end() && !it->second.is_none()) {
      deleted.append(it->second);
      deleted_cross_replaced.append(it->second);
    } else {
      deleted.append(py::str(fid));
      deleted_cross_replaced.append(py::str(fid));
    }
  }

  for (const auto &fid : dangling_deleted) {
    auto it = fid_to_obj.find(fid);
    if (it != fid_to_obj.end() && !it->second.is_none()) {
      deleted.append(it->second);
      deleted_dangling.append(it->second);
    } else {
      deleted.append(py::str(fid));
      deleted_dangling.append(py::str(fid));
    }
  }

  py::list new_lines;
  for (const auto &fl : current_lines) {
    if (original_fids.find(fl.fid) != original_fids.end()) {
      continue;
    }
    // output each part as a LineString feature
    for (const auto &pts : fl.parts) {
      if (pts.size() < 2) {
        continue;
      }
      py::dict f;
      f["type"] = "Feature";
      f["id"] = fl.fid;
      py::dict props;
      const std::string src = source_from_fid(fl.fid);
      props["source_id"] = py::str(src);
      auto it = connector_meta.find(fl.fid);
      if (it != connector_meta.end()) {
        props["note"] = "dangling_connect";
        props["source_id"] = py::str(it->second.first);
        props["target_id"] = py::str(it->second.second);
      } else {
        props["note"] = "repaired_split";
      }
      props["cost_factor"] = 10.0;
      f["properties"] = props;
      py::dict geom;
      geom["type"] = "LineString";
      py::list coords;
      for (const auto &p : pts) {
        coords.append(py::make_tuple(p.x, p.y));
      }
      geom["coordinates"] = coords;
      f["geometry"] = geom;
      new_lines.append(std::move(f));
    }
  }

  const std::size_t deleted_total = static_cast<std::size_t>(deleted.size());
  const std::size_t new_total = static_cast<std::size_t>(new_lines.size());

  std::string msg;
  if (deleted_total > 0) {
    msg += "建议删除线要素 " + std::to_string(deleted_total) + " 条（短悬挂=" +
           std::to_string(dangling_deleted.size()) + "；交叉被替换=" + std::to_string(replaced_original.size()) + "）";
  }
  if (new_total > 0) {
    if (!msg.empty()) {
      msg += "；";
    }
    msg += "生成绿色修复线段 " + std::to_string(new_total) + " 条（cost_factor=10.0）";
  }
  if (iters_done > 1) {
    if (!msg.empty()) {
      msg += "；";
    }
    msg += "迭代修复轮次 " + std::to_string(iters_done) + "/" + std::to_string(max_iters);
  }
  if (trimmed_spurs > 0 || connectors_added > 0) {
    if (!msg.empty()) {
      msg += "；";
    }
    msg += "悬挂处理：剪尾=" + std::to_string(trimmed_spurs) + "；补连接=" + std::to_string(connectors_added);
  }
  if (msg.empty()) {
    msg = "未发现可自动修复的问题";
  }

  py::dict counts;
  counts["dangling_deleted"] = static_cast<int>(dangling_deleted.size());
  counts["cross_replaced"] = static_cast<int>(replaced_original.size());
  counts["new_lines"] = static_cast<int>(new_lines.size());
  counts["iters"] = iters_report;
  counts["trimmed_spurs"] = trimmed_spurs;
  counts["connectors_added"] = connectors_added;

  py::dict out;
  out["action"] = "layer_repair";
  out["message"] = msg;
  out["deleted"] = deleted;
  out["deleted_dangling"] = deleted_dangling;
  out["deleted_cross_replaced"] = deleted_cross_replaced;
  out["new_lines"] = new_lines;
  out["counts"] = counts;
  out["cpp_only_noding"] = false;
  out["trim_m"] = trim_m;
  out["connect_m"] = connect_m;
  out["dangling_delete_threshold_m"] = dangling_delete_threshold_m;
  return out;
}

} // namespace

py::dict check_topology_layer_cpp(const py::list &features, const py::object &bbox, int ndigits) {
  auto lines = extract_lines(features, bbox);
  const auto res = topology::check_topology_layer(lines, ndigits);

  py::list issues;
  for (const auto &iss : res.issues) {
    py::dict it;
    it["kind"] = iss.kind;
    it["point"] = py::make_tuple(iss.point.x, iss.point.y);
    it["message"] = iss.message;
    issues.append(std::move(it));
  }

  py::dict stats;
  stats["lines"] = res.stats.lines;
  stats["segments"] = res.stats.segments;
  stats["cell"] = res.stats.cell;
  stats["seg_query_calls"] = res.stats.seg_query_calls;
  stats["seg_candidates_total"] = res.stats.seg_candidates_total;
  stats["seg_pairs_tested"] = res.stats.seg_pairs_tested;
  stats["seg_pairs_bbox_pass"] = res.stats.seg_pairs_bbox_pass;
  stats["seg_intersections_found"] = res.stats.seg_intersections_found;
  stats["seg_skipped_both_have_node"] = res.stats.seg_skipped_both_have_node;
  stats["seg_issues_endpoint_on_segment"] = res.stats.seg_issues_endpoint_on_segment;
  stats["seg_issues_cross_without_node"] = res.stats.seg_issues_cross_without_node;

  py::dict out;
  out["issues"] = issues;
  out["stats"] = stats;
  return out;
}

PYBIND11_MODULE(topology_cpp, m) {
  m.doc() = "Topology check core (C++)";
  m.def(
      "check_topology_layer",
      &check_topology_layer_cpp,
      py::arg("features"),
      py::arg("bbox") = py::none(),
      py::arg("ndigits") = 6,
      "Check topology for an entire line layer; returns {issues, stats}.");

  m.def(
      "repair_layer",
      &repair_layer_cpp,
      py::arg("features"),
      py::arg("issues"),
      py::arg("dangling_delete_threshold_m") = 100.0,
      py::arg("ndigits") = 6,
      "Repair a line layer by iterative noding (C++). Returns a dict compatible with Python repair_layer.");
}
