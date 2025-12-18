#include "topology_check.h"

#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <utility>

// Reuse existing R-tree implementation for bbox-based candidate filtering.
// NOTE: R_tree.cpp is included as a translation-unit dependency (current project style).
#include "../R_tree.cpp"

namespace topology {

using geom_common::BBox;
using geom_common::Pt;

static inline bool on_segment(const Pt &a, const Pt &b, const Pt &p, double eps_dist) {
  // colinearity (within distance tolerance) + bounding-box
  const double minx = std::min(a.x, b.x) - eps_dist;
  const double maxx = std::max(a.x, b.x) + eps_dist;
  const double miny = std::min(a.y, b.y) - eps_dist;
  const double maxy = std::max(a.y, b.y) + eps_dist;
  if (p.x < minx || p.x > maxx || p.y < miny || p.y > maxy) {
    return false;
  }
  const Pt ap{p.x - a.x, p.y - a.y};
  const Pt ab{b.x - a.x, b.y - a.y};
  const double ab_len = std::hypot(ab.x, ab.y);
  if (ab_len <= 1e-18) {
    return (std::hypot(ap.x, ap.y) <= eps_dist);
  }
  const double cross = geom_common::cross2(ab, ap);
  // distance from p to line ab is |cross|/|ab|
  return (std::fabs(cross) / ab_len) <= eps_dist;
}

// Segment intersection compatible with Python version:
// - detects proper intersections
// - also detects endpoint-on-segment when segments are colinear/parallel
// - ignores true overlaps (infinite intersections)
static inline bool segment_intersection_any(const Pt &p, const Pt &p2, const Pt &q, const Pt &q2, Pt &out, double eps_dist) {
  const Pt r = geom_common::sub(p2, p);
  const Pt s = geom_common::sub(q2, q);
  const double denom = geom_common::cross2(r, s);
  constexpr double eps_denom = 1e-12;
  const double eps_t = 1e-12;

  if (std::fabs(denom) < eps_denom) {
    // parallel/colinear: only treat endpoint touches as intersection
    if (on_segment(p, p2, q, eps_dist)) {
      out = q;
      return true;
    }
    if (on_segment(p, p2, q2, eps_dist)) {
      out = q2;
      return true;
    }
    if (on_segment(q, q2, p, eps_dist)) {
      out = p;
      return true;
    }
    if (on_segment(q, q2, p2, eps_dist)) {
      out = p2;
      return true;
    }
    return false;
  }

  const Pt qp = geom_common::sub(q, p);
  const double t = geom_common::cross2(qp, s) / denom;
  const double u = geom_common::cross2(qp, r) / denom;
  if (t < -eps_t || t > 1.0 + eps_t || u < -eps_t || u > 1.0 + eps_t) {
    return false;
  }
  out = Pt{p.x + t * r.x, p.y + t * r.y};

  // Snap-to-endpoints to stabilize rounding-based endpoint classification.
  const double eps2 = eps_dist * eps_dist;
  auto snap_if_close = [&](const Pt &cand) {
    const double dx = out.x - cand.x;
    const double dy = out.y - cand.y;
    if ((dx * dx + dy * dy) <= eps2) {
      out = cand;
    }
  };
  snap_if_close(p);
  snap_if_close(p2);
  snap_if_close(q);
  snap_if_close(q2);
  return true;
}

struct Seg {
  Pt a;
  Pt b;
  BBox bb;
  std::string fid;
  std::uint64_t akey;
  std::uint64_t bkey;
};

static BBox bbox_from_pts(const std::vector<Pt> &pts) {
  auto bb = geom_common::bbox_empty();
  for (const auto &p : pts) {
    geom_common::bbox_expand(bb, p);
  }
  return bb;
}

CheckResult check_topology_layer(const std::vector<FeatureLines> &lines_in, int ndigits) {
  const double scale = std::pow(10.0, ndigits);

  CheckResult out;
  out.stats.lines = static_cast<int>(lines_in.size());

  // 端点度
  std::unordered_map<std::uint64_t, int> degree;
  degree.reserve(lines_in.size() * 4);

  for (const auto &fl : lines_in) {
    for (const auto &pts : fl.parts) {
      if (pts.size() < 2) {
        continue;
      }
      const auto akey = geom_common::round_key(pts.front(), scale);
      const auto bkey = geom_common::round_key(pts.back(), scale);
      degree[akey] += 1;
      degree[bkey] += 1;
    }
  }

  // 悬空端点
  for (const auto &fl : lines_in) {
    for (const auto &pts : fl.parts) {
      if (pts.size() < 2) {
        continue;
      }
      const auto akey = geom_common::round_key(pts.front(), scale);
      const auto bkey = geom_common::round_key(pts.back(), scale);
      if (degree[akey] <= 1) {
        out.issues.push_back(Issue{"dangling_endpoint", pts.front(), std::string("悬挂端点：fid=") + fl.fid});
      }
      if (degree[bkey] <= 1) {
        out.issues.push_back(Issue{"dangling_endpoint", pts.back(), std::string("悬挂端点：fid=") + fl.fid});
      }
    }
  }

  // 构建线段级索引 + 每条线的“端点集合”（仅首尾端点），用于判定交点处是否已有节点。
  struct SegRec {
    Pt a;
    Pt b;
    BBox bb;
    std::string fid;
    int seg_idx_in_fid = -1; // for dedup/self-intersection keying
  };

  std::unordered_map<std::string, std::unordered_set<std::uint64_t>> fid_endpoints;
  fid_endpoints.reserve(lines_in.size());

  std::vector<SegRec> segs;
  segs.reserve(1024);

  bool bb_first = true;
  BBox layer_bb{};
  int total_segments = 0;

  for (const auto &fl : lines_in) {
    if (bb_first) {
      layer_bb = fl.bb;
      bb_first = false;
    } else {
      geom_common::bbox_merge(layer_bb, fl.bb);
    }

    auto &epset = fid_endpoints[fl.fid];
    // rough reserve: sum of all pts
    std::size_t pts_sum = 0;
    for (const auto &pts : fl.parts) {
      pts_sum += pts.size();
    }
    if (pts_sum > 0 && epset.empty()) {
      epset.reserve(8);
    }

    int seg_idx = 0;
    for (const auto &pts : fl.parts) {
      if (pts.size() < 2) {
        continue;
      }
      epset.insert(geom_common::round_key(pts.front(), scale));
      epset.insert(geom_common::round_key(pts.back(), scale));
      for (std::size_t i = 1; i < pts.size(); i++) {
        const Pt a = pts[i - 1];
        const Pt b = pts[i];
        auto bb = geom_common::bbox_empty();
        geom_common::bbox_expand(bb, a);
        geom_common::bbox_expand(bb, b);
        segs.push_back(SegRec{a, b, bb, fl.fid, seg_idx});
        seg_idx += 1;
        total_segments += 1;
      }
    }
  }

  out.stats.segments = total_segments;
  if (total_segments <= 0) {
    out.stats.cell = 0.0;
    return out;
  }

  const double cell = geom_common::grid_cell_size(layer_bb);
  out.stats.cell = cell;

  // 构建线段级 R-tree：候选过滤更精确，避免线级 bbox 过大导致漏检/误判。
  R_tree seg_index;
  for (int i = 0; i < static_cast<int>(segs.size()); i++) {
    const auto &sr = segs[static_cast<std::size_t>(i)];
    seg_index.insert_segment(sr.a.x, sr.a.y, sr.b.x, sr.b.y, i);
  }

  struct CrossKey {
    std::uint64_t pkey;
    std::string a;
    std::string b;
    int ia = -1;
    int ib = -1;

    bool operator==(const CrossKey &o) const noexcept {
      return pkey == o.pkey && a == o.a && b == o.b && ia == o.ia && ib == o.ib;
    }
  };
  struct CrossHash {
    std::size_t operator()(const CrossKey &k) const noexcept {
      std::size_t h = std::hash<std::uint64_t>{}(k.pkey);
      h ^= (std::hash<std::string>{}(k.a) << 1);
      h ^= (std::hash<std::string>{}(k.b) << 2);
      h ^= (std::hash<int>{}(k.ia) << 3);
      h ^= (std::hash<int>{}(k.ib) << 4);
      return h;
    }
  };
  std::unordered_set<CrossKey, CrossHash> seen;

  std::vector<int> cand_ids;
  cand_ids.reserve(128);

  // 对于每条线段，通过 bbox 查询候选线段；然后执行精确线段相交。
  for (int i = 0; i < static_cast<int>(segs.size()); i++) {
    const auto &sa = segs[static_cast<std::size_t>(i)];
    out.stats.seg_query_calls += 1;
    cand_ids.clear();
    seg_index.query_box_ids(sa.bb.minx, sa.bb.miny, sa.bb.maxx, sa.bb.maxy, cand_ids);
    if (cand_ids.empty()) {
      continue;
    }
    std::sort(cand_ids.begin(), cand_ids.end());
    cand_ids.erase(std::unique(cand_ids.begin(), cand_ids.end()), cand_ids.end());
    out.stats.seg_candidates_total += static_cast<std::int64_t>(cand_ids.size());

    for (const int j : cand_ids) {
      if (j <= i || j >= static_cast<int>(segs.size())) {
        continue;
      }
      out.stats.seg_pairs_tested += 1;
      const auto &sb = segs[static_cast<std::size_t>(j)];

      // Match Python checker semantics: ignore self intersections within the same feature.
      if (sa.fid == sb.fid) {
        continue;
      }
      if (!geom_common::bbox_intersects(sa.bb, sb.bb)) {
        continue;
      }
      out.stats.seg_pairs_bbox_pass += 1;

      Pt ip{};
      const double eps_dist = 0.5 / scale;
      if (!segment_intersection_any(sa.a, sa.b, sb.a, sb.b, ip, eps_dist)) {
        continue;
      }
      out.stats.seg_intersections_found += 1;

      const auto ipkey = geom_common::round_key(ip, scale);
      const auto &a_vs = fid_endpoints[sa.fid];
      const auto &b_vs = fid_endpoints[sb.fid];

      const bool a_has_node = (a_vs.find(ipkey) != a_vs.end());
      const bool b_has_node = (b_vs.find(ipkey) != b_vs.end());

      // 两边都已有节点：共享顶点（包含首尾相接/内部顶点相接），不算拓扑问题。
      if (a_has_node && b_has_node) {
        out.stats.seg_skipped_both_have_node += 1;
        continue;
      }

      // 仅一边已有节点：点落在对方线段内部（等价于“端点/顶点落在线段内部”），需要补节点。
      if (a_has_node ^ b_has_node) {
        const std::string end_fid = a_has_node ? sa.fid : sb.fid;
        const std::string seg_fid = a_has_node ? sb.fid : sa.fid;
        CrossKey ck{ipkey, end_fid, seg_fid, -1, -1};
        if (seen.find(ck) != seen.end()) {
          continue;
        }
        seen.insert(ck);
        out.issues.push_back(
            Issue{"endpoint_on_segment", ip, std::string("点落在线段内部：node=") + end_fid + " on " + seg_fid});
        out.stats.seg_issues_endpoint_on_segment += 1;
        continue;
      }

      // 两边都没有节点：真正的交叉无节点。
      std::string a = sa.fid;
      std::string b = sb.fid;
      if (a > b) {
        std::swap(a, b);
      }

      CrossKey ck{ipkey, a, b, -1, -1};
      if (seen.find(ck) != seen.end()) {
        continue;
      }
      seen.insert(ck);
      out.issues.push_back(Issue{"cross_without_node", ip, std::string("线相交无节点：") + a + " x " + b});
      out.stats.seg_issues_cross_without_node += 1;
    }
  }

  return out;
}

} // namespace topology
