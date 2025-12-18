#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace geom_common {

struct Pt {
  double x;
  double y;
};

struct BBox {
  double minx{std::numeric_limits<double>::infinity()};
  double miny{std::numeric_limits<double>::infinity()};
  double maxx{-std::numeric_limits<double>::infinity()};
  double maxy{-std::numeric_limits<double>::infinity()};
};

static inline bool bbox_intersects(const BBox &a, const BBox &b) {
  return !(a.maxx < b.minx || a.minx > b.maxx || a.maxy < b.miny || a.miny > b.maxy);
}

static inline double cross2(const Pt &a, const Pt &b) { return a.x * b.y - a.y * b.x; }

static inline Pt sub(const Pt &a, const Pt &b) { return Pt{a.x - b.x, a.y - b.y}; }

// Segment intersection (proper intersection only; ignores colinear/overlap) for minimal topology checks.
static inline bool segment_intersection(const Pt &p, const Pt &p2, const Pt &q, const Pt &q2, Pt &out) {
  const Pt r = sub(p2, p);
  const Pt s = sub(q2, q);
  const double denom = cross2(r, s);
  const double eps = 1e-12;
  if (std::fabs(denom) < eps) {
    return false;
  }
  const Pt qp = sub(q, p);
  const double t = cross2(qp, s) / denom;
  const double u = cross2(qp, r) / denom;
  if (t < 0.0 || t > 1.0 || u < 0.0 || u > 1.0) {
    return false;
  }
  out = Pt{p.x + t * r.x, p.y + t * r.y};
  return true;
}

static inline std::uint64_t pack_key(std::int64_t ix, std::int64_t iy) {
  const std::uint64_t ux = static_cast<std::uint64_t>(static_cast<std::uint32_t>(ix));
  const std::uint64_t uy = static_cast<std::uint64_t>(static_cast<std::uint32_t>(iy));
  return (ux << 32) ^ uy;
}

static inline std::uint64_t round_key(const Pt &p, double scale) {
  const auto ix = static_cast<std::int64_t>(std::llround(p.x * scale));
  const auto iy = static_cast<std::int64_t>(std::llround(p.y * scale));
  return pack_key(ix, iy);
}

static inline double clamp(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }

static inline double grid_cell_size(const BBox &bb) {
  const double spanx = std::max(1e-12, bb.maxx - bb.minx);
  const double spany = std::max(1e-12, bb.maxy - bb.miny);
  const double size = std::max(spanx, spany) / 50.0;
  return clamp(size, 0.001, 0.05);
}

static inline std::int64_t grid_i(double v, double cell) {
  return static_cast<std::int64_t>(std::floor(v / cell));
}

static inline BBox bbox_empty() {
  return BBox{std::numeric_limits<double>::infinity(),
              std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity()};
}

static inline void bbox_expand(BBox &bb, const Pt &p) {
  bb.minx = std::min(bb.minx, p.x);
  bb.miny = std::min(bb.miny, p.y);
  bb.maxx = std::max(bb.maxx, p.x);
  bb.maxy = std::max(bb.maxy, p.y);
}

static inline void bbox_merge(BBox &bb, const BBox &other) {
  bb.minx = std::min(bb.minx, other.minx);
  bb.miny = std::min(bb.miny, other.miny);
  bb.maxx = std::max(bb.maxx, other.maxx);
  bb.maxy = std::max(bb.maxy, other.maxy);
}

} // namespace geom_common
