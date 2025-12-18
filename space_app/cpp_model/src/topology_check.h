#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "geom_common.h"

namespace topology {

struct FeatureLines {
  std::string fid;
  std::vector<std::vector<geom_common::Pt>> parts;
  geom_common::BBox bb;
  std::unordered_set<std::uint64_t> endpoints;
};

struct Issue {
  std::string kind;
  geom_common::Pt point;
  std::string message;
};

struct Stats {
  int lines = 0;
  int segments = 0;
  double cell = 0.0;

  // Debug counters for checker quality/perf
  std::int64_t seg_query_calls = 0;
  std::int64_t seg_candidates_total = 0;
  std::int64_t seg_pairs_tested = 0;
  std::int64_t seg_pairs_bbox_pass = 0;
  std::int64_t seg_intersections_found = 0;
  std::int64_t seg_skipped_both_have_node = 0;
  std::int64_t seg_issues_endpoint_on_segment = 0;
  std::int64_t seg_issues_cross_without_node = 0;
};

struct CheckResult {
  std::vector<Issue> issues;
  Stats stats;
};

CheckResult check_topology_layer(const std::vector<FeatureLines> &lines, int ndigits);

} // namespace topology
