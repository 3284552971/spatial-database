#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

inline uint64_t edge_key(int u, int v) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(u)) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(v));
}

double haversine_m(double lon1, double lat1, double lon2, double lat2) {
  // Great-circle distance on sphere (meters) - good enough for heuristic.
  constexpr double R = 6371000.0;
  constexpr double PI = 3.14159265358979323846;
  const double phi1 = lat1 * PI / 180.0;
  const double phi2 = lat2 * PI / 180.0;
  const double dphi = (lat2 - lat1) * PI / 180.0;
  const double dl = (lon2 - lon1) * PI / 180.0;
  const double s1 = std::sin(dphi / 2.0);
  const double s2 = std::sin(dl / 2.0);
  const double h = s1 * s1 + std::cos(phi1) * std::cos(phi2) * s2 * s2;
  return 2.0 * R * std::atan2(std::sqrt(h), std::sqrt(1.0 - h));
}

struct PathResult {
  double dist_m = std::numeric_limits<double>::infinity();
  std::vector<int> nodes;
};

}  // namespace

class RoadGraph {
 public:
  RoadGraph(std::vector<std::pair<double, double>> nodes,
            std::vector<std::tuple<int, int, double>> edges)
      : nodes_(std::move(nodes)) {
    adj_.assign(nodes_.size(), {});
    for (const auto& e : edges) {
      const int u = std::get<0>(e);
      const int v = std::get<1>(e);
      const double w = std::get<2>(e);
      if (u < 0 || v < 0) continue;
      if (static_cast<size_t>(u) >= nodes_.size() || static_cast<size_t>(v) >= nodes_.size()) continue;
      if (!(w > 0.0)) continue;
      add_edge(u, v, w);
      add_edge(v, u, w);
    }
  }

  size_t node_count() const { return nodes_.size(); }
  size_t edge_count() const { return weight_.size(); }

  py::dict k_shortest_paths(int start, int goal, int k, const std::string& algo) const {
    py::dict empty;
    empty["paths"] = py::list();
    empty["distances"] = py::list();
    if (start < 0 || goal < 0 || k <= 0) {
      return empty;
    }
    if (static_cast<size_t>(start) >= nodes_.size() || static_cast<size_t>(goal) >= nodes_.size()) {
      return empty;
    }

    const std::string a = normalize_algo(algo);
    std::vector<PathResult> out = yen_k_shortest(start, goal, k, a);

    py::list paths;
    py::list dists;
    for (const auto& pr : out) {
      if (!std::isfinite(pr.dist_m) || pr.nodes.size() < 2) continue;
      paths.append(pr.nodes);
      dists.append(pr.dist_m);
    }
    py::dict res;
    res["paths"] = paths;
    res["distances"] = dists;
    return res;
  }

 private:
  std::vector<std::pair<double, double>> nodes_;
  std::vector<std::vector<std::pair<int, double>>> adj_;
  std::unordered_map<uint64_t, double> weight_;

  void add_edge(int u, int v, double w) {
    adj_[static_cast<size_t>(u)].push_back(std::make_pair(v, w));
    const uint64_t k = edge_key(u, v);
    auto it = weight_.find(k);
    if (it == weight_.end() || w < it->second) {
      weight_[k] = w;
    }
  }

  static std::string normalize_algo(std::string algo) {
    std::string out;
    out.reserve(algo.size());
    for (char c : algo) out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    if (out == "floyd") return "dijkstra";  // fallback
    if (out != "dijkstra" && out != "astar") return "dijkstra";
    return out;
  }

  double heuristic(int node, int goal) const {
    const auto& a = nodes_[static_cast<size_t>(node)];
    const auto& b = nodes_[static_cast<size_t>(goal)];
    return haversine_m(a.first, a.second, b.first, b.second);
  }

  PathResult shortest_path(
      int start,
      int goal,
      const std::string& algo,
      const std::vector<char>& banned_nodes,
      const std::unordered_set<uint64_t>& banned_edges) const {
    PathResult res;
    if (banned_nodes[static_cast<size_t>(start)] || banned_nodes[static_cast<size_t>(goal)]) {
      return res;
    }

    const size_t n = nodes_.size();
    std::vector<double> dist(n, std::numeric_limits<double>::infinity());
    std::vector<int> parent(n, -1);
    std::vector<char> visited(n, 0);

    struct QItem {
      double f;
      int node;
      double g;
    };
    struct Cmp {
      bool operator()(const QItem& a, const QItem& b) const { return a.f > b.f; }
    };
    std::priority_queue<QItem, std::vector<QItem>, Cmp> pq;

    dist[static_cast<size_t>(start)] = 0.0;
    const double h0 = (algo == "astar") ? heuristic(start, goal) : 0.0;
    pq.push(QItem{h0, start, 0.0});

    while (!pq.empty()) {
      const QItem cur = pq.top();
      pq.pop();
      const int u = cur.node;
      if (visited[static_cast<size_t>(u)]) continue;
      visited[static_cast<size_t>(u)] = 1;
      if (u == goal) break;

      const double du = dist[static_cast<size_t>(u)];
      for (const auto& nb : adj_[static_cast<size_t>(u)]) {
        const int v = nb.first;
        if (banned_nodes[static_cast<size_t>(v)]) continue;
        const uint64_t ek = edge_key(u, v);
        if (banned_edges.find(ek) != banned_edges.end()) continue;
        auto wit = weight_.find(ek);
        if (wit == weight_.end()) continue;
        const double w = wit->second;
        const double nd = du + w;
        if (nd < dist[static_cast<size_t>(v)]) {
          dist[static_cast<size_t>(v)] = nd;
          parent[static_cast<size_t>(v)] = u;
          const double hv = (algo == "astar") ? heuristic(v, goal) : 0.0;
          pq.push(QItem{nd + hv, v, nd});
        }
      }
    }

    if (!std::isfinite(dist[static_cast<size_t>(goal)])) {
      return res;
    }

    // reconstruct
    std::vector<int> path;
    for (int cur = goal; cur != -1; cur = parent[static_cast<size_t>(cur)]) {
      path.push_back(cur);
      if (cur == start) break;
    }
    if (path.empty() || path.back() != start) {
      return PathResult{};
    }
    std::reverse(path.begin(), path.end());
    res.dist_m = dist[static_cast<size_t>(goal)];
    res.nodes = std::move(path);
    return res;
  }

  std::vector<PathResult> yen_k_shortest(int start, int goal, int k, const std::string& algo) const {
    const size_t n = nodes_.size();
    std::vector<char> empty_banned_nodes(n, 0);
    std::unordered_set<uint64_t> empty_banned_edges;

    PathResult first = shortest_path(start, goal, algo, empty_banned_nodes, empty_banned_edges);
    if (!std::isfinite(first.dist_m) || first.nodes.size() < 2) {
      return {};
    }

    std::vector<PathResult> A;
    A.push_back(std::move(first));
    if (k == 1) return A;

    // precompute prefix costs along A[0]
    const std::vector<int>& base = A[0].nodes;
    std::vector<double> prefix_cost(base.size(), 0.0);
    for (size_t i = 1; i < base.size(); i++) {
      const uint64_t ek = edge_key(base[i - 1], base[i]);
      auto it = weight_.find(ek);
      if (it == weight_.end()) break;
      prefix_cost[i] = prefix_cost[i - 1] + it->second;
    }

    struct Cand {
      double cost;
      std::vector<int> path;
    };
    struct CCmp {
      bool operator()(const Cand& a, const Cand& b) const { return a.cost > b.cost; }
    };
    std::priority_queue<Cand, std::vector<Cand>, CCmp> B;
    std::unordered_set<std::string> seen;

    auto path_sig = [](const std::vector<int>& p) {
      std::string s;
      s.reserve(p.size() * 6);
      for (int v : p) {
        s.append(std::to_string(v));
        s.push_back(',');
      }
      return s;
    };

    for (size_t i = 0; i + 1 < base.size(); i++) {
      const int spur_node = base[i];
      const std::vector<int> root_path(base.begin(), base.begin() + static_cast<long>(i + 1));

      std::vector<char> banned_nodes(n, 0);
      for (size_t j = 0; j < i; j++) {
        banned_nodes[static_cast<size_t>(base[j])] = 1;
      }
      std::unordered_set<uint64_t> banned_edges;

      // remove the edge that would recreate the same prefix for the already found shortest path
      banned_edges.insert(edge_key(base[i], base[i + 1]));

      PathResult spur = shortest_path(spur_node, goal, algo, banned_nodes, banned_edges);
      if (!std::isfinite(spur.dist_m) || spur.nodes.size() < 2) continue;

      std::vector<int> total = root_path;
      total.pop_back();
      total.insert(total.end(), spur.nodes.begin(), spur.nodes.end());

      const std::string sig = path_sig(total);
      if (seen.find(sig) != seen.end()) continue;
      seen.insert(sig);

      const double root_cost = prefix_cost[i];
      const double total_cost = root_cost + spur.dist_m;
      B.push(Cand{total_cost, std::move(total)});
    }

    while (A.size() < static_cast<size_t>(k) && !B.empty()) {
      Cand best = B.top();
      B.pop();
      PathResult pr;
      pr.dist_m = best.cost;
      pr.nodes = std::move(best.path);
      A.push_back(std::move(pr));
      break;  // K=2 current需求；更大 K 可继续扩展
    }

    return A;
  }
};

PYBIND11_MODULE(pathfinding_cpp, m) {
  m.doc() = "Shortest path / K-shortest paths for repaired road network (pybind11).";

  py::class_<RoadGraph>(m, "RoadGraph")
      .def(py::init<std::vector<std::pair<double, double>>, std::vector<std::tuple<int, int, double>>>(),
           py::arg("nodes"), py::arg("edges"))
      .def("node_count", &RoadGraph::node_count)
      .def("edge_count", &RoadGraph::edge_count)
      .def("k_shortest_paths", &RoadGraph::k_shortest_paths, py::arg("start"), py::arg("goal"), py::arg("k"),
           py::arg("algo"));
}
