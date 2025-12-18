#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEarthRadiusM = 6371000.0;

inline double deg2rad(double d) { return d * (kPi / 180.0); }

struct XY {
  double x;
  double y;
};

static XY lonlat_to_xy_m(double lon_deg, double lat_deg, double lat0_deg) {
  const double lon = deg2rad(lon_deg);
  const double lat = deg2rad(lat_deg);
  const double lat0 = deg2rad(lat0_deg);
  const double x = kEarthRadiusM * lon * std::cos(lat0);
  const double y = kEarthRadiusM * lat;
  return XY{x, y};
}

inline std::int64_t grid_key(std::int32_t ix, std::int32_t iy) {
  const std::uint64_t a = static_cast<std::uint32_t>(ix);
  const std::uint64_t b = static_cast<std::uint32_t>(iy);
  return static_cast<std::int64_t>((a << 32) | b);
}

struct GridIndex {
  double cell_m;
  double inv_cell_m;
  std::vector<double> nx;
  std::vector<double> ny;
  std::unordered_map<std::int64_t, std::vector<int>> buckets;

  GridIndex() : cell_m(50.0), inv_cell_m(1.0 / 50.0) {}

  void build(const std::vector<double>& xs, const std::vector<double>& ys, double cell) {
    if (xs.size() != ys.size()) {
      throw std::runtime_error("nodes x/y size mismatch");
    }
    if (xs.empty()) {
      throw std::runtime_error("empty nodes");
    }
    cell_m = (cell > 0.0) ? cell : 50.0;
    inv_cell_m = 1.0 / cell_m;
    nx = xs;
    ny = ys;
    buckets.clear();
    buckets.reserve(xs.size() / 8 + 1);

    for (int i = 0; i < static_cast<int>(xs.size()); i++) {
      const std::int32_t ix = static_cast<std::int32_t>(std::floor(xs[i] * inv_cell_m));
      const std::int32_t iy = static_cast<std::int32_t>(std::floor(ys[i] * inv_cell_m));
      buckets[grid_key(ix, iy)].push_back(i);
    }
  }

  std::vector<int> nearest_k(double x, double y, int k, int max_ring = 30) const {
    if (k <= 0) {
      return {};
    }
    k = std::min<int>(k, static_cast<int>(nx.size()));
    std::vector<std::pair<double, int>> best;  // (dist2, idx)
    best.reserve(k);
    auto worst_it = [&best]() -> std::vector<std::pair<double, int>>::iterator {
      return std::max_element(best.begin(), best.end(), [](auto& a, auto& b) { return a.first < b.first; });
    };

    const std::int32_t cx = static_cast<std::int32_t>(std::floor(x * inv_cell_m));
    const std::int32_t cy = static_cast<std::int32_t>(std::floor(y * inv_cell_m));

    for (int r = 0; r <= max_ring; r++) {
      for (int dx = -r; dx <= r; dx++) {
        for (int dy = -r; dy <= r; dy++) {
          if (std::abs(dx) != r && std::abs(dy) != r) {
            continue;  // only border cells to avoid re-scanning
          }
          const std::int32_t ix = cx + dx;
          const std::int32_t iy = cy + dy;
          auto it = buckets.find(grid_key(ix, iy));
          if (it == buckets.end()) {
            continue;
          }
          for (int idx : it->second) {
            const double ddx = nx[idx] - x;
            const double ddy = ny[idx] - y;
            const double d2 = ddx * ddx + ddy * ddy;
            if (static_cast<int>(best.size()) < k) {
              best.push_back({d2, idx});
            } else {
              auto w = worst_it();
              if (d2 < w->first) {
                *w = {d2, idx};
              }
            }
          }
        }
      }
      if (static_cast<int>(best.size()) >= k && r >= 2) {
        break;  // approximate early stop; dense road networks should hit quickly
      }
    }

    if (best.empty()) {
      return {};
    }
    std::sort(best.begin(), best.end(), [](auto& a, auto& b) { return a.first < b.first; });
    std::vector<int> out;
    out.reserve(k);
    for (int i = 0; i < std::min<int>(k, static_cast<int>(best.size())); i++) {
      out.push_back(best[i].second);
    }
    return out;
  }
};

static void kalman_cv_xy(
    const std::vector<double>& t,
    const std::vector<double>& mx,
    const std::vector<double>& my,
    std::vector<double>& out_speed_mps,
    std::vector<double>& out_heading_deg) {
  const std::size_t n = mx.size();
  if (my.size() != n || t.size() != n) {
    throw std::runtime_error("kalman input size mismatch");
  }
  out_speed_mps.assign(n, 0.0);
  out_heading_deg.assign(n, 0.0);
  if (n == 0) {
    return;
  }

  // State: [x, y, vx, vy]
  double x = mx[0];
  double y = my[0];
  double vx = 0.0;
  double vy = 0.0;
  if (n >= 2) {
    const double dt0 = std::max(1e-3, t[1] - t[0]);
    vx = (mx[1] - mx[0]) / dt0;
    vy = (my[1] - my[0]) / dt0;
  }

  double P[16] = {0};
  P[0] = 10.0;
  P[5] = 10.0;
  P[10] = 100.0;
  P[15] = 100.0;

  const double sigma_meas = 8.0;    // meters
  const double R = sigma_meas * sigma_meas;
  const double sigma_acc = 2.0;     // m/s^2
  const double q = sigma_acc * sigma_acc;

  auto emit = [&](std::size_t i) {
    const double sp = std::sqrt(vx * vx + vy * vy);
    double ang = 0.0;
    if (sp > 1e-6) {
      ang = std::atan2(vy, vx) * (180.0 / kPi);
      if (ang < 0.0) {
        ang += 360.0;
      }
      if (ang >= 360.0) {
        ang = std::fmod(ang, 360.0);
      }
    }
    out_speed_mps[i] = sp;
    out_heading_deg[i] = ang;
  };

  emit(0);

  for (std::size_t i = 1; i < n; i++) {
    double dt = t[i] - t[i - 1];
    if (!(dt > 0.0)) {
      dt = 1e-3;
    }

    // Predict step
    x = x + vx * dt;
    y = y + vy * dt;

    // F matrix
    const double F[16] = {
        1, 0, dt, 0,
        0, 1, 0, dt,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };

    // Q matrix (CV + acceleration noise)
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt2 * dt2;
    const double Q[16] = {
        q * (dt4 / 4.0), 0, q * (dt3 / 2.0), 0,
        0, q * (dt4 / 4.0), 0, q * (dt3 / 2.0),
        q * (dt3 / 2.0), 0, q * dt2, 0,
        0, q * (dt3 / 2.0), 0, q * dt2,
    };

    // P = F P F^T + Q
    double FP[16] = {0};
    for (int r = 0; r < 4; r++) {
      for (int c = 0; c < 4; c++) {
        double s = 0.0;
        for (int k = 0; k < 4; k++) {
          s += F[r * 4 + k] * P[k * 4 + c];
        }
        FP[r * 4 + c] = s;
      }
    }
    double FPFt[16] = {0};
    for (int r = 0; r < 4; r++) {
      for (int c = 0; c < 4; c++) {
        double s = 0.0;
        for (int k = 0; k < 4; k++) {
          s += FP[r * 4 + k] * F[c * 4 + k];  // F^T
        }
        FPFt[r * 4 + c] = s + Q[r * 4 + c];
      }
    }
    for (int j = 0; j < 16; j++) {
      P[j] = FPFt[j];
    }

    // Update step with z = [mx, my]
    const double z0 = mx[i];
    const double z1 = my[i];
    const double y0 = z0 - x;
    const double y1 = z1 - y;

    // S = HPH^T + R, H selects x,y so:
    // S = [[P00+R, P01],[P10, P11+R]]
    const double S00 = P[0] + R;
    const double S01 = P[1];
    const double S10 = P[4];
    const double S11 = P[5] + R;
    const double det = S00 * S11 - S01 * S10;
    if (std::abs(det) < 1e-12) {
      emit(i);
      continue;
    }
    const double invS00 = S11 / det;
    const double invS01 = -S01 / det;
    const double invS10 = -S10 / det;
    const double invS11 = S00 / det;

    // K = P H^T invS; H^T picks first two columns of P
    // K is 4x2
    double K[8] = {0};
    for (int r = 0; r < 4; r++) {
      const double P_r0 = P[r * 4 + 0];
      const double P_r1 = P[r * 4 + 1];
      K[r * 2 + 0] = P_r0 * invS00 + P_r1 * invS10;
      K[r * 2 + 1] = P_r0 * invS01 + P_r1 * invS11;
    }

    x += K[0] * y0 + K[1] * y1;
    y += K[2] * y0 + K[3] * y1;
    vx += K[4] * y0 + K[5] * y1;
    vy += K[6] * y0 + K[7] * y1;

    // P = P - K * (H*P) ; H*P takes first two rows of P
    double Pnew[16] = {0};
    for (int r = 0; r < 4; r++) {
      for (int c = 0; c < 4; c++) {
        const double k0 = K[r * 2 + 0];
        const double k1 = K[r * 2 + 1];
        Pnew[r * 4 + c] = P[r * 4 + c] - k0 * P[0 * 4 + c] - k1 * P[1 * 4 + c];
      }
    }
    for (int j = 0; j < 16; j++) {
      P[j] = Pnew[j];
    }

    emit(i);
  }
}

}  // namespace

static py::dict hmm_match_and_kalman(
    const py::list& obs_lon,
    const py::list& obs_lat,
    const py::list& obs_t,
    const py::list& nodes_lon,
    const py::list& nodes_lat,
    int k_candidates = 2,
    double grid_cell_m = 50.0,
    double sigma_emission_m = 15.0,
    double sigma_transition_m = 80.0) {
  const std::size_t n = obs_lon.size();
  if (obs_lat.size() != n || obs_t.size() != n) {
    throw std::runtime_error("obs lon/lat/t size mismatch");
  }
  const std::size_t m = nodes_lon.size();
  if (nodes_lat.size() != m) {
    throw std::runtime_error("nodes lon/lat size mismatch");
  }
  if (n == 0) {
    throw std::runtime_error("empty trajectory");
  }
  if (m == 0) {
    throw std::runtime_error("empty road nodes");
  }
  if (k_candidates <= 0) {
    k_candidates = 1;
  }
  k_candidates = std::min<int>(k_candidates, static_cast<int>(m));

  std::vector<double> olon(n), olat(n), ot(n);
  for (std::size_t i = 0; i < n; i++) {
    olon[i] = py::cast<double>(obs_lon[i]);
    olat[i] = py::cast<double>(obs_lat[i]);
    ot[i] = py::cast<double>(obs_t[i]);
  }

  const double lat0 = olat[0];
  std::vector<double> ox(n), oy(n);
  for (std::size_t i = 0; i < n; i++) {
    const XY p = lonlat_to_xy_m(olon[i], olat[i], lat0);
    ox[i] = p.x;
    oy[i] = p.y;
  }

  std::vector<double> nlon(m), nlat(m), nx(m), ny(m);
  for (std::size_t i = 0; i < m; i++) {
    nlon[i] = py::cast<double>(nodes_lon[i]);
    nlat[i] = py::cast<double>(nodes_lat[i]);
    const XY p = lonlat_to_xy_m(nlon[i], nlat[i], lat0);
    nx[i] = p.x;
    ny[i] = p.y;
  }

  GridIndex grid;
  grid.build(nx, ny, grid_cell_m);

  const double sigma_e2 = sigma_emission_m * sigma_emission_m;
  const double sigma_t2 = sigma_transition_m * sigma_transition_m;
  if (!(sigma_e2 > 0.0) || !(sigma_t2 > 0.0)) {
    throw std::runtime_error("sigma must be > 0");
  }

  // Candidate indices per timestep: cand[i*K + k]
  const int K = k_candidates;
  std::vector<int> cand(static_cast<std::size_t>(K) * n, -1);
  std::vector<double> emit_cost(static_cast<std::size_t>(K) * n, 0.0);
  for (std::size_t i = 0; i < n; i++) {
    const auto ids = grid.nearest_k(ox[i], oy[i], K);
    if (ids.empty()) {
      throw std::runtime_error("failed to find candidate nodes");
    }
    for (int k = 0; k < K; k++) {
      const int idx = ids[std::min<int>(k, static_cast<int>(ids.size()) - 1)];
      cand[i * K + k] = idx;
      const double dx = nx[idx] - ox[i];
      const double dy = ny[idx] - oy[i];
      const double d2 = dx * dx + dy * dy;
      emit_cost[i * K + k] = d2 / (2.0 * sigma_e2);
    }
  }

  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> prev(K, inf), cur(K, inf);
  std::vector<std::uint8_t> back(static_cast<std::size_t>(K) * n, 0);

  for (int k = 0; k < K; k++) {
    prev[k] = emit_cost[0 * K + k];
  }

  for (std::size_t i = 1; i < n; i++) {
    const double dxo = ox[i] - ox[i - 1];
    const double dyo = oy[i] - oy[i - 1];
    const double d_obs = std::sqrt(dxo * dxo + dyo * dyo);

    for (int k = 0; k < K; k++) {
      double best = inf;
      std::uint8_t arg = 0;
      const int nk = cand[i * K + k];

      for (int pk = 0; pk < K; pk++) {
        const int pn = cand[(i - 1) * K + pk];
        const double dxn = nx[nk] - nx[pn];
        const double dyn = ny[nk] - ny[pn];
        const double d_can = std::sqrt(dxn * dxn + dyn * dyn);
        const double delta = d_can - d_obs;
        const double trans = (delta * delta) / (2.0 * sigma_t2);
        const double cost = prev[pk] + trans + emit_cost[i * K + k];
        if (cost < best) {
          best = cost;
          arg = static_cast<std::uint8_t>(pk);
        }
      }
      cur[k] = best;
      back[i * K + k] = arg;
    }
    prev.swap(cur);
  }

  // Backtrack best path
  int endk = 0;
  double best_end = prev[0];
  for (int k = 1; k < K; k++) {
    if (prev[k] < best_end) {
      best_end = prev[k];
      endk = k;
    }
  }

  std::vector<int> path_k(n, 0);
  path_k[n - 1] = endk;
  for (std::size_t i = n - 1; i >= 1; i--) {
    const std::uint8_t pk = back[i * K + path_k[i]];
    path_k[i - 1] = static_cast<int>(pk);
    if (i == 1) {
      break;
    }
  }

  std::vector<double> matched_lon(n), matched_lat(n), matched_x(n), matched_y(n);
  std::vector<int> matched_node_index(n, -1);
  for (std::size_t i = 0; i < n; i++) {
    const int nk = cand[i * K + path_k[i]];
    matched_node_index[i] = nk;
    matched_lon[i] = nlon[nk];
    matched_lat[i] = nlat[nk];
    matched_x[i] = nx[nk];
    matched_y[i] = ny[nk];
  }

  std::vector<double> est_speed_mps;
  std::vector<double> est_heading_deg;
  kalman_cv_xy(ot, matched_x, matched_y, est_speed_mps, est_heading_deg);

  py::dict out;
  out["matched_lon"] = matched_lon;
  out["matched_lat"] = matched_lat;
  out["est_speed_mps"] = est_speed_mps;
  out["est_heading_deg"] = est_heading_deg;
  out["matched_node_index"] = matched_node_index;
  out["cost"] = best_end;
  out["k_candidates"] = K;
  return out;
}

PYBIND11_MODULE(trajectory_cpp, m) {
  m.doc() = "Trajectory HMM map matching + Kalman (no external libs)";
  m.def("hmm_match_and_kalman", &hmm_match_and_kalman,
        py::arg("obs_lon"),
        py::arg("obs_lat"),
        py::arg("obs_t"),
        py::arg("nodes_lon"),
        py::arg("nodes_lat"),
        py::arg("k_candidates") = 2,
        py::arg("grid_cell_m") = 50.0,
        py::arg("sigma_emission_m") = 15.0,
        py::arg("sigma_transition_m") = 80.0);
}

