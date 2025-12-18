#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "R_tree.cpp"  // Reuse existing implementation; consider header split later.

namespace py = pybind11;

// Convert Python dict -> attribute map (string -> string).
static std::unordered_map<std::string, std::string> dict_to_attrs(const py::dict& d) {
    std::unordered_map<std::string, std::string> out;
    for (auto item : d) {
        std::string key = py::str(item.first);
        py::object val = py::reinterpret_borrow<py::object>(item.second);
        if (py::isinstance<py::str>(val)) {
            out[key] = val.cast<std::string>();
        } else if (py::isinstance<py::bool_>(val)) {
            out[key] = val.cast<bool>() ? "true" : "false";
        } else if (py::isinstance<py::int_>(val)) {
            out[key] = std::to_string(val.cast<long long>());
        } else if (py::isinstance<py::float_>(val)) {
            out[key] = std::to_string(val.cast<double>());
        } else {
            out[key] = py::str(val); // Fallback serialization.
        }
    }
    return out;
}

// Build a Python dict for a feature. Structure documented here for Python callers:
// {
//   "id": int,
//   "type": "point"|"polyline"|"polygon",
//   "bbox": {"minx": float, "miny": float, "maxx": float, "maxy": float},
//   "attributes": {<string>: <string>}
// }
static py::dict feature_to_dict(const Feature& f) {
    py::dict d;
    d["id"] = f.id;
    d["type"] = f.geometry_type();
    py::dict bbox;
    bbox["minx"] = f.bbox.minx;
    bbox["miny"] = f.bbox.miny;
    bbox["maxx"] = f.bbox.maxx;
    bbox["maxy"] = f.bbox.maxy;
    d["bbox"] = bbox;
    py::dict attrs;
    for (const auto& kv : f.attributes) {
        attrs[kv.first.c_str()] = kv.second;
    }
    d["attributes"] = attrs;
    return d;
}

class RTreeIndex {
public:
    bool load_from_file(const std::string& path) {
        return tree_.load_from_file(path);
    }

    bool save_serialized(const std::string& path) {
        return tree_.save_serialized(path);
    }

    bool load_serialized(const std::string& path) {
        return tree_.load_serialized(path);
    }

    void insert(const std::string& geom_type,
                const std::vector<std::vector<double>>& coords,
                const py::dict& attrs = py::dict()) {
        tree_.insert_from_coordinates(geom_type, coords, dict_to_attrs(attrs));
    }

    py::list query_box(double minx, double miny, double maxx, double maxy, const std::string& type_filter = "") {
        std::vector<Feature> results;
        tree_.query_box(minx, miny, maxx, maxy, results, type_filter);
        py::list out;
        for (const auto& f : results) {
            out.append(feature_to_dict(f));
        }
        return out;
    }

    py::list query_circle(double cx, double cy, double radius, const std::string& type_filter = "") {
        std::vector<Feature> results;
        tree_.query_circle(cx, cy, radius, results, type_filter);
        py::list out;
        for (const auto& f : results) {
            out.append(feature_to_dict(f));
        }
        return out;
    }

    bool delete_by_attribute(const std::string& key, const std::string& value) {
        return tree_.delete_by_attribute(key, value);
    }

private:
    R_tree tree_;
};

PYBIND11_MODULE(rtree_module, m) {
    m.doc() = "Minimal R-tree bindings with attribute-based delete and box/circle queries.";

    py::class_<RTreeIndex>(m, "RTreeIndex")
        .def(py::init<>())
        .def("load_from_file", &RTreeIndex::load_from_file, py::arg("path"),
             "Load features from a GeoJSON/simple JSON file.")
        .def("insert", &RTreeIndex::insert, py::arg("geom_type"), py::arg("coords"), py::arg("attrs") = py::dict(),
             "Insert a geometry of type 'point'|'polyline'|'polygon' with optional attributes (dict of string->string).")
        .def("query_box", &RTreeIndex::query_box, py::arg("minx"), py::arg("miny"), py::arg("maxx"), py::arg("maxy"), py::arg("type_filter") = "",
             "Query features intersecting an axis-aligned box; type_filter can be 'point', 'polyline', or 'polygon'. Returns list of dict as documented in feature_to_dict().")
        .def("query_circle", &RTreeIndex::query_circle, py::arg("cx"), py::arg("cy"), py::arg("radius"), py::arg("type_filter") = "",
             "Query features intersecting a circle; type_filter works like query_box. Returns list of dict.")
        .def("delete_by_attribute", &RTreeIndex::delete_by_attribute, py::arg("key"), py::arg("value"),
               "Delete the first feature whose attributes contain key==value. Returns True if deleted.")
           .def("save_serialized", &RTreeIndex::save_serialized, py::arg("path"),
               "Persist the current R-tree features as JSON for fast reload.")
           .def("load_serialized", &RTreeIndex::load_serialized, py::arg("path"),
               "Load a previously serialized R-tree (JSON produced by save_serialized).");
}
