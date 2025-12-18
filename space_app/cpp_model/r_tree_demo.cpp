#include "R_tree.cpp" // 或者改为头/源分离后包含头
#include <iostream>

int main() {
    R_tree tree;
    tree.insert_from_coordinates("point", {{1.0, 2.0}});
    // Example path (project-relative): space_app/shenzhen/road_sz.geojson
    tree.load_from_file("space_app/shenzhen/road_sz.geojson");
    std::cout << "Loaded tree.\n";
    return 0;
}
