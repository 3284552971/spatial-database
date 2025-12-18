cmake_minimum_required(VERSION 3.15)
project(register_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Homebrew prefix on macOS if nothing is provided, so `cmake -B build` works out of the box.
if(APPLE)
	if(NOT DEFINED CMAKE_PREFIX_PATH OR CMAKE_PREFIX_PATH STREQUAL "")
		set(CMAKE_PREFIX_PATH "/opt/homebrew" CACHE PATH "Prefix path for dependencies")
	endif()
endif()

find_package(yaml-cpp CONFIG REQUIRED)

# Ensure pybind11 picks the active Python (honors Python_EXECUTABLE if provided).
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# Put all build artifacts under this build directory (space_app/cpp_model/build)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

add_library(r_tree STATIC R_tree.cpp)
target_include_directories(r_tree PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(r_tree PUBLIC nlohmann_json::nlohmann_json)


# Python bindings for the R-tree
pybind11_add_module(rtree_module rtree_bindings.cpp)
target_include_directories(rtree_module PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(rtree_module PRIVATE nlohmann_json::nlohmann_json)

# Topology core (C++)
pybind11_add_module(topology_cpp topology_bindings.cpp src/topology_check.cpp)
target_include_directories(topology_cpp PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Shortest path / K-shortest paths (C++)
pybind11_add_module(pathfinding_cpp pathfinding_bindings.cpp)
target_include_directories(pathfinding_cpp PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Trajectory: HMM map matching + Kalman (C++)
pybind11_add_module(trajectory_cpp trajectory_bindings.cpp)
target_include_directories(trajectory_cpp PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/include
)
