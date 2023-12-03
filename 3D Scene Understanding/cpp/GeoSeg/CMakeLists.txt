cmake_minimum_required(VERSION 3.5)
project(point_sam)

message("start make GeoSeg")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_STANDARD 17)

set(ALL_TARGET_LIBRARIES "")
include(cmake/boost.cmake)
include(cmake/global_definition.cmake)
include(cmake/glog.cmake)
include(cmake/eigen.cmake)
include(cmake/pcl.cmake)
include(cmake/yaml.cmake)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
)

file(GLOB_RECURSE ALL_SRCS "src/*.cpp")
add_library(${PROJECT_NAME} SHARED
        ${ALL_SRCS})
target_link_libraries(${PROJECT_NAME}
        ${ALL_TARGET_LIBRARIES}
        )

add_executable(plane_seg examples/cpp/plane_seg.cpp)
target_link_libraries(plane_seg ${PROJECT_NAME})

