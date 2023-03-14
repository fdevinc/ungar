add_library(Eigen3 INTERFACE)
add_library(Eigen3::Eigen ALIAS Eigen3)

target_include_directories(Eigen3 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
