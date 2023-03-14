cmake_minimum_required(VERSION 3.20)

project(Ungar VERSION 1.0.0)

option(UNGAR_RELEASE "Enable Ungar release mode." OFF)

if(MSVC)
  add_compile_options("/std:c++latest" "/Zc:preprocessor" "/W4"
                      "$<$<CONFIG:RELEASE>:/O2>" "/wd5105")
else()
  add_compile_options(
    "-std=c++20" # "-Wall" "-Wextra"
    "-pedantic"
    "-ftemplate-backtrace-limit=1"
    "-fconstexpr-depth=2147483647"
    "-fconstexpr-loop-limit=2147483647"
    "-fconstexpr-cache-depth=2147483647"
    "-fconstexpr-ops-limit=2147483647"
    "$<$<CONFIG:RELEASE>:-O3>")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(external)

add_library(ungar INTERFACE)
add_library(ungar::ungar ALIAS ungar)

target_include_directories(
  ungar
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_BINARY_DIR}/external/cppad/include
            ${CMAKE_BINARY_DIR}/external/cppadcodegen/include)
target_link_libraries(
  ungar
  INTERFACE spdlog::spdlog
            ${cppad_lib}
            ${CMAKE_DL_LIBS}
            Eigen3::Eigen
            Boost::hana
            finitediff::finitediff
            osqp-cpp)

set(UNGAR_CODEGEN_FOLDER "${CMAKE_BINARY_DIR}/ungar_codegen")
if(UNGAR_RELEASE)
  target_compile_definitions(
    ungar
    INTERFACE "UNGAR_CODEGEN_FOLDER=\"${UNGAR_CODEGEN_FOLDER}\""
    INTERFACE "UNGAR_RELEASE")
else()
  target_compile_definitions(
    ungar
    INTERFACE "UNGAR_CODEGEN_FOLDER=\"${UNGAR_CODEGEN_FOLDER}\""
    INTERFACE "_GLIBCXX_ASSERTIONS")
endif()

add_executable(function.test test/autodiff/function.test.cpp)
target_link_libraries(function.test ungar::ungar gtest)

add_executable(utils.test test/utils/utils.test.cpp)
target_link_libraries(utils.test ungar::ungar gtest)

add_executable(variable.test test/variable.test.cpp)
target_link_libraries(variable.test ungar::ungar gtest)

add_executable(soft_sqp.test test/optimization/soft_sqp.test.cpp)
target_link_libraries(soft_sqp.test ungar::ungar gtest)
