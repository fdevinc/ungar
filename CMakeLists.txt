cmake_minimum_required(VERSION 3.20)

project(Ungar VERSION 1.0.0)

include(cmake/ungar_utils.cmake)

ungar_option(RELEASE "Enable Ungar release mode." OFF)
ungar_option(TESTS "Enable Ungar tests." ON)
ungar_option(EXAMPLES "Enable Ungar examples." ON)

ungar_option(BUILD_AD "Build with automatic differentiation." ON)
ungar_option(SQP "Build with sequential QP." ON)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(external)

add_library(ungar INTERFACE)
add_library(ungar::ungar ALIAS ungar)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
  message(ERROR "Clang not currently supported")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  target_compile_options(ungar INTERFACE
    "-std=c++20" # "-Wall" "-Wextra"
    "-pedantic"
    "-ftemplate-backtrace-limit=1"
    "-fconstexpr-depth=2147483647"
    "-fconstexpr-loop-limit=2147483647"
    "-fconstexpr-cache-depth=2147483647"
    "-fconstexpr-ops-limit=2147483647"
    "$<$<CONFIG:RELEASE>:-O3>")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # using Intel C++
  message(ERROR "Intel not currently supported")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # using Visual Studio C++
  target_compile_options(ungar INTERFACE"/std:c++latest" "/Zc:preprocessor" "/W4"
                      "$<$<CONFIG:RELEASE>:/O2>" "/wd5105")
else()
  message(ERROR "Unknown compiler not currently supported")
endif()

list(APPEND UNGAR_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")
list(APPEND UNGAR_LINK_LIBRARIES spdlog::spdlog ${CMAKE_DL_LIBS} Eigen3::Eigen Boost::hana finitediff::finitediff)

if(${UNGAR_BUILD_AD})
  list(APPEND UNGAR_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/external/cppad/include" "${CMAKE_BINARY_DIR}/external/cppadcodegen/include")
  list(APPEND UNGAR_LINK_LIBRARIES ${cppad_lib})
endif()

if(${UNGAR_BUILD_SQP})
  list(APPEND UNGAR_LINK_LIBRARIES osqp-cpp)
endif()

target_include_directories(
  ungar
  INTERFACE ${UNGAR_INCLUDE_DIRECTORIES})
target_link_libraries(
  ungar
  INTERFACE ${UNGAR_LINK_LIBRARIES})

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

if(${UNGAR_TESTS})
  add_executable(utils.test test/utils/utils.test.cpp)
  target_link_libraries(utils.test ungar::ungar gtest)

  add_executable(variable.test test/variable.test.cpp)
  target_link_libraries(variable.test ungar::ungar gtest)

  if(${UNGAR_BUILD_AD})
    add_executable(function.test test/autodiff/function.test.cpp)
    target_link_libraries(function.test ungar::ungar gtest)
  endif()

  if(${UNGAR_BUILD_SQP})
    add_executable(soft_sqp.test test/optimization/soft_sqp.test.cpp)
    target_link_libraries(soft_sqp.test ungar::ungar gtest)
  endif()
endif()


if(${UNGAR_EXAMPLES})
add_executable(variable.example example/variable.example.cpp)
  target_link_libraries(variable.example ungar::ungar)

  add_executable(variable_map.example example/variable_map.example.cpp)
  target_link_libraries(variable_map.example ungar::ungar)

  if(${UNGAR_BUILD_AD})
    add_executable(function.example example/autodiff/function.example.cpp)
    target_link_libraries(function.example ungar::ungar)
  endif()
endif()
