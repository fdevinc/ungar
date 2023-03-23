if(UNGAR_BUILD_TESTS
   OR UNGAR_ENABLE_LOGGING
   OR UNGAR_ENABLE_AUTODIFF
   OR UNGAR_ENABLE_OPTIMIZATION)
  include(FetchContent)
endif()
if(UNGAR_BUILD_TESTS)
  include(GoogleTest)
endif()

if(NOT TARGET Eigen3::Eigen)
  add_subdirectory(eigen3)
endif()

if(NOT TARGET Boost::hana)
  add_subdirectory(hana)
endif()

if(UNGAR_BUILD_TESTS)
  message(STATUS "Fetching googletest...")
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE)
  FetchContent_MakeAvailable(googletest)
endif()

if(UNGAR_ENABLE_LOGGING)
  message(STATUS "Fetching spdlog...")
  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.9.2
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE)
  option(SPDLOG_BUILD_SHARED OFF)
  option(SPDLOG_BUILD_ALL OFF)
  FetchContent_MakeAvailable(spdlog)
endif()

if(UNGAR_ENABLE_AUTODIFF)
  message(STATUS "Fetching finitediff...")
  FetchContent_Declare(
    finitediff
    GIT_REPOSITORY https://github.com/zfergus/finite-diff.git
    GIT_TAG main
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE)
  option(FINITE_DIFF_BUILD_UNIT_TESTS OFF)
  FetchContent_MakeAvailable(finitediff)
endif()

if(UNGAR_ENABLE_OPTIMIZATION)
  message(STATUS "Fetching osqpcpp...")
  FetchContent_Declare(
    osqpcpp
    GIT_REPOSITORY git@github.com:google/osqp-cpp.git
    GIT_TAG master
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE)
  FetchContent_MakeAvailable(osqpcpp)
endif()

if(UNGAR_ENABLE_AUTODIFF)
  configure_file(CMakeLists.txt.in
                 ${CMAKE_BINARY_DIR}/external-build/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} .
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/external-build)
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/external-build)
  find_library(cppad_lib cppad_lib HINTS ${CMAKE_BINARY_DIR}/external/cppad/lib
                                         REQUIRED)
endif()
