if(UNGAR_BUILD_TESTS
   OR UNGAR_ENABLE_LOGGING
   OR UNGAR_ENABLE_AUTODIFF
   OR UNGAR_ENABLE_OPTIMIZATION)
  include(FetchContent)

  # Prevent warning about 'DOWNLOAD_EXTRACT_TIMESTAMP'.
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
  endif()
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
    URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
    URL_HASH
      SHA256=ffa17fbc5953900994e2deec164bb8949879ea09b411e07f215bfbb1f87f4632)
  set(BUILD_GMOCK OFF)
  FetchContent_MakeAvailable(googletest)
endif()

if(UNGAR_ENABLE_LOGGING)
  message(STATUS "Fetching spdlog...")
  FetchContent_Declare(
    spdlog
    URL https://github.com/gabime/spdlog/archive/refs/tags/v1.9.2.zip
    URL_HASH
      SHA256=130bd593c33e2e2abba095b551db6a05f5e4a5a19c03ab31256c38fa218aa0a6)
  set(SPDLOG_BUILD_SHARED OFF)
  set(SPDLOG_BUILD_ALL OFF)
  FetchContent_MakeAvailable(spdlog)
endif()

if(UNGAR_ENABLE_AUTODIFF)
  message(STATUS "Fetching finitediff...")
  FetchContent_Declare(
    finitediff
    URL https://github.com/zfergus/finite-diff/archive/refs/tags/v1.0.1.zip
    URL_HASH
      SHA256=772cb25e314bd957959ae42e8c00f312e9b4a216c6e49424c57e91f02833e968)
  set(FINITE_DIFF_BUILD_UNIT_TESTS OFF)
  FetchContent_MakeAvailable(finitediff)
endif()

if(UNGAR_ENABLE_OPTIMIZATION)
  # Download osqp to prevent osqpcpp from doing it unsafely.
  if(NOT TARGET osqpstatic)
    message(STATUS "Fetching osqp...")
    FetchContent_Declare(
      osqp
      GIT_REPOSITORY https://github.com/oxfordcontrol/osqp.git
      GIT_TAG v0.6.3
      GIT_SHALLOW TRUE
      GIT_PROGRESS TRUE)
    FetchContent_MakeAvailable(osqp)
  endif()

  message(STATUS "Fetching osqpcpp...")
  FetchContent_Declare(
    osqpcpp
    URL https://github.com/google/osqp-cpp/archive/refs/heads/master.zip
    URL_HASH
      SHA256=5158b39ea402a6f12fec9a01a3b03a6da418d95253a8e7747bad888d3284914f)
  set(OSQP-CPP_BUILD_TESTS OFF)
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
