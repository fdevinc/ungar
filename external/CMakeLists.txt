# ##############################################################################
# Define utility function.
# ##############################################################################
# Download and install an external project according to a provided configuration
# file.
function(download_external_project external_project_config_file
         external_project_name)
  # Create external project 'CMakeLists.txt' file inside the build folder.
  configure_file(
    ${external_project_config_file}
    "${CMAKE_BINARY_DIR}/external/${external_project_name}/CMakeLists.txt"
    @ONLY)

  # Create external project build directory 'ep-build'.
  file(MAKE_DIRECTORY
       ${CMAKE_BINARY_DIR}/external/${external_project_name}/ep-build)

  # Build external project.
  execute_process(
    COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" ..
    WORKING_DIRECTORY
      ${CMAKE_BINARY_DIR}/external/${external_project_name}/ep-build)
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build .
    WORKING_DIRECTORY
      ${CMAKE_BINARY_DIR}/external/${external_project_name}/ep-build)
endfunction()

# ##############################################################################
# Define dependency directories.
# ##############################################################################
# Eigen.
if(DEFINED Eigen3_ROOT)
  set(EIGEN3_ROOT_DIRECTORY ${Eigen3_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_BUNDLED_EIGEN ON)
  set(BUNDLED_EIGEN_FILENAME
      ${CMAKE_CURRENT_LIST_DIR}/config/eigen/eigen-3.4.0.zip)
  set(EIGEN3_ROOT_DIRECTORY ${CMAKE_BINARY_DIR}/external/eigen/install)
endif()

# Hana.
if(DEFINED Hana_ROOT)
  set(HANA_ROOT_DIRECTORY ${Hana_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_BUNDLED_HANA ON)
  set(BUNDLED_HANA_FILENAME
      ${CMAKE_CURRENT_LIST_DIR}/config/hana/hana-boost-1.84.0.zip)
  set(HANA_ROOT_DIRECTORY ${CMAKE_BINARY_DIR}/external/hana/install)
endif()

# GoogleTest.
if(DEFINED GTest_ROOT)
  set(GTEST_ROOT_DIRECTORY ${GTest_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_GOOGLETEST ON)
  set(GTEST_ROOT_DIRECTORY ${CMAKE_BINARY_DIR}/external/googletest/install)
endif()

# spdlog.
if(DEFINED spdlog_ROOT)
  set(SPDLOG_ROOT_DIRECTORY ${spdlog_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_SPDLOG ON)
  set(SPDLOG_ROOT_DIRECTORY ${CMAKE_BINARY_DIR}/external/spdlog/install)
endif()

# FiniteDiff.
if(DEFINED FiniteDiff_ROOT)
  set(FINITEDIFF_ROOT_DIRECTORY ${FiniteDiff_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_FINITEDIFF ON)
  set(FINITEDIFF_ROOT_DIRECTORY
      ${CMAKE_BINARY_DIR}/external/finite-diff/install)
endif()

# Abseil.
if(DEFINED absl_ROOT)
  set(ABSL_ROOT_DIRECTORY ${absl_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_ABSEIL ON)
  set(ABSL_ROOT_DIRECTORY ${CMAKE_BINARY_DIR}/external/abseil-cpp/install)
endif()

# OSQP.
if(DEFINED osqp_ROOT)
  set(OSQP_ROOT_DIRECTORY ${osqp_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_OSQP ON)
  set(OSQP_ROOT_DIRECTORY ${CMAKE_BINARY_DIR}/external/osqp/install)
endif()

# osqp-cpp.
if(DEFINED osqp-cpp_ROOT)
  set(OSQP_CPP_ROOT_DIRECTORY ${osqp-cpp_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_OSQPCPP ON)
  set(OSQP_CPP_ROOT_DIRECTORY ${CMAKE_BINARY_DIR}/external/osqp-cpp/install)
endif()

# CppAD.
if(DEFINED CppAD_ROOT)
  set(CPPAD_ROOT_DIRECTORY ${CppAD_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_CPPAD ON)
  set(CPPAD_ROOT_DIRECTORY ${CMAKE_BINARY_DIR}/external/CppAD/install)
endif()

# CppADCodeGen.
if(DEFINED CppADCodeGen_ROOT)
  set(CPPADCODEGEN_ROOT_DIRECTORY ${CppADCodeGen_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_CPPADCODEGEN ON)
  set(CPPADCODEGEN_ROOT_DIRECTORY
      ${CMAKE_BINARY_DIR}/external/CppADCodeGen/install)
endif()

# ===========================================================================
# 1) CORE DEPENDENCIES
# ##############################################################################
# Add Eigen.
# ##############################################################################
message(STATUS "--------------------------------------------------")
if(USE_BUNDLED_EIGEN)
  message(STATUS "Using bundled Eigen...")
  download_external_project(config/eigen/CMakeLists.txt.in eigen)
  find_package(Eigen3 REQUIRED PATHS
               ${EIGEN3_ROOT_DIRECTORY}/share/eigen3/cmake NO_DEFAULT_PATH)
else()
  message(STATUS "Using system-wide Eigen...")
  find_package(Eigen3 3.4.0 REQUIRED NO_MODULE)
endif()
set_target_properties(Eigen3::Eigen PROPERTIES IMPORTED_GLOBAL TRUE)

# If the Eigen package is found, it defines the variable 'EIGEN3_ROOT_DIR'.
# However, this feature is deprecated and 'EIGEN3_ROOT_DIR' may not be defined.
if(DEFINED EIGEN3_ROOT_DIR)
  message(STATUS "Eigen found at '${EIGEN3_ROOT_DIR}'.")
elseif(DEFINED EIGEN3_ROOT_DIRECTORY)
  message(STATUS "Eigen found at '${EIGEN3_ROOT_DIRECTORY}'.")
else()
  message(STATUS "Eigen found.")
endif()

# ##############################################################################
# Add Hana.
# ##############################################################################
message(STATUS "--------------------------------------------------")
if(USE_BUNDLED_HANA)
  message(STATUS "Using bundled Hana...")
  download_external_project(config/hana/CMakeLists.txt.in hana)
  find_package(Hana REQUIRED PATHS ${HANA_ROOT_DIRECTORY}/lib/cmake/hana
               NO_DEFAULT_PATH)
else()
  message(STATUS "Using system-wide Hana...")
  find_package(Hana REQUIRED)
endif()
set_target_properties(hana PROPERTIES IMPORTED_GLOBAL TRUE)

if(DEFINED HANA_ROOT_DIRECTORY)
  message(STATUS "Hana found at '${HANA_ROOT_DIRECTORY}'.")
else()
  message(STATUS "Hana found.")
endif()

# ===========================================================================
# 2) TESTING
if(UNGAR_BUILD_TESTS)
  # ############################################################################
  # Add GoogleTest.
  # ############################################################################
  message(STATUS "--------------------------------------------------")
  include(GoogleTest)

  if(USE_DOWNLOADED_GOOGLETEST)
    message(STATUS "Using locally downloaded GoogleTest...")
    download_external_project(config/googletest/CMakeLists.txt.in googletest)
    find_package(GTest CONFIG REQUIRED PATHS
                 ${GTEST_ROOT_DIRECTORY}/lib/cmake/GTest NO_DEFAULT_PATH)
  else()
    message(STATUS "Using system-wide GoogleTest...")
    find_package(GTest CONFIG REQUIRED)
  endif()
  set_target_properties(GTest::gtest PROPERTIES IMPORTED_GLOBAL TRUE)

  if(DEFINED GTEST_ROOT_DIRECTORY)
    message(STATUS "GoogleTest found at '${GTEST_ROOT_DIRECTORY}'.")
  else()
    message(STATUS "GoogleTest found.")
  endif()

endif(UNGAR_BUILD_TESTS)

# ===========================================================================
# 3) LOGGING
if(UNGAR_ENABLE_LOGGING)
  # ############################################################################
  # Add spdlog.
  # ############################################################################
  message(STATUS "--------------------------------------------------")
  if(USE_DOWNLOADED_SPDLOG)
    message(STATUS "Using locally downloaded spdlog...")
    download_external_project(config/spdlog/CMakeLists.txt.in spdlog)
    find_package(spdlog REQUIRED PATHS
                 ${SPDLOG_ROOT_DIRECTORY}/lib/cmake/spdlog NO_DEFAULT_PATH)
  else()
    message(STATUS "Using system-wide spdlog...")
    find_package(spdlog REQUIRED)
  endif()
  set_target_properties(spdlog::spdlog PROPERTIES IMPORTED_GLOBAL TRUE)
  set_target_properties(spdlog::spdlog_header_only PROPERTIES IMPORTED_GLOBAL
                                                              TRUE)

  if(DEFINED SPDLOG_ROOT_DIRECTORY)
    message(STATUS "spdlog found at '${SPDLOG_ROOT_DIRECTORY}'.")
  else()
    message(STATUS "spdlog found.")
  endif()

endif(UNGAR_ENABLE_LOGGING)

# ===========================================================================
# 4) AUTOMATIC DIFFERENTIATION
if(UNGAR_ENABLE_AUTODIFF)
  # ############################################################################
  # Add FiniteDiff.
  # ############################################################################
  message(STATUS "--------------------------------------------------")
  if(USE_DOWNLOADED_FINITEDIFF)
    message(STATUS "Using locally downloaded FiniteDiff...")
    download_external_project(config/finite-diff/CMakeLists.txt.in finite-diff)
    find_package(
      FiniteDiff REQUIRED PATHS
      ${FINITEDIFF_ROOT_DIRECTORY}/lib/cmake/finitediff NO_DEFAULT_PATH)
  else()
    message(STATUS "Using system-wide FiniteDiff...")
    find_package(FiniteDiff REQUIRED)
  endif()
  set_target_properties(finitediff::finitediff PROPERTIES IMPORTED_GLOBAL TRUE)

  if(DEFINED FINITEDIFF_ROOT_DIRECTORY)
    message(STATUS "FiniteDiff found at '${FINITEDIFF_ROOT_DIRECTORY}'.")
  else()
    message(STATUS "FiniteDiff found.")
  endif()

  # ############################################################################
  # Add CppAD.
  # ############################################################################
  message(STATUS "--------------------------------------------------")
  if(USE_DOWNLOADED_CPPAD)
    message(STATUS "Using locally downloaded CppAD...")
    download_external_project(config/CppAD/CMakeLists.txt.in CppAD)
  else()
    message(STATUS "Using system-wide CppAD...")
  endif()

  if(DEFINED CPPAD_ROOT_DIRECTORY)
    set(CPPAD_LIB_DIRECTORY ${CPPAD_ROOT_DIRECTORY}/lib)
    set(CPPAD_INCLUDE_DIRECTORY ${CPPAD_ROOT_DIRECTORY}/include)

    find_library(
      CppAD_LIBRARY cppad_lib
      PATHS ${CPPAD_LIB_DIRECTORY} REQUIRED
      NO_DEFAULT_PATH)
    find_path(
      CppAD_INCLUDE_DIR cppad/cppad.hpp
      PATHS ${CPPAD_INCLUDE_DIRECTORY} REQUIRED
      NO_DEFAULT_PATH)
  else()
    find_library(CppAD_LIBRARY cppad_lib REQUIRED)
    find_path(CppAD_INCLUDE_DIR cppad/cppad.hpp REQUIRED)
  endif()

  if(DEFINED CPPAD_ROOT_DIRECTORY)
    message(STATUS "CppAD found at '${CPPAD_ROOT_DIRECTORY}'.")
  else()
    message(STATUS "CppAD found.")
  endif()

  # Create target for CppAD.
  add_library(CppAD SHARED IMPORTED GLOBAL)
  add_library(CppAD::CppAD ALIAS CppAD)
  target_include_directories(CppAD INTERFACE ${CppAD_INCLUDE_DIR})

  get_filename_component(CppAD_LIBRARY_NAME ${CppAD_LIBRARY} NAME)
  get_filename_component(CppAD_LIBRARY_REAL_PATH ${CppAD_LIBRARY} REALPATH)

  set_target_properties(CppAD PROPERTIES IMPORTED_SONAME ${CppAD_LIBRARY_NAME})
  set_target_properties(CppAD PROPERTIES IMPORTED_LOCATION
                                         ${CppAD_LIBRARY_REAL_PATH})

  # ############################################################################
  # Add CppADCodeGen.
  # ############################################################################
  message(STATUS "--------------------------------------------------")
  if(USE_DOWNLOADED_CPPADCODEGEN)
    message(STATUS "Using locally downloaded CppADCodeGen...")
    download_external_project(config/CppADCodeGen/CMakeLists.txt.in
                              CppADCodeGen)
  else()
    message(STATUS "Using system-wide CppADCodeGen...")
  endif()

  if(DEFINED CPPADCODEGEN_ROOT_DIRECTORY)
    set(CPPADCODEGEN_INCLUDE_DIRECTORY ${CPPADCODEGEN_ROOT_DIRECTORY}/include)
    find_path(
      CppADCodeGen_INCLUDE_DIR cppad/cg.hpp
      PATHS ${CPPADCODEGEN_INCLUDE_DIRECTORY} REQUIRED
      NO_DEFAULT_PATH)
  else()
    find_path(CppADCodeGen_INCLUDE_DIR cppad/cg.hpp REQUIRED)
  endif()

  if(DEFINED CPPADCODEGEN_ROOT_DIRECTORY)
    message(STATUS "CppADCodeGen found at '${CPPADCODEGEN_ROOT_DIRECTORY}'.")
  else()
    message(STATUS "CppADCodeGen found.")
  endif()

  # Create target for CppADCodeGen.
  add_library(CppADCodeGen INTERFACE IMPORTED GLOBAL)
  add_library(CppADCodeGen::CppADCodeGen ALIAS CppADCodeGen)
  target_include_directories(CppADCodeGen INTERFACE ${CppADCodeGen_INCLUDE_DIR})

endif(UNGAR_ENABLE_AUTODIFF)

# ===========================================================================
# 5) OPTIMIZATION
if(UNGAR_ENABLE_OPTIMIZATION)
  # ############################################################################
  # Add Abseil.
  # ############################################################################
  message(STATUS "--------------------------------------------------")
  if(USE_DOWNLOADED_ABSEIL)
    message(STATUS "Using locally downloaded Abseil...")
    download_external_project(config/abseil-cpp/CMakeLists.txt.in abseil-cpp)
    find_package(absl REQUIRED PATHS ${ABSL_ROOT_DIRECTORY}/lib/cmake/absl
                 NO_DEFAULT_PATH)
  else()
    message(STATUS "Using system-wide Abseil...")
    find_package(absl REQUIRED)
  endif()

  if(DEFINED ABSL_ROOT_DIRECTORY)
    message(STATUS "Abseil found at '${ABSL_ROOT_DIRECTORY}'.")
  else()
    message(STATUS "Abseil found.")
  endif()

  # ############################################################################
  # Add OSQP.
  # ############################################################################
  message(STATUS "--------------------------------------------------")
  if(USE_DOWNLOADED_OSQP)
    message(STATUS "Using locally downloaded OSQP...")
    download_external_project(config/osqp/CMakeLists.txt.in osqp)
    find_package(osqp REQUIRED PATHS ${OSQP_ROOT_DIRECTORY}/lib/cmake/osqp
                 NO_DEFAULT_PATH)
  else()
    message(STATUS "Using system-wide OSQP...")
    find_package(osqp REQUIRED)
  endif()

  if(DEFINED OSQP_ROOT_DIRECTORY)
    message(STATUS "OSQP found at '${OSQP_ROOT_DIRECTORY}'.")
  else()
    message(STATUS "OSQP found.")
  endif()

  # ############################################################################
  # Add osqp-cpp.
  # ############################################################################
  message(STATUS "--------------------------------------------------")
  if(USE_DOWNLOADED_OSQPCPP)
    message(STATUS "Using locally downloaded osqp-cpp...")
    download_external_project(config/osqp-cpp/CMakeLists.txt.in osqp-cpp)
    find_package(osqp-cpp REQUIRED PATHS
                 ${OSQP_CPP_ROOT_DIRECTORY}/lib/cmake/osqp-cpp NO_DEFAULT_PATH)
  else()
    message(STATUS "Using system-wide osqp-cpp...")
    find_package(osqp-cpp REQUIRED)
  endif()
  set_target_properties(osqp-cpp::osqp-cpp PROPERTIES IMPORTED_GLOBAL TRUE)

  if(DEFINED OSQP_CPP_ROOT_DIRECTORY)
    message(STATUS "osqp-cpp found at '${OSQP_CPP_ROOT_DIRECTORY}'.")
  else()
    message(STATUS "osqp-cpp found.")
  endif()

endif(UNGAR_ENABLE_OPTIMIZATION)
