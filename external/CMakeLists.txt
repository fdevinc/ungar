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
    "${CMAKE_CURRENT_BINARY_DIR}/${external_project_name}/CMakeLists.txt" @ONLY)

  # Create external project build directory 'ep-build'.
  file(MAKE_DIRECTORY
       ${CMAKE_CURRENT_BINARY_DIR}/${external_project_name}/ep-build)

  # Build external project.
  execute_process(
    COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" ..
    WORKING_DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}/${external_project_name}/ep-build)
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build .
    WORKING_DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}/${external_project_name}/ep-build)
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
  set(EIGEN3_ROOT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/eigen/install)
endif()

# Hana.
if(DEFINED Hana_ROOT)
  set(HANA_ROOT_DIRECTORY ${Hana_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_BUNDLED_HANA ON)
  set(BUNDLED_HANA_FILENAME
      ${CMAKE_CURRENT_LIST_DIR}/config/hana/hana-boost-1.84.0.zip)
  set(HANA_ROOT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/hana/install)
endif()

# Preprocessor.
if(DEFINED Preprocessor_ROOT)
  set(PREPROCESSOR_ROOT_DIRECTORY ${Preprocessor_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_BUNDLED_PREPROCESSOR ON)
  set(BUNDLED_PREPROCESSOR_FILENAME
      ${CMAKE_CURRENT_LIST_DIR}/config/preprocessor/preprocessor-1.84.0-ungar.zip
  )
  set(PREPROCESSOR_ROOT_DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}/preprocessor/install)
endif()

# GoogleTest.
if(DEFINED GTest_ROOT)
  set(GTEST_ROOT_DIRECTORY ${GTest_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_GOOGLETEST ON)
  set(GTEST_ROOT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest/install)
endif()

# spdlog.
if(DEFINED spdlog_ROOT)
  set(SPDLOG_ROOT_DIRECTORY ${spdlog_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_SPDLOG ON)
  set(SPDLOG_ROOT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/spdlog/install)
endif()

# FiniteDiff.
if(DEFINED FiniteDiff_ROOT)
  set(FINITEDIFF_ROOT_DIRECTORY ${FiniteDiff_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_FINITEDIFF ON)
  set(FINITEDIFF_ROOT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/finite-diff/install)
endif()

# CppAD.
if(DEFINED CppAD_ROOT)
  set(CPPAD_ROOT_DIRECTORY ${CppAD_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_CPPAD ON)
  set(CPPAD_ROOT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CppAD/install)
endif()

# CppADCodeGen.
if(DEFINED CppADCodeGen_ROOT)
  set(CPPADCODEGEN_ROOT_DIRECTORY ${CppADCodeGen_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_CPPADCODEGEN ON)
  set(CPPADCODEGEN_ROOT_DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}/CppADCodeGen/install)
endif()

# Abseil.
if(DEFINED absl_ROOT)
  set(ABSL_ROOT_DIRECTORY ${absl_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_ABSEIL ON)
  set(ABSL_ROOT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/abseil-cpp/install)
endif()

# OSQP.
if(DEFINED osqp_ROOT)
  set(OSQP_ROOT_DIRECTORY ${osqp_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_OSQP ON)
  set(OSQP_ROOT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/osqp/install)
endif()

# osqp-cpp.
if(DEFINED osqp-cpp_ROOT)
  set(OSQP_CPP_ROOT_DIRECTORY ${osqp-cpp_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_OSQP_CPP ON)
  set(OSQP_CPP_ROOT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/osqp-cpp/install)
endif()

# Boost.
if(DEFINED Boost_ROOT)
  set(BOOST_ROOT_DIRECTORY ${Boost_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_BOOST ON)
  set(BOOST_ROOT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/boost/install)
endif()

# urdfdom_headers.
if(DEFINED urdfdom_headers_ROOT)
  set(URDFDOM_HEADERS_ROOT_DIRECTORY ${urdfdom_headers_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_URDFDOM_HEADERS ON)
  set(URDFDOM_HEADERS_ROOT_DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}/urdfdom_headers/install)
endif()

# console_bridge.
if(DEFINED console_bridge_ROOT)
  set(CONSOLE_BRIDGE_ROOT_DIRECTORY ${console_bridge_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_CONSOLE_BRIDGE ON)
  set(CONSOLE_BRIDGE_ROOT_DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}/console_bridge/install)
endif()

# TinyXML2.
if(DEFINED TinyXML2_ROOT)
  set(TINYXML2_ROOT_DIRECTORY ${TinyXML2_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_TINYXML2 ON)
  set(TINYXML2_ROOT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tinyxml2/install)
endif()

# urdfdom.
if(DEFINED urdfdom_ROOT)
  set(URDFDOM_ROOT_DIRECTORY ${urdfdom_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_URDFDOM ON)
  set(URDFDOM_ROOT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/urdfdom/install)
endif()

# Pinocchio.
if(DEFINED pinocchio_ROOT)
  set(PINOCCHIO_ROOT_DIRECTORY ${pinocchio_ROOT})
elseif(NOT UNGAR_USE_SYSTEM_LIBRARIES)
  set(USE_DOWNLOADED_PINOCCHIO ON)
  set(PINOCCHIO_ROOT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pinocchio/install)
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
  find_package(Eigen3 3.4.0 REQUIRED CONFIG)
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

# ##############################################################################
# Add Preprocessor.
# ##############################################################################
message(STATUS "--------------------------------------------------")
if(USE_BUNDLED_PREPROCESSOR)
  message(STATUS "Using bundled Preprocessor...")
  download_external_project(config/preprocessor/CMakeLists.txt.in preprocessor)
  find_package(
    Preprocessor REQUIRED PATHS
    ${PREPROCESSOR_ROOT_DIRECTORY}/lib/cmake/preprocessor NO_DEFAULT_PATH)
else()
  message(STATUS "Using system-wide Preprocessor...")
  find_package(Preprocessor REQUIRED)
endif()
set_target_properties(Boost::preprocessor PROPERTIES IMPORTED_GLOBAL TRUE)

if(DEFINED PREPROCESSOR_ROOT_DIRECTORY)
  message(STATUS "Preprocessor found at '${PREPROCESSOR_ROOT_DIRECTORY}'.")
else()
  message(STATUS "Preprocessor found.")
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
  if(USE_DOWNLOADED_OSQP_CPP)
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

# ===========================================================================
# 5) PINOCCHIO
if(UNGAR_ENABLE_PINOCCHIO)
  # ############################################################################
  # Add Boost.
  # ############################################################################
  message(STATUS "--------------------------------------------------")
  if(USE_DOWNLOADED_BOOST)
    message(STATUS "Using locally downloaded Boost...")
    download_external_project(config/boost/CMakeLists.txt.in boost)
    set(Boost_USE_STATIC_LIBS ON)
    set(Boost_USE_STATIC_RUNTIME ON)
    find_package(Boost REQUIRED PATHS
                 ${BOOST_ROOT_DIRECTORY}/lib/cmake/Boost-1.84.0 NO_DEFAULT_PATH)
  else()
    message(STATUS "Using system-wide Boost...")
    find_package(Boost 1.77.0 REQUIRED)
  endif()
  set_target_properties(Boost::boost PROPERTIES IMPORTED_GLOBAL TRUE)

  if(DEFINED BOOST_ROOT_DIRECTORY)
    message(STATUS "Boost found at '${BOOST_ROOT_DIRECTORY}'.")
  else()
    message(STATUS "Boost found.")
  endif()

  # ############################################################################
  # Add urdfdom_headers.
  # ############################################################################
  message(STATUS "--------------------------------------------------")
  if(USE_DOWNLOADED_URDFDOM_HEADERS)
    message(STATUS "Using locally downloaded urdfdom_headers...")
    download_external_project(config/urdfdom_headers/CMakeLists.txt.in
                              urdfdom_headers)
    find_package(
      urdfdom_headers REQUIRED PATHS
      ${URDFDOM_HEADERS_ROOT_DIRECTORY}/lib/urdfdom_headers/cmake
      NO_DEFAULT_PATH)
  else()
    message(STATUS "Using system-wide urdfdom_headers...")
    find_package(urdfdom_headers REQUIRED)
  endif()

  if(DEFINED URDFDOM_HEADERS_ROOT_DIRECTORY)
    message(
      STATUS "urdfdom_headers found at '${URDFDOM_HEADERS_ROOT_DIRECTORY}'.")
  else()
    message(STATUS "urdfdom_headers found.")
  endif()

  # ############################################################################
  # Add console_bridge.
  # ############################################################################
  message(STATUS "--------------------------------------------------")
  if(USE_DOWNLOADED_CONSOLE_BRIDGE)
    message(STATUS "Using locally downloaded console_bridge...")
    download_external_project(config/console_bridge/CMakeLists.txt.in
                              console_bridge)
    find_package(
      console_bridge REQUIRED PATHS
      ${CONSOLE_BRIDGE_ROOT_DIRECTORY}/lib/console_bridge/cmake NO_DEFAULT_PATH)
  else()
    message(STATUS "Using system-wide console_bridge...")
    find_package(console_bridge REQUIRED)
  endif()

  if(DEFINED CONSOLE_BRIDGE_ROOT_DIRECTORY)
    message(
      STATUS "console_bridge found at '${CONSOLE_BRIDGE_ROOT_DIRECTORY}'.")
  else()
    message(STATUS "console_bridge found.")
  endif()

  # ############################################################################
  # Add TinyXML2.
  # ############################################################################
  message(STATUS "--------------------------------------------------")
  if(USE_DOWNLOADED_TINYXML2)
    message(STATUS "Using locally downloaded TinyXML2...")
    download_external_project(config/tinyxml2/CMakeLists.txt.in tinyxml2)
    find_package(TinyXML2 REQUIRED PATHS
                 ${TINYXML2_ROOT_DIRECTORY}/lib/cmake/tinyxml2 NO_DEFAULT_PATH)
  else()
    message(STATUS "Using system-wide TinyXML2...")
    find_package(TinyXML2 REQUIRED)
  endif()

  if(DEFINED TINYXML2_ROOT_DIRECTORY)
    message(STATUS "TinyXML2 found at '${TINYXML2_ROOT_DIRECTORY}'.")
  else()
    message(STATUS "TinyXML2 found.")
  endif()

  # ############################################################################
  # Add urdfdom.
  # ############################################################################
  message(STATUS "--------------------------------------------------")
  if(USE_DOWNLOADED_URDFDOM)
    message(STATUS "Using locally downloaded urdfdom...")
    download_external_project(config/urdfdom/CMakeLists.txt.in urdfdom)

    # Set preferred paths for urdfdom's dependency.
    set(urdfdom_headers_ROOT ${URDFDOM_HEADERS_ROOT_DIRECTORY})

    find_package(urdfdom REQUIRED PATHS
                 ${URDFDOM_ROOT_DIRECTORY}/lib/urdfdom/cmake NO_DEFAULT_PATH)
  else()
    message(STATUS "Using system-wide urdfdom...")
    find_package(urdfdom REQUIRED)
  endif()

  if(DEFINED URDFDOM_ROOT_DIRECTORY)
    message(STATUS "urdfdom found at '${URDFDOM_ROOT_DIRECTORY}'.")
  else()
    message(STATUS "urdfdom found.")
  endif()

  # ############################################################################
  # Add Pinocchio.
  # ############################################################################
  message(STATUS "--------------------------------------------------")
  if(USE_DOWNLOADED_PINOCCHIO)
    message(STATUS "Using locally downloaded Pinocchio...")
    download_external_project(config/pinocchio/CMakeLists.txt.in pinocchio)

    # Set C++ standard and preferred paths for Pinocchio's dependencies.
    set(CMAKE_CXX_STANDARD 20)
    set(Boost_NO_WARN_NEW_VERSIONS ON)

    set(Boost_ROOT ${BOOST_ROOT_DIRECTORY})
    set(Eigen3_ROOT ${EIGEN3_ROOT_DIRECTORY})
    set(console_bridge_ROOT ${CONSOLE_BRIDGE_ROOT_DIRECTORY})
    set(urdfdom_headers_ROOT ${URDFDOM_HEADERS_ROOT_DIRECTORY})
    set(urdfdom_ROOT ${URDFDOM_ROOT_DIRECTORY})

    find_package(
      pinocchio REQUIRED PATHS ${PINOCCHIO_ROOT_DIRECTORY}/lib/cmake/pinocchio
      NO_DEFAULT_PATH)
  else()
    message(STATUS "Using system-wide Pinocchio...")
    find_package(pinocchio REQUIRED)
  endif()
  set_target_properties(pinocchio::pinocchio PROPERTIES IMPORTED_GLOBAL TRUE)

  if(DEFINED PINOCCHIO_ROOT_DIRECTORY)
    message(STATUS "Pinocchio found at '${PINOCCHIO_ROOT_DIRECTORY}'.")
  else()
    message(STATUS "Pinocchio found.")
  endif()

endif(UNGAR_ENABLE_PINOCCHIO)
