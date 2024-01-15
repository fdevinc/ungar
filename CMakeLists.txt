cmake_minimum_required(VERSION 3.20)

project(Ungar VERSION 1.0.1)

add_library(ungar INTERFACE)
add_library(ungar::ungar ALIAS ungar)
target_compile_features(ungar INTERFACE cxx_std_20)

# ##############################################################################
# Set up CMake options.
# ##############################################################################
include(CMakeDependentOption)

option(
  UNGAR_USE_SYSTEM_LIBRARIES
  "Prevent automatic download of missing external dependencies. To selectively use system-wide packages, provide their installation prefixes as CMake variables of the form '<PackageName>_ROOT', where <PackageName> is the (case-preserved) name given to the find_package() call and _ROOT is literal."
  OFF)
option(UNGAR_BUILD_TESTS "Add targets for unit testing." OFF)
option(UNGAR_BUILD_EXAMPLES "Add targets showing Ungar's features." OFF)
option(UNGAR_INSTALL "Install CMake targets during install step." OFF)
option(UNGAR_ENABLE_OPTIMIZATION
       "Enable Ungar's nonlinear programming interface." OFF)
cmake_dependent_option(
  UNGAR_ENABLE_LOGGING
  "Enable Ungar's logging functionalities. Automatically enabled if any of the flags 'UNGAR_ENABLE_AUTODIFF', 'UNGAR_ENABLE_OPTIMIZATION', and 'UNGAR_BUILD_EXAMPLES' is set to 'ON'."
  OFF
  "NOT UNGAR_ENABLE_AUTODIFF;NOT UNGAR_ENABLE_OPTIMIZATION;NOT UNGAR_BUILD_EXAMPLES"
  ON)
cmake_dependent_option(
  UNGAR_ENABLE_AUTODIFF
  "Enable Ungar's autodiff interface. Automatically enabled if 'UNGAR_ENABLE_OPTIMIZATION' is set to 'ON'."
  OFF
  "NOT UNGAR_ENABLE_OPTIMIZATION"
  ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(STATUS "--------------------------------------------------")
message(STATUS "Configuring Ungar with the following options:")
message(STATUS "    Enable logging         : ${UNGAR_ENABLE_LOGGING}")
message(STATUS "    Enable autodiff        : ${UNGAR_ENABLE_AUTODIFF}")
message(STATUS "    Enable optimization    : ${UNGAR_ENABLE_OPTIMIZATION}")
message(STATUS "    Build examples         : ${UNGAR_BUILD_EXAMPLES}")
message(STATUS "    Build tests            : ${UNGAR_BUILD_TESTS}")
message(STATUS "    Install                : ${UNGAR_INSTALL}")
message(STATUS "    Use system libraries   : ${UNGAR_USE_SYSTEM_LIBRARIES}")

# Add external dependencies.
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
add_subdirectory(external)
message(STATUS "--------------------------------------------------")

# ##############################################################################
# Add compile options.
# ##############################################################################
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang") # Clang.
  target_compile_options(ungar INTERFACE "-pedantic" "$<$<CONFIG:RELEASE>:-O2>")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU") # GNU Compiler Collection (GCC).
  target_compile_options(
    ungar
    INTERFACE "-pedantic"
              "-ftemplate-backtrace-limit=1"
              "-fconstexpr-depth=2147483647"
              "-fconstexpr-loop-limit=2147483647"
              "-fconstexpr-cache-depth=2147483647"
              "-fconstexpr-ops-limit=2147483647"
              "$<$<CONFIG:RELEASE>:-O3>")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel") # Intel C++ Compiler.
  message(WARNING "Intel C++ Compiler is not supported.")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") # MSVC.
  target_compile_options(ungar INTERFACE "/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
  message(WARNING "${CMAKE_CXX_COMPILER_ID} is not supported.")
endif()

# ##############################################################################
# Set Ungar's include directories and link libraries.
# ##############################################################################
# Core modules.
list(APPEND UNGAR_INCLUDE_DIRECTORIES
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
     $<INSTALL_INTERFACE:include>)
list(APPEND UNGAR_LINK_LIBRARIES Eigen3::Eigen hana)

# Optional modules.
if(UNGAR_ENABLE_LOGGING)
  list(APPEND UNGAR_LINK_LIBRARIES spdlog::spdlog)
endif()
if(UNGAR_ENABLE_AUTODIFF)
  list(APPEND UNGAR_LINK_LIBRARIES CppAD::CppAD CppADCodeGen::CppADCodeGen
       ${CMAKE_DL_LIBS} finitediff::finitediff)
endif()
if(UNGAR_ENABLE_OPTIMIZATION)
  list(APPEND UNGAR_LINK_LIBRARIES osqp-cpp::osqp-cpp)
endif()

target_include_directories(ungar INTERFACE ${UNGAR_INCLUDE_DIRECTORIES})
target_link_libraries(ungar INTERFACE ${UNGAR_LINK_LIBRARIES})

# ##############################################################################
# Set Ungar's compile definitions.
# ##############################################################################
target_compile_definitions(
  ungar
  INTERFACE $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>
            $<$<CONFIG:RelWithDebInfo>:_GLIBCXX_ASSERTIONS>
            $<$<CONFIG:MinSizeRel>:UNGAR_CONFIG_ENABLE_RELEASE_MODE>
            $<$<CONFIG:Release>:UNGAR_CONFIG_ENABLE_RELEASE_MODE>)

if(UNGAR_ENABLE_LOGGING)
  target_compile_definitions(ungar INTERFACE -DUNGAR_CONFIG_ENABLE_LOGGING)
endif()

if(UNGAR_ENABLE_AUTODIFF)
  target_compile_definitions(
    ungar
    INTERFACE -DUNGAR_CONFIG_ENABLE_AUTODIFF
    INTERFACE "UNGAR_CODEGEN_FOLDER=\"${CMAKE_BINARY_DIR}/ungar_codegen\"")
endif()

if(UNGAR_ENABLE_OPTIMIZATION)
  target_compile_definitions(ungar INTERFACE -DUNGAR_CONFIG_ENABLE_OPTIMIZATION)
endif()

# ##############################################################################
# Add optional targets.
# ##############################################################################
if(UNGAR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(UNGAR_BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

if(UNGAR_INSTALL)
  if(UNGAR_ENABLE_AUTODIFF)
    if(NOT DEFINED CppAD_ROOT)
      message(
        STATUS
          "To install Ungar, the prefix path of 'CppAD' must be provided as a CMake variable named 'CppAD_ROOT'. Using default value: '/usr/local'."
      )
      set(CppAD_ROOT /usr/local)
    endif()
    if(NOT DEFINED CppADCodeGen_ROOT)
      message(
        STATUS
          "To install Ungar, the prefix path of 'CppADCodeGen' must be provided as a CMake variable named 'CppADCodeGen_ROOT'. Using default value: '/usr/local'."
      )
      set(CppADCodeGen_ROOT /usr/local)
    endif()
  endif(UNGAR_ENABLE_AUTODIFF)

  include(GNUInstallDirs)

  # Install library headers.
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  # Install library binaries.
  install(
    TARGETS ungar
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  # Install library configuration files.
  include(CMakePackageConfigHelpers)

  configure_package_config_file(
    UngarConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ungar)

  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY SameMajorVersion)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ungar)

  # Perform installation.
  install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ungar::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ungar)
endif(UNGAR_INSTALL)
