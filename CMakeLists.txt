cmake_minimum_required(VERSION 3.20)

project(Ungar VERSION 1.0.0)

add_library(ungar INTERFACE)
add_library(ungar::ungar ALIAS ungar)
target_compile_features(ungar INTERFACE cxx_std_17)

# ##############################################################################
# Set up CMake options.
# ##############################################################################
option(UNGAR_BUILD_TESTS "Add targets for unit testing." OFF)
option(UNGAR_BUILD_EXAMPLES "Add targets showing Ungar's features." OFF)
option(UNGAR_ENABLE_LOGGING "Enable Ungar's logging functionalities." OFF)
option(UNGAR_ENABLE_AUTODIFF "Enable Ungar's autodiff interface." OFF)
option(UNGAR_ENABLE_OPTIMIZATION
       "Enable Ungar's nonlinear programming interface." OFF)

if(UNGAR_ENABLE_AUTODIFF AND NOT UNGAR_ENABLE_LOGGING)
  message(
    FATAL_ERROR
      "Ungar's autodiff interface requires logging functionalities."
      " To enable logging, set the CMake flag UNGAR_ENABLE_LOGGING to 'ON'.")
endif()
if(UNGAR_ENABLE_OPTIMIZATION
   AND NOT UNGAR_ENABLE_LOGGING
   AND NOT UNGAR_ENABLE_AUTODIFF)
  message(
    FATAL_ERROR
      "Ungar's optimization interface requires logging and autodiff functionalities."
      " To enable both, set the CMake flags UNGAR_ENABLE_LOGGING and UNGAR_ENABLE_AUTODIFF to 'ON'."
  )
endif()
if(UNGAR_BUILD_EXAMPLES AND NOT UNGAR_ENABLE_LOGGING)
  message(
    FATAL_ERROR
      "Ungar's examples require logging functionalities."
      " To enable logging, set the CMake flag UNGAR_ENABLE_LOGGING to 'ON'.")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add external dependencies.
add_subdirectory(external)

# ##############################################################################
# Add compile options.
# ##############################################################################
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang") # Clang.
  target_compile_options(ungar INTERFACE "-pedantic" "$<$<CONFIG:RELEASE>:-O2>")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU") # GNU Compiler Collection (GCC).
  target_compile_options(ungar INTERFACE "-pedantic" "$<$<CONFIG:RELEASE>:-O3>")
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
list(APPEND UNGAR_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")
list(APPEND UNGAR_LINK_LIBRARIES gcem nanorange Eigen3::Eigen Boost::hana)

# Optional modules.
if(UNGAR_ENABLE_LOGGING)
  list(APPEND UNGAR_LINK_LIBRARIES spdlog::spdlog)
endif()
if(UNGAR_ENABLE_AUTODIFF)
  list(APPEND UNGAR_INCLUDE_DIRECTORIES
       "${CMAKE_BINARY_DIR}/external/cppad/include"
       "${CMAKE_BINARY_DIR}/external/cppadcodegen/include")
  list(APPEND UNGAR_LINK_LIBRARIES ${cppad_lib} ${CMAKE_DL_LIBS}
       finitediff::finitediff)
endif()
if(UNGAR_ENABLE_OPTIMIZATION)
  list(APPEND UNGAR_LINK_LIBRARIES osqp-cpp)
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
  add_subdirectory(test)
endif()

if(UNGAR_BUILD_EXAMPLES)
  add_subdirectory(example)
endif()
