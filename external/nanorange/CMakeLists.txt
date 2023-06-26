add_library(nanorange INTERFACE)

target_include_directories(nanorange SYSTEM
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
