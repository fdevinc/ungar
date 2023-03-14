add_library(boost_hana INTERFACE)
add_library(Boost::hana ALIAS boost_hana)

target_include_directories(boost_hana
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
