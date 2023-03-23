add_executable(variable.example variable.example.cpp)
target_link_libraries(variable.example ungar::ungar)

add_executable(variable_map.example variable_map.example.cpp)
target_link_libraries(variable_map.example ungar::ungar)

if(UNGAR_ENABLE_AUTODIFF)
  add_executable(function.example autodiff/function.example.cpp)
  target_link_libraries(function.example ungar::ungar)
endif()
