add_executable(variable.example variable.example.cpp)
target_link_libraries(variable.example ungar::ungar)

add_executable(variable_map.example variable_map.example.cpp)
target_link_libraries(variable_map.example ungar::ungar)

if(UNGAR_ENABLE_AUTODIFF)
  add_executable(function.example autodiff/function.example.cpp)
  target_link_libraries(function.example ungar::ungar)
endif()

if(UNGAR_ENABLE_OPTIMIZATION)
  add_executable(quadrotor.example mpc/quadrotor.example.cpp)
  target_link_libraries(quadrotor.example ungar::ungar)

  add_executable(quadruped.example mpc/quadruped.example.cpp)
  target_link_libraries(quadruped.example ungar::ungar)

  add_executable(rc_car.example mpc/rc_car.example.cpp)
  target_link_libraries(rc_car.example ungar::ungar)
endif()

# Add examples to the test targets if testing is enabled.
if(UNGAR_BUILD_TESTS)
  add_test(NAME variable.example COMMAND variable.example)
  add_test(NAME variable_map.example COMMAND variable_map.example)

  if(UNGAR_ENABLE_AUTODIFF)
    add_test(NAME function.example COMMAND function.example)
  endif()

  if(UNGAR_ENABLE_OPTIMIZATION)
    add_test(NAME quadrotor.example COMMAND quadrotor.example)
    add_test(NAME quadruped.example COMMAND quadruped.example)
    add_test(NAME rc_car.example COMMAND rc_car.example)
  endif()
endif()
