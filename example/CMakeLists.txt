add_executable(ungar.variable.example variable.example.cpp)
target_link_libraries(ungar.variable.example ungar::ungar)

add_executable(ungar.variable_map.example variable_map.example.cpp)
target_link_libraries(ungar.variable_map.example ungar::ungar)

if(UNGAR_ENABLE_AUTODIFF)
  add_executable(ungar.function.example autodiff/function.example.cpp)
  target_link_libraries(ungar.function.example ungar::ungar)
endif()

if(UNGAR_ENABLE_OPTIMIZATION)
  add_executable(ungar.quadrotor.example mpc/quadrotor.example.cpp)
  target_link_libraries(ungar.quadrotor.example ungar::ungar)

  add_executable(ungar.quadruped.example mpc/quadruped.example.cpp)
  target_link_libraries(ungar.quadruped.example ungar::ungar)

  add_executable(ungar.rc_car.example mpc/rc_car.example.cpp)
  target_link_libraries(ungar.rc_car.example ungar::ungar)
endif()

# Add examples to the test targets if testing is enabled.
if(UNGAR_BUILD_TESTS)
  add_test(NAME ungar.variable.example COMMAND ungar.variable.example)
  add_test(NAME ungar.variable_map.example COMMAND ungar.variable_map.example)

  if(UNGAR_ENABLE_AUTODIFF)
    add_test(NAME ungar.function.example COMMAND ungar.function.example)
  endif()

  if(UNGAR_ENABLE_OPTIMIZATION)
    add_test(NAME ungar.quadrotor.example COMMAND ungar.quadrotor.example)
    add_test(NAME ungar.quadruped.example COMMAND ungar.quadruped.example)
    add_test(NAME ungar.rc_car.example COMMAND ungar.rc_car.example)
  endif()
endif()
