include(CTest)

add_executable(utils.test utils/utils.test.cpp)
target_link_libraries(utils.test ungar::ungar gtest)
add_test(NAME utils.test COMMAND utils.test)

add_executable(variable.test variable.test.cpp)
target_link_libraries(variable.test ungar::ungar gtest)
add_test(NAME variable.test COMMAND variable.test)

if(UNGAR_ENABLE_AUTODIFF)
  add_executable(function.test autodiff/function.test.cpp)
  target_link_libraries(function.test ungar::ungar gtest)
  add_test(NAME function.test COMMAND function.test)
endif()

if(UNGAR_ENABLE_OPTIMIZATION)
  add_executable(soft_sqp.test optimization/soft_sqp.test.cpp)
  target_link_libraries(soft_sqp.test ungar::ungar gtest)
  add_test(NAME soft_sqp.test COMMAND soft_sqp.test)
endif()
