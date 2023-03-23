add_executable(utils.test utils/utils.test.cpp)
target_link_libraries(utils.test ungar::ungar gtest)

add_executable(variable.test variable.test.cpp)
target_link_libraries(variable.test ungar::ungar gtest)

if(UNGAR_ENABLE_AUTODIFF)
  add_executable(function.test autodiff/function.test.cpp)
  target_link_libraries(function.test ungar::ungar gtest)
endif()

if(UNGAR_ENABLE_OPTIMIZATION)
  add_executable(soft_sqp.test optimization/soft_sqp.test.cpp)
  target_link_libraries(soft_sqp.test ungar::ungar gtest)
endif()
