include(CTest)

add_executable(ungar.utils.test utils/utils.test.cpp)
target_link_libraries(ungar.utils.test ungar::ungar GTest::gtest)
add_test(NAME ungar.utils.test COMMAND ungar.utils.test)

add_executable(ungar.variable.test variable.test.cpp)
target_link_libraries(ungar.variable.test ungar::ungar GTest::gtest)
add_test(NAME ungar.variable.test COMMAND ungar.variable.test)

if(UNGAR_ENABLE_AUTODIFF)
  add_executable(ungar.function.test autodiff/function.test.cpp)
  target_link_libraries(ungar.function.test ungar::ungar GTest::gtest)
  add_test(NAME ungar.function.test COMMAND ungar.function.test)
endif()

if(UNGAR_ENABLE_OPTIMIZATION)
  add_executable(ungar.soft_sqp.test optimization/soft_sqp.test.cpp)
  target_link_libraries(ungar.soft_sqp.test ungar::ungar GTest::gtest)
  add_test(NAME ungar.soft_sqp.test COMMAND ungar.soft_sqp.test)
endif()
