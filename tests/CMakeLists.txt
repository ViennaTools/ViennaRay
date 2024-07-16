add_custom_target(ViennaRay_Tests ALL)

if(WIN32)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/tests>)
  viennacore_setup_embree_env(ViennaRay_Tests ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  viennacore_setup_tbb_env(ViennaRay_Tests ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

viennacore_add_subdirs(${CMAKE_CURRENT_LIST_DIR})
