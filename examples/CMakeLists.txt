add_custom_target(ViennaRay_Examples ALL)

if(WIN32 AND NOT VIENNARAY_DISABLE_COPY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/examples>)
  viennacore_setup_embree_env(ViennaRay_Examples ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  viennacore_setup_tbb_env(ViennaRay_Examples ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

viennacore_add_subdirs(${CMAKE_CURRENT_LIST_DIR})
