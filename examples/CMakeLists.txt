add_custom_target(ViennaRay_Examples ALL)

if(WIN32 AND NOT VIENNARAY_DISABLE_COPY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/examples>)

  if(NOT VIENNARAY_SYSTEM_EMBREE)
    setup_embree_env(ViennaRay_Examples ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  endif()

  if(NOT VIENNARAY_SYSTEM_TBB)
    setup_tbb_env(ViennaRay_Examples ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  endif()
endif()

file(
  GLOB entries
  LIST_DIRECTORIES true
  "*")

foreach(entry ${entries})
  if(NOT IS_DIRECTORY ${entry})
    continue()
  endif()

  message(STATUS "[ViennaRay] Adding Example ${entry}")
  add_subdirectory(${entry})
endforeach()
