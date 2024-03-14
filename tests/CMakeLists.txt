add_custom_target(ViennaRay_Tests ALL)

if(WIN32
   AND NOT VIENNARAY_DISABLE_COPY
   AND NOT VIENNARAY_SYSTEM_EMBREE)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:${PROJECT_BINARY_DIR}/tests>)
  setup_embree_env(ViennaRay_Tests ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

file(
  GLOB entries
  LIST_DIRECTORIES true
  "*")

foreach(entry ${entries})
  if(NOT IS_DIRECTORY ${entry})
    continue()
  endif()

  message(STATUS "[ViennaRay] Adding Test ${entry}")
  add_subdirectory(${entry})
endforeach()
