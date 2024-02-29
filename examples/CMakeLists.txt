add_custom_target(ViennaRay_Examples ALL)

include("../cmake/embree.cmake")
setup_embree_env(ViennaRay_Examples "examples")

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
