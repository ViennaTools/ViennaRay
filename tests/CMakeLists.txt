add_custom_target(ViennaRay_Tests ALL)

include("../cmake/embree.cmake")
setup_embree_env(ViennaRay_Tests "tests")

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
