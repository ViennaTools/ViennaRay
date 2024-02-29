add_custom_target(ViennaRay_Examples ALL)

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
