cmake_minimum_required(VERSION 3.4)

project(ViennaRayExamples)

include("${CMAKE_SOURCE_DIR}/cmake/prepare.cmake")

find_package(ViennaRay CONFIG PATHS ${ViennaRay_BINARY_DIR})
if(NOT ViennaRay_FOUND)
  message(
    FATAL_ERROR
      "One of the dependencies hasn't been built yet and thus can't be used by the buildExamples target.\n\
  Try the following steps to fix the problem:\n\
  1. Turn off the examples target by disabling the VIENNARAY_BUILD_EXAMPLES option when configuring the project\n\
  (e.g. with the `-D VIENNARAY_BUILD_EXAMPLES=OFF` flag)\n\
  2. Build the dependencies target: e.g. `make buildDependencies` when using Makefiles\n\
  3. Turn on the examples target again by re-enabling the VIENNARAY_BUILD_EXAMPLES option\n\
  (e.g. with the `-D VIENNARAY_BUILD_EXAMPLES=ON` flag)\n\
  4. Now the configure step should pick up the built dependencies and proceed as intended.")
endif()

list(PREPEND VIENNARAY_INCLUDE_DIRS ${VIENNARAY_BUILD_INCLUDE_DIRS})

# Exclude the buildTests target from the `all` target
set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL TRUE)

add_custom_target(buildExamples)

subdirlist(EXAMPLE_DIRS ${CMAKE_CURRENT_LIST_DIR})

# add all examples
foreach(subdir ${EXAMPLE_DIRS})
  add_subdirectory(${subdir})
endforeach()
