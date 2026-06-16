cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(trenchSplitting LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaRay)

configure_file(config.txt config.txt COPYONLY)

# When GPU support is enabled, make sure the callable wrapper OptiX IR and
# normalisation PTX kernels are built before this example runs.
if(TARGET ViennaRayCallableWrapper)
  add_dependencies(${PROJECT_NAME} ViennaRayCallableWrapper
                   ${VIENNARAY_GPU_DEPENDENCIES})
endif()

add_dependencies(ViennaRay_Examples ${PROJECT_NAME})
