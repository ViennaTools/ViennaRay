project(GPU_rngSeed LANGUAGES CXX)

add_gpu_executable(${PROJECT_NAME} target_name rngSeed.cpp)

add_dependencies(ViennaRay-GPU_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)

configure_file(${CMAKE_SOURCE_DIR}/examples/triangle3D/trenchMesh.dat
               ${CMAKE_CURRENT_BINARY_DIR}/trenchMesh.dat COPYONLY)