project(GPU_sourceDirection LANGUAGES CXX)

viennacore_add_optixir(TestSourcePipelineTriangle ${CMAKE_CURRENT_SOURCE_DIR}/TestSourcePipelineTriangle.cu
                       DEPENDS ${VIENNARAY_PIPELINE_HEADERS})
add_executable(${PROJECT_NAME} sourceDirection.cpp)
add_dependencies(${PROJECT_NAME} TestSourcePipelineTriangle)
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaRay)

add_dependencies(ViennaRay-GPU_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
