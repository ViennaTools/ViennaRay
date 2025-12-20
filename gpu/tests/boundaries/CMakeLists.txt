project(boundaries LANGUAGES CXX)

viennacore_add_optixir(viennaray_test_pipeline ${CMAKE_CURRENT_SOURCE_DIR}/TestPipelineTriangle.cu)
add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
add_dependencies(${PROJECT_NAME} viennaray_test_pipeline)
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaRay)

add_dependencies(ViennaRay-GPU_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
