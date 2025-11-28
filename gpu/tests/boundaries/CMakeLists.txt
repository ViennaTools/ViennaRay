project(boundaries LANGUAGES CXX)

generate_pipeline(${PROJECT_NAME} generated_files TestPipelineTriangle.cu)
add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp ${generated_files})
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaRay)

add_dependencies(ViennaRay-GPU_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
