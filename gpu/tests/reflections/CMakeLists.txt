project(testReflections LANGUAGES CXX)

# wrap CUDA kernel only
generate_kernel(generated_files "testReflections.cu")

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp" ${generated_files})
target_include_directories(${PROJECT_NAME} PRIVATE ${OptiX_INCLUDE})
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaRay ${VIENNACORE_GPU_LIBS})

add_dependencies(ViennaRay-GPU_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
