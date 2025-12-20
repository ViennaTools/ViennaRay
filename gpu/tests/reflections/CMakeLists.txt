project(testReflections LANGUAGES CXX)

# wrap CUDA kernel only
viennacore_add_ptx(viennaray_test_reflections ${CMAKE_CURRENT_SOURCE_DIR}/testReflections.cu)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
add_dependencies(${PROJECT_NAME} viennaray_test_reflections)
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaRay)

add_dependencies(ViennaRay-GPU_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
