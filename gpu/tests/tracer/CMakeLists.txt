project(tracer LANGUAGES CXX)

add_gpu_executable(${PROJECT_NAME} target_name ${PROJECT_NAME}.cpp)

add_dependencies(ViennaRay-GPU_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
