project(tracer LANGUAGES CXX)

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
add_dependencies(${PROJECT_NAME} ${VIENNARAY_GPU_DEPENDENCIES})

add_dependencies(ViennaRay-GPU_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
