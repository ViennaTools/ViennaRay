project(testReflections LANGUAGES CXX)

# wrap CUDA kernel only
#message(STATUS "VIENNARAY_GPU_INCLUDE_DIR: " ${VIENNARAY_GPU_INCLUDE_DIR})
#
#generate_kernel(${PROJECT_NAME} generated_files "testReflections.cu")

#include_directories(${ViennaCore_SOURCE_DIR}/include/viennacore)
#include_directories(${OptiX_INCLUDE})
#include_directories(${VIENNARAY_GPU_INCLUDE_DIR})

cuda_wrap_srcs(
        ${PROJECT_NAME}
        PTX
        generated_files
        "testReflections.cu"
        OPTIONS
        verbose)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp" ${generated_files})
#target_include_directories(${PROJECT_NAME} PRIVATE ${VIENNARAY_GPU_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaRay)

add_dependencies(ViennaRay-GPU_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
