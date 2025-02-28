project(testReflections LANGUAGES CXX)

include_directories(${OptiX_INCLUDE})
include_directories(${VIENNARAY_GPU_INCLUDE_DIRS})
include_directories(${ViennaCore_SOURCE_DIR}/include/viennacore)

# wrap CUDA kernel only
cuda_wrap_srcs(${PROJECT_NAME} # target name
               PTX generated_files "testReflections.cu")

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp" ${generated_files})
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaRay ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
target_compile_definitions(
  ${PROJECT_NAME} PRIVATE VIENNACORE_COMPILE_GPU VIENNARAY_KERNELS_PATH_DEFINE=${VIENNARAY_PTX_DIR})

add_dependencies(ViennaRay-GPU_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
