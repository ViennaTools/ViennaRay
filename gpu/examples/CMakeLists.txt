project(ViennaRay-GPU_Examples)
include("../../cmake/generate_ptx.cmake")

add_gpu_executable(trenchGPU target_name trench.cpp
                   ${CMAKE_SOURCE_DIR}/gpu/pipelines/TestPipeline.cu)
