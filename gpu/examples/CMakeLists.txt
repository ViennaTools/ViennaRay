project(ViennaRay-GPU_Examples)

add_gpu_executable(trenchGPU target_name trench.cpp)
configure_file(Resources/trenchMesh.dat ${CMAKE_CURRENT_BINARY_DIR}/trenchMesh.dat COPYONLY)
