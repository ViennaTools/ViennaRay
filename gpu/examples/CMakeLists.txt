project(ViennaRay-GPU_Examples)

configure_file(Resources/trenchMesh.dat ${CMAKE_CURRENT_BINARY_DIR}/trenchMesh.dat COPYONLY)

add_gpu_executable(trenchGPU target_name trench.cpp ${VIENNARAY_PIPELINE_DIR}/TestPipeline.cu)
