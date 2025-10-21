project(ViennaRay-GPU_Examples)

add_gpu_executable(GPU_trenchTriangles target_name trenchTriangles.cpp)
configure_file(Resources/trenchMesh.dat ${CMAKE_CURRENT_BINARY_DIR}/trenchMesh.dat COPYONLY)

add_gpu_executable(GPU_trenchDisks target_name trenchDisks.cpp)
configure_file(${CMAKE_SOURCE_DIR}/examples/trench/trenchGrid3D.dat ${CMAKE_CURRENT_BINARY_DIR}/trenchGrid3D.dat COPYONLY)
