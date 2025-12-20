project(ViennaRay-GPU_Examples)

add_custom_target(ViennaRay-GPU_Examples ALL)

add_executable(GPU_trenchTriangles trenchTriangles.cpp)
target_link_libraries(GPU_trenchTriangles PRIVATE ViennaRay)
add_dependencies(${PROJECT_NAME} ${VIENNARAY_GPU_DEPENDENCIES})
configure_file(${CMAKE_SOURCE_DIR}/examples/triangle3D/trenchMesh.dat
               ${CMAKE_CURRENT_BINARY_DIR}/trenchMesh.dat COPYONLY)
add_dependencies(ViennaRay-GPU_Examples GPU_trenchTriangles)

add_executable(GPU_trenchDisks trenchDisks.cpp)
target_link_libraries(GPU_trenchDisks PRIVATE ViennaRay)
add_dependencies(${PROJECT_NAME} ${VIENNARAY_GPU_DEPENDENCIES})
configure_file(${CMAKE_SOURCE_DIR}/examples/disk3D/trenchGrid3D.dat
               ${CMAKE_CURRENT_BINARY_DIR}/trenchGrid3D.dat COPYONLY)
add_dependencies(ViennaRay-GPU_Examples GPU_trenchDisks)

add_executable(GPU_trenchLines trenchLines.cpp)
target_link_libraries(GPU_trenchLines PRIVATE ViennaRay)
add_dependencies(${PROJECT_NAME} ${VIENNARAY_GPU_DEPENDENCIES})
configure_file(${CMAKE_SOURCE_DIR}/examples/triangle2D/lineMesh.dat
               ${CMAKE_CURRENT_BINARY_DIR}/lineMesh.dat COPYONLY)
add_dependencies(ViennaRay-GPU_Examples GPU_trenchLines)
