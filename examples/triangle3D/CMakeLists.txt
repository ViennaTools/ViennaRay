project(triangle3D LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaRay)
configure_file(${CMAKE_SOURCE_DIR}/gpu/examples/Resources/trenchMesh.dat ${CMAKE_CURRENT_BINARY_DIR}/trenchMesh.dat COPYONLY)

add_dependencies(ViennaRay_Examples ${PROJECT_NAME})
