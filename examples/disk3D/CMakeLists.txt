project(disk3D LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaRay)
configure_file(trenchGrid3D.dat ${CMAKE_CURRENT_BINARY_DIR}/trenchGrid3D.dat COPYONLY)

add_dependencies(ViennaRay_Examples ${PROJECT_NAME})
