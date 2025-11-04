project(disk2D LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaRay)
configure_file(trenchGrid2D.dat ${CMAKE_CURRENT_BINARY_DIR}/trenchGrid2D.dat COPYONLY)

add_dependencies(ViennaRay_Examples ${PROJECT_NAME})
