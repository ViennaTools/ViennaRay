project(linesToTriangles LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaRay)
configure_file(${CMAKE_SOURCE_DIR}/examples/triangle2D/lineMesh.dat
               ${CMAKE_CURRENT_BINARY_DIR}/lineMesh.dat COPYONLY)

add_dependencies(ViennaRay_Tests ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)
