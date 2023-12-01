cmake_minimum_required(VERSION 3.4)

project("trench")

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC "${VIENNARAY_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME} PRIVATE ${VIENNARAY_LIBRARIES})
configure_file(trenchGrid3D.dat ${CMAKE_CURRENT_BINARY_DIR}/trenchGrid3D.dat COPYONLY)

# Add to example target
add_dependencies(buildExamples ${PROJECT_NAME})
