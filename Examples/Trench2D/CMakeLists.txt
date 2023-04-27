cmake_minimum_required(VERSION 3.14)

project("Trench2D")

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC "${VIENNARAY_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME} PRIVATE ${VIENNARAY_LIBRARIES})
configure_file(trenchGrid2D.dat ${CMAKE_CURRENT_BINARY_DIR}/trenchGrid2D.dat COPYONLY)

# Add to example target
add_dependencies(buildExamples ${PROJECT_NAME})
