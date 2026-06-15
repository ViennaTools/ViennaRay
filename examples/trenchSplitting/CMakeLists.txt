cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(trenchSplitting LANGUAGES CXX)

add_executable(${PROJECT_NAME} "${PROJECT_NAME}.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ViennaRay)

configure_file(config.txt config.txt COPYONLY)

add_dependencies(ViennaRay_Examples ${PROJECT_NAME})
