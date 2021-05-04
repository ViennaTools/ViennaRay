cmake_minimum_required(VERSION 3.4)

project("AdvectTrench3D")

find_package(ViennaHRLE REQUIRED)
find_package(ViennaLS REQUIRED)
find_package(embree 3.0 REQUIRED)

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC "${VIENNARAY_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME} PRIVATE ${VIENNARAY_LIBRARIES})