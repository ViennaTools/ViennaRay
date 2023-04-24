cmake_minimum_required(VERSION 3.14)

project(ViennaRayPython)

include("${CMAKE_SOURCE_DIR}/cmake/prepare.cmake")

# Point find_package to the binary directory instead of the install location
find_package(ViennaRay REQUIRED PATHS ${ViennaRay_BINARY_DIR} NO_DEFAULT_PATH)

list(PREPEND VIENNARAY_INCLUDE_DIRS ${VIENNARAY_BUILD_INCLUDE_DIRS})

message(STATUS "ViennaRay version: ${VIENNARAY_VERSION}")

set(PYBIND11_PYTHON_VERSION 3
    CACHE STRING "Python version")

find_package(pybind11 REQUIRED PATHS ${pybind11_DIR} NO_DEFAULT_PATH)

set(VIENNARAY_PYTHON_SOURCE pyWrap.cpp)
set(VIENNARAY_PYTHON_MODULE_NAME "viennaray")

# ##################################################################################################
# BUILD 2D PYTHON LIBRARY
# ##################################################################################################
set(VIENNARAY_PYTHON_MODULE_NAME_2D "_${VIENNARAY_PYTHON_MODULE_NAME}2d")

pybind11_add_module(${VIENNARAY_PYTHON_MODULE_NAME_2D} ${VIENNARAY_PYTHON_SOURCE})
target_include_directories(${VIENNARAY_PYTHON_MODULE_NAME_2D} PUBLIC ${VIENNARAY_INCLUDE_DIRS})
target_link_libraries(${VIENNARAY_PYTHON_MODULE_NAME_2D} PRIVATE ${VIENNARAY_LIBRARIES})
target_compile_definitions(${VIENNARAY_PYTHON_MODULE_NAME_2D}
                            PRIVATE
                            -DVIENNARAY_PYTHON_DIMENSION=2
                            -DVIENNARAY_MODULE_NAME=${VIENNARAY_PYTHON_MODULE_NAME_2D})

if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(MODULE_NAME ${VIENNARAY_PYTHON_MODULE_NAME_2D})
    set_target_properties(${VIENNARAY_PYTHON_MODULE_NAME_2D} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/viennaray2d)
    configure_file(__init__.py.in ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/viennaray2d/__init__.py)
endif()

# ##################################################################################################
# BUILD 3D PYTHON LIBRARY
# ##################################################################################################
set(VIENNARAY_PYTHON_MODULE_NAME_3D "_${VIENNARAY_PYTHON_MODULE_NAME}3d")

pybind11_add_module(${VIENNARAY_PYTHON_MODULE_NAME_3D} ${VIENNARAY_PYTHON_SOURCE})
target_include_directories(${VIENNARAY_PYTHON_MODULE_NAME_3D} PUBLIC ${VIENNARAY_INCLUDE_DIRS})
target_link_libraries(${VIENNARAY_PYTHON_MODULE_NAME_3D} PRIVATE ${VIENNARAY_LIBRARIES})
target_compile_definitions(${VIENNARAY_PYTHON_MODULE_NAME_3D}
                            PRIVATE
                            -DVIENNARAY_PYTHON_DIMENSION=3
                            -DVIENNARAY_MODULE_NAME=${VIENNARAY_PYTHON_MODULE_NAME_3D})

if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(MODULE_NAME ${VIENNARAY_PYTHON_MODULE_NAME_3D})
    set_target_properties(${VIENNARAY_PYTHON_MODULE_NAME_3D} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/viennaray3d)
    configure_file(__init__.py.in ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/viennaray3d/__init__.py)
endif()