cmake_minimum_required(VERSION 3.4)

project(
  "ViennaRay"
  VERSION 0.0.1)

add_definitions(-DVIENNARAY_VERSION=${PROJECT_VERSION})

include(GNUInstallDirs)

# c++17 for inlince constexpr variables
SET(CMAKE_CXX_STANDARD "17")

# set default build type
SET(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# tell VS to export all symbols to its dll files
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE CACHE BOOL "Export all symbols")
endif()

# OpenMP Support
find_package(OpenMP REQUIRED)
list(APPEND VIENNARAY_LIBRARIES OpenMP::OpenMP_CXX)

# ViennaLS and ViennaHRLE
set(ViennaHRLE_DIR $ENV{VIENNAHRLE_DIR})
set(ViennaLS_DIR $ENV{VIENNALS_DIR})
find_package(ViennaHRLE REQUIRED)
find_package(ViennaLS REQUIRED)

set(
    VIENNARAY_INCLUDE_DIRS 
    "${VIENNALS_INCLUDE_DIRS}" 
    "${${PROJECT_NAME}_SOURCE_DIR}/include" 
    )

# embree
if(NOT embree_DIR)
    set(embree_DIR $ENV{embree_DIR})
endif(NOT embree_DIR)

find_package(embree 3.0 REQUIRED)
if(embree_FOUND)
    message(STATUS "Found embree")
    list(APPEND VIENNARAY_LIBRARIES embree)
else(embree_FOUND)
    message(FATAL_ERROR "No embree isntall found.")
endif(embree_FOUND)

# VTK
if(NOT VTK_DIR)
    set(VTK_DIR $ENV{VTK_DIR})
endif(NOT VTK_DIR)

find_package(VTK REQUIRED)
if(VTK_FOUND)
    message(STATUS "Found VTK")
    add_definitions(-DVIENNARAY_USE_VTK)
    include(${VTK_USE_FILE})
    # only link needed vtk libraries
    set(VTK_LIBRARIES vtksys;vtkIOCore;vtkexpat;vtklz4;vtkzlib;vtklzma;vtkdoubleconversion;vtkCommonMisc;vtkCommonSystem;vtkIOXML)
    list(APPEND VIENNARAY_LIBRARIES ${VTK_LIBRARIES})
else(VTK_FOUND)
  message(FATAL_ERROR "No VTK install found.")
endif(VTK_FOUND)

#################################################
# BUILD EXAMPLES
#################################################
option(VIENNARAY_BUILD_EXAMPLES "Build examples." OFF)
if(VIENNARAY_BUILD_EXAMPLES)
  add_subdirectory(Examples)
endif(VIENNARAY_BUILD_EXAMPLES)


#################################################
# BUILD TESTS (Smaller examples for testing)
#################################################
option(VIENNARAY_BUILD_TESTS "Build tests." OFF)
if(VIENNARAY_BUILD_TESTS)
  enable_testing()
  add_subdirectory(Tests)
endif(VIENNARAY_BUILD_TESTS)