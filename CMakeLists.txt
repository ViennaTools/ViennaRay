cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(
  ViennaRay
  LANGUAGES CXX
  VERSION 2.1.3)

# --------------------------------------------------------------------------------------------------------
# Library switches
# --------------------------------------------------------------------------------------------------------

option(EMBREE_ISPC_SUPPORT "Enable Intel's SPMD Compiler" OFF)
option(EMBREE_RAY_MASK "Enable Embree Ray Masking" OFF)

option(VIENNARAY_USE_WDIST "Enable weighted distribution of ray weights" OFF)
option(VIENNARAY_PRINT_PROGRESS "Enable progress bar" OFF)

option(VIENNARAY_BUILD_EXAMPLES "Build examples" OFF)
option(VIENNARAY_BUILD_TESTS "Build tests" OFF)

# --------------------------------------------------------------------------------------------------------
# Library options
# --------------------------------------------------------------------------------------------------------

set(VIENNARAY_EMBREE_VERSION
    4
    CACHE STRING "The embree version to use")

set(VIENNARAY_EMBREE_TAG
    "Default"
    CACHE STRING "The remote embree version to use")

if(VIENNARAY_EMBREE_TAG STREQUAL "Default")
  if(VIENNARAY_EMBREE_VERSION EQUAL 4)
    set(VIENNARAY_EMBREE_TAG "v4.3.1")
  elseif(VIENNARAY_EMBREE_VERSION EQUAL 3)
    set(VIENNARAY_EMBREE_TAG "v3.13.0")
  else()
    message(WARNING "[ViennaRay] Please manually specify VIENNARAY_EMBREE_TAG")
  endif()
endif()

if(VIENNARAY_EMBREE_VERSION GREATER_EQUAL 4)
  message(STATUS "[ViennaRay] Using Embree Ray Masking on Embree >= 4")
  set(EMBREE_RAY_MASK ON)
endif()

# --------------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------------

include("cmake/embree.cmake")
include("cmake/tbb.cmake")

# --------------------------------------------------------------------------------------------------------
# Setup Library
# --------------------------------------------------------------------------------------------------------

add_library(${PROJECT_NAME} INTERFACE)
add_library(ViennaTools::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
target_compile_definitions(${PROJECT_NAME}
                           INTERFACE VIENNARAY_EMBREE_VERSION=${VIENNARAY_EMBREE_VERSION})

set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES CXX_STANDARD 17
             CXX_EXTENSIONS OFF
             CXX_STANDARD_REQUIRED ON
             WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(VIENNARAY_USE_WDIST)
  message(STATUS "[ViennaRay] Using weighted distribution of ray weights")
  target_compile_definitions(${PROJECT_NAME} INTERFACE VIENNARAY_USE_WDIST)
endif()

if(VIENNARAY_PRINT_PROGRESS)
  target_compile_definitions(${PROJECT_NAME} INTERFACE VIENNARAY_PRINT_PROGRESS)
endif()

if(EMBREE_RAY_MASK)
  message(STATUS "[ViennaRay] Using embree ray masking")
  target_compile_definitions(${PROJECT_NAME} INTERFACE VIENNARAY_USE_RAY_MASKING)
endif()

# --------------------------------------------------------------------------------------------------------
# Include directories
# --------------------------------------------------------------------------------------------------------

target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/viennaray>
                            $<INSTALL_INTERFACE:include/viennaray-${PROJECT_VERSION}>)

# --------------------------------------------------------------------------------------------------------
# Setup Dependencies
# --------------------------------------------------------------------------------------------------------

include("cmake/cpm.cmake")

CPMAddPackage(
  NAME Core
  GIT_TAG main # TODO: Create Tag
  GIT_REPOSITORY "https://github.com/ViennaTools/ViennaCore"
  OPTIONS "VIENNA_CORE_FORMAT_EXCLUDE docs/")

CPMAddPackage(
  NAME PackageProject
  VERSION 1.11.1
  GIT_REPOSITORY "https://github.com/TheLartians/PackageProject.cmake")

if(NOT VIENNARAY_SYSTEM_TBB)
  CPMFindPackage(
    NAME TBB
    VERSION 2020.0
    GIT_TAG v2021.11.0
    GIT_REPOSITORY "https://github.com/oneapi-src/oneTBB"
    OPTIONS "TBB_TEST OFF" "TBB_STRICT OFF")
endif()

CPMFindPackage(
  NAME embree
  VERSION ${VIENNARAY_EMBREE_VERSION}
  GIT_TAG ${VIENNARAY_EMBREE_TAG}
  GIT_REPOSITORY "https://github.com/embree/embree"
  OPTIONS "EMBREE_TUTORIALS OFF")

find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} INTERFACE embree OpenMP::OpenMP_CXX)

# --------------------------------------------------------------------------------------------------------
# Setup Examples
# --------------------------------------------------------------------------------------------------------

if(VIENNARAY_BUILD_EXAMPLES)
  message(STATUS "[ViennaRay] Building Examples")
  add_subdirectory(examples)
endif()

# --------------------------------------------------------------------------------------------------------
# Setup Tests
# --------------------------------------------------------------------------------------------------------

if(VIENNARAY_BUILD_TESTS)
  message(STATUS "[ViennaRay] Building Tests")

  enable_testing()
  add_subdirectory(tests)
endif()

# --------------------------------------------------------------------------------------------------------
# Install Target
# --------------------------------------------------------------------------------------------------------

packageProject(
  NAME ${PROJECT_NAME} NAMESPACE ViennaTools
  VERSION ${PROJECT_VERSION}
  BINARY_DIR ${PROJECT_BINARY_DIR}
  INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/viennaray
  INCLUDE_DESTINATION include/viennaray-${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
  DEPENDENCIES "embree;TBB;OpenMP")
