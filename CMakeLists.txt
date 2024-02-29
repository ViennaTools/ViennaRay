cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(
  ViennaRay
  LANGUAGES CXX
  VERSION 2.0.0)

# --------------------------------------------------------------------------------------------------------
# Library options
# --------------------------------------------------------------------------------------------------------

option(EMBREE_ISPC_SUPPORT "Enable Intel's SPMD Compiler" OFF)
option(EMBREE_RAY_MASK "Enable Embree Ray Masking" OFF)

option(VIENNARAY_COPY_LIBRARIES "Copy the required library files to the built tests and examples" ON)
option(VIENNARAY_USE_WDIST "Enable weighted distribution of ray weights" OFF)
option(VIENNARAY_BUILD_EXAMPLES "Build examples" OFF)
option(VIENNARAY_BUILD_TESTS "Build tests" OFF)

# --------------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------------

include("cmake/openmp.cmake")

if(MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

# --------------------------------------------------------------------------------------------------------
# Setup Library
# --------------------------------------------------------------------------------------------------------

add_library(${PROJECT_NAME} INTERFACE)
add_library(ViennaTools::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES CXX_STANDARD 17
             CXX_EXTENSIONS OFF
             CXX_STANDARD_REQUIRED ON)

if(VIENNARAY_USE_WDIST)
  message(STATUS "[ViennaRay] Using weighted distribution of ray weights")
  target_compile_definitions(${PROJECT_NAME} PUBLIC VIENNARAY_USE_WDIST)
endif()

if(EMBREE_RAY_MASK)
  message(STATUS "[ViennaRay] Using embree ray masking")
  target_compile_definitions(${PROJECT_NAME} PUBLIC VIENNARAY_USE_RAY_MASKING)
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
  NAME PackageProject
  VERSION 1.11.1
  GIT_REPOSITORY "https://github.com/TheLartians/PackageProject.cmake")

CPMFindPackage(
  NAME tbb
  VERSION 2020.0
  GIT_TAG v2021.11.0
  GIT_REPOSITORY "https://github.com/oneapi-src/oneTBB"
  OPTIONS "TBB_TEST OFF" "TBB_STRICT OFF")

CPMFindPackage(
  NAME embree
  VERSION 3.0
  GIT_TAG v3.13.0
  GIT_REPOSITORY "https://github.com/embree/embree"
  OPTIONS "EMBREE_TUTORIALS OFF")

find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} INTERFACE OpenMP::OpenMP_CXX embree)

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
  DEPENDENCIES "embree 3.0; tbb 2020.0")
