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

option(VIENNARAY_USE_WDIST "Enable weighted distribution of ray weights" OFF)
option(VIENNARAY_DISABLE_COPY "Disable Embree Environment setup" OFF)
option(VIENNARAY_BUILD_EXAMPLES "Build examples" OFF)
option(VIENNARAY_BUILD_TESTS "Build tests" OFF)

# --------------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------------

include("cmake/openmp.cmake")

set(VIENNARAY_EMBREE_VERSION 4)
find_package(embree ${VIENNARAY_EMBREE_VERSION} QUIET)
set(VIENNARAY_SYSTEM_EMBREE ${embree_FOUND})
if(VIENNARAY_SYSTEM_EMBREE)
  message(STATUS "[ViennaRay] Found embree 4 system installation")
else(VIENNARAY_SYSTEM_EMBREE)
  # look for embree 3
  find_package(embree 3 QUIET)
  set(VIENNARAY_SYSTEM_EMBREE ${embree_FOUND})
  if(VIENNARAY_SYSTEM_EMBREE)
    message(STATUS "[ViennaRay] Found embree 3 system installation")
    set(VIENNARAY_EMBREE_VERSION 3)
  endif()
endif()

if(NOT VIENNARAY_SYSTEM_EMBREE)
  message(STATUS "[ViennaRay] Using remote embree")
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
             CXX_STANDARD_REQUIRED ON
             WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_compile_definitions(${PROJECT_NAME}
                           INTERFACE VIENNARAY_EMBREE_VERSION=${VIENNARAY_EMBREE_VERSION})

if(VIENNARAY_USE_WDIST)
  message(STATUS "[ViennaRay] Using weighted distribution of ray weights")
  target_compile_definitions(${PROJECT_NAME} INTERFACE VIENNARAY_USE_WDIST)
endif()

if(EMBREE_RAY_MASK OR VIENNARAY_EMBREE_VERSION EQUAL 4)
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
  NAME PackageProject
  VERSION 1.11.1
  GIT_REPOSITORY "https://github.com/TheLartians/PackageProject.cmake")

if(NOT VIENNARAY_SYSTEM_EMBREE)
  CPMFindPackage(
    NAME TBB
    VERSION 2020.0
    GIT_TAG v2021.11.0
    GIT_REPOSITORY "https://github.com/oneapi-src/oneTBB"
    OPTIONS "TBB_TEST OFF" "TBB_STRICT OFF")
endif()

if(VIENNARAY_EMBREE_VERSION EQUAL 3)
  CPMFindPackage(NAME embree VERSION ${VIENNARAY_EMBREE_VERSION})
else(VIENNARAY_EMBREE_VERSION)
  CPMFindPackage(
    NAME embree
    VERSION ${VIENNARAY_EMBREE_VERSION}
    GIT_TAG v4.3.1
    GIT_REPOSITORY "https://github.com/embree/embree"
    OPTIONS "EMBREE_TUTORIALS OFF")
endif()

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
  DEPENDENCIES "embree;TBB")
