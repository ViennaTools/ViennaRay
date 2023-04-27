cmake_minimum_required(VERSION 3.14)

project("ViennaRay" VERSION 1.2.0)

# Store version in cache
set(VIENNARAY_VERSION
    ${PROJECT_VERSION}
    CACHE STRING "The version of ViennaRay" FORCE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD "17")

add_definitions(-DVIENNARAY_VERSION=${PROJECT_VERSION})

include(GNUInstallDirs)

# set default build type
set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "${DEFAULT_BUILD_TYPE}"
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                               "RelWithDebInfo")
endif()

# tell VS to export all symbols to its dll files
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS
      TRUE
      CACHE BOOL "Export all symbols")
endif()

# ##################################################################################################
# CONFIGURATION OPTIONS
# ##################################################################################################
# Build the included examples
option(VIENNARAY_BUILD_EXAMPLES "Build examples." OFF)

# Build the included tests
option(VIENNARAY_BUILD_TESTS "Build tests." OFF)

# Build Python bindings
option(VIENNARAY_BUILD_PYTHON "Build for python3.x." OFF)

# Enable weighted distribution of ray weights
option(VIENNARAY_USE_WDIST "Use weighted distribution of ray weights." OFF)

# Statically link dependencies
option(VIENNARAY_STATIC_BUILD "Build dependencies as static libraries." OFF)

# ##################################################################################################
# If you want to use the Intel SPMD Compiler, set that here
# ##################################################################################################
set(EMBREE_ISPC_SUPPORT
    OFF
    CACHE BOOL "Set whether or not you use Intel's SPMD Compiler")

# ##################################################################################################
# AUTOMATIC DEPENDENCY PREPARATION
# ##################################################################################################

# With the stampfile mechanism, cmake automatically retriggers the configure step after the
# buildDependencies targed has been executed. Thus all dependencies that were built with
# buildDependencies should then be made available by the find_package calls.
set(STAMPFILE ${CMAKE_BINARY_DIR}/target.stamp)
# When the buildDependencies target is executed, the stamp file is touched
add_custom_target(buildDependencies COMMAND ${CMAKE_COMMAND} -E touch ${STAMPFILE})

# Include the external dependencies
include(ExternalProject)
if(NOT DEFINED VIENNARAY_DEPENDENCIES_DIR)
  set(DEPENDENCIES_DIR ${CMAKE_SOURCE_DIR}/dependencies)
else()
  set(DEPENDENCIES_DIR
      ${VIENNARAY_DEPENDENCIES_DIR}
      CACHE PATH "Directory for downloading, building and installing external dependencies")
endif()
add_subdirectory(external/upstream)

# Create the initial stamp file
file(TOUCH ${STAMPFILE})
# Include the stamp file, so that cmake is forced to re-run once the file has been touched
include(${STAMPFILE})

# Store the binary directory for use in subprojects
set(ViennaRay_BINARY_DIR
    ${PROJECT_BINARY_DIR}
    CACHE PATH "Path to local ViennaRay installation" FORCE)

# ##################################################################################################
# DEPENDENCY CHECKS
# ##################################################################################################
set(DEPENDENCIES_FOUND TRUE)

find_package(OpenMP REQUIRED)
list(APPEND VIENNARAY_LIBRARIES OpenMP::OpenMP_CXX)

list(APPEND VIENNARAY_DEPENDENCIES "embree_external")
find_package(
  embree
  3.0
  QUIET
  CONFIG
  PATHS
  ${embree_DIR}
  NO_DEFAULT_PATH)
if(NOT embree_FOUND)
  set(DEPENDENCIES_FOUND FALSE)
endif()

if(VIENNARAY_BUILD_PYTHON)
  list(APPEND VIENNARAY_DEPENDENCIES "pybind11_external")
  find_package(pybind11 QUIET PATHS ${pybind11_DIR} NO_DEFAULT_PATH)
  if(NOT pybind11_FOUND)
    set(DEPENDENCIES_FOUND FALSE)
  endif()
endif()

if(DEPENDENCIES_FOUND)
  # Remove the buildDependencies target from the ALL target to prevent unnecessary re-builds
  set_target_properties(buildDependencies PROPERTIES EXCLUDE_FROM_ALL true)
else()
  message(WARNING "Not all dependencies were found. Execute buildDependencies target first.")
  # Add the buildDependencies target to the ALL target
  set_target_properties(buildDependencies PROPERTIES EXCLUDE_FROM_ALL false)
endif()

# ##################################################################################################
# MAIN LIBRARY TARGET AND ADDITIONAL COMPILE DEFINITIONS
# ##################################################################################################
message(STATUS "embree ray masking: ${EMBREE_RAY_MASK}")

# Enables Ray Masking in ViennaRay, if embree is compiled with ray masking enabled
if(EMBREE_RAY_MASK)
  add_compile_definitions(VIENNARAY_USE_RAY_MASKING)
endif()

if(VIENNARAY_USE_WDIST)
  add_compile_definitions(VIENNARAY_USE_WDIST)
endif()

add_library(${PROJECT_NAME} INTERFACE)

# ##################################################################################################
# DIRECTORY CONFIGURATIONS
# ##################################################################################################
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME}/")
endif()
set(LOCAL_CONFIG_DIR "lib/cmake/${PROJECT_NAME}")

# This variable is used by the example, test, python and precompiled library target, since those are
# compiled before the project is installed in its proper install location.
set(VIENNARAY_BUILD_INCLUDE_DIRS "${${PROJECT_NAME}_SOURCE_DIR}/include")

# Adding the install interface generator expression makes sure that the include files are installed
# to the proper location (provided by GNUInstallDirs)
target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${VIENNARAY_BUILD_INCLUDE_DIRS}>
                            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_11)

# ##################################################################################################
# CMAKE CONFIG FILE SETUP
# ##################################################################################################
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" INSTALL_DESTINATION ${LOCAL_CONFIG_DIR})

# ##################################################################################################
# BUILD EXAMPLES
# ##################################################################################################
if(VIENNARAY_BUILD_EXAMPLES AND DEPENDENCIES_FOUND)
  add_subdirectory(Examples)
endif()

# ##################################################################################################
# BUILD TESTS (Smaller examples for testing)
# ##################################################################################################
if(VIENNARAY_BUILD_TESTS AND DEPENDENCIES_FOUND)
  enable_testing()
  add_subdirectory(Tests)
endif()

# ##################################################################################################
# BUILD PYTHON MODULE
# ##################################################################################################
if(VIENNARAY_BUILD_PYTHON AND DEPENDENCIES_FOUND)
  add_subdirectory(Python)
endif()

# ##################################################################################################
# INSTALL
# ##################################################################################################
# install config files locations are provided by GNUInstallDirs
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}_Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(
  EXPORT ${PROJECT_NAME}_Targets
  FILE ${PROJECT_NAME}Targets.cmake
  DESTINATION ${LOCAL_CONFIG_DIR})

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${LOCAL_CONFIG_DIR})

# install include files
file(GLOB_RECURSE HEADER_FILES "${PROJECT_SOURCE_DIR}/include/*.hpp")
install(FILES ${HEADER_FILES} DESTINATION include)
