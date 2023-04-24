cmake_minimum_required(VERSION 3.14)

project("ViennaRay" VERSION 1.2.0)

add_definitions(-DVIENNARAY_VERSION=${PROJECT_VERSION})

include(GNUInstallDirs)

# c++17 for inline constexpr variables
set(CMAKE_CXX_STANDARD "17")

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

set(ViennaRay_BINARY_DIR
    ${PROJECT_BINARY_DIR}
    CACHE PATH "Path to local ViennaRay installation" FORCE)

# OpenMP Support
find_package(OpenMP REQUIRED)
list(APPEND VIENNARAY_LIBRARIES OpenMP::OpenMP_CXX)

# ##################################################################################################
# If you want to use the Intel SPMD Compiler, set that here
# ##################################################################################################
set(EMBREE_ISPC_SUPPORT
    OFF
    CACHE BOOL "Set whether or not you use Intel's SPMD Compiler")
# ##################################################################################################

# build dependencies
set(DEPENDENCIES_DIR ${CMAKE_SOURCE_DIR}/dependencies)

# Include all external dependencies
include(ExternalProject)
add_custom_target(buildDependencies)
add_subdirectory(external/upstream)

message(STATUS "embree ray masking: ${EMBREE_RAY_MASK}")

# Enable Ray Masking if embree is compiled with ray masking enabled
if(EMBREE_RAY_MASK)
  add_compile_definitions(VIENNARAY_USE_RAY_MASKING)
endif(EMBREE_RAY_MASK)

set(VIENNARAY_BUILD_INCLUDE_DIRS "${${PROJECT_NAME}_SOURCE_DIR}/include")

option(VIENNARAY_USE_WDIST "Use weighted distribution of ray weights." OFF)
if(VIENNARAY_USE_WDIST)
  add_compile_definitions(VIENNARAY_USE_WDIST)
endif(VIENNARAY_USE_WDIST)

add_library(${PROJECT_NAME} INTERFACE)

# set the correct paths for installation
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME}/")
endif()
set(LOCAL_CONFIG_DIR "lib/cmake/${PROJECT_NAME}")

# Adding the install interface generator expression makes sure that the include files are installed
# to the proper location (provided by GNUInstallDirs)
set(VIENNARAY_BUILD_INCLUDE_DIRS "${${PROJECT_NAME}_SOURCE_DIR}/include")
target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${VIENNARAY_BUILD_INCLUDE_DIRS}>
                            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

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
option(VIENNARAY_BUILD_EXAMPLES "Build examples." OFF)
if(VIENNARAY_BUILD_EXAMPLES)
  add_subdirectory(Examples)
endif(VIENNARAY_BUILD_EXAMPLES)

# ##################################################################################################
# BUILD TESTS (Smaller examples for testing)
# ##################################################################################################
option(VIENNARAY_BUILD_TESTS "Build tests." OFF)
if(VIENNARAY_BUILD_TESTS)
  enable_testing()
  add_subdirectory(Tests)
endif(VIENNARAY_BUILD_TESTS)

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
