
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was ViennaRayConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)

# ViennaRay requires C++17
set(CMAKE_CXX_STANDARD "17")

# ##################################################################################################
# compiler dependent settings for ViennaRay
# ##################################################################################################
find_dependency(OpenMP)
list(APPEND VIENNARAY_LIBRARIES OpenMP::OpenMP_CXX)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # disable-new-dtags sets RPATH which searches for libs recursively, instead of RUNPATH which is
  # not needed for g++ to link correctly
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd\"4267\" /wd\"4244\"")
endif()

set(VIENNARAY_INCLUDE_DIRS "/usr/local/ViennaRay//include")

if(OFF)
  add_compile_definitions(VIENNARAY_USE_WDIST)
endif(OFF)

set(embree_DIR embree_DIR-NOTFOUND)
find_dependency(embree 3.0 PATHS ${embree_DIR} NO_DEFAULT_PATH)
list(APPEND VIENNARAY_LIBRARIES embree)

# Enable Ray Masking if embree is compiled with ray masking enabled
if(OFF)
  add_compile_definitions(VIENNARAY_USE_RAY_MASKING)
endif(OFF)

if(OFF)
  set(VIENNARAY_STATIC_BUILD ON)
endif()

check_required_components("ViennaRay")
