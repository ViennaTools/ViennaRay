# ##################################################################################################
# Check embree dependency
# ##################################################################################################
# Check if the variable is a cache variable
get_property(
  is_cached
  CACHE embree_DIR
  PROPERTY TYPE)
# If it is not from cache and defined at this point, the variable was passed as a -D command line
# argument
if("${is_cached}" STREQUAL "" AND DEFINED embree_DIR)
  # Required for windows to convert backslashes to forward slashes
  file(TO_CMAKE_PATH "${embree_DIR}" embree_DIR)
  set(embree_DIR
      "${embree_DIR}"
      CACHE PATH "Path to embree installation" FORCE)
  find_package(embree 3.0 REQUIRED PATHS ${embree_DIR} NO_DEFAULT_PATH)
  message(STATUS "Found embree at ${embree_DIR}.")
  add_library(embree_external INTERFACE) # dummy target
else()
  # First try to locate any local build
  find_package(embree 3.0 QUIET PATHS "${DEPENDENCIES_DIR}/Install/embree_external" NO_DEFAULT_PATH)

  # If that fails look for a system installation
  if(NOT embree_FOUND)
    find_package(embree 3.0 QUIET)
  endif()
  if(embree_FOUND)
    message(STATUS "Found embree at ${embree_DIR}.")
    add_library(embree_external INTERFACE) # dummy target
  else()
    message(
      STATUS
        "Couldn't find embree. Dependency will be built with the buildDependencies target (e.g. `make buildDependencies`)."
    )
    # Override CXX standard, since Embree does not confirm to CXX20.
    set(CMAKE_CXX_STANDARD 17)

    # This cmake policy is for the git_submodules parameter below
    cmake_policy(SET CMP0097 NEW)

    # Get the number of cores
    include(ProcessorCount)
    ProcessorCount(NPROC)

    ExternalProject_Add(
      embree_external
      GIT_REPOSITORY https://github.com/embree/embree.git
      GIT_TAG # hash of tags/v3.13.0
              7c53133eb21424f7f0ae1e25bf357e358feaf6ab
      GIT_SHALLOW true
      INSTALL_DIR "${DEPENDENCIES_DIR}/Install/embree_external"
      # Embree still lists a (an empty) submodule even though it does not exist (anymore). CMake
      # >=3.16 can prevent download (error) of submodules by specifying an empty string to
      # git_submodule and setting cmake policy CMP0097 to NEW.
      GIT_SUBMODULES ""
      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                 -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                 -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                 -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
                 -DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}
                 -DCMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}
                 -DCMAKE_SHARED_LINKER_FLAGS=-lstdc++
                 -DEMBREE_TUTORIALS=OFF
                 -DEMBREE_BACKFACE_CULLING=OFF
                 -DEMBREE_FILTER_FUNCTION=ON # setting this option is probably not necessary
                 -DCMAKE_BUILD_TYPE=Release
      CMAKE_CACHE_ARGS
        -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}
        -DEMBREE_ISPC_SUPPORT:BOOL=${EMBREE_ISPC_SUPPORT}
        # We use OpenMP, Embree calls that internal (even though it is external to Embree)
        -DEMBREE_TASKING_SYSTEM:STRING=INTERNAL
      BUILD_COMMAND ${CMAKE_COMMAND} --build . --parallel ${NPROC}
      # Do not install If one installs it and tries to use the installed library, then there exists
      # a linkage problem with TBB. Simply use the generated files in the binary directory.
      USES_TERMINAL_DOWNLOAD 1
      USES_TERMINAL_UPDATE 1
      GIT_PROGRESS 1)

    set(EMBREE_RAY_MASK OFF)

    # Get install and build directory from external project
    ExternalProject_Get_Property(embree_external INSTALL_DIR BINARY_DIR)

    set(embree_DIR "${INSTALL_DIR}")

    # add to buildDependencies target
    add_dependencies(buildDependencies embree_external)
  endif()
  set(embree_DIR
      "${embree_DIR}"
      CACHE PATH "Path to embree installation" FORCE)
endif()

set(EMBREE_RAY_MASK
    ${EMBREE_RAY_MASK}
    CACHE BOOL "Whether embree uses ray masking" FORCE)
