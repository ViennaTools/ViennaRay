# ##################################################################################################
# Check embree dependency
# ##################################################################################################

# if this is a static build, also build vtk statically
if(VIENNARAY_STATIC_BUILD)
  set(EMBREE_EXTERNAL_INSTALL "static_embree_external")
  message(STATUS "Static build: Using static embree lib.")
else()
  set(EMBREE_EXTERNAL_INSTALL "embree_external")
endif()

# If the path to an installation was provided by the user
if(DEFINED embree_DIR AND NOT DEFINED CACHE{embree_DIR})
  # Required for windows to convert backslashes to forward slashes
  file(TO_CMAKE_PATH "${embree_DIR}" embree_DIR)
  set(embree_DIR
      "${embree_DIR}"
      CACHE PATH "Path to embree installation" FORCE)
  find_package(embree 3.0 REQUIRED PATHS ${embree_DIR} NO_DEFAULT_PATH)
  if(VIENNARAY_STATIC_BUILD)
    # Check if the library can be statically linked
    check_all_type(ALL_STATIC "STATIC_LIBRARY" "embree")
    if(NOT ${ALL_STATIC})
      message(
        FATAL_ERROR
          "The detected installation of embree was not built using EMBREE_STATIC_LIB=ON. A static version of the embree library will be built from source."
      )
    endif()
  endif()
  message(STATUS "Found embree at ${embree_DIR}.")
  add_library(embree_external INTERFACE) # dummy target
else()
  # First try to locate any local build
  find_package(embree 3.0 QUIET PATHS "${DEPENDENCIES_DIR}/Install/${EMBREE_EXTERNAL_INSTALL}"
               NO_DEFAULT_PATH)

  # If that fails look for a system installation
  if(NOT embree_FOUND)
    find_package(embree 3.0 QUIET)
  endif()
  if(VIENNARAY_STATIC_BUILD)
    # Check if the library can be statically linked
    check_all_type(ALL_STATIC "STATIC_LIBRARY" "embree")
    if(NOT ${ALL_STATIC})
      set(embree_FOUND FALSE)
      message(
        STATUS
          "The detected installation of embree was not built using EMBREE_STATIC_LIB=ON. A static version of the embree library will be built from source."
      )
    endif()
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
      TMP_DIR "${DEPENDENCIES_DIR}/tmp/${EMBREE_EXTERNAL_INSTALL}"
      STAMP_DIR "${DEPENDENCIES_DIR}/Stamp/${EMBREE_EXTERNAL_INSTALL}"
      BINARY_DIR "${DEPENDENCIES_DIR}/Build/${EMBREE_EXTERNAL_INSTALL}"
      INSTALL_DIR "${DEPENDENCIES_DIR}/Install/${EMBREE_EXTERNAL_INSTALL}"
      # Embree still lists a (an empty) submodule even though it does not exist (anymore). CMake
      # >=3.16 can prevent download (error) of submodules by specifying an empty string to
      # git_submodule and setting cmake plicy CMP0097 to NEW.
      GIT_SUBMODULES ""
      CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
                 -DEMBREE_STATIC_LIB=${VIENNARAY_STATIC_BUILD}
                 -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                 -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                 -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                 -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
                 -DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}
                 -DCMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}
                 -DCMAKE_SHARED_LINKER_FLAGS=-lstdc++
                 -DEMBREE_TUTORIALS=OFF
                 -DEMBREE_BACKFACE_CULLING=OFF
                 -DEMBREE_FILTER_FUNCTION=ON # setting this option is probably not necessary
      CMAKE_CACHE_ARGS
        -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}
        -DEMBREE_ISPC_SUPPORT:BOOL=${EMBREE_ISPC_SUPPORT}
        # We use OpenMP, Embree calls that internal (eventhough it is external to Embree)
        -DEMBREE_TASKING_SYSTEM:STRING=INTERNAL
      BUILD_COMMAND ${CMAKE_COMMAND} --build . --parallel ${NPROC}
      USES_TERMINAL_DOWNLOAD 1
      USES_TERMINAL_UPDATE 1
      GIT_PROGRESS 1)

    set(EMBREE_RAY_MASK OFF)

    # Get install and build directory from external project
    ExternalProject_Get_Property(embree_external INSTALL_DIR)

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
    CACHE BOOL "Wheter embree uses ray masking" FORCE)
