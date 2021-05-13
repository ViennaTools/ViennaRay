# Override CXX standard, since Embree does not confirm to CXX20.
set(CMAKE_CXX_STANDARD 17)

# This cmake policy is for the git_submodules parameter below
cmake_policy(SET CMP0097 NEW)

externalproject_add(embree-external
  GIT_REPOSITORY
    https://github.com/embree/embree.git
  GIT_TAG
    ## Hash of tags/v3.5.2
    #e766297adcdaf8e5d8dd996c15aed753255230fd
    #
    ## Hash of tags/v3.6.1
    #d17aec1491a6c664f4da2f0b8ba03c171cbf36a1
    #
    ## tags/v3.10.0
    #0e78b682dc631575b79f1cc43863931e8d411e3b
    #
    ## hash of tags/v3.11.0
    cc8c3aac00cdb63fb1acb44bcd0fa89a17139533
    #
  GIT_SHALLOW
    true
    # Embree still lists a (an empty) submodule eventhough it does not exist (anymore).
    # CMake >=3.16 can prevent download (error) of submodules by specifying an empty
    # string to git_submodule and setting cmake plicy CMP0097 to NEW.
  GIT_SUBMODULES
    ""
  CMAKE_ARGS
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

    #-DCMAKE_BUILD_TYPE=Debug
  CMAKE_CACHE_ARGS
    #-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
    -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}
    -DEMBREE_ISPC_SUPPORT:BOOL=${EMBREE_ISPC_SUPPORT}
    # We use OpenMP, Embree calls that internal (eventhough it is external to Embree)
    -DEMBREE_TASKING_SYSTEM:STRING=INTERNAL
  # Do not install
  # If one installes it and tries to use the installed library, then there exists a linkage
  # problem with TBB. Simply use the generated files in the binary directory.
  USES_TERMINAL_DOWNLOAD
    1
  USES_TERMINAL_UPDATE
    1
  GIT_PROGRESS
    1
)

# The following might be simplified, because we are not using TBB anymore.

# Get install and build directory from external project
externalproject_get_property(embree-external INSTALL_DIR BINARY_DIR)

set(
  embree_DIR "${INSTALL_DIR}"
  CACHE PATH "Path to embree installation"
  FORCE
  )
set(
  embree_INCLUDEDIR "${embree_DIR}/include"
  CACHE PATH "Path to embree include directories"
  FORCE
  )
set(
  embree_LIBRARYDIR "${embree_DIR}/lib"
  CACHE PATH "Path to embree library directories"
  FORCE
  )