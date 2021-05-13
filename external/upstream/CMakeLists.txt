## external project base directory for all projects in this hierarchy
set_property(DIRECTORY PROPERTY EP_BASE ${DEPENDENCIES_DIR})

# build embree if no path was specified
if(NOT DEFINED embree_DIR)
    add_subdirectory(embree)
    message(STATUS "embree_DIR: ${embree_DIR}")
endif()