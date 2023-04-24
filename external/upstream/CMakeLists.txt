# external project base directory for all projects in this hierarchy
set_property(DIRECTORY PROPERTY EP_BASE ${DEPENDENCIES_DIR})

if(VIENNARAY_BUILD_PYTHON)
  add_subdirectory(pybind11)
endif()

add_subdirectory(embree)
