#### Set variables
set(VIENNARAY_CUDA_KERNELS
    "${PROJECT_SOURCE_DIR}/gpu/kernels/normKernels.cu"
    CACHE STRING "ViennaRay CUDA kernel source files.")
set(VIENNARAY_PIPELINE_DIR
    "${PROJECT_SOURCE_DIR}/gpu/pipelines"
    CACHE STRING "ViennaRay pipeline directory.")

# Incldue directories for PTX and optix-ir compilation (force cache to update with new entries)
if(NOT VIENNARAY_PASSED_FIRST_CONFIGURE)
  set(VIENNACORE_NVCC_INCLUDE_DIRS
      ${VIENNACORE_NVCC_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include/viennaray/gpu
      CACHE STRING "PTX include directories" FORCE)
  if(VIENNARAY_GPU_DOUBLE_PRECISION)
    set(VIENNACORE_NVCC_DEFINES
        ${VIENNACORE_NVCC_DEFINES} "VIENNARAY_GPU_DOUBLE_PRECISION=1"
        CACHE STRING "PTX compile definitions" FORCE)
  endif()
endif()

# The optixir commands do not scan headers, so a change to LaunchParams or the
# SBT records leaves every module stale while the host struct moves under it --
# the device then reads the old layout and misbehaves with no build error. List
# the shared headers explicitly so a layout change rebuilds the pipelines.
# CACHE so consumers (ViennaPS's callable wrapper) can depend on them too:
# a plain variable does not cross the directory scope CPM adds this in.
set(VIENNARAY_GPU_SHARED_HEADERS
    "${PROJECT_SOURCE_DIR}/include/viennaray/gpu/raygLaunchParams.hpp;${PROJECT_SOURCE_DIR}/include/viennaray/gpu/raygSBTRecords.hpp;${PROJECT_SOURCE_DIR}/include/viennaray/gpu/raygPerRayData.hpp;${PROJECT_SOURCE_DIR}/include/viennaray/gpu/raygReflection.hpp;${PROJECT_SOURCE_DIR}/include/viennaray/gpu/raygSource.hpp"
    CACHE STRING "ViennaRay shared GPU device headers." FORCE)
# Add the general pipelines and callable wrapper
viennacore_add_optixir(GeneralPipelineDisk ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineDisk.cu
                       DEPENDS "${VIENNARAY_GPU_SHARED_HEADERS}")
viennacore_add_optixir(GeneralPipelineTriangle ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineTriangle.cu
                       DEPENDS "${VIENNARAY_GPU_SHARED_HEADERS}")
viennacore_add_optixir(GeneralPipelineLine ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineLine.cu
                       DEPENDS "${VIENNARAY_GPU_SHARED_HEADERS}")
viennacore_add_optixir(GeneralPipelineCell ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineCell.cu
                       DEPENDS "${VIENNARAY_GPU_SHARED_HEADERS}")
viennacore_add_optixir(ViennaRayCallableWrapper ${VIENNARAY_PIPELINE_DIR}/CallableWrapper.cu
                       DEPENDS "${VIENNARAY_PIPELINE_DIR}/Particle.cuh"
                               "${VIENNARAY_GPU_SHARED_HEADERS}")

# Add the norm kernels
viennacore_add_ptx(normKernels ${VIENNARAY_CUDA_KERNELS})

# CMake target to collect all GPU dependencies for tests and examples
set(VIENNARAY_GPU_DEPENDENCIES
    GeneralPipelineDisk GeneralPipelineTriangle GeneralPipelineLine
    GeneralPipelineCell ViennaRayCallableWrapper normKernels
    CACHE STRING "ViennaRay GPU dependencies." FORCE)
# FORCE: the list names build targets, and a stale cached list from before a
# geometry kind existed silently drops its pipeline from every consumer.
# Install PTX files
install(DIRECTORY "${VIENNACORE_NVCC_PTX_DIR}/" DESTINATION "lib/ptx")

if(VIENNARAY_BUILD_EXAMPLES)
  message(STATUS "[ViennaRay] Adding GPU Examples")
  add_subdirectory(examples)
endif(VIENNARAY_BUILD_EXAMPLES)

if(VIENNARAY_BUILD_TESTS)
  message(STATUS "[ViennaRay] Adding GPU Tests")
  add_subdirectory(tests)
endif()

set(VIENNARAY_PASSED_FIRST_CONFIGURE
    ON
    CACHE INTERNAL "Indicates whether the first configure has been passed.")
