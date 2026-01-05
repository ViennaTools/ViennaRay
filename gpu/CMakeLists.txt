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

# Add the general pipelines and callable wrapper
viennacore_add_optixir(GeneralPipelineDisk ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineDisk.cu)
viennacore_add_optixir(GeneralPipelineTriangle ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineTriangle.cu)
viennacore_add_optixir(GeneralPipelineLine ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineLine.cu)
viennacore_add_optixir(ViennaRayCallableWrapper ${VIENNARAY_PIPELINE_DIR}/CallableWrapper.cu)

# Add the norm kernels
viennacore_add_ptx(normKernels ${VIENNARAY_CUDA_KERNELS})

# CMake target to collect all GPU dependencies for tests and examples
set(VIENNARAY_GPU_DEPENDENCIES
    GeneralPipelineDisk GeneralPipelineTriangle GeneralPipelineLine normKernels
    CACHE STRING "ViennaRay GPU dependencies.")
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
