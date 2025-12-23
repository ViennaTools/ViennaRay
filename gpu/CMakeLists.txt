#### Set variables
set(VIENNARAY_GPU_INCLUDE
    "${PROJECT_SOURCE_DIR}/gpu/include"
    CACHE STRING "ViennaRay GPU headers.")
set(VIENNARAY_CUDA_KERNELS
    "${PROJECT_SOURCE_DIR}/gpu/kernels/normKernels.cu"
    CACHE STRING "ViennaRay CUDA kernel source files.")
set(VIENNARAY_PIPELINE_DIR
    "${PROJECT_SOURCE_DIR}/gpu/pipelines"
    CACHE STRING "ViennaRay pipeline directory.")

target_include_directories(${PROJECT_NAME}
                           INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/gpu/include>)

# Incldue directories for PTX and optix-ir compilation (force cache to update with new entries)
set(VIENNACORE_PTX_INCLUDE_DIRS
    ${VIENNACORE_PTX_INCLUDE_DIRS} ${VIENNARAY_GPU_INCLUDE}
    CACHE STRING "PTX include directories" FORCE)
if(VIENNARAY_GPU_DOUBLE_PRECISION)
  set(VIENNACORE_PTX_DEFINES
      ${VIENNACORE_PTX_DEFINES} "VIENNARAY_GPU_DOUBLE_PRECISION=1"
      CACHE STRING "PTX compile definitions" FORCE)
endif()

# Add the general pipelines and callable wrapper
viennacore_add_optixir(diskPipeline ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineDisk.cu)
viennacore_add_optixir(trianglePipeline ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineTriangle.cu)
viennacore_add_optixir(linePipeline ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineLine.cu)
viennacore_add_optixir(ViennaRayCallableWrapper ${VIENNARAY_PIPELINE_DIR}/CallableWrapper.cu)

# Add the norm kernels
viennacore_add_ptx(normKernels ${VIENNARAY_CUDA_KERNELS})

set(VIENNARAY_GPU_DEPENDENCIES diskPipeline trianglePipeline linePipeline ViennaRayCallableWrapper
                               normKernels)

if(VIENNARAY_BUILD_EXAMPLES)
  message(STATUS "[ViennaRay] Adding GPU Examples")
  add_subdirectory(examples)
endif(VIENNARAY_BUILD_EXAMPLES)

if(VIENNARAY_BUILD_TESTS)
  message(STATUS "[ViennaRay] Adding GPU Tests")
  add_subdirectory(tests)
endif()
