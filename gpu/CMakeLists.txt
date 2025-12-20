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

if(NOT DEFINED VIENNACORE_PTX_INCLUDE_DIRS)
  set(VIENNACORE_PTX_INCLUDE_DIRS
      ""
      CACHE STRING "ViennaCore PTX include directories")
endif()
list(APPEND VIENNACORE_PTX_INCLUDE_DIRS "${VIENNARAY_GPU_INCLUDE}")

if(VIENNARAY_GPU_DOUBLE_PRECISION)
  if(NOT DEFINED VIENNACORE_PTX_DEFINES)
    set(VIENNACORE_PTX_DEFINES
        ""
        CACHE STRING "ViennaCore PTX compile definitions")
  endif()
  list(APPEND VIENNACORE_PTX_DEFINES "VIENNARAY_GPU_DOUBLE_PRECISION=1")
endif()

# Add the general pipelines and callable wrapper
viennacore_add_optixir(disk_pipeline ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineDisk.cu)
viennacore_add_optixir(triangle_pipeline ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineTriangle.cu)
viennacore_add_optixir(line_pipeline ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineLine.cu)
viennacore_add_optixir(callable_wrapper ${VIENNARAY_PIPELINE_DIR}/CallableWrapper.cu)

# Add the norm kernels
viennacore_add_ptx(norm_kernels ${VIENNARAY_CUDA_KERNELS})

set(VIENNARAY_GPU_DEPENDENCIES disk_pipeline triangle_pipeline line_pipeline callable_wrapper
                               norm_kernels)

if(VIENNARAY_BUILD_EXAMPLES)
  message(STATUS "[ViennaRay] Adding GPU Examples")
  add_subdirectory(examples)
endif(VIENNARAY_BUILD_EXAMPLES)

if(VIENNARAY_BUILD_TESTS)
  message(STATUS "[ViennaRay] Adding GPU Tests")
  add_subdirectory(tests)
endif()
