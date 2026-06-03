#### Set variables
set(VIENNARAY_CUDA_KERNELS
    "${PROJECT_SOURCE_DIR}/gpu/kernels/normKernels.cu"
    CACHE STRING "ViennaRay CUDA kernel source files.")
set(VIENNARAY_PIPELINE_DIR
    "${PROJECT_SOURCE_DIR}/gpu/pipelines"
    CACHE STRING "ViennaRay pipeline directory.")

# Include directories and definitions for PTX and optix-ir compilation.
set(_VIENNARAY_NVCC_INCLUDE_DIRS ${VIENNACORE_NVCC_INCLUDE_DIRS}
                                 ${PROJECT_SOURCE_DIR}/include/viennaray/gpu)
list(REMOVE_DUPLICATES _VIENNARAY_NVCC_INCLUDE_DIRS)
set(VIENNACORE_NVCC_INCLUDE_DIRS
    ${_VIENNARAY_NVCC_INCLUDE_DIRS}
    CACHE STRING "PTX include directories" FORCE)

set(_VIENNARAY_NVCC_DEFINES ${VIENNACORE_NVCC_DEFINES})
list(REMOVE_ITEM _VIENNARAY_NVCC_DEFINES "VIENNARAY_GPU_DOUBLE_PRECISION=1"
     "VIENNARAY_GPU_DOUBLE_PRECISION" "VIENNARAY_BENCHMARK=1"
     "VIENNARAY_BENCHMARK")
if(VIENNARAY_GPU_DOUBLE_PRECISION)
  list(APPEND _VIENNARAY_NVCC_DEFINES "VIENNARAY_GPU_DOUBLE_PRECISION=1")
endif()
if(VIENNARAY_BENCHMARK)
  list(APPEND _VIENNARAY_NVCC_DEFINES "VIENNARAY_BENCHMARK=1")
endif()
list(REMOVE_DUPLICATES _VIENNARAY_NVCC_DEFINES)
set(VIENNACORE_NVCC_DEFINES
    ${_VIENNARAY_NVCC_DEFINES}
    CACHE STRING "PTX compile definitions" FORCE)

# Add the general pipelines and callable wrapper
viennacore_add_optixir(GeneralPipelineDisk ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineDisk.cu)
viennacore_add_optixir(GeneralPipelineTriangle ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineTriangle.cu)
viennacore_add_optixir(GeneralPipelineLine ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineLine.cu)
viennacore_add_optixir(ViennaRayCallableWrapper ${VIENNARAY_PIPELINE_DIR}/CallableWrapper.cu
                       DEPENDS "${VIENNARAY_PIPELINE_DIR}/Particle.cuh")

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
