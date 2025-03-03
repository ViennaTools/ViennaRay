project(ViennaRay-GPU)

option(VIENNARAY_ENABLE_OPTIXIR_SUPPORT
        "Enable support for generating OptiX-IR targeted input files" ON)

if (CUDA_VERSION VERSION_LESS 11.7)
    if (VIENNARAY_ENABLE_OPTIXIR_SUPPORT)
        message(
                SEND_ERROR
                "VIENNARAY_ENABLE_OPTIXIR_SUPPORT is not supported in CUDA versions less than 11.7")
    endif ()
endif ()

if (VIENNARAY_ENABLE_OPTIXIR_SUPPORT)
    option(VIENNARAY_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" ON)
    option(VIENNARAY_GENERATE_PTX "Generate PTX OptiX shaders" OFF)
else ()
    option(VIENNARAY_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" OFF)
    option(VIENNARAY_GENERATE_PTX "Generate PTX OptiX shaders" ON)
endif ()

#### Set variables
set(VIENNARAY_GPU_INCLUDE_DIR
        "${PROJECT_SOURCE_DIR}/include"
        CACHE STRING "ViennaRay GPU headers."
)
set(VIENNARAY_CUDA_KERNELS
        "${PROJECT_SOURCE_DIR}/kernels/normKernels.cu"
        CACHE STRING "Path to the ViennaRay CUDA kernel source file.")

if (VIENNARAY_BUILD_EXAMPLES)
    message(STATUS "[ViennaRay] Adding GPU Examples")
    add_subdirectory(examples)
endif (VIENNARAY_BUILD_EXAMPLES)

if (VIENNARAY_BUILD_TESTS)
    message(STATUS "[ViennaRay] Adding GPU Tests")
    add_subdirectory(tests)
endif ()


