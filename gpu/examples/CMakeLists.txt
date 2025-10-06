project(ViennaRay-GPU_Examples)

configure_file(Resources/trenchMesh.dat ${CMAKE_CURRENT_BINARY_DIR}/trenchMesh.dat COPYONLY)

add_gpu_executable(trenchGPU target_name trench.cpp ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineDisk.cu
                                                    ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineTriangle.cu
                                                    ${VIENNARAY_PIPELINE_DIR}/GeneralPipelineLine.cu
                                                    ${VIENNARAY_PIPELINE_DIR}/CallableWrapper.cu)
