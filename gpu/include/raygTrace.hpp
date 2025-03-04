#pragma once

#include <cuda.h>
#include <optix_stubs.h>

#include <cstring>
#include <filesystem>

#include <rayUtil.hpp>

#include "raygBoundary.hpp"
#include "raygLaunchParams.hpp"
#include "raygParticle.hpp"
#include "raygSBTRecords.hpp"
#include "raygTriangleGeometry.hpp"

#include <set>
#include <vcChecks.hpp>
#include <vcContext.hpp>
#include <vcCudaBuffer.hpp>
#include <vcLaunchKernel.hpp>
#include <vcLoadModules.hpp>
#include <vcUtil.hpp>

namespace viennaray::gpu {

using namespace viennacore;

template <class T, int D> class Trace {
public:
  /// constructor - performs all setup, including initializing
  /// optix, creates module, pipeline, programs, SBT, etc.
  explicit Trace(Context &passedContext) : context(passedContext) {
    initRayTracer();
  }

  void setGeometry(const TriangleMesh &passedMesh) {
    geometry.buildAccel(context, passedMesh, launchParams);
  }

  void setPipeline(std::string fileName, const std::filesystem::path &path) {
    // check if filename ends in .optixir
    if (fileName.find(".optixir") == std::string::npos) {
      if (fileName.find(".ptx") == std::string::npos)
        fileName += ".optixir";
    }

    pipelineFile = path / fileName;

    if (!std::filesystem::exists(pipelineFile)) {
      Logger::getInstance()
          .addError("Pipeline file " + fileName + " not found.")
          .print();
    }
  }

  void insertNextParticle(const Particle<T> &particle) {
    particles.push_back(particle);
  }

  void apply() {

    // if (numCellData != 0 && cellDataBuffer.sizeInBytes == 0) {
    //   cellDataBuffer.allocInit(numCellData * launchParams.numElements,
    //                            float(0));
    // }
    assert(cellDataBuffer.sizeInBytes / sizeof(float) ==
           numCellData * launchParams.numElements);

    // resize our cuda result buffer
    resultBuffer.allocInit(launchParams.numElements * numRates, float(0));
    launchParams.resultBuffer = (float *)resultBuffer.dPointer();

    if (materialIdsBuffer.sizeInBytes != 0) {
      launchParams.materialIds = (int *)materialIdsBuffer.dPointer();
    }

    if (useRandomSeed) {
      std::random_device rd;
      std::uniform_int_distribution<unsigned int> gen;
      launchParams.seed = gen(rd);
    } else {
      launchParams.seed = runNumber++;
    }

    int numPointsPerDim =
        static_cast<int>(std::sqrt(static_cast<T>(launchParams.numElements)));

    if (numberOfRaysFixed > 0) {
      numPointsPerDim = 1;
      numberOfRaysPerPoint = numberOfRaysFixed;
    }

    numRays = numPointsPerDim * numPointsPerDim * numberOfRaysPerPoint;
    if (numRays > (1 << 29)) {
      Logger::getInstance()
          .addWarning("Too many rays for single launch: " +
                      util::prettyDouble(numRays))
          .print();
      numberOfRaysPerPoint = (1 << 29) / (numPointsPerDim * numPointsPerDim);
      numRays = numPointsPerDim * numPointsPerDim * numberOfRaysPerPoint;
    }
    Logger::getInstance()
        .addDebug("Number of rays: " + util::prettyDouble(numRays))
        .print();

    // set up material specific sticking probabilities
    materialStickingBuffer.resize(particles.size());
    for (size_t i = 0; i < particles.size(); i++) {
      if (!particles[i].materialSticking.empty()) {
        if (uniqueMaterialIds.empty() || materialIdsBuffer.sizeInBytes == 0) {
          Logger::getInstance()
              .addError("Material IDs not set, when using material dependent "
                        "sticking.")
              .print();
        }
        std::vector<float> materialSticking(uniqueMaterialIds.size());
        unsigned currentId = 0;
        for (auto &matId : uniqueMaterialIds) {
          if (particles[i].materialSticking.find(matId) ==
              particles[i].materialSticking.end()) {
            materialSticking[currentId++] =
                static_cast<float>(particles[i].sticking);
          } else {
            materialSticking[currentId++] =
                static_cast<float>(particles[i].materialSticking[matId]);
          }
        }
        materialStickingBuffer[i].allocUpload(materialSticking);
      }
    }

    for (size_t i = 0; i < particles.size(); i++) {
      launchParams.particleIdx = static_cast<unsigned>(i);
      launchParams.cosineExponent =
          static_cast<float>(particles[i].cosineExponent);
      launchParams.sticking = static_cast<float>(particles[i].sticking);
      if (!particles[i].materialSticking.empty()) {
        assert(materialStickingBuffer[i].sizeInBytes != 0);
        launchParams.materialSticking =
            (float *)materialStickingBuffer[i].dPointer();
      }
      Vec3Df direction{static_cast<float>(particles[i].direction[0]),
                       static_cast<float>(particles[i].direction[1]),
                       static_cast<float>(particles[i].direction[2])};
      launchParams.source.directionBasis =
          rayInternal::getOrthonormalBasis<float>(direction);
      launchParamsBuffer.upload(&launchParams, 1);

      CUstream stream;
      CUDA_CHECK(StreamCreate(&stream));
      generateSBT(i);
      OPTIX_CHECK(optixLaunch(pipelines[i], stream,
                              /*! parameters and SBT */
                              launchParamsBuffer.dPointer(),
                              launchParamsBuffer.sizeInBytes, &sbts[i],
                              /*! dimensions of the launch: */
                              numPointsPerDim, numPointsPerDim,
                              numberOfRaysPerPoint));
    }

    // std::cout << util::prettyDouble(numRays * particles.size()) << std::endl;
    // float *temp = new float[launchParams.numElements];
    // resultBuffer.download(temp, launchParams.numElements);
    // for (int i = 0; i < launchParams.numElements; i++) {
    //   std::cout << temp[i] << " ";
    // }
    // delete temp;

    // sync - maybe remove in future
    cudaDeviceSynchronize();
    normalize();
  }

  void setElementData(CudaBuffer &passedCellDataBuffer, unsigned numData) {
    assert(passedCellDataBuffer.sizeInBytes / sizeof(float) / numData ==
           launchParams.numElements);
    cellDataBuffer = passedCellDataBuffer;
    numCellData = numData;
  }

  template <class NumericType>
  void setMaterialIds(const std::vector<NumericType> &materialIds,
                      const bool mapToConsecutive = true) {
    assert(materialIds.size() == launchParams.numElements);

    if (mapToConsecutive) {
      uniqueMaterialIds.clear();
      for (auto &matId : materialIds) {
        uniqueMaterialIds.insert(static_cast<int>(matId));
      }
      std::unordered_map<NumericType, unsigned> materialIdMap;
      int currentId = 0;
      for (auto &uniqueMaterialId : uniqueMaterialIds) {
        materialIdMap[uniqueMaterialId] = currentId++;
      }
      assert(currentId == materialIdMap.size());

      std::vector<int> materialIdsMapped(launchParams.numElements);
#pragma omp parallel for
      for (size_t i = 0; i < launchParams.numElements; i++) {
        materialIdsMapped[i] = materialIdMap[materialIds[i]];
      }
      materialIdsBuffer.allocUpload(materialIdsMapped);
    } else {
      std::vector<int> materialIdsMapped(launchParams.numElements);
      for (size_t i = 0; i < launchParams.numElements; i++) {
        materialIdsMapped[i] = static_cast<int>(materialIds[i]);
      }
      materialIdsBuffer.allocUpload(materialIdsMapped);
    }
  }

  void setNumberOfRaysPerPoint(const size_t pNumRays) {
    numberOfRaysPerPoint = pNumRays;
  }

  void setNumberOfRaysFixed(const size_t pNumRays) {
    numberOfRaysFixed = pNumRays;
  }

  void setUseRandomSeeds(const bool set) { useRandomSeed = set; }

  void getFlux(float *flux, int particleIdx, int dataIdx) {
    unsigned int offset = 0;
    for (size_t i = 0; i < particles.size(); i++) {
      if (particleIdx > i)
        offset += particles[i].dataLabels.size();
    }
    std::vector<float> temp(numRates * launchParams.numElements);
    resultBuffer.download(temp.data(), launchParams.numElements * numRates);
    offset = (offset + dataIdx) * launchParams.numElements;
    std::memcpy(flux, &temp[offset], launchParams.numElements * sizeof(float));
  }

  void setUseCellData(unsigned numData) { numCellData = numData; }

  void setPeriodicBoundary(const bool periodic) {
    launchParams.periodicBoundary = periodic;
  }

  void freeBuffers() {
    resultBuffer.free();
    hitgroupRecordBuffer.free();
    missRecordBuffer.free();
    raygenRecordBuffer.free();
    dataPerParticleBuffer.free();
    geometry.freeBuffers();
  }

  unsigned int prepareParticlePrograms() {
    if (particles.empty()) {
      Logger::getInstance().addWarning("No particles defined.").print();
      return 0;
    }

    createModule();
    createRaygenPrograms();
    createMissPrograms();
    createHitgroupPrograms();
    createPipelines();
    if (sbts.empty()) {
      for (size_t i = 0; i < particles.size(); i++) {
        OptixShaderBindingTable sbt = {};
        sbts.push_back(sbt);
      }
    }
    numRates = 0;
    std::vector<unsigned int> dataPerParticle;
    for (const auto &p : particles) {
      dataPerParticle.push_back(p.dataLabels.size());
      numRates += p.dataLabels.size();
    }
    dataPerParticleBuffer.allocUpload(dataPerParticle);
    launchParams.dataPerParticle =
        (unsigned int *)dataPerParticleBuffer.dPointer();
    Logger::getInstance()
        .addDebug("Number of flux arrays: " + std::to_string(numRates))
        .print();

    return numRates;
  }

  CudaBuffer &getData() { return cellDataBuffer; }

  CudaBuffer &getResults() { return resultBuffer; }

  [[nodiscard]] std::size_t getNumberOfRays() const { return numRays; }

  std::vector<Particle<T>> &getParticles() { return particles; }

  [[nodiscard]] unsigned int getNumberOfRates() const { return numRates; }

  [[nodiscard]] unsigned int getNumberOfElements() const {
    return launchParams.numElements;
  }

  void setParameters(CUdeviceptr d_params) {
    launchParams.customData = (void *)d_params;
  }

  auto &getParameterBuffer() { return parameterBuffer; }

protected:
  void normalize() {
    float sourceArea =
        (launchParams.source.maxPoint[0] - launchParams.source.minPoint[0]) *
        (launchParams.source.maxPoint[1] - launchParams.source.minPoint[1]);
    assert(resultBuffer.sizeInBytes != 0 &&
           "Normalization: Result buffer not initialized.");
    CUdeviceptr d_data = resultBuffer.dPointer();
    CUdeviceptr d_vertex = geometry.geometryVertexBuffer.dPointer();
    CUdeviceptr d_index = geometry.geometryIndexBuffer.dPointer();
    void *kernel_args[] = {
        &d_data,     &d_vertex, &d_index, &launchParams.numElements,
        &sourceArea, &numRays,  &numRates};

    LaunchKernel::launch(normModuleName, normKernelName, kernel_args, context);
  }

  void initRayTracer() {
    context.addModule(normModuleName);
    launchParamsBuffer.alloc(sizeof(launchParams));
    normKernelName.push_back(NumericType);
  }

  /// creates the module that contains all the programs we are going to use. We
  /// use a single module from a single .cu file
  void createModule() {
    moduleCompileOptions.maxRegisterCount =
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName =
        globalParamsName.c_str();

    pipelineLinkOptions.maxTraceDepth = 2;

    char log[2048];
    size_t sizeof_log = sizeof(log);

    size_t inputSize = 0;
    auto pipelineInput = getInputData(pipelineFile.c_str(), inputSize);

    OPTIX_CHECK(optixModuleCreate(context.optix, &moduleCompileOptions,
                                  &pipelineCompileOptions, pipelineInput,
                                  inputSize, log, &sizeof_log, &module));
    // if (sizeof_log > 1)
    //   PRINT(log);
  }

  /// does all setup for the raygen program
  void createRaygenPrograms() {
    raygenPGs.resize(particles.size());
    for (size_t i = 0; i < particles.size(); i++) {
      std::string entryFunctionName = "__raygen__" + particles[i].name;
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc pgDesc = {};
      pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      pgDesc.raygen.module = module;
      pgDesc.raygen.entryFunctionName = entryFunctionName.c_str();

      // OptixProgramGroup raypg;
      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(context.optix, &pgDesc, 1, &pgOptions,
                                          log, &sizeof_log, &raygenPGs[i]));
      // if (sizeof_log > 1)
      //   PRINT(log);
    }
  }

  /// does all setup for the miss program
  void createMissPrograms() {
    missPGs.resize(particles.size());
    for (size_t i = 0; i < particles.size(); i++) {
      std::string entryFunctionName = "__miss__" + particles[i].name;
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc pgDesc = {};
      pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      pgDesc.miss.module = module;
      pgDesc.miss.entryFunctionName = entryFunctionName.c_str();

      // OptixProgramGroup raypg;
      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(context.optix, &pgDesc, 1, &pgOptions,
                                          log, &sizeof_log, &missPGs[i]));
      // if (sizeof_log > 1)
      //   PRINT(log);
    }
  }

  /// does all setup for the hitgroup program
  void createHitgroupPrograms() {
    hitgroupPGs.resize(particles.size());
    for (size_t i = 0; i < particles.size(); i++) {
      std::string entryFunctionName = "__closesthit__" + particles[i].name;
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc pgDesc = {};
      pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      pgDesc.hitgroup.moduleCH = module;
      pgDesc.hitgroup.entryFunctionNameCH = entryFunctionName.c_str();

      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(context.optix, &pgDesc, 1, &pgOptions,
                                          log, &sizeof_log, &hitgroupPGs[i]));
      // if (sizeof_log > 1)
      //   PRINT(log);
    }
  }

  /// assembles the full pipeline of all programs
  void createPipelines() {
    pipelines.resize(particles.size());
    for (size_t i = 0; i < particles.size(); i++) {
      std::vector<OptixProgramGroup> programGroups;
      programGroups.push_back(raygenPGs[i]);
      programGroups.push_back(missPGs[i]);
      programGroups.push_back(hitgroupPGs[i]);

      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixPipelineCreate(
          context.optix, &pipelineCompileOptions, &pipelineLinkOptions,
          programGroups.data(), static_cast<int>(programGroups.size()), log,
          &sizeof_log, &pipelines[i]));
      // #ifndef NDEBUG
      //       if (sizeof_log > 1)
      //         PRINT(log);
      // #endif
    }
    // probably not needed in current implementation but maybe something to
    // think about in future OPTIX_CHECK(optixPipelineSetStackSize(
    //     pipeline,
    //     2 * 1024, // The direct stack size requirement for direct callables
    //               // invoked from IS or AH.
    //     2 * 1024, // The direct stack size requirement for direct callables
    //               // invoked from RG, MS, or CH.
    //     2 * 1024, // The continuation stack requirement.
    //     1         // The maximum depth of a traversable graph passed to trace
    //     ));
  }

  /// constructs the shader binding table
  void generateSBT(const size_t i) {
    // build raygen record
    RaygenRecord raygenRecord = {};
    optixSbtRecordPackHeader(raygenPGs[i], &raygenRecord);
    raygenRecord.data = nullptr;
    raygenRecordBuffer.allocUploadSingle(raygenRecord);
    sbts[i].raygenRecord = raygenRecordBuffer.dPointer();

    // build miss record
    MissRecord missRecord = {};
    optixSbtRecordPackHeader(missPGs[i], &missRecord);
    missRecord.data = nullptr;
    missRecordBuffer.allocUploadSingle(missRecord);
    sbts[i].missRecordBase = missRecordBuffer.dPointer();
    sbts[i].missRecordStrideInBytes = sizeof(MissRecord);
    sbts[i].missRecordCount = 1;

    // build hitgroup records
    std::vector<HitgroupRecord> hitgroupRecords;

    // geometry hitgroup
    HitgroupRecord geometryHitgroupRecord = {};
    optixSbtRecordPackHeader(hitgroupPGs[i], &geometryHitgroupRecord);
    geometryHitgroupRecord.data.vertex =
        (Vec3Df *)geometry.geometryVertexBuffer.dPointer();
    geometryHitgroupRecord.data.index =
        (Vec3D<unsigned> *)geometry.geometryIndexBuffer.dPointer();
    geometryHitgroupRecord.data.isBoundary = false;
    geometryHitgroupRecord.data.cellData = (void *)cellDataBuffer.dPointer();
    hitgroupRecords.push_back(geometryHitgroupRecord);

    // boundary hitgroup
    HitgroupRecord boundaryHitgroupRecord = {};
    optixSbtRecordPackHeader(hitgroupPGs[i], &boundaryHitgroupRecord);
    boundaryHitgroupRecord.data.vertex =
        (Vec3Df *)geometry.boundaryVertexBuffer.dPointer();
    boundaryHitgroupRecord.data.index =
        (Vec3D<unsigned> *)geometry.boundaryIndexBuffer.dPointer();
    boundaryHitgroupRecord.data.isBoundary = true;
    hitgroupRecords.push_back(boundaryHitgroupRecord);

    hitgroupRecordBuffer.allocUpload(hitgroupRecords);
    sbts[i].hitgroupRecordBase = hitgroupRecordBuffer.dPointer();
    sbts[i].hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbts[i].hitgroupRecordCount = 2;
  }

protected:
  // context for cuda kernels
  Context &context;
  std::filesystem::path pipelineFile;

  // geometry
  TriangleGeometry geometry;
  std::set<int> uniqueMaterialIds;
  CudaBuffer materialIdsBuffer;

  // particles
  unsigned int numRates = 0;
  std::vector<Particle<T>> particles;
  CudaBuffer dataPerParticleBuffer;               // same for all particles
  CudaBuffer parameterBuffer;                     // same for all particles
  std::vector<CudaBuffer> materialStickingBuffer; // different for particles

  // sbt data
  CudaBuffer cellDataBuffer;

  std::vector<OptixPipeline> pipelines;
  OptixPipelineCompileOptions pipelineCompileOptions = {};
  OptixPipelineLinkOptions pipelineLinkOptions = {};

  OptixModule module{};
  OptixModuleCompileOptions moduleCompileOptions = {};

  // program groups, and the SBT built around
  std::vector<OptixProgramGroup> raygenPGs;
  CudaBuffer raygenRecordBuffer;
  std::vector<OptixProgramGroup> missPGs;
  CudaBuffer missRecordBuffer;
  std::vector<OptixProgramGroup> hitgroupPGs;
  CudaBuffer hitgroupRecordBuffer;
  std::vector<OptixShaderBindingTable> sbts;

  // launch parameters, on the host, constant for all particles
  LaunchParams launchParams;
  CudaBuffer launchParamsBuffer;

  // results Buffer
  CudaBuffer resultBuffer;

  bool useRandomSeed = false;
  unsigned numCellData = 0;
  unsigned numberOfRaysPerPoint = 3000;
  unsigned numberOfRaysFixed = 0;
  int runNumber = 0;

  size_t numRays = 0;
  std::string globalParamsName = "launchParams";

  const std::string normModuleName = "normKernels.ptx";
  std::string normKernelName = "normalize_surface_";

  static constexpr char NumericType =
      'f'; // std::is_same_v<T, float> ? 'f' : 'd';
};

} // namespace viennaray::gpu
