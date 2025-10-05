#pragma once

#include <cuda.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cstring>
#include <filesystem>

#include <rayParticle.hpp>
#include <rayPointNeighborhood.hpp>
#include <rayUtil.hpp>

#include "raygLaunchParams.hpp"
#include "raygSBTRecords.hpp"

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
  explicit Trace(std::shared_ptr<DeviceContext> &passedContext,
                 std::string &&geometryType)
      : context(passedContext), geometryType_(std::move(geometryType)) {
    initRayTracer();
  }

  explicit Trace(std::string &&geometryType, unsigned deviceID = 0)
      : geometryType_(std::move(geometryType)) {
    context = DeviceContext::getContextFromRegistry(deviceID);
    if (!context) {
      Logger::getInstance()
          .addError("No context found for device ID " +
                    std::to_string(deviceID) +
                    ". Create and register a context first.")
          .print();
    }
    initRayTracer();
  }

  ~Trace() { freeBuffers(); }

  virtual void setGeometry(const LineMesh &passedMesh) {}
  virtual void setGeometry(const DiskMesh &passedMesh) {}
  virtual void setGeometry(const TriangleMesh &passedMesh) {}

  void setPipeline(std::string fileName, const std::filesystem::path &path) {
    // check if filename ends in .optixir
    if (fileName.find(".optixir") == std::string::npos) {
      if (fileName.find(".ptx") == std::string::npos)
        fileName += ".optixir";
    }

    std::filesystem::path p(fileName);
    std::string base = p.stem().string();
    std::string ext = p.extension().string();
    std::string finalName = base + geometryType_ + ext;

    pipelineFile = path / finalName;

    if (!std::filesystem::exists(pipelineFile)) {
      Logger::getInstance()
          .addError("Pipeline file " + finalName + " not found.")
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

    launchParams.gridDelta = gridDelta;

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

    // Every particle gets its own stream and launch parameters
    std::vector<cudaStream_t> streams(particles.size());
    launchParamsBuffers.resize(particles.size());

    if (particleMap_.empty()) {
      Logger::getInstance()
          .addError("No particle name->particleType mapping provided.")
          .print();
    }

    for (size_t i = 0; i < particles.size(); i++) {
      auto it = particleMap_.find(particles[i].name);
      if (it == particleMap_.end()) {
        Logger::getInstance()
            .addError("Unknown particle name: " + particles[i].name)
            .print();
      }
      launchParams.particleType = it->second;
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

      launchParamsBuffers[i].alloc(sizeof(launchParams));
      launchParamsBuffers[i].upload(&launchParams, 1);

      CUDA_CHECK(StreamCreate(&streams[i]));
    }

    generateSBT();

    // TODO: Multiple streams seem to give same performance as single stream
    for (size_t i = 0; i < particles.size(); i++) {
      OPTIX_CHECK(optixLaunch(pipeline, streams[i],
                              /*! parameters and SBT */
                              launchParamsBuffers[i].dPointer(),
                              launchParamsBuffers[i].sizeInBytes, &sbt,
                              /*! dimensions of the launch: */
                              numberOfRaysPerPoint, numPointsPerDim,
                              numPointsPerDim));
    }
    // std::cout << util::prettyDouble(numRays * particles.size()) << std::endl;
    // float *temp = new float[launchParams.numElements];
    // resultBuffer.download(temp, launchParams.numElements);
    // for (int i = 0; i < launchParams.numElements; i++) {
    //   std::cout << temp[i] << " ";
    // }
    // delete temp;

    // sync - maybe remove in future
    // cudaDeviceSynchronize();
    for (auto &s : streams) {
      CUDA_CHECK(StreamSynchronize(s));
      CUDA_CHECK(StreamDestroy(s));
    }
    normalize();
    results.resize(launchParams.numElements * numRates);
    // cudaDeviceSynchronize(); // download is sync anyway
    resultBuffer.download(results.data(), launchParams.numElements * numRates);
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

  void
  setParticleCallableMap(std::tuple<std::unordered_map<std::string, unsigned>,
                                    std::vector<viennaray::gpu::CallableConfig>>
                             maps) {
    particleMap_ = std::get<0>(maps);
    callableMap_ = std::get<1>(maps);
  }

  void getFlux(float *flux, int particleIdx, int dataIdx,
               int smoothingNeighbors = 0) {
    unsigned int offset = 0;
    for (size_t i = 0; i < particles.size(); i++) {
      if (particleIdx > i)
        offset += particles[i].dataLabels.size();
    }
    offset = (offset + dataIdx) * launchParams.numElements;
    std::vector<float> temp(launchParams.numElements);
    std::memcpy(temp.data(), results.data() + offset,
                launchParams.numElements * sizeof(float));
    if (smoothingNeighbors > 0)
      smoothFlux(temp, smoothingNeighbors);
    std::memcpy(flux, temp.data(), launchParams.numElements * sizeof(float));
  }

  virtual void smoothFlux(std::vector<float> &flux, int smoothingNeighbors) {}

  void setUseCellData(unsigned numData) { numCellData = numData; }

  void setPeriodicBoundary(const bool periodic) {
    launchParams.periodicBoundary = periodic;
  }

  void freeBuffers() {
    resultBuffer.free();
    hitgroupRecordBuffer.free();
    missRecordBuffer.free();
    raygenRecordBuffer.free();
    directCallableRecordBuffer.free();
    dataPerParticleBuffer.free();
    for (auto &buffer : launchParamsBuffers) {
      buffer.free();
    }
    materialIdsBuffer.free();
    for (auto &buffer : materialStickingBuffer) {
      buffer.free();
    }
    neighborsBuffer.free();
    areaBuffer.free();
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
    createDirectCallablePrograms();
    createPipelines();

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

protected:
  virtual void normalize() {}

  void initRayTracer() {
    context->addModule(normModuleName);
    // launchParamsBuffer.alloc(sizeof(launchParams));
    normKernelName.append(geometryType_ + "_");
    normKernelName.push_back(NumericType);
  }

  /// creates the module that contains all the programs we are going to use.
  /// We use a single module from a single .cu file
  void createModule() {
    moduleCompileOptions.maxRegisterCount =
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    // moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    // moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 4; // TODO: what is this
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName =
        globalParamsName.c_str();

    pipelineLinkOptions.maxTraceDepth = 1;

    char log[2048];
    size_t sizeof_log = sizeof(log);

    size_t inputSize = 0;
    auto pipelineInput = getInputData(pipelineFile.c_str(), inputSize);

    OPTIX_CHECK(optixModuleCreate(context->optix, &moduleCompileOptions,
                                  &pipelineCompileOptions, pipelineInput,
                                  inputSize, log, &sizeof_log, &module));
    // if (sizeof_log > 1)
    //   PRINT(log);
  }

  /// does all setup for the raygen program
  void createRaygenPrograms() {
    std::string entryFunctionName = "__raygen__";
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = entryFunctionName.c_str();

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context->optix, &pgDesc, 1, &pgOptions,
                                        log, &sizeof_log, &raygenPG));
    // if (sizeof_log > 1)
    //   PRINT(log);
  }

  /// does all setup for the miss program
  void createMissPrograms() {
    std::string entryFunctionName = "__miss__";
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = entryFunctionName.c_str();

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context->optix, &pgDesc, 1, &pgOptions,
                                        log, &sizeof_log, &missPG));
    // if (sizeof_log > 1)
    //   PRINT(log);
  }

  /// does all setup for the hitgroup program
  void createHitgroupPrograms() {
    std::string entryFunctionNameIS = "__intersection__";
    std::string entryFunctionNameCH = "__closesthit__";
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = entryFunctionNameCH.c_str();

    if (geometryType_ != "Triangle") {
      pgDesc.hitgroup.moduleIS = module;
      pgDesc.hitgroup.entryFunctionNameIS = entryFunctionNameIS.c_str();
    }

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context->optix, &pgDesc, 1, &pgOptions,
                                        log, &sizeof_log, &hitgroupPG));
    // if (sizeof_log > 1)
    //   PRINT(log);
  }

  /// does all setup for the direct callables
  void createDirectCallablePrograms() {
    if (callableMap_.empty()) {
      Logger::getInstance()
          .addError("No particleType->callable mapping provided.")
          .print();
    }
    unsigned numCallables =
        maxParticleTypes * static_cast<unsigned>(CallableSlot::COUNT);
    std::vector<std::string> entryFunctionNames(numCallables,
                                                "__direct_callable__noop");
    for (const auto &cfg : callableMap_) {
      entryFunctionNames[callableIndex(cfg.particle, cfg.slot)] = cfg.callable;
    }

    directCallablePGs.resize(numCallables);
    for (size_t i = 0; i < numCallables; i++) {
      OptixProgramGroupOptions dcOptions = {};
      OptixProgramGroupDesc dcDesc = {};
      dcDesc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
      dcDesc.callables.moduleDC = module;
      dcDesc.callables.entryFunctionNameDC = entryFunctionNames[i].c_str();

      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(context->optix, &dcDesc, 1,
                                          &dcOptions, log, &sizeof_log,
                                          &directCallablePGs[i]));
      // if (sizeof_log > 1)
      //   PRINT(log);
    }
  }

  /// assembles the full pipeline of all programs
  void createPipelines() {
    std::vector<OptixProgramGroup> programGroups;
    programGroups.push_back(raygenPG);
    programGroups.push_back(missPG);
    programGroups.push_back(hitgroupPG);

    for (size_t j = 0; j < directCallablePGs.size(); j++) {
      programGroups.push_back(directCallablePGs[j]);
    }

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(context->optix, &pipelineCompileOptions,
                                    &pipelineLinkOptions, programGroups.data(),
                                    static_cast<int>(programGroups.size()), log,
                                    &sizeof_log, &pipeline));
    // #ifndef NDEBUG
    //       if (sizeof_log > 1)
    //         PRINT(log);
    // #endif

    OptixStackSizes stackSizes = {};
    for (auto &pg : programGroups) {
      optixUtilAccumulateStackSizes(pg, &stackSizes, pipeline);
    }

    unsigned int dcStackFromTrav = 0;
    unsigned int dcStackFromState = 0;
    unsigned int continuationStack = 0;

    // These need to be adjusted when using nested callables
    // or recursive tracing
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stackSizes,
        pipelineLinkOptions.maxTraceDepth, // OptixTrace recursion depth
        0,                                 // continuation callable depth
        1,                                 // direct callable depth
        &dcStackFromTrav, &dcStackFromState, &continuationStack));

    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline,
        dcStackFromTrav,   // stack size for DirectCallables from IS or AH.
        dcStackFromState,  // stack size for DirectCallables from RG, MS or CH.
        continuationStack, // continuation stack size
        1));               // nested traversable graph depth
  }

  virtual void buildHitGroups() {}

  /// constructs the shader binding table
  void generateSBT() {
    // build raygen record
    RaygenRecord raygenRecord = {};
    optixSbtRecordPackHeader(raygenPG, &raygenRecord);
    raygenRecord.data = nullptr;
    raygenRecordBuffer.allocUploadSingle(raygenRecord);
    sbt.raygenRecord = raygenRecordBuffer.dPointer();

    // build miss record
    MissRecord missRecord = {};
    optixSbtRecordPackHeader(missPG, &missRecord);
    missRecord.data = nullptr;
    missRecordBuffer.allocUploadSingle(missRecord);
    sbt.missRecordBase = missRecordBuffer.dPointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = 1;

    // build geometry specific hitgroup records
    buildHitGroups();

    // callable programs
    std::vector<CallableRecord> callableRecords(directCallablePGs.size());
    for (size_t j = 0; j < directCallablePGs.size(); ++j) {
      CallableRecord callableRecord = {};
      optixSbtRecordPackHeader(directCallablePGs[j], &callableRecord);
      callableRecords[j] = callableRecord;
    }
    directCallableRecordBuffer.allocUpload(callableRecords);

    sbt.callablesRecordBase = directCallableRecordBuffer.dPointer();
    sbt.callablesRecordStrideInBytes = sizeof(CallableRecord);
    sbt.callablesRecordCount =
        static_cast<unsigned int>(directCallablePGs.size());
  }

protected:
  // context for cuda kernels
  std::shared_ptr<DeviceContext> context;
  std::filesystem::path pipelineFile;

  // Disk specific
  PointNeighborhood<float, D> pointNeighborhood_;

  std::string geometryType_;
  std::unordered_map<std::string, unsigned> particleMap_;
  std::vector<CallableConfig> callableMap_;

  std::set<int> uniqueMaterialIds;
  CudaBuffer materialIdsBuffer;

  CudaBuffer neighborsBuffer;
  float gridDelta = 0.0f;

  CudaBuffer areaBuffer;

  Vec3Df minBox;
  Vec3Df maxBox;

  // particles
  unsigned int maxParticleTypes = 5; // max nr. of different particle types
                                     // nr. of particles per type is not limited
  unsigned int numRates = 0;
  std::vector<Particle<T>> particles;
  CudaBuffer dataPerParticleBuffer;               // same for all particles
  std::vector<CudaBuffer> materialStickingBuffer; // different for particles

  // sbt data
  CudaBuffer cellDataBuffer;

  OptixPipeline pipeline{};
  OptixPipelineCompileOptions pipelineCompileOptions = {};
  OptixPipelineLinkOptions pipelineLinkOptions = {};

  OptixModule module{};
  OptixModuleCompileOptions moduleCompileOptions = {};

  // program groups, and the SBT built around
  OptixProgramGroup raygenPG;
  CudaBuffer raygenRecordBuffer;
  OptixProgramGroup missPG;
  CudaBuffer missRecordBuffer;
  OptixProgramGroup hitgroupPG;
  CudaBuffer hitgroupRecordBuffer;
  std::vector<OptixProgramGroup> directCallablePGs;
  CudaBuffer directCallableRecordBuffer;
  OptixShaderBindingTable sbt = {};

  // launch parameters, on the host, constant for all particles
  LaunchParams launchParams;
  std::vector<CudaBuffer> launchParamsBuffers;

  // results Buffer
  CudaBuffer resultBuffer;
  std::vector<float> results;

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
