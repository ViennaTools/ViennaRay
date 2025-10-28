#pragma once

#include <cuda.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cstring>
#include <filesystem>

#include <rayParticle.hpp>
#include <rayPointNeighborhood.hpp>
#include <rayUtil.hpp>

#include "raygCallableConfig.hpp"
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
  Trace(std::shared_ptr<DeviceContext> &passedContext,
        std::string &&geometryType)
      : context_(passedContext), geometryType_(std::move(geometryType)) {
    initRayTracer();
  }

  Trace(std::string &&geometryType, unsigned deviceID = 0)
      : geometryType_(std::move(geometryType)) {
    context_ = DeviceContext::getContextFromRegistry(deviceID);
    if (!context_) {
      Logger::getInstance()
          .addError("No context found for device ID " +
                    std::to_string(deviceID) +
                    ". Create and register a context first.")
          .print();
    }
    initRayTracer();
  }

  ~Trace() { freeBuffers(); }

  void setCallables(std::string fileName, const std::filesystem::path &path) {
    // check if filename ends in .optixir
    if (fileName.find(".optixir") == std::string::npos) {
      if (fileName.find(".ptx") == std::string::npos)
        fileName += ".optixir";
    }

    std::filesystem::path p(fileName);
    std::string base = p.stem().string();
    std::string ext = p.extension().string();
    std::string finalName = base + ext;

    callableFile_ = path / finalName;

    if (!std::filesystem::exists(callableFile_)) {
      Logger::getInstance()
          .addError("Callable file " + finalName + " not found.")
          .print();
    }
  }

  void insertNextParticle(const Particle<T> &particle) {
    particles_.push_back(particle);
  }

  void apply() {
    if (particles_.empty()) {
      Logger::getInstance()
          .addError("No particles inserted. Use insertNextParticle first.")
          .print();
    }

    if (cellDataBuffer_.sizeInBytes / sizeof(float) !=
        numCellData * launchParams.numElements) {
      Logger::getInstance()
          .addError("Cell data buffer size does not match the expected size.")
          .print();
    }

    // resize our cuda result buffer
    resultBuffer.allocInit(launchParams.numElements * numFluxes_, float(0));
    launchParams.resultBuffer = (float *)resultBuffer.dPointer();

    if (materialIdsBuffer_.sizeInBytes != 0) {
      launchParams.materialIds = (int *)materialIdsBuffer_.dPointer();
    }

    launchParams.seed = config_.rngSeed + config_.runNumber++;
    if (config_.useRandomSeed) {
      std::random_device rd;
      std::uniform_int_distribution<unsigned int> gen;
      launchParams.seed = gen(rd);
    }

    launchParams.tThreshold = 1.1 * gridDelta_; // TODO: find the best value

    int numPointsPerDim =
        static_cast<int>(std::sqrt(static_cast<T>(launchParams.numElements)));

    if (config_.numRaysFixed > 0) {
      numPointsPerDim = 1;
      config_.numRaysPerPoint = config_.numRaysFixed;
    }

    numRays = numPointsPerDim * numPointsPerDim * config_.numRaysPerPoint;
    if (numRays > (1 << 29)) {
      Logger::getInstance()
          .addWarning("Too many rays for single launch: " +
                      util::prettyDouble(numRays))
          .print();
      config_.numRaysPerPoint = (1 << 29) / (numPointsPerDim * numPointsPerDim);
      numRays = numPointsPerDim * numPointsPerDim * config_.numRaysPerPoint;
    }
    Logger::getInstance()
        .addDebug("Number of rays: " + util::prettyDouble(numRays))
        .print();

    // set up material specific sticking probabilities
    materialStickingBuffer_.resize(particles_.size());
    for (size_t i = 0; i < particles_.size(); i++) {
      if (!particles_[i].materialSticking.empty()) {
        if (uniqueMaterialIds_.empty() || materialIdsBuffer_.sizeInBytes == 0) {
          Logger::getInstance()
              .addError("Material IDs not set, when using material dependent "
                        "sticking.")
              .print();
        }
        std::vector<float> materialSticking(uniqueMaterialIds_.size());
        unsigned currentId = 0;
        for (auto &matId : uniqueMaterialIds_) {
          if (particles_[i].materialSticking.find(matId) ==
              particles_[i].materialSticking.end()) {
            materialSticking[currentId++] =
                static_cast<float>(particles_[i].sticking);
          } else {
            materialSticking[currentId++] =
                static_cast<float>(particles_[i].materialSticking[matId]);
          }
        }
        materialStickingBuffer_[i].allocUpload(materialSticking);
      }
    }

    // Every particle gets its own stream and launch parameters
    std::vector<cudaStream_t> streams(particles_.size());
    launchParamsBuffers.resize(particles_.size());

    if (particleMap_.empty()) {
      Logger::getInstance()
          .addError("No particle name->particleType mapping provided.")
          .print();
    }

    for (size_t i = 0; i < particles_.size(); i++) {
      auto it = particleMap_.find(particles_[i].name);
      if (it == particleMap_.end()) {
        Logger::getInstance()
            .addError("Unknown particle name: " + particles_[i].name)
            .print();
      }
      launchParams.particleType = it->second;
      launchParams.particleIdx = static_cast<unsigned>(i);
      launchParams.cosineExponent =
          static_cast<float>(particles_[i].cosineExponent);
      launchParams.sticking = static_cast<float>(particles_[i].sticking);
      if (!particles_[i].materialSticking.empty()) {
        assert(materialStickingBuffer_[i].sizeInBytes != 0);
        launchParams.materialSticking =
            (float *)materialStickingBuffer_[i].dPointer();
      }

      if (particles_[i].useCustomDirection) {
        Vec3Df direction{static_cast<float>(particles_[i].direction[0]),
                         static_cast<float>(particles_[i].direction[1]),
                         static_cast<float>(particles_[i].direction[2])};
        launchParams.source.directionBasis =
            rayInternal::getOrthonormalBasis<float>(direction);
        launchParams.source.customDirectionBasis = true;
      }

      launchParamsBuffers[i].alloc(sizeof(launchParams));
      launchParamsBuffers[i].upload(&launchParams, 1);

      CUDA_CHECK(StreamCreate(&streams[i]));
    }

    generateSBT();

#ifndef NDEBUG // Launch on single stream in debug mode
    for (size_t i = 0; i < particles_.size(); i++) {
      OPTIX_CHECK(optixLaunch(pipeline_, streams[0],
                              /*! parameters and SBT */
                              launchParamsBuffers[i].dPointer(),
                              launchParamsBuffers[i].sizeInBytes, &sbt,
                              /*! dimensions of the launch: */
                              config_.numRaysPerPoint, numPointsPerDim,
                              numPointsPerDim));
    }
#else // Launch on multiple streams in release mode
    for (size_t i = 0; i < particles_.size(); i++) {
      OPTIX_CHECK(optixLaunch(pipeline_, streams[i],
                              /*! parameters and SBT */
                              launchParamsBuffers[i].dPointer(),
                              launchParamsBuffers[i].sizeInBytes, &sbt,
                              /*! dimensions of the launch: */
                              config_.numRaysPerPoint, numPointsPerDim,
                              numPointsPerDim));
    }
#endif

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
    results.resize(launchParams.numElements * numFluxes_);
    // cudaDeviceSynchronize(); // download is sync anyway
    resultBuffer.download(results.data(),
                          launchParams.numElements * numFluxes_);
  }

  void setElementData(CudaBuffer &passedCellDataBuffer, unsigned numData) {
    assert(passedCellDataBuffer.sizeInBytes / sizeof(float) / numData ==
           launchParams.numElements);
    cellDataBuffer_ = passedCellDataBuffer;
    numCellData = numData;
  }

  template <class NumericType>
  void setMaterialIds(const std::vector<NumericType> &materialIds,
                      const bool mapToConsecutive = true) {
    assert(materialIds.size() == launchParams.numElements);

    if (mapToConsecutive) {
      uniqueMaterialIds_.clear();
      for (auto &matId : materialIds) {
        uniqueMaterialIds_.insert(static_cast<int>(matId));
      }
      std::unordered_map<NumericType, unsigned> materialIdMap;
      int currentId = 0;
      for (auto &uniqueMaterialId : uniqueMaterialIds_) {
        materialIdMap[uniqueMaterialId] = currentId++;
      }
      assert(currentId == materialIdMap.size());

      std::vector<int> materialIdsMapped(launchParams.numElements);
#pragma omp parallel for
      for (size_t i = 0; i < launchParams.numElements; i++) {
        materialIdsMapped[i] = materialIdMap[materialIds[i]];
      }
      materialIdsBuffer_.allocUpload(materialIdsMapped);
    } else {
      std::vector<int> materialIdsMapped(launchParams.numElements);
      for (size_t i = 0; i < launchParams.numElements; i++) {
        materialIdsMapped[i] = static_cast<int>(materialIds[i]);
      }
      materialIdsBuffer_.allocUpload(materialIdsMapped);
    }
  }

  void setNumberOfRaysPerPoint(const size_t pNumRays) {
    config_.numRaysPerPoint = pNumRays;
  }

  void setNumberOfRaysFixed(const size_t pNumRays) {
    config_.numRaysFixed = pNumRays;
  }

  void setUseRandomSeeds(const bool set) { config_.useRandomSeed = set; }

  void setRngSeed(const unsigned seed) {
    config_.rngSeed = seed;
    config_.useRandomSeed = false;
  }

  void
  setParticleCallableMap(std::tuple<std::unordered_map<std::string, unsigned>,
                                    std::vector<viennaray::gpu::CallableConfig>>
                             maps) {
    particleMap_ = std::get<0>(maps);
    callableMap_ = std::get<1>(maps);
  }

  size_t getNumberOfRays() const { return numRays; }

  void getFlux(float *flux, int particleIdx, int dataIdx,
               int smoothingNeighbors = 0) {
    unsigned int offset = 0;
    for (size_t i = 0; i < particles_.size(); i++) {
      if (particleIdx > i)
        offset += particles_[i].dataLabels.size();
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
    dataPerParticleBuffer_.free();
    for (auto &buffer : launchParamsBuffers) {
      buffer.free();
    }
    materialIdsBuffer_.free();
    for (auto &buffer : materialStickingBuffer_) {
      buffer.free();
    }
    areaBuffer_.free();
  }

  unsigned int prepareParticlePrograms() {
    if (particles_.empty()) {
      Logger::getInstance().addWarning("No particles defined.").print();
      return 0;
    }

    createModules();
    createRaygenPrograms();
    createMissPrograms();
    createHitgroupPrograms();
    createDirectCallablePrograms();
    createPipelines();

    numFluxes_ = 0;
    std::vector<unsigned int> dataPerParticle;
    for (const auto &p : particles_) {
      dataPerParticle.push_back(p.dataLabels.size());
      numFluxes_ += p.dataLabels.size();
    }
    dataPerParticleBuffer_.allocUpload(dataPerParticle);
    launchParams.dataPerParticle =
        (unsigned int *)dataPerParticleBuffer_.dPointer();
    Logger::getInstance()
        .addDebug("Number of flux arrays: " + std::to_string(numFluxes_))
        .print();

    return numFluxes_;
  }

  CudaBuffer &getData() { return cellDataBuffer_; }

  CudaBuffer &getResults() { return resultBuffer; }

  std::vector<Particle<T>> &getParticles() { return particles_; }

  [[nodiscard]] unsigned int getNumberOfRates() const { return numFluxes_; }

  [[nodiscard]] unsigned int getNumberOfElements() const {
    return launchParams.numElements;
  }

  void setParameters(CUdeviceptr d_params) {
    launchParams.customData = (void *)d_params;
  }

protected:
  virtual void normalize() {}

  void initRayTracer() {
    context_->addModule(normModuleName);
    normKernelName.append(geometryType_ + "_f");
    // launchParamsBuffer.alloc(sizeof(launchParams));
    // normKernelName.push_back(NumericType);
  }

  /// Creates the modules that contain all the programs we are going to use.
  /// We use one module for the pipeline programs, and one for the direct
  /// callables
  void createModules() {
    moduleCompileOptions_.maxRegisterCount =
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
    moduleCompileOptions_.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOptions_.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    moduleCompileOptions_.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions_.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
    pipelineCompileOptions_ = {};
    pipelineCompileOptions_.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions_.usesMotionBlur = false;
    pipelineCompileOptions_.numPayloadValues = 2;
    pipelineCompileOptions_.numAttributeValues = 0;
    pipelineCompileOptions_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions_.pipelineLaunchParamsVariableName =
        globalParamsName.c_str();

    pipelineLinkOptions_.maxTraceDepth = 1;

    size_t inputSize = 0;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    std::string pipelineFile = "GeneralPipeline" + geometryType_ + ".optixir";
    std::filesystem::path pipelinePath = context_->modulePath / pipelineFile;
    if (!std::filesystem::exists(pipelinePath)) {
      Logger::getInstance()
          .addError("Pipeline file " + pipelinePath.string() + " not found.")
          .print();
    }

    auto pipelineInput = getInputData(pipelinePath.c_str(), inputSize);
    if (!pipelineInput) {
      Logger::getInstance()
          .addError("Pipeline file " + pipelinePath.string() + " not found.")
          .print();
    }

    OPTIX_CHECK(optixModuleCreate(context_->optix, &moduleCompileOptions_,
                                  &pipelineCompileOptions_, pipelineInput,
                                  inputSize, log, &sizeof_log, &module_));
    // if (sizeof_log > 1)
    //   PRINT(log);

    char logCallable[2048];
    size_t sizeof_log_callable = sizeof(logCallable);

    auto callableInput = getInputData(callableFile_.c_str(), inputSize);
    if (!callableInput) {
      Logger::getInstance()
          .addError("Callable file " + callableFile_.string() + " not found.")
          .print();
    }

    OPTIX_CHECK(optixModuleCreate(context_->optix, &moduleCompileOptions_,
                                  &pipelineCompileOptions_, callableInput,
                                  inputSize, logCallable, &sizeof_log_callable,
                                  &moduleCallable_));
    // if (sizeof_log_callable > 1) {
    //   std::cout << "Callable module log: " << logCallable << std::endl;
    // }
  }

  /// does all setup for the raygen program
  void createRaygenPrograms() {
    std::string entryFunctionName = "__raygen__";
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module_;
    pgDesc.raygen.entryFunctionName = entryFunctionName.c_str();

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context_->optix, &pgDesc, 1, &pgOptions,
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
    pgDesc.miss.module = module_;
    pgDesc.miss.entryFunctionName = entryFunctionName.c_str();

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context_->optix, &pgDesc, 1, &pgOptions,
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
    pgDesc.hitgroup.moduleCH = module_;
    pgDesc.hitgroup.entryFunctionNameCH = entryFunctionNameCH.c_str();

    if (geometryType_ != "Triangle") {
      pgDesc.hitgroup.moduleIS = module_;
      pgDesc.hitgroup.entryFunctionNameIS = entryFunctionNameIS.c_str();
    }

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context_->optix, &pgDesc, 1, &pgOptions,
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
    unsigned maxParticleTypeId = 0;
    for (const auto &p : particleMap_) {
      if (p.second >= maxParticleTypeId)
        maxParticleTypeId = p.second;
    }
    unsigned numCallables =
        (maxParticleTypeId + 1) * static_cast<unsigned>(CallableSlot::COUNT);
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
      dcDesc.callables.moduleDC = moduleCallable_;
      dcDesc.callables.entryFunctionNameDC = entryFunctionNames[i].c_str();

      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(context_->optix, &dcDesc, 1,
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
    OPTIX_CHECK(optixPipelineCreate(context_->optix, &pipelineCompileOptions_,
                                    &pipelineLinkOptions_, programGroups.data(),
                                    static_cast<int>(programGroups.size()), log,
                                    &sizeof_log, &pipeline_));
    // #ifndef NDEBUG
    //       if (sizeof_log > 1)
    //         PRINT(log);
    // #endif

    OptixStackSizes stackSizes = {};
    for (auto &pg : programGroups) {
      optixUtilAccumulateStackSizes(pg, &stackSizes, pipeline_);
    }

    unsigned int dcStackFromTrav = 0;
    unsigned int dcStackFromState = 0;
    unsigned int continuationStack = 0;

    // These need to be adjusted when using nested callables
    // or recursive tracing
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stackSizes,
        pipelineLinkOptions_.maxTraceDepth, // OptixTrace recursion depth
        0,                                  // continuation callable depth
        1,                                  // direct callable depth
        &dcStackFromTrav, &dcStackFromState, &continuationStack));

    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline_,
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
  std::shared_ptr<DeviceContext> context_;
  std::filesystem::path callableFile_;

  std::string geometryType_;
  std::unordered_map<std::string, unsigned> particleMap_;
  std::vector<CallableConfig> callableMap_;

  std::set<int> uniqueMaterialIds_;
  CudaBuffer materialIdsBuffer_;

  float gridDelta_ = 0.0f;

  CudaBuffer areaBuffer_;

  // particles
  unsigned int numFluxes_ = 0;
  std::vector<Particle<T>> particles_;
  CudaBuffer dataPerParticleBuffer_;               // same for all particles
  std::vector<CudaBuffer> materialStickingBuffer_; // different for particles

  // sbt data
  CudaBuffer cellDataBuffer_;

  OptixPipeline pipeline_{};
  OptixPipelineCompileOptions pipelineCompileOptions_ = {};
  OptixPipelineLinkOptions pipelineLinkOptions_ = {};

  OptixModule module_{};
  OptixModule moduleCallable_{};
  OptixModuleCompileOptions moduleCompileOptions_ = {};

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

  rayInternal::KernelConfig config_;

  size_t numRays;
  unsigned numCellData = 0;
  const std::string globalParamsName = "launchParams";

  const std::string normModuleName = "normKernels.ptx";
  std::string normKernelName = "normalize_surface_";
};

} // namespace viennaray::gpu
