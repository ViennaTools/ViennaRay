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
  Trace(std::shared_ptr<DeviceContext> const &passedContext,
        std::string &&geometryType)
      : context_(passedContext), geometryType_(std::move(geometryType)) {
    initRayTracer();
  }

  Trace(std::string &&geometryType, const int deviceID = 0)
      : geometryType_(std::move(geometryType)) {
    context_ = DeviceContext::getContextFromRegistry(deviceID);
    if (!context_) {
      VIENNACORE_LOG_ERROR("No context found for device ID " +
                           std::to_string(deviceID) +
                           ". Create and register a context first.");
    }
    initRayTracer();
  }

  virtual ~Trace() {
    freeBuffers();
    destroyMembers();
  }

  void setPipelineFileName(const std::string &fileName) {
    pipelineFileName_ = fileName;
  }

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
      VIENNACORE_LOG_ERROR("Callable file " + finalName + " not found.");
    }
  }

  void insertNextParticle(const Particle<T> &particle) {
    particles_.push_back(particle);
  }

  void apply() {
    if (particles_.empty()) {
      VIENNACORE_LOG_ERROR(
          "No particles inserted. Use insertNextParticle first.");
    }

    if (cellDataBuffer_.sizeInBytes / sizeof(float) !=
        numCellData_ * launchParams_.numElements) {
      VIENNACORE_LOG_ERROR(
          "Cell data buffer size does not match the expected size.");
    }

    // Resize our cuda result buffer
    resultBuffer_.allocInit(launchParams_.numElements * numFluxes_,
                            ResultType(0));
    launchParams_.resultBuffer = (ResultType *)resultBuffer_.dPointer();

    if (materialIdsBuffer_.sizeInBytes != 0) {
      launchParams_.materialIds = (int *)materialIdsBuffer_.dPointer();
    }

    launchParams_.seed = config_.rngSeed + config_.runNumber++;
    if (config_.useRandomSeed) {
      std::random_device rd;
      std::uniform_int_distribution<unsigned int> gen;
      launchParams_.seed = gen(rd);
    }

    // Threshold value for neighbor detection in disk-based geometries
    assert(gridDelta_ > 0.0f);
    launchParams_.tThreshold = 1.1 * gridDelta_; // TODO: find the best value

    launchParams_.maxReflections = config_.maxReflections;
    launchParams_.maxBoundaryHits = config_.maxBoundaryHits;

    int numPointsPerDim = static_cast<int>(
        std::sqrt(static_cast<double>(launchParams_.numElements)));

    if (config_.numRaysFixed > 0) {
      numPointsPerDim = 1;
      config_.numRaysPerPoint = config_.numRaysFixed;
    }

    numRays_ = numPointsPerDim * numPointsPerDim * config_.numRaysPerPoint;
    if (numRays_ > (1 << 29)) {
      VIENNACORE_LOG_WARNING("Too many rays for single launch: " +
                             util::prettyDouble(numRays_));
      config_.numRaysPerPoint = (1 << 29) / (numPointsPerDim * numPointsPerDim);
      numRays_ = numPointsPerDim * numPointsPerDim * config_.numRaysPerPoint;
    }
    VIENNACORE_LOG_DEBUG("Number of rays: " + util::prettyDouble(numRays_));

    // set up material specific sticking probabilities
    materialStickingBuffer_.resize(particles_.size());
    for (size_t i = 0; i < particles_.size(); i++) {
      if (!particles_[i].materialSticking.empty()) {
        if (uniqueMaterialIds_.empty() || materialIdsBuffer_.sizeInBytes == 0) {
          VIENNACORE_LOG_ERROR(
              "Material IDs not set, when using material dependent "
              "sticking.");
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
    launchParamsBuffers_.resize(particles_.size());

    if (particleMap_.empty()) {
      VIENNACORE_LOG_ERROR("No particle name->particleType mapping provided.");
    }

    for (size_t i = 0; i < particles_.size(); i++) {
      auto it = particleMap_.find(particles_[i].name);
      if (it == particleMap_.end()) {
        VIENNACORE_LOG_ERROR("Unknown particle name: " + particles_[i].name);
      }
      launchParams_.particleType = it->second;
      launchParams_.particleIdx = static_cast<unsigned>(i);
      launchParams_.cosineExponent =
          static_cast<float>(particles_[i].cosineExponent);
      launchParams_.sticking = static_cast<float>(particles_[i].sticking);
      if (!particles_[i].materialSticking.empty()) {
        assert(materialStickingBuffer_[i].sizeInBytes != 0);
        launchParams_.materialSticking =
            (float *)materialStickingBuffer_[i].dPointer();
      }

      if (particles_[i].useCustomDirection) {
        Vec3Df direction{static_cast<float>(particles_[i].direction[0]),
                         static_cast<float>(particles_[i].direction[1]),
                         static_cast<float>(particles_[i].direction[2])};
        launchParams_.source.directionBasis =
            rayInternal::getOrthonormalBasis<float>(direction);
        launchParams_.source.customDirectionBasis = true;
      }

      launchParamsBuffers_[i].allocUploadSingle(launchParams_);

      CUDA_CHECK(StreamCreate(&streams[i]));
    }

    generateSBT();

#ifndef NDEBUG // Launch on single stream in debug mode
    for (size_t i = 0; i < particles_.size(); i++) {
      OPTIX_CHECK(optixLaunch(
          pipeline_, streams[0],
          /*! parameters and SBT */
          launchParamsBuffers_[i].dPointer(),
          launchParamsBuffers_[i].sizeInBytes, &shaderBindingTable_,
          /*! dimensions of the launch: */
          config_.numRaysPerPoint, numPointsPerDim, numPointsPerDim));
    }
#else // Launch on multiple streams in release mode
    for (size_t i = 0; i < particles_.size(); i++) {
      OPTIX_CHECK(optixLaunch(
          pipeline_, streams[i],
          /*! parameters and SBT */
          launchParamsBuffers_[i].dPointer(),
          launchParamsBuffers_[i].sizeInBytes, &shaderBindingTable_,
          /*! dimensions of the launch: */
          config_.numRaysPerPoint, numPointsPerDim, numPointsPerDim));
    }
#endif

    // sync
    for (auto &s : streams) {
      CUDA_CHECK(StreamSynchronize(s));
      CUDA_CHECK(StreamDestroy(s));
    }

    resultsDownloaded_ = false;
  }

  void setElementData(const CudaBuffer &passedCellDataBuffer,
                      const unsigned numData) {
    if (passedCellDataBuffer.sizeInBytes / sizeof(float) / numData !=
        launchParams_.numElements) {
      VIENNACORE_LOG_WARNING(
          "Passed cell data does not match number of elements.");
    }
    cellDataBuffer_ = passedCellDataBuffer;
#ifndef NDEBUG
    // In debug mode, we set the buffer as reference to avoid accidental frees
    cellDataBuffer_.isRef = true;
#endif
    numCellData_ = numData;
  }

  template <class NumericType>
  void setMaterialIds(const std::vector<NumericType> &materialIds,
                      const bool mapToConsecutive = true) {
    assert(materialIds.size() == launchParams_.numElements);

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

      std::vector<int> materialIdsMapped(launchParams_.numElements);
#pragma omp parallel for
      for (int i = 0; i < launchParams_.numElements; i++) {
        materialIdsMapped[i] = materialIdMap[materialIds[i]];
      }
      materialIdsBuffer_.allocUpload(materialIdsMapped);
    } else {
      std::vector<int> materialIdsMapped(launchParams_.numElements);
      for (int i = 0; i < launchParams_.numElements; i++) {
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

  void setMaxReflections(const unsigned pMaxReflections) {
    config_.maxReflections = pMaxReflections;
  }

  void setMaxBoundaryHits(const unsigned pMaxBoundaryHits) {
    config_.maxBoundaryHits = pMaxBoundaryHits;
  }

  void setUseRandomSeeds(const bool set) { config_.useRandomSeed = set; }

  void setRngSeed(const unsigned seed) {
    config_.rngSeed = seed;
    config_.useRandomSeed = false;
  }

  void setParticleCallableMap(
      std::tuple<std::unordered_map<std::string, unsigned>,
                 std::vector<viennaray::gpu::CallableConfig>> const &maps) {
    particleMap_ = std::get<0>(maps);
    callableMap_ = std::get<1>(maps);
  }

  size_t getNumberOfRays() const { return numRays_; }

  std::vector<ResultType> getFlux(int particleIdx, int dataIdx,
                                  int smoothingNeighbors = 0) {
    if (!resultsDownloaded_) {
      results_.resize(launchParams_.numElements * numFluxes_);
      resultBuffer_.download(results_.data(),
                             launchParams_.numElements * numFluxes_);
      resultsDownloaded_ = true;
    }

    std::vector<ResultType> flux(launchParams_.numElements);
    unsigned int offset = 0;
    for (size_t i = 0; i < particles_.size(); i++) {
      if (particleIdx > i)
        offset += particles_[i].dataLabels.size();
    }
    offset = (offset + dataIdx) * launchParams_.numElements;
    std::memcpy(flux.data(), results_.data() + offset,
                launchParams_.numElements * sizeof(ResultType));
    if (smoothingNeighbors > 0)
      smoothFlux(flux, smoothingNeighbors);
    return flux;
  }

  void setUseCellData(unsigned numData) { numCellData_ = numData; }

  void setPeriodicBoundary(const bool periodic) {
    launchParams_.periodicBoundary = periodic;
  }

  void setIgnoreBoundary(const bool ignore) { ignoreBoundary_ = ignore; }

  void freeBuffers() {
    resultBuffer_.free();
    hitgroupRecordBuffer_.free();
    missRecordBuffer_.free();
    raygenRecordBuffer_.free();
    directCallableRecordBuffer_.free();
    dataPerParticleBuffer_.free();
    for (auto &buffer : launchParamsBuffers_) {
      buffer.free();
    }
    materialIdsBuffer_.free();
    for (auto &buffer : materialStickingBuffer_) {
      buffer.free();
    }
  }

  void destroyMembers() {
    if (pipeline_) {
      optixPipelineDestroy(pipeline_);
      pipeline_ = nullptr;
    }
    if (module_) {
      optixModuleDestroy(module_);
      module_ = nullptr;
    }
    if (moduleCallable_) {
      optixModuleDestroy(moduleCallable_);
      moduleCallable_ = nullptr;
    }
    if (raygenPG_) {
      optixProgramGroupDestroy(raygenPG_);
      raygenPG_ = nullptr;
    }
    if (missPG_) {
      optixProgramGroupDestroy(missPG_);
      missPG_ = nullptr;
    }
    if (hitgroupPG_) {
      optixProgramGroupDestroy(hitgroupPG_);
      hitgroupPG_ = nullptr;
    }
    for (auto &pg : directCallablePGs_) {
      if (pg) {
        optixProgramGroupDestroy(pg);
      }
    }
    directCallablePGs_.clear();
  }

  unsigned int prepareParticlePrograms() {
    if (particles_.empty()) {
      VIENNACORE_LOG_WARNING("No particles defined.");
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
    launchParams_.dataPerParticle =
        (unsigned int *)dataPerParticleBuffer_.dPointer();
    VIENNACORE_LOG_DEBUG("Number of flux arrays: " +
                         std::to_string(numFluxes_));

    return numFluxes_;
  }

  [[nodiscard]] CudaBuffer &getData() { return cellDataBuffer_; }

  [[nodiscard]] CudaBuffer &getResultBuffer() { return resultBuffer_; }

  [[nodiscard]] std::vector<std::vector<ResultType>> getResults() {
    downloadResults();
    std::vector<std::vector<ResultType>> resultArrays;
    resultArrays.resize(numFluxes_);
    for (unsigned int i = 0; i < numFluxes_; ++i) {
      resultArrays[i].resize(launchParams_.numElements);
      std::memcpy(resultArrays[i].data(),
                  results_.data() + i * launchParams_.numElements,
                  launchParams_.numElements * sizeof(ResultType));
    }
    return resultArrays;
  }

  [[nodiscard]] std::vector<Particle<T>> &getParticles() { return particles_; }

  [[nodiscard]] unsigned int getNumberOfRates() const { return numFluxes_; }

  [[nodiscard]] unsigned int getNumberOfElements() const {
    return launchParams_.numElements;
  }

  void setParameters(CUdeviceptr d_params) {
    launchParams_.customData = (void *)d_params;
  }

  void downloadResults() {
    if (!resultsDownloaded_) {
      results_.resize(launchParams_.numElements * numFluxes_);
      resultBuffer_.download(results_.data(),
                             launchParams_.numElements * numFluxes_);
      resultsDownloaded_ = true;
    }
  }

  // To be implemented by derived classes
  virtual void smoothFlux(std::vector<ResultType> &flux,
                          int smoothingNeighbors) {}

  virtual void normalizeResults() = 0;

protected:
  virtual void buildHitGroups() = 0;

private:
  void initRayTracer() {
    launchParams_.D = D;
    context_->addModule(normModuleName_);
    normKernelName_.append(geometryType_);
  }

  void createProgramGroup(const OptixProgramGroupDesc *pgDesc,
                          const OptixProgramGroupOptions *pgOptions,
                          OptixProgramGroup *prog) {
#ifdef VIENNACORE_CUDA_LOG_DEBUG
    char log[2048];
    size_t sizeof_log = sizeof(log);
    auto result = optixProgramGroupCreate(context_->optix, pgDesc, 1, pgOptions,
                                          log, &sizeof_log, prog);
    if (sizeof_log > 1) {
      size_t len = std::min(sizeof_log, sizeof(log) - 1);
      log[len] = '\0';
      std::cerr << "Program group log:\n" << log << std::endl;
    }
    OPTIX_CHECK_RESULT(result);
#else
    OPTIX_CHECK(optixProgramGroupCreate(context_->optix, pgDesc, 1, pgOptions,
                                        NULL, NULL, prog));
#endif
  }

  void createModule(const char *input, size_t inputSize, OptixModule *module) {
#ifdef VIENNACORE_CUDA_LOG_DEBUG
    char log[8192];
    size_t sizeof_log = sizeof(log);
    auto res = optixModuleCreate(context_->optix, &moduleCompileOptions_,
                                 &pipelineCompileOptions_, input, inputSize,
                                 log, &sizeof_log, module);
    if (sizeof_log > 1) {
      size_t len = std::min(sizeof_log, sizeof(log) - 1);
      log[len] = '\0';
      std::cerr << "Module log:\n" << log << std::endl;
    }
    OPTIX_CHECK_RESULT(res);
#else
    OPTIX_CHECK(optixModuleCreate(context_->optix, &moduleCompileOptions_,
                                  &pipelineCompileOptions_, input, inputSize,
                                  NULL, NULL, module));
#endif
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
        globalParamsName_.c_str();

    size_t inputSize = 0;
    std::string pipelineFile = pipelineFileName_ + geometryType_ + ".optixir";
    std::filesystem::path pipelinePath = context_->modulePath / pipelineFile;
    if (!std::filesystem::exists(pipelinePath)) {
      VIENNACORE_LOG_ERROR("Pipeline file " + pipelinePath.string() +
                           " not found.");
    }

    std::string path_str = pipelinePath.string(); // explicit conversion
    auto pipelineInput = getInputData(path_str.c_str(), inputSize);
    if (!pipelineInput) {
      VIENNACORE_LOG_ERROR("Pipeline file " + pipelinePath.string() +
                           " not found.");
    }

    createModule(pipelineInput, inputSize, &module_);

    if (callableFile_.empty()) {
      VIENNACORE_LOG_WARNING("No callable file set.");
      return;
    }
    std::string callable_path_str =
        callableFile_.string(); // explicit conversion
    auto callableInput = getInputData(callable_path_str.c_str(), inputSize);
    if (!callableInput) {
      VIENNACORE_LOG_ERROR("Callable file " + callableFile_.string() +
                           " not found.");
    }

    createModule(callableInput, inputSize, &moduleCallable_);
  }

  /// does all setup for the raygen program
  void createRaygenPrograms() {
    std::string entryFunctionName = "__raygen__";
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module_;
    pgDesc.raygen.entryFunctionName = entryFunctionName.c_str();
    createProgramGroup(&pgDesc, &pgOptions, &raygenPG_);
  }

  /// does all setup for the miss program
  void createMissPrograms() {
    std::string entryFunctionName = "__miss__";
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module_;
    pgDesc.miss.entryFunctionName = entryFunctionName.c_str();
    createProgramGroup(&pgDesc, &pgOptions, &missPG_);
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

    createProgramGroup(&pgDesc, &pgOptions, &hitgroupPG_);
  }

  /// does all setup for the direct callables
  void createDirectCallablePrograms() {
    if (callableMap_.empty()) {
      VIENNACORE_LOG_WARNING("No particleType->callable mapping provided.");
      return;
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

    directCallablePGs_.resize(numCallables);
    for (size_t i = 0; i < numCallables; i++) {
      OptixProgramGroupOptions dcOptions = {};
      OptixProgramGroupDesc dcDesc = {};
      dcDesc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
      dcDesc.callables.moduleDC = moduleCallable_;
      dcDesc.callables.entryFunctionNameDC = entryFunctionNames[i].c_str();
      createProgramGroup(&dcDesc, &dcOptions, &directCallablePGs_[i]);
    }
  }

  /// assembles the full pipeline of all programs
  void createPipelines() {
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;

    std::vector<OptixProgramGroup> programGroups;
    programGroups.push_back(raygenPG_);
    programGroups.push_back(missPG_);
    programGroups.push_back(hitgroupPG_);

    for (auto const &directCallablePG : directCallablePGs_) {
      programGroups.push_back(directCallablePG);
    }

#ifdef VIENNACORE_CUDA_LOG_DEBUG
    char log[2048];
    size_t sizeof_log = sizeof(log);
    auto resPipeline =
        optixPipelineCreate(context_->optix, &pipelineCompileOptions_,
                            &pipelineLinkOptions, programGroups.data(),
                            programGroups.size(), log, &sizeof_log, &pipeline_);
    if (sizeof_log > 1) {
      size_t len = std::min(sizeof_log, sizeof(log) - 1);
      log[len] = '\0';
      std::cerr << "Pipeline creation log:\n" << log << std::endl;
    }
    OPTIX_CHECK_RESULT(resPipeline);
#else
    OPTIX_CHECK(optixPipelineCreate(
        context_->optix, &pipelineCompileOptions_, &pipelineLinkOptions,
        programGroups.data(), programGroups.size(), NULL, NULL, &pipeline_));
#endif

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
        pipelineLinkOptions.maxTraceDepth, // OptixTrace recursion depth
        0,                                 // continuation callable depth
        1,                                 // direct callable depth
        &dcStackFromTrav, &dcStackFromState, &continuationStack));

    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline_,
        dcStackFromTrav,   // stack size for DirectCallables from IS or AH.
        dcStackFromState,  // stack size for DirectCallables from RG, MS or CH.
        continuationStack, // continuation stack size
        1));               // nested traversable graph depth
  }

  /// constructs the shader binding table
  void generateSBT() {
    // build raygen record
    RaygenRecord raygenRecord = {};
    optixSbtRecordPackHeader(raygenPG_, &raygenRecord);
    raygenRecord.data = nullptr;
    raygenRecordBuffer_.allocUploadSingle(raygenRecord);
    shaderBindingTable_.raygenRecord = raygenRecordBuffer_.dPointer();

    // build miss record
    MissRecord missRecord = {};
    optixSbtRecordPackHeader(missPG_, &missRecord);
    missRecord.data = nullptr;
    missRecordBuffer_.allocUploadSingle(missRecord);
    shaderBindingTable_.missRecordBase = missRecordBuffer_.dPointer();
    shaderBindingTable_.missRecordStrideInBytes = sizeof(MissRecord);
    shaderBindingTable_.missRecordCount = 1;

    // build geometry specific hitgroup records
    buildHitGroups();

    // build callable programs
    if (!directCallablePGs_.empty()) {
      std::vector<CallableRecord> callableRecords(directCallablePGs_.size());
      for (size_t j = 0; j < directCallablePGs_.size(); ++j) {
        CallableRecord callableRecord = {};
        optixSbtRecordPackHeader(directCallablePGs_[j], &callableRecord);
        callableRecords[j] = callableRecord;
      }
      directCallableRecordBuffer_.allocUpload(callableRecords);

      shaderBindingTable_.callablesRecordBase =
          directCallableRecordBuffer_.dPointer();
      shaderBindingTable_.callablesRecordStrideInBytes = sizeof(CallableRecord);
      shaderBindingTable_.callablesRecordCount =
          static_cast<unsigned int>(directCallablePGs_.size());
    } else {
      assert(false && "No direct callables found.");
    }
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

  // particles
  unsigned int numFluxes_ = 0;
  std::vector<Particle<T>> particles_;
  CudaBuffer dataPerParticleBuffer_;               // same for all particles
  std::vector<CudaBuffer> materialStickingBuffer_; // different for particles

  // sbt data
  CudaBuffer cellDataBuffer_;

  OptixPipeline pipeline_{};
  OptixPipelineCompileOptions pipelineCompileOptions_ = {};

  OptixModule module_{};
  OptixModule moduleCallable_{};
  OptixModuleCompileOptions moduleCompileOptions_ = {};

  // program groups, and the SBT built around
  OptixProgramGroup raygenPG_{};
  CudaBuffer raygenRecordBuffer_;
  OptixProgramGroup missPG_{};
  CudaBuffer missRecordBuffer_;
  OptixProgramGroup hitgroupPG_{};
  CudaBuffer hitgroupRecordBuffer_;
  std::vector<OptixProgramGroup> directCallablePGs_;
  CudaBuffer directCallableRecordBuffer_;
  OptixShaderBindingTable shaderBindingTable_{};

  // launch parameters
  LaunchParams launchParams_;
  std::vector<CudaBuffer> launchParamsBuffers_; // one per particle

  // results Buffer
  CudaBuffer resultBuffer_;
  std::vector<ResultType> results_;

  rayInternal::KernelConfig config_;
  bool ignoreBoundary_ = false;
  bool resultsDownloaded_ = false;

  size_t numRays_ = 0;
  unsigned numCellData_ = 0;
  const std::string globalParamsName_ = "launchParams";

  const std::string normModuleName_ = "normKernels.ptx";
  std::string normKernelName_ = "normalize_surface_";
  std::string pipelineFileName_ = "GeneralPipeline";
};

} // namespace viennaray::gpu
