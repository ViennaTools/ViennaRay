#pragma once

// #include "optix_types.h"
#include <cuda.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cstring>
#include <filesystem>

#include <rayParticle.hpp>
#include <rayUtil.hpp>

#include "raygBoundary.hpp"
#include "raygLaunchParams.hpp"
#include "raygSBTRecords.hpp"
#include "raygTriangleGeometry.hpp"

#include "raygDiskGeometry.hpp"
#include <rayBoundary.hpp>
#include <rayDiskBoundingBoxIntersector.hpp>
#include <rayGeometry.hpp>

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
  explicit Trace(std::shared_ptr<DeviceContext> &passedContext)
      : context(passedContext) {
    initRayTracer();
  }

  explicit Trace(unsigned deviceID = 0) {
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

  void setGeometry(const TriangleMesh &passedMesh) {
    assert(context);
    triangleGeometry.buildAccel(*context, passedMesh, launchParams);
    geometryType = "Triangle";
  }

  void setGeometry(const DiskMesh &passedMesh) {
    assert(context);
    minBox = static_cast<Vec3Df>(passedMesh.minimumExtent);
    maxBox = static_cast<Vec3Df>(passedMesh.maximumExtent);
    if constexpr (D == 2) {
      minBox[2] = -passedMesh.gridDelta;
      maxBox[2] = passedMesh.gridDelta;
    }
    gridDelta = static_cast<float>(passedMesh.gridDelta);
    launchParams.D = D;
    diskMesh = passedMesh;
    pointNeighborhood.template init<3>(passedMesh.points, 2 * passedMesh.radius,
                                       passedMesh.minimumExtent,
                                       passedMesh.maximumExtent);
    diskGeometry.buildAccel(*context, passedMesh, launchParams);
    geometryType = "Disk";
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

    if (geometryType == "Disk") {
      // Has to be higher than expected due to more neighbors at corners
      int maxNeighbors = (D == 2) ? 4 : 20;
      std::vector<int> neighborIdx;
      for (int i = 0; i < getNumberOfElements(); ++i) {
        std::vector<unsigned int> neighbors =
            pointNeighborhood.getNeighborIndices(i);
        if (neighbors.size() > maxNeighbors) {
          Logger::getInstance()
              .addError("More neighbors (" + std::to_string(neighbors.size()) +
                        ") than maxNeighbors (" + std::to_string(maxNeighbors) +
                        ")! Increase maxNeighbors.")
              .print();
        }
        for (int j = 0; j < maxNeighbors; ++j) {
          int id = (j < neighbors.size()) ? neighbors[j] : -1;
          neighborIdx.push_back(id);
        }
      }

      neighborsBuffer.allocUpload(neighborIdx);
      launchParams.neighbors = (int *)neighborsBuffer.dPointer();
      launchParams.maxNeighbors = maxNeighbors;
    }

    // Every particle gets its own stream and launch parameters
    std::vector<cudaStream_t> streams(particles.size());
    launchParamsBuffers.resize(particles.size());

    for (size_t i = 0; i < particles.size(); i++) {
      launchParams.particleIdx = static_cast<unsigned>(i);
      // TODO: make this more robust
      if (particles[i].name == "Neutral" || particles[i].name == "Particle" ||
          particles[i].name == "SingleParticle") {
        launchParams.particleType = ParticleType::NEUTRAL;
      } else if (particles[i].name == "Ion") {
        launchParams.particleType = ParticleType::ION;
      } else {
        Logger::getInstance()
            .addError("Unknown particle name: " + particles[i].name)
            .print();
        launchParams.particleType = ParticleType::UNDEFINED;
      }

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

    generateSBT(0);

    // TODO: Multiple streams seem to give same performance as single stream
    for (size_t i = 0; i < particles.size(); i++) {
      OPTIX_CHECK(optixLaunch(pipelines[0], streams[0],
                              /*! parameters and SBT */
                              launchParamsBuffers[i].dPointer(),
                              launchParamsBuffers[i].sizeInBytes, &sbts[0],
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
    directCallableRecordBuffer.free();
    dataPerParticleBuffer.free();
    for (auto &buffer : launchParamsBuffers) {
      buffer.free();
    }
    materialIdsBuffer.free();
    for (auto &buffer : materialStickingBuffer) {
      buffer.free();
    }
    triangleGeometry.freeBuffers();
    diskGeometry.freeBuffers();
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

protected:
  void normalize() {
    float sourceArea = 0.f;
    if constexpr (D == 2) {
      sourceArea =
          (launchParams.source.maxPoint[0] - launchParams.source.minPoint[0]);
    } else {
      sourceArea =
          (launchParams.source.maxPoint[0] - launchParams.source.minPoint[0]) *
          (launchParams.source.maxPoint[1] - launchParams.source.minPoint[1]);
    }
    assert(resultBuffer.sizeInBytes != 0 &&
           "Normalization: Result buffer not initialized.");

    if (geometryType == "Triangle") {
      CUdeviceptr d_data = resultBuffer.dPointer();
      CUdeviceptr d_vertex = triangleGeometry.geometryVertexBuffer.dPointer();
      CUdeviceptr d_index = triangleGeometry.geometryIndexBuffer.dPointer();
      void *kernel_args[] = {
          &d_data,     &d_vertex, &d_index, &launchParams.numElements,
          &sourceArea, &numRays,  &numRates};
      LaunchKernel::launch(normModuleName, normKernelName, kernel_args,
                           *context);

    } else if (geometryType == "Disk") {
      CUdeviceptr d_data = resultBuffer.dPointer();
      CUdeviceptr d_points = diskGeometry.geometryPointBuffer.dPointer();
      CUdeviceptr d_normals = diskGeometry.geometryNormalBuffer.dPointer();

      // calculate areas on host and send to device for now
      Vec2D<Vec3Df> bdBox = {minBox, maxBox};
      std::vector<float> areas(launchParams.numElements);
      DiskBoundingBoxXYIntersector<float> bdDiskIntersector(bdBox);

      // 0 = REFLECTIVE, 1 = PERIODIC, 2 = IGNORE
      std::array<BoundaryCondition, 2> boundaryConds = {
          BoundaryCondition::REFLECTIVE, BoundaryCondition::REFLECTIVE};
      const std::array<int, 2> boundaryDirs = {0, 1};
      constexpr float eps = 1e-4f;
#pragma omp for
      for (long idx = 0; idx < launchParams.numElements; ++idx) {
        std::array<float, 4> disk{0.f, 0.f, 0.f, diskMesh.radius};
        Vec3Df coord = diskMesh.points[idx];
        Vec3Df normal = diskMesh.normals[idx];
        disk[0] = coord[0];
        disk[1] = coord[1];
        disk[2] = coord[2];

        if constexpr (D == 3) {
          areas[idx] = disk[3] * disk[3] * M_PIf; // full disk area
          if (boundaryConds[boundaryDirs[0]] == BoundaryCondition::IGNORE &&
              boundaryConds[boundaryDirs[1]] == BoundaryCondition::IGNORE) {
            // no boundaries
            continue;
          }

          if (boundaryDirs[0] != 2 && boundaryDirs[1] != 2) {
            // Disk-BBox intersection only works with boundaries in x and y
            // direction
            areas[idx] = bdDiskIntersector.areaInside(disk, normal);
            continue;
          }
        } else { // 2D
          areas[idx] = 2 * disk[3];

          // test min boundary
          if ((boundaryConds[boundaryDirs[0]] != BoundaryCondition::IGNORE) &&
              (std::abs(disk[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) <
               disk[3])) {
            T insideTest =
                1 - normal[boundaryDirs[0]] * normal[boundaryDirs[0]];
            if (insideTest > eps) {
              insideTest =
                  std::abs(disk[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) /
                  std::sqrt(insideTest);
              if (insideTest < disk[3]) {
                areas[idx] -= disk[3] - insideTest;
              }
            }
          }

          // test max boundary
          if ((boundaryConds[boundaryDirs[0]] != BoundaryCondition::IGNORE) &&
              (std::abs(disk[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) <
               disk[3])) {
            T insideTest =
                1 - normal[boundaryDirs[0]] * normal[boundaryDirs[0]];
            if (insideTest > eps) {
              insideTest =
                  std::abs(disk[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) /
                  std::sqrt(insideTest);
              if (insideTest < disk[3]) {
                areas[idx] -= disk[3] - insideTest;
              }
            }
          }
        }
      }
      areaBuffer.allocUpload(areas);
      CUdeviceptr d_areas = areaBuffer.dPointer();

      normKernelName = "normalize_surface_disk_f";
      void *kernel_args[] = {&d_data,     &d_areas, &launchParams.numElements,
                             &sourceArea, &numRays, &numRates};
      LaunchKernel::launch(normModuleName, normKernelName, kernel_args,
                           *context);
    }
  }

  void initRayTracer() {
    context->addModule(normModuleName);
    // launchParamsBuffer.alloc(sizeof(launchParams));
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
    raygenPGs.resize(particles.size());
    for (size_t i = 0; i < particles.size(); i++) {
      std::string entryFunctionName = "__raygen__";
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc pgDesc = {};
      pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      pgDesc.raygen.module = module;
      pgDesc.raygen.entryFunctionName = entryFunctionName.c_str();

      // OptixProgramGroup raypg;
      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(context->optix, &pgDesc, 1,
                                          &pgOptions, log, &sizeof_log,
                                          &raygenPGs[i]));
      // if (sizeof_log > 1)
      //   PRINT(log);
    }
  }

  /// does all setup for the miss program
  void createMissPrograms() {
    missPGs.resize(particles.size());
    for (size_t i = 0; i < particles.size(); i++) {
      std::string entryFunctionName = "__miss__";
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc pgDesc = {};
      pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      pgDesc.miss.module = module;
      pgDesc.miss.entryFunctionName = entryFunctionName.c_str();

      // OptixProgramGroup raypg;
      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(context->optix, &pgDesc, 1,
                                          &pgOptions, log, &sizeof_log,
                                          &missPGs[i]));
      // if (sizeof_log > 1)
      //   PRINT(log);
    }
  }

  /// does all setup for the hitgroup program
  void createHitgroupPrograms() {
    hitgroupPGs.resize(particles.size());
    for (size_t i = 0; i < particles.size(); i++) {
      std::string entryFunctionNameIS = "__intersection__";
      std::string entryFunctionNameCH = "__closesthit__";
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc pgDesc = {};
      pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
      pgDesc.hitgroup.moduleCH = module;
      pgDesc.hitgroup.entryFunctionNameCH = entryFunctionNameCH.c_str();

      if (geometryType != "Triangle") {
        pgDesc.hitgroup.moduleIS = module;
        pgDesc.hitgroup.entryFunctionNameIS = entryFunctionNameIS.c_str();
      }

      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(context->optix, &pgDesc, 1,
                                          &pgOptions, log, &sizeof_log,
                                          &hitgroupPGs[i]));
      // if (sizeof_log > 1)
      //   PRINT(log);
    }
  }

  /// does all setup for the direct callables
  void createDirectCallablePrograms() {
    unsigned numCallables = static_cast<unsigned>(ParticleType::COUNT) *
                            static_cast<unsigned>(CallableSlot::COUNT);
    std::vector<std::string> entryFunctionNames(numCallables,
                                                "__direct_callable__noop");
    // TODO: move this to a separate function/file
    std::string processName = "MultiParticleProcess";
    if (processName == "DefaultProcess") {
      processName = "SingleParticleProcess";
      Logger::getInstance()
          .addWarning(
              "No process name set, using 'SingleParticleProcess' as default.")
          .print();
    }

    if (processName == "SingleParticleProcess") {
      entryFunctionNames[callableIndex(ParticleType::NEUTRAL,
                                       CallableSlot::COLLISION)] =
          "__direct_callable__singleNeutralCollision";
      entryFunctionNames[callableIndex(ParticleType::NEUTRAL,
                                       CallableSlot::REFLECTION)] =
          "__direct_callable__singleNeutralReflection";
    }

    if (processName == "MultiParticleProcess") {
      entryFunctionNames[callableIndex(ParticleType::NEUTRAL,
                                       CallableSlot::COLLISION)] =
          "__direct_callable__multiNeutralCollision";
      entryFunctionNames[callableIndex(ParticleType::NEUTRAL,
                                       CallableSlot::REFLECTION)] =
          "__direct_callable__multiNeutralReflection";
      entryFunctionNames[callableIndex(ParticleType::ION,
                                       CallableSlot::COLLISION)] =
          "__direct_callable__multiIonCollision";
      entryFunctionNames[callableIndex(ParticleType::ION,
                                       CallableSlot::REFLECTION)] =
          "__direct_callable__multiIonReflection";
      entryFunctionNames[callableIndex(ParticleType::ION, CallableSlot::INIT)] =
          "__direct_callable__multiIonInit";
    }

    if (processName == "SF6O2Etching" || processName == "HBrO2Etching") {
      entryFunctionNames[callableIndex(ParticleType::NEUTRAL,
                                       CallableSlot::COLLISION)] =
          "__direct_callable__plasmaNeutralCollision";
      entryFunctionNames[callableIndex(ParticleType::NEUTRAL,
                                       CallableSlot::REFLECTION)] =
          "__direct_callable__plasmaNeutralReflection";
      entryFunctionNames[callableIndex(ParticleType::ION,
                                       CallableSlot::COLLISION)] =
          "__direct_callable__plasmaIonCollision";
      entryFunctionNames[callableIndex(ParticleType::ION,
                                       CallableSlot::REFLECTION)] =
          "__direct_callable__plasmaIonReflection";
      entryFunctionNames[callableIndex(ParticleType::ION, CallableSlot::INIT)] =
          "__direct_callable__plasmaIonInit";
    }

    directCallablePGs.resize(entryFunctionNames.size());
    for (size_t i = 0; i < entryFunctionNames.size(); i++) {
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
    pipelines.resize(particles.size());
    for (size_t i = 0; i < particles.size(); i++) {
      std::vector<OptixProgramGroup> programGroups;
      programGroups.push_back(raygenPGs[i]);
      programGroups.push_back(missPGs[i]);
      programGroups.push_back(hitgroupPGs[i]);

      for (size_t j = 0; j < directCallablePGs.size(); j++) {
        programGroups.push_back(directCallablePGs[j]);
      }

      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixPipelineCreate(
          context->optix, &pipelineCompileOptions, &pipelineLinkOptions,
          programGroups.data(), static_cast<int>(programGroups.size()), log,
          &sizeof_log, &pipelines[0]));
      // #ifndef NDEBUG
      //       if (sizeof_log > 1)
      //         PRINT(log);
      // #endif

      OptixStackSizes stackSizes = {};
      for (auto &pg : programGroups) {
        optixUtilAccumulateStackSizes(pg, &stackSizes, pipelines[0]);
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
          pipelines[0],
          dcStackFromTrav,  // stack size for DirectCallables from IS or AH.
          dcStackFromState, // stack size for DirectCallables from RG, MS or CH.
          continuationStack, // continuation stack size
          1));               // nested traversable graph depth
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
    // Triangle hitgroup
    if (geometryType == "Triangle") {
      std::vector<HitgroupRecord> hitgroupRecords;
      HitgroupRecord geometryHitgroupRecord = {};
      optixSbtRecordPackHeader(hitgroupPGs[i], &geometryHitgroupRecord);
      geometryHitgroupRecord.data.vertex =
          (Vec3Df *)triangleGeometry.geometryVertexBuffer.dPointer();
      geometryHitgroupRecord.data.index =
          (Vec3D<unsigned> *)triangleGeometry.geometryIndexBuffer.dPointer();
      geometryHitgroupRecord.data.isBoundary = false;
      geometryHitgroupRecord.data.cellData = (void *)cellDataBuffer.dPointer();
      hitgroupRecords.push_back(geometryHitgroupRecord);

      // boundary hitgroup
      HitgroupRecord boundaryHitgroupRecord = {};
      optixSbtRecordPackHeader(hitgroupPGs[i], &boundaryHitgroupRecord);
      boundaryHitgroupRecord.data.vertex =
          (Vec3Df *)triangleGeometry.boundaryVertexBuffer.dPointer();
      boundaryHitgroupRecord.data.index =
          (Vec3D<unsigned> *)triangleGeometry.boundaryIndexBuffer.dPointer();
      boundaryHitgroupRecord.data.isBoundary = true;
      hitgroupRecords.push_back(boundaryHitgroupRecord);

      hitgroupRecordBuffer.allocUpload(hitgroupRecords);
      sbts[i].hitgroupRecordBase = hitgroupRecordBuffer.dPointer();
      sbts[i].hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
      sbts[i].hitgroupRecordCount = 2;
    } // Disk hitgroup
    else if (geometryType == "Disk") {
      std::vector<HitgroupRecordDisk> hitgroupRecords;
      HitgroupRecordDisk geometryHitgroupRecord = {};
      optixSbtRecordPackHeader(hitgroupPGs[i], &geometryHitgroupRecord);
      geometryHitgroupRecord.data.point =
          (Vec3Df *)diskGeometry.geometryPointBuffer.dPointer();
      geometryHitgroupRecord.data.normal =
          (Vec3Df *)diskGeometry.geometryNormalBuffer.dPointer();
      geometryHitgroupRecord.data.radius = diskMesh.radius;
      geometryHitgroupRecord.data.isBoundary = false;
      geometryHitgroupRecord.data.cellData = (void *)cellDataBuffer.dPointer();
      hitgroupRecords.push_back(geometryHitgroupRecord);

      // boundary hitgroup
      HitgroupRecordDisk boundaryHitgroupRecord = {};
      optixSbtRecordPackHeader(hitgroupPGs[i], &boundaryHitgroupRecord);
      boundaryHitgroupRecord.data.point =
          (Vec3Df *)diskGeometry.boundaryPointBuffer.dPointer();
      boundaryHitgroupRecord.data.normal =
          (Vec3Df *)diskGeometry.boundaryNormalBuffer.dPointer();
      boundaryHitgroupRecord.data.radius = diskGeometry.boundaryRadius;
      boundaryHitgroupRecord.data.isBoundary = true;
      hitgroupRecords.push_back(boundaryHitgroupRecord);

      hitgroupRecordBuffer.allocUpload(hitgroupRecords);
      sbts[i].hitgroupRecordBase = hitgroupRecordBuffer.dPointer();
      sbts[i].hitgroupRecordStrideInBytes = sizeof(HitgroupRecordDisk);
      sbts[i].hitgroupRecordCount = 2;

      // callable programs
      std::vector<CallableRecord> callableRecords(directCallablePGs.size());
      for (size_t j = 0; j < directCallablePGs.size(); ++j) {
        CallableRecord callableRecord = {};
        optixSbtRecordPackHeader(directCallablePGs[j], &callableRecord);
        callableRecords[j] = callableRecord;
      }
      directCallableRecordBuffer.allocUpload(callableRecords);

      sbts[i].callablesRecordBase = directCallableRecordBuffer.dPointer();
      sbts[i].callablesRecordStrideInBytes = sizeof(CallableRecord);
      sbts[i].callablesRecordCount =
          static_cast<unsigned int>(directCallablePGs.size());
    } else {
      std::cout << "Unknown geometry type: " << geometryType << std::endl;
      std::exit(1);
    }
  }

protected:
  // context for cuda kernels
  std::shared_ptr<DeviceContext> context;
  std::filesystem::path pipelineFile;

  // triangleGeometry
  TriangleGeometry triangleGeometry;
  DiskGeometry<D> diskGeometry;

  // Disk specific
  DiskMesh diskMesh;
  PointNeighborhood<float, D> pointNeighborhood;

  std::string geometryType = "Disk";

  std::set<int> uniqueMaterialIds;
  CudaBuffer materialIdsBuffer;

  CudaBuffer neighborsBuffer;
  float gridDelta = 0.0f;

  CudaBuffer areaBuffer;

  Vec3Df minBox;
  Vec3Df maxBox;

  // particles
  unsigned int numRates = 0;
  std::vector<Particle<T>> particles;
  CudaBuffer dataPerParticleBuffer;               // same for all particles
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
  std::vector<OptixProgramGroup> directCallablePGs;
  CudaBuffer directCallableRecordBuffer;
  std::vector<OptixShaderBindingTable> sbts;

  // launch parameters, on the host, constant for all particles
  LaunchParams launchParams;
  std::vector<CudaBuffer> launchParamsBuffers;

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
