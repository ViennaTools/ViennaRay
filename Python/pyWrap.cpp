/*
  This file is used to generate the python module of viennaray.
  It uses pybind11 to create the modules.

  All necessary headers are included here and the interface
  of the classes which should be exposed defined
*/

// correct module name macro
#define TOKENPASTE_INTERNAL(x, y, z) x##y##z
#define TOKENPASTE(x, y, z) TOKENPASTE_INTERNAL(x, y, z)
#define STRINGIZE2(s) #s
#define STRINGIZE(s) STRINGIZE2(s)
#define VIENNARAY_MODULE_VERSION STRINGIZE(VIENNARAY_VERSION)

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <rayParticle.hpp>
#include <pybind11/stl_bind.h>
#include <omp.h>
#include <rayRNG.hpp>
#include <rayReflection.hpp>
#include <rayTracingData.hpp>
#include <rayUtil.hpp>
#include <rayTrace.hpp>

// always use double for python export
typedef double T;
// get dimension from cmake define
constexpr int D = VIENNARAY_PYTHON_DIMENSION;

namespace py = pybind11;

// The binding for the rayParticle class (now this can be used as the base for the CRTP)
// used this as the constructor for the rayParticle is protected, and it is used when inheriting from it
template <typename Derived, typename T>
class PyrayParticle : public rayParticle<Derived, T> {
public:
    using Base = rayParticle<Derived, T>;
    using Base::clone;

    void initNew(rayRNG& Rng) override {
        Base::initNew(Rng);
    }

    std::pair<T, rayTriple<T>> surfaceReflection(T rayWeight, const rayTriple<T>& rayDir,
        const rayTriple<T>& geomNormal, const unsigned int primId, const int materialId,
        const rayTracingData<T>* globalData, rayRNG& Rng) override {
        using Pair_T_rayTriple = std::pair<T, rayTriple<T>>;
        return Base::surfaceReflection(rayWeight, rayDir, geomNormal, primId, materialId, globalData, Rng);
    }

    void surfaceCollision(T rayWeight, const rayTriple<T>& rayDir, const rayTriple<T>& geomNormal,
        const unsigned int primID, const int materialId, rayTracingData<T>& localData,
        const rayTracingData<T>* globalData, rayRNG& Rng) override {
        Base::surfaceCollision(rayWeight, rayDir, geomNormal, primID, materialId, localData, globalData, Rng);
    }

    int getRequiredLocalDataSize() const override {
        return Base::getRequiredLocalDataSize();
    }

    [[nodiscard]] T getSourceDistributionPower() const override {
        return Base::getSourceDistributionPower();
    }

    std::vector<std::string> getLocalDataLabels() const override {
        return Base::getLocalDataLabels();
    }

protected:
    using Base::Base;
};

// Particle Class' rayParticle.clone() will give a PyParticle class
template <int D> class PyParticle : public rayParticle<PyParticle<D>, T> {
  using ClassName = rayParticle<PyParticle<D>, T>;

public:
  void surfaceCollision(T rayWeight, const rayTriple<T> &rayDir,
                        const rayTriple<T> &geomNormal,
                        const unsigned int primID, const int materialID,
                        rayTracingData<T> &localData,
                        const rayTracingData<T> *globalData,
                        rayRNG &Rng) override final {
    PYBIND11_OVERRIDE(void, ClassName, surfaceCollision, rayWeight, rayDir,
                      geomNormal, primID, materialID, localData, globalData,
                      Rng);
  }

  std::pair<T, rayTriple<T>>
  surfaceReflection(T rayWeight, const rayTriple<T> &rayDir,
                    const rayTriple<T> &geomNormal, const unsigned int primID,
                    const int materialID, const rayTracingData<T> *globalData,
                    rayRNG &Rng) override final {
    using Pair = std::pair<T, rayTriple<T>>;
    PYBIND11_OVERRIDE(Pair, ClassName, surfaceReflection, rayWeight, rayDir,
                      geomNormal, primID, materialID, globalData, Rng);
  }

  void initNew(rayRNG &RNG) override final {
    PYBIND11_OVERRIDE(void, ClassName, initNew, RNG);
  }

  int getRequiredLocalDataSize() const override final {
    PYBIND11_OVERRIDE(int, ClassName, getRequiredLocalDataSize);
  }

  T getSourceDistributionPower() const override final {
    PYBIND11_OVERRIDE(T, ClassName, getSourceDistributionPower);
  }

  std::vector<std::string> getLocalDataLabels() const override final {
    PYBIND11_OVERRIDE(std::vector<std::string>, ClassName, getLocalDataLabels);
  }
};

// module specification
PYBIND11_MODULE(VIENNARAY_MODULE_NAME, module) {
  module.doc() =
      "ViennaRay is a flux calculation library for topography simulations, "
      "based on Intel®'s ray tracing kernel [Embree](https://www.embree.org/). "
      "It is designed to provide efficient and high-performance ray "
      "tracing, while maintaining a simple and easy to use interface. "
      "ViennaRay was developed and optimized for use in conjunction with "
      "[ViennaLS](https://github.com/ViennaTools/ViennaLS), which provides "
      "the necessary geometry representation. It is however possible to "
      "use this as a standalone library, with self-designed geometries.";

  // set version string of python module
  module.attr("__version__") = VIENNARAY_MODULE_VERSION;

  // wrap omp_set_num_threads to control number of threads
  module.def("setNumThreads", &omp_set_num_threads);

  module.def("rayReflectionSpecular", &rayReflectionSpecular<T>, "Perform a specular reflection of a ray",
    py::arg("ray_dir"), py::arg("geom_normal"));

  // rayTracingDataMergeEnum
  py::enum_<rayTracingDataMergeEnum>(module, "rayTracingDataMergeEnum")
    .value("SUM", rayTracingDataMergeEnum::SUM)
    .value("APPEND", rayTracingDataMergeEnum::APPEND)
    .value("AVERAGE", rayTracingDataMergeEnum::AVERAGE)
    .export_values();

  // rayTraceDirection
  py::enum_<rayTraceDirection>(module, "rayTraceDirection")
    .value("POS_X", rayTraceDirection::POS_X)
    .value("NEG_X", rayTraceDirection::NEG_X)
    .value("POS_Y", rayTraceDirection::POS_Y)
    .value("NEG_Y", rayTraceDirection::NEG_Y)
    .value("POS_Z", rayTraceDirection::POS_Z)
    .value("NEG_Z", rayTraceDirection::NEG_Z)
    .export_values();


// rayTracingData
  py::class_<rayTracingData<T>,std::shared_ptr<rayTracingData<T>>>(module, "rayTracingData")
    .def(py::init<>())
    .def("appendVectorData", &rayTracingData<T>::appendVectorData)
    .def("setNumberOfVectorData", &rayTracingData<T>::setNumberOfVectorData)
    .def("setNumberOfScalarData", &rayTracingData<T>::setNumberOfScalarData)
    .def("setScalarData",&rayTracingData<T>::setScalarData)

    .def("setVectorData", (void (rayTracingData<T>::*)(int, std::vector<T>&, std::string)) &rayTracingData<T>::setVectorData,
        py::arg("num"), py::arg("vector"), py::arg("label") = "vectorData")
    .def("setVectorData", (void (rayTracingData<T>::*)(int, std::vector<T>&&, std::string)) &rayTracingData<T>::setVectorData,
        py::arg("num"), py::arg("vector"), py::arg("label") = "vectorData")
    .def("setVectorData", (void (rayTracingData<T>::*)(int, size_t, T, std::string)) &rayTracingData<T>::setVectorData,
        py::arg("num"), py::arg("size"), py::arg("value"), py::arg("label") = "vectorData")

    .def("resizeAllVectorData", &rayTracingData<T>::resizeAllVectorData, py::arg("size"), py::arg("val") = 0)

    .def("setVectorMergeType", (void (rayTracingData<T>::*)(const std::vector<rayTracingDataMergeEnum>&)) &rayTracingData<T>::setVectorMergeType)
    .def("setVectorMergeType", (void (rayTracingData<T>::*)(int, rayTracingDataMergeEnum)) &rayTracingData<T>::setVectorMergeType)

    .def("setScalarMergeType", (void (rayTracingData<T>::*)(const std::vector<rayTracingDataMergeEnum>&)) &rayTracingData<T>::setScalarMergeType)
    .def("setScalarMergeType", (void (rayTracingData<T>::*)(int, rayTracingDataMergeEnum)) &rayTracingData<T>::setScalarMergeType)

    .def("getVectorData", (std::vector<T> &(rayTracingData<T>::*)(int)) &rayTracingData<T>::getVectorData)
    .def("getVectorData", (std::vector<T> &(rayTracingData<T>::*)(std::string)) &rayTracingData<T>::getVectorData)
    .def("getVectorData", (std::vector<std::vector<T>>& (rayTracingData<T>::*)()) &rayTracingData<T>::getVectorData)

    .def("getScalarData", (T &(rayTracingData<T>::*)(int)) &rayTracingData<T>::getScalarData)
    .def("getScalarData", (T &(rayTracingData<T>::*)(std::string)) &rayTracingData<T>::getScalarData)
    .def("getScalarData", (std::vector<T>& (rayTracingData<T>::*)()) &rayTracingData<T>::getScalarData)

    .def("getVectorDataLabel", &rayTracingData<T>::getVectorDataLabel)
    .def("getScalarDataLabel", &rayTracingData<T>::getScalarDataLabel)
    .def("getVectorDataIndex", &rayTracingData<T>::getVectorDataIndex)
    .def("getScalarDataIndex", &rayTracingData<T>::getScalarDataIndex)
    .def("getVectorMergeType", &rayTracingData<T>::getVectorMergeType)
    .def("getScalarMergeType", &rayTracingData<T>::getScalarMergeType);

  // RNG
  // I implemented for the seed only for the parameter "value", not for the seed sequence object
  py::class_<std::mt19937_64,std::shared_ptr<std::mt19937_64>>(module, "rayRNG")
    .def(py::init<>())
    .def("seed", [](std::mt19937_64 &self, std::mt19937_64::result_type value){ self.seed(value); })
    .def("__call__", &std::mt19937_64::min)
    .def("__call__", &std::mt19937_64::max)
    .def("__call__", &std::mt19937_64::operator());
  //One possible flaw, we can't use holder type std::shared_ptr, as the clone function won't
  //work for these classes
  py::class_<rayAbstractParticle<double>>(module, "rayAbstractParticle");
  
  //The binding for the rayParticle Class, having as template the rayTestParticle class
  //This is used so that we instantiate a rayParticle class that is of type <rayTestParticle>, for testing
  py::class_< rayParticle<rayTestParticle<T>,T> , rayAbstractParticle<T> , PyrayParticle<rayTestParticle<T>,T>>(module, "rayParticleInheritTest")
    .def("clone",&rayParticle<rayTestParticle<T>,T>::clone)
    .def("initNew",&rayParticle<rayTestParticle<T>,T>::initNew)
    .def("getRequiredLocalDataSize",&rayParticle<rayTestParticle<T>,T>::getRequiredLocalDataSize)
    .def("getLocalDataLabels",&rayParticle<rayTestParticle<T>,T>::getLocalDataLabels)
    .def("getSourceDistributionPower",&rayParticle<rayTestParticle<T>,T>::getSourceDistributionPower)
    .def("surfaceCollision",&rayParticle<rayTestParticle<T>,T>::surfaceCollision)
    .def("surfaceReflection",&rayParticle<rayTestParticle<T>,T>::surfaceReflection)
    .def(py::init<>());

  //rayTestParticle
  py::class_<rayTestParticle<T>, rayParticle<rayTestParticle<T>,T>>(module, "rayTest")
    .def(py::init<>())
    .def("surfaceReflection",&rayTestParticle<T>::surfaceReflection)
    .def("surfaceCollision",&rayTestParticle<T>::surfaceCollision)
    .def("getRequiredLocalDataSize",&rayTestParticle<T>::getRequiredLocalDataSize)
    .def("getSourceDistributionPower",&rayTestParticle<T>::getSourceDistributionPower)
    .def("getLocalDataLabels",&rayTestParticle<T>::getLocalDataLabels);


  //rayParticle that has as template PyParticle class
  //This is used for instantiating a rayParticle class that is of type <PyParticle>  
  py::class_< rayParticle<PyParticle<D>,T> , rayAbstractParticle<T> , PyrayParticle<PyParticle<D>,T>>(module, "rayParticle")
    .def("clone",&rayParticle<PyParticle<D>,T>::clone)
    .def("initNew",&rayParticle<PyParticle<D>,T>::initNew)
    .def("getRequiredLocalDataSize",&rayParticle<PyParticle<D>,T>::getRequiredLocalDataSize)
    .def("getLocalDataLabels",&rayParticle<PyParticle<D>,T>::getLocalDataLabels)
    .def("getSourceDistributionPower",&rayParticle<PyParticle<D>,T>::getSourceDistributionPower)
    .def("surfaceCollision",&rayParticle<PyParticle<D>,T>::surfaceCollision)
    .def("surfaceReflection",&rayParticle<PyParticle<D>,T>::surfaceReflection)
    .def(py::init<>());

  // Shim to instantiate the particle class
  pybind11::class_<PyParticle<D>,rayParticle<PyParticle<D>,T>>(module,
                                                                "PyParticle")
    .def(pybind11::init<>())
    .def("clone",&PyParticle<D>::clone)
    .def("surfaceCollision",&PyParticle<D>::surfaceCollision)
    .def("surfaceReflection",&PyParticle<D>::surfaceReflection)
    .def("initNew",&PyParticle<D>::initNew)
    .def("getRequiredLocalDataSize",&PyParticle<D>::getRequiredLocalDataSize)
    .def("getSourceDistributionPower",&PyParticle<D>::getSourceDistributionPower)
    .def("getLocalDataLabels",&PyParticle<D>::getLocalDataLabels);

  module.def("rayTriple", [](T a,T b,T c) { return std::array<T, 3>{{a, b, c}}; });
  module.def("rayPair", [](T a,T b ) { return std::array<T, 2>{{a, b}}; });
  module.def("rayQuadruple", [](T a,T b,T c,T d) { return std::array<T, 4>{{a, b, c, d}}; });

  // Wrapping for rayTraceInfo
  pybind11::class_<rayTraceInfo,std::shared_ptr<rayTraceInfo>>(module, "rayTraceInfo")
    .def(pybind11::init<>())
    .def_readwrite("numRays", &rayTraceInfo::numRays)
    .def_readwrite("totalRaysTraced", &rayTraceInfo::totalRaysTraced)
    .def_readwrite("totalDiskHits", &rayTraceInfo::totalDiskHits)
    .def_readwrite("nonGeometryHits", &rayTraceInfo::nonGeometryHits)
    .def_readwrite("geometryHits", &rayTraceInfo::geometryHits)
    .def_readwrite("time", &rayTraceInfo::time)
    .def_readwrite("warning", &rayTraceInfo::warning)
    .def_readwrite("error", &rayTraceInfo::error);



  // enum for rayUtil
  py::enum_<rayNormalizationType>(module, "rayNormalizationType")
    .value("SOURCE", rayNormalizationType::SOURCE)
    .value("MAX", rayNormalizationType::MAX)
    .export_values();
    
  // enum for rayBoundCondition
  py::enum_<rayTraceBoundary>(module, "rayTraceBoundary")
    .value("REFLECTIVE", rayTraceBoundary::REFLECTIVE)
    .value("PERIODIC", rayTraceBoundary::PERIODIC)
    .value("IGNORE", rayTraceBoundary::IGNORE)
    .export_values();

  // namespace for rayUtil
  pybind11::module PYrayUtil = module.def_submodule("rayUtil", "namespace for rayUtil");
  PYrayUtil.def("Sum",
    [](const rayTriple<T>& pVecA, const rayTriple<T>& pVecB) {
        return rayInternal::Sum(pVecA, pVecB);
    }, pybind11::arg("pVecA"), pybind11::arg("pVecB"), pybind11::return_value_policy::take_ownership)
    .def("Sum",
        [](const rayTriple<T>& pVecA, const rayTriple<T>& pVecB, const rayTriple<T>& pT) {
            return rayInternal::Sum(pVecA, pVecB, pT);
        }, pybind11::arg("pVecA"), pybind11::arg("pVecB"), pybind11::arg("pt"), pybind11::return_value_policy::take_ownership)
    .def("Diff",
        [](const rayTriple<T>& pVecA, const rayTriple<T>& pVecB) {
            return rayInternal::Diff(pVecA, pVecB);
        }, pybind11::arg("pVecA"), pybind11::arg("pVecB"), pybind11::return_value_policy::take_ownership)
    .def("Diff",
        [](const rayPair<T>& pVecA, const rayPair<T>& pVecB) {
            return rayInternal::Diff(pVecA, pVecB);
        }, pybind11::arg("pVecA"), pybind11::arg("pVecB"), pybind11::return_value_policy::take_ownership)
    .def("DotProduct",
        [](const rayTriple<T>& pVecA, const rayTriple<T>& pVecB) {
            return rayInternal::DotProduct(pVecA, pVecB);
        }, pybind11::arg("pVecA"), pybind11::arg("pVecB"), pybind11::return_value_policy::take_ownership)
    .def("CrossProduct",
        [](const rayTriple<T>& pVecA, const rayTriple<T>& pVecB) {
            return rayInternal::CrossProduct(pVecA, pVecB);
        }, pybind11::arg("pVecA"), pybind11::arg("pVecB"), pybind11::return_value_policy::take_ownership)
    .def("Norm",
        [](const std::array<T, D>& vec) {
            return rayInternal::Norm<T, D>(vec);
        }, pybind11::arg("vec"), pybind11::return_value_policy::take_ownership)
    .def("Normalize",
        [](std::array<T, D>& vec) {
            return rayInternal::Normalize(vec);
        }, pybind11::arg("vec"), pybind11::return_value_policy::take_ownership)
    .def("Normalize",
        [](const std::array<T, D>& vec) {
            return rayInternal::Normalize(vec);
        }, pybind11::arg("vec"), pybind11::return_value_policy::take_ownership)
    .def("Inv",
        [](const rayTriple<T>& vec) {
            return rayInternal::Inv(vec);
        }, pybind11::arg("vec"), pybind11::return_value_policy::take_ownership)
    .def("Scale",
        [](const T pF, rayTriple<T>& pT) {
            return rayInternal::Scale(pF, pT);
        }, pybind11::arg("pF"), pybind11::arg("pT"), pybind11::return_value_policy::take_ownership)
    .def("Scale",
        [](const T pF, const rayTriple<T>& pT) {
            return rayInternal::Scale(pF, pT);
        }, pybind11::arg("pF"), pybind11::arg("pT"), pybind11::return_value_policy::take_ownership)
    .def("Distance",
        [](const std::array<T, D>& pVecA, const std::array<T, D>& pVecB) {
            return rayInternal::Distance(pVecA, pVecB);
        }, pybind11::arg("pVecA"), pybind11::arg("pVecB"), pybind11::return_value_policy::take_ownership)
    .def("ComputeNormal",
        [](const rayTriple<rayTriple<T>>& planeCoords) {
            return rayInternal::ComputeNormal(planeCoords);
        }, pybind11::arg("planeCoords"), pybind11::return_value_policy::take_ownership)
    .def("IsNormalized",
        [](const rayTriple<T>& vec) {
            return rayInternal::IsNormalized(vec);
        }, pybind11::arg("vec"), pybind11::return_value_policy::take_ownership)
    .def("adjustBoundingBox",
        [](rayPair<rayTriple<T>>& bdBox, rayTraceDirection direction, T discRadius) {
            rayInternal::adjustBoundingBox<T, D>(bdBox, direction, discRadius);
        }, pybind11::arg("bdBox"), pybind11::arg("direction"), pybind11::arg("discRadius"))
    .def("getTraceSettings",
        [](rayTraceDirection sourceDir) {
            return rayInternal::getTraceSettings(sourceDir);
        }, pybind11::arg("sourceDir"), pybind11::return_value_policy::take_ownership)
    .def("PickRandomPointOnUnitSphere",
        [](rayRNG& RNG) {
            return rayInternal::PickRandomPointOnUnitSphere<T>(RNG);
        }, pybind11::arg("RNG"), pybind11::return_value_policy::take_ownership)
    .def("getOrthonormalBasis",
        [](const rayTriple<T>& pVector) {
            return rayInternal::getOrthonormalBasis(pVector);
        }, pybind11::arg("pVector"), pybind11::return_value_policy::take_ownership)
    .def("createPlaneGrid",
        [](const T gridDelta, const T extent, const std::array<int, 3>& direction,
            std::vector<std::array<T, 3>>& points, std::vector<std::array<T, 3>>& normals) {
            return rayInternal::createPlaneGrid(gridDelta, extent, direction, points, normals);
        }, pybind11::arg("gridDelta"), pybind11::arg("extent"), pybind11::arg("direction"),
        pybind11::arg("points"), pybind11::arg("normals"), pybind11::return_value_policy::take_ownership)
    .def("readGridFromFile",
        [](const std::string& fileName, T& gridDelta, std::vector<rayTriple<T>>& points,
            std::vector<rayTriple<T>>& normals) {
            return rayInternal::readGridFromFile(fileName, gridDelta, points, normals);
        }, pybind11::arg("fileName"), pybind11::arg("gridDelta"), pybind11::arg("points"),
        pybind11::arg("normals"), pybind11::return_value_policy::take_ownership)
    .def("writeVTK",
        [](const std::string& filename, const std::vector<rayTriple<T>>& points,
            const std::vector<T>& mcestimates) {
            return rayInternal::writeVTK(filename, points, mcestimates);
        }, pybind11::arg("filename"), pybind11::arg("points"), pybind11::arg("mcestimates"),
        pybind11::return_value_policy::take_ownership)
    .def("createSourceGrid",
        [](const rayPair<rayTriple<T>>& pBdBox, const size_t pNumPoints, const T pGridDelta,
            const std::array<int, 5>& pTraceSettings) {
            return rayInternal::createSourceGrid<T, D>(pBdBox, pNumPoints, pGridDelta, pTraceSettings);
        }, pybind11::arg("pBdBox"), pybind11::arg("pNumPoints"), pybind11::arg("pGridDelta"),
        pybind11::arg("pTraceSettings"), pybind11::return_value_policy::take_ownership)
    .def("printBoundingBox",
        [](const rayPair<rayTriple<T>>& bdBox) {
            return rayInternal::printBoundingBox<T>(bdBox);
        }, pybind11::arg("bdBox"))
    .def("printTriple",
        [](const rayTriple<T>& vec, bool endl = true) {
            return rayInternal::printTriple<T>(vec, endl);
        }, pybind11::arg("vec"), pybind11::arg("endl"))
    .def("printPair",
        [](const rayPair<T>& vec) {
            return rayInternal::printPair<T>(vec);
        }, pybind11::arg("vec"))
    .def("timeStampNow",
        []() {
            rayInternal::timeStampNow<std::chrono::milliseconds>();
        }, pybind11::return_value_policy::take_ownership);


  py::class_<rayTrace<T, D>,std::shared_ptr<rayTrace<T, D>>>rayTraceBind(module, "rayTrace");
    
    rayTraceBind.def(py::init<>())
    .def("apply", &rayTrace<T, D>::apply)
    .def("setMaterialIds", &rayTrace<T, D>::setMaterialIds<T>)
    // for this function, the python doesn't detect that a vector is passed,
    // and therefore we have to pass on a list from python ,and then cast it to 
    // rayTraceBoundary
    .def("setBoundaryConditions",[](rayTrace<T, D>& self, py::list boundaries) {
            rayTraceBoundary pBoundaryConds[D];
            for (size_t i = 0; i < D; ++i) {
                    pBoundaryConds[i] = boundaries[i].cast<rayTraceBoundary>();
                }
            self.setBoundaryConditions(pBoundaryConds);
        })
    .def("setNumberOfRaysPerPoint", &rayTrace<T, D>::setNumberOfRaysPerPoint)
    .def("setNumberOfRaysFixed", &rayTrace<T, D>::setNumberOfRaysFixed)
    .def("setSourceDirection", &rayTrace<T, D>::setSourceDirection)
    .def("setUseRandomSeeds", &rayTrace<T, D>::setUseRandomSeeds)
    .def("setCalculateFlux", &rayTrace<T, D>::setCalculateFlux)
    .def("setCheckRelativeError", &rayTrace<T, D>::setCheckRelativeError)
    .def("getTotalFlux", &rayTrace<T, D>::getTotalFlux)
    .def("getNormalizedFlux", &rayTrace<T, D>::getNormalizedFlux)
    .def("normalizeFlux", &rayTrace<T, D>::normalizeFlux)
    .def("smoothFlux", &rayTrace<T, D>::smoothFlux)
    .def("getHitCounts", &rayTrace<T, D>::getHitCounts)
    .def("getRelativeError", &rayTrace<T, D>::getRelativeError)
    .def("getDiskAreas", &rayTrace<T, D>::getDiskAreas)
    .def("getLocalData", &rayTrace<T, D>::getLocalData)
    .def("getGlobalData", &rayTrace<T, D>::getGlobalData)
    .def("setGlobalData", &rayTrace<T, D>::setGlobalData)
    .def("getRayTraceInfo", &rayTrace<T, D>::getRayTraceInfo);

    // For this, it is written that it is illegal to pass unique_ptr<> to functions and then wrap them
    // it is written here:https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html?highlight=unique_ptr#std-unique-ptr
    // for this approach we pass a smart pointer and then use the copy constructor to make it into an unique one
    rayTraceBind.def("setParticleType",
    [](rayTrace<T, D> &rT,
        std::shared_ptr<PyParticle<D>> &passedParticle) {
        if (passedParticle) {
            auto particle =
                std::make_unique<PyParticle<D>>(*passedParticle.get());
            rT.setParticleType(particle);
        }
    });


  // Done in preprocessing, as during compilation, it would be wrapped automatically
  // and the static_assert((D != 3 || Dim != 2) && "Setting 2D geometry in 3D trace object"); in the rayTrace.hpp
  // would fail
    #if VIENNARAY_PYTHON_DIMENSION == 3
            rayTraceBind.def("setGeometry", [](rayTrace<T, D> &instance, 
                           std::vector<std::array<T, 3>> &points, 
                           std::vector<std::array<T, 3>> &normals,
                           const T gridDelta) {
        return instance.setGeometry<3>(points, normals, gridDelta);
    });
            rayTraceBind.def("setGeometry", [](rayTrace<T, D> &instance, 
                           std::vector<std::array<T, 3>> &points, 
                           std::vector<std::array<T, 3>> &normals,
                           const T gridDelta, const T diskRadii) {
        return instance.setGeometry<3>(points, normals, gridDelta, diskRadii);
    });
    #endif
    #if VIENNARAY_PYTHON_DIMENSION == 2
                    rayTraceBind.def("setGeometry", [](rayTrace<T, D> &instance, 
                           std::vector<std::array<T, 3>> &points, 
                           std::vector<std::array<T, 3>> &normals,
                           const T gridDelta) {
        return instance.setGeometry<3>(points, normals, gridDelta);
    });
            rayTraceBind.def("setGeometry", [](rayTrace<T, D> &instance, 
                           std::vector<std::array<T, 3>> &points, 
                           std::vector<std::array<T, 3>> &normals,
                           const T gridDelta, const T diskRadii) {
        return instance.setGeometry<3>(points, normals, gridDelta, diskRadii);
    });
                rayTraceBind.def("setGeometry", [](rayTrace<T, D> &instance, 
                           std::vector<std::array<T, 2>> &points, 
                           std::vector<std::array<T, 2>> &normals,
                           const T gridDelta) {
        return instance.setGeometry<2>(points, normals, gridDelta);
    });
            rayTraceBind.def("setGeometry", [](rayTrace<T, D> &instance, 
                           std::vector<std::array<T, 2>> &points, 
                           std::vector<std::array<T, 2>> &normals,
                           const T gridDelta, const T diskRadii) {
        return instance.setGeometry<2>(points, normals, gridDelta, diskRadii);
    });
    #endif



}