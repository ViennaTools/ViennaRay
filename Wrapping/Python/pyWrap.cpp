/*
  This file is used to generate the python module of ViennaRay.
  It uses pybind11 to create the modules.
*/

// correct module name macro
#define TOKENPASTE_INTERNAL(x, y, z) x##y##z
#define TOKENPASTE(x, y, z) TOKENPASTE_INTERNAL(x, y, z)
#define VIENNARAY_MODULE_NAME \
     TOKENPASTE(viennaRay, VIENNARAY_PYTHON_DIMENSION, d)
#define STRINGIZE2(s) #s
#define STRINGIZE(s) STRINGIZE2(s)
#define VIENNARAY_MODULE_VERSION STRINGIZE(VIENNARAY_VERSION)

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// all header files which define API functions
#include <rayParticle.hpp>
#include <rayTrace.hpp>
#include <rayTraceDirection.hpp>

// always use float for python export
typedef float T;
// get dimension from cmake define
constexpr int D = VIENNARAY_PYTHON_DIMENSION;

// define trampoline classes for interface functions
// ALSO NEED TO ADD TRAMPOLINE CLASSES FOR CLASSES
// WHICH HOLD REFERENCES TO INTERFACE(ABSTRACT) CLASSES

// BASE CLASS WRAPPERS
// trampoline class for the abstract particle class
class PyrayAbstractParticle : public rayAbstractParticle<T>
{
     // inherit constructor
     using rayAbstractParticle<T>::rayAbstractParticle;

     std::unique_ptr<rayAbstractParticle> clone() const override
     {
          PYBIND11_OVERRIDE_PURE(std::unique_ptr<rayAbstractParticle<T>>,
                                 rayAbstractParticle<T>, clone, );
     }

     void initNew(rayRNG &Rng) override {
          PYBIND11_OVERRIDE_PURE(void, rayAbstractParticle<T>, initNew, Rng);
     }

     std::pair<T, rayTriple<T>>
     surfaceReflection(T rayWeight, const rayTriple<T> &rayDir,
                       const rayTriple<T> &geomNormal, const unsigned int primId,
                       const int materialId, const rayTracingData<T> *globalData,
                       rayRNG &Rng) override
     {
          PYBIND11_OVERRIDE_PURE(std::pair<T, rayTriple<T>>, rayAbstractParticle<T>,
                                 surfaceReflection, rayWeight, rayDir, geomNormal,
                                 primId, materialId, globalData, Rng);
     }

     void surfaceCollision(T rayWeight,
                           const rayTriple<T> &rayDir,
                           const rayTriple<T> &geomNormal,
                           const unsigned int primID, const int materialId,
                           rayTracingData<T> &localData,
                           const rayTracingData<T> *globalData,
                           rayRNG &Rng) override
     {
          PYBIND11_OVERRIDE_PURE(void, rayAbstractParticle<T>, surfaceCollision,
                                 rayWeight, rayDir, geomNormal, primID, materialId,
                                 localData, globalData, Rng);
     }

     int getRequiredLocalDataSize() const override
     {
          PYBIND11_OVERRIDE_PURE(int, rayAbstractParticle<T>,
                                 getRequiredLocalDataSize, );
     }

     T getSourceDistributionPower() const override
     {
          PYBIND11_OVERRIDE_PURE(T, rayAbstractParticle<T>,
                                 getSourceDistributionPower, );
     }

     std::vector<std::string> getLocalDataLabels() const override
     {
          PYBIND11_OVERRIDE_PURE(std::vector<std::string>, rayAbstractParticle<T>,
                                 getLocalDataLabels, );
     }
}

// module specification
PYBIND11_MODULE(VIENNARAY_MODULE_NAME, module)
{
     module.doc() =
         " ViennaRay is a flux calculation library for topography simulations, "
         " based in Intel®'s ray tracing kernel Embree. "
         " It is designed to provide efficient and high-performance ray tracing, "
         " while maintaining a simple and easy to use interface. ";

     // set version string of python module
     module.attr("__version__") = VIENNARAY_MODULE_VERSION;

     // wrap omp_set_num_threads to control number of threads
     module.def("setNumThreads", &omp_set_num_threads);

     // rayTrace
     pybind11::class_<rayTrace<T, D>>(module, "rayTrace")
         // constructor
         .def(pybind11::init())
         // member functions
         .def("apply", &rayTrace<T, D>::apply, "Run the ray tracer.")
         .def("setParticleType",
              &rayTrace<T, D>::setParticleType<PyrayAbstractParticle>,
              "Set the particle type used for ray tracing. The particle is a "
              "user defined object that has to interface the rayParticle "
              "class.")
         .def("setGeometry",
              pybind11::overload_cast<std::vector<std::array<T, D>> &,
                                      std::vector<std::array<T, D>> &, const T>(
                  &rayTrace<T, D>::setGeometry<D>),
              "Set the ray tracing geometry.")
         .def("setGeometry",
              pybind11::overload_cast<std::vector<std::array<T, D>> &,
                                      std::vector<std::array<T, D>> &, const T>(
                  &rayTrace<T, D>::setGeometry<D>),
              "Set the ray tracing geometry. Specify the disk radius manually.")
         .def("setMaterialIds", &rayTrace<T, D>::setMaterialIds<T>,
              "Set material ID's for each geometry point. If not set, all "
              "material ID's are default 0.")
         .def("setBoundaryConditions", &rayTrace<T, D>::setBoundaryConditions,
              "Set the boundary conditions. There has to be a boundary "
              "condition defined for each space dimension, however the "
              "boundary condition in direction of the tracing direction is "
              "ignored.")
         .def("setNumberOfRaysPerPoint", &rayTrace<T, D>::setNumberOfRaysPerPoint,
              "Set the number of rays per geometry point. The total number of "
              "rays, that are traced, is the set number set here times the "
              "number of points in the geometry.")
         .def("setNumberOfRaysFixed", &rayTrace<T, D>::setNumberOfRaysFixed,
              "Set the number of total rays traced to a fixed amount, "
              "independent of the geometry")
         .def("setSourceDirection", &rayTrace<T, D>::setSourceDirection,
              "Set the source direction, where the rays should be traced from.")
         .def("setUseRandomSeeds", &rayTrace<T, D>::setUseRandomSeeds,
              "Set whether random seeds for the internal random number "
              "generators should be used.")
         .def("setCalculateFlux", &rayTrace<T, D>::setCalculateFlux,
              "Set whether the flux and hit counts should be recorded. If not "
              "needed, this should be turned off to increase performance. If "
              "set to false, the functions getTotalFlux(), "
              "getNormalizedFlux(), getHitCounts() and getRelativeError() can "
              "not be used.")
         .def("getTotalFlux", &rayTrace<T, D>::getTotalFlux,
              "Returns the total flux on each disk.")
         .def("getNormalizedFlux", &rayTrace<T, D>::getNormalizedFlux,
              "Returns the normalized flux on each disk.")
         .def("normalizeFlux", &rayTrace<T, D>::normalizeFlux,
              "Helper function to normalize the recorded flux in a "
              "post-processing step. The flux can be normalized to the source "
              "flux and the maximum recorded value.")
         .def("smoothFlux", &rayTrace<T, D>::smoothFlux,
              "Helper function to smooth the recorded flux by averaging over "
              "the neighborhood in a post-processing step.")
         .def("getHitCounts", &rayTrace<T, D>::getHitCounts,
              "Returns the total number of hits for each geometry point.")
         .def("getRelativeError", &rayTrace<T, D>::getRelativeError,
              "Returns the relative error of the flux for each geometry point.")
         .def("getDiskAreas", &rayTrace<T, D>::getDiskAreas,
              "Returns the disk area for each geometry point")
         .def("getLocalData", &rayTrace<T, D>::getLocalData)
         .def("getGlobalData", &rayTrace<T, D>::getGlobalData)
         .def("setGlobalData", &rayTrace<T, D>::setGlobalData)
         .def("getRayTraceInfo", &rayTrace<T, D>::getRayTraceInfo);

     //   // enums
     pybind11::enum_<rayTraceDirection>(module, "rayTraceDirection")
         .value("POS_X", rayTraceDirection::POS_X)
         .value("NEG_X", rayTraceDirection::NEG_X)
         .value("POS_Y", rayTraceDirection::POS_Y)
         .value("NEG_Y", rayTraceDirection::NEG_Y)
         .value("POS_Z", rayTraceDirection::POS_Z)
         .value("NEG_Z", rayTraceDirection::NEG_Z);
}
