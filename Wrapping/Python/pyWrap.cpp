/*
  This file is used to generate the python module of ViennaRay.
  It uses pybind11 to create the modules.
*/

// correct module name macro
#define TOKENPASTE_INTERNAL(x, y, z) x##y##z
#define TOKENPASTE(x, y, z) TOKENPASTE_INTERNAL(x, y, z)
#define VIENNARAY_MODULE_NAME TOKENPASTE(viennaRay, VIENNARAY_PYTHON_DIMENSION, d)
#define STRINGIZE2(s) #s
#define STRINGIZE(s) STRINGIZE2(s)
#define VIENNARAY_MODULE_VERSION STRINGIZE(VIENNARAY_VERSION)

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// all header files which define API functions
#include <rayTrace.hpp>
#include <rayParticle.hpp>

// always use float for python export
typedef float T;
// get dimension from cmake define
constexpr int D = VIENNARAY_PYTHON_DIMENSION;

// PYBIND11_DECLARE_HOLDER_TYPE(TemplateType, lsSmartPointer<TemplateType>);

// define trampoline classes for interface functions
// ALSO NEED TO ADD TRAMPOLINE CLASSES FOR CLASSES
// WHICH HOLD REFERENCES TO INTERFACE(ABSTRACT) CLASSES

// BASE CLASS WRAPPERS
// trampoline class for the abstract particle class
class PyrayAbstractParticle : public rayAbstractParticle<T>
{
  // inherit constructor
  using rayAbstractParticle<T>::rayAbstractParticle;

  std::unique_ptr<rayAbstractParticle> clone() override
  {
    PYBIND11_OVERRIDE_PURE(
        std::unique_ptr<rayAbstractParticle<T>>,
        rayAbstractParticle<T>,
        clone, );
  }

  std::pair<T, rayTriple<T>>
  surfaceReflection(T rayWeight, const rayTriple<T> &rayDir,
                    const rayTriple<T> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<T> *globalData,
                    rayRNG &Rng) override
  {
    PYBIND11_OVERRIDE_PURE(
        std::pair<T, rayTriple<T>>,
        rayAbstractParticle<T>,
        surfaceReflection,
        rayWeight, rayDir, geomNormal, primId, materialId, globalData, Rng);
  }

  void
  surfaceCollision(NumericType rayWeight,
                   const rayTriple<NumericType> &rayDir,
                   const rayTriple<NumericType> &geomNormal,
                   const unsigned int primID, const int materialId,
                   rayTracingData<NumericType> &localData,
                   const rayTracingData<NumericType> *globalData,
                   rayRNG &Rng) override
  {
    PYBIND11_OVERRIDE_PURE(
        void,
        rayAbstractParticle<T>,
        surfaceCollision,
        rayWeight, rayDir, geomNormal, primID, materialId, localData, globalData, Rng);
  }

  int getRequiredLocalDataSize() override
  {
    PYBIND11_OVERRIDE_PURE(
        int,
        rayAbstractParticle<T>,
        getRequiredLocalDataSize, );
  }

  T getSourceDistributionPower() override
  {
    PYBIND11_OVERRIDE_PURE(
        T,
        rayAbstractParticle<T>,
        getSourceDistributionPower, );
  }

  std::vector<std::string>
  getLocalDataLabels() override
  {
    PYBIND11_OVERRIDE_PURE(
        std::vector<std::string>,
        rayAbstractParticle<T>,
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
  // pybind11::class_<rayTrace<T, D>,

  // lsAdvect
  //   pybind11::class_<lsAdvect<T, D>, lsSmartPointer<lsAdvect<T, D>>>(module,
  //                                                                    "lsAdvect")
  //       // constructors
  //       .def(pybind11::init(&lsSmartPointer<lsAdvect<T, D>>::New<>))
  //       .def(pybind11::init(&lsSmartPointer<lsAdvect<T, D>>::New<
  //                           lsSmartPointer<lsDomain<T, D>> &>))
  //       .def(pybind11::init(&lsSmartPointer<lsAdvect<T, D>>::New<
  //                           lsSmartPointer<lsVelocityField<T>> &>))
  //       .def(pybind11::init(&lsSmartPointer<lsAdvect<T, D>>::New<
  //                           lsSmartPointer<lsDomain<T, D>> &,
  //                           lsSmartPointer<lsVelocityField<T>> &>))
  //       // getters and setters
  //       .def("insertNextLevelSet", &lsAdvect<T, D>::insertNextLevelSet,
  //            "Insert next level set to use for advection.")
  //       .def("setVelocityField",
  //            &lsAdvect<T, D>::setVelocityField<PylsVelocityField>,
  //            "Set the velocity to use for advection.")
  //       .def("setAdvectionTime", &lsAdvect<T, D>::setAdvectionTime,
  //            "Set the time until when the level set should be advected.")
  //       .def("setTimeStepRatio", &lsAdvect<T, D>::setTimeStepRatio,
  //            "Set the maximum time step size relative to grid size. Advection is "
  //            "only stable for <0.5.")
  //       .def("setCalculateNormalVectors",
  //            &lsAdvect<T, D>::setCalculateNormalVectors,
  //            "Set whether normal vectors are needed for the supplied velocity "
  //            "field.")
  //       .def("setIgnoreVoids", &lsAdvect<T, D>::setIgnoreVoids,
  //            "Set whether voids in the geometry should be ignored during "
  //            "advection or not.")
  //       .def("getAdvectedTime", &lsAdvect<T, D>::getAdvectedTime,
  //            "Get the time passed during advection.")
  //       .def("getNumberOfTimeSteps", &lsAdvect<T, D>::getNumberOfTimeSteps,
  //            "Get how many advection steps were performed after the last apply() "
  //            "call.")
  //       .def("getTimeStepRatio", &lsAdvect<T, D>::getTimeStepRatio,
  //            "Get the time step ratio used for advection.")
  //       .def("getCalculateNormalVectors",
  //            &lsAdvect<T, D>::getCalculateNormalVectors,
  //            "Get whether normal vectors are computed during advection.")
  //       .def("setIntegrationScheme", &lsAdvect<T, D>::setIntegrationScheme,
  //            "Set the integration scheme to use during advection.")
  //       .def("setDissipationAlpha", &lsAdvect<T, D>::setDissipationAlpha,
  //            "Set the dissipation value to use for Lax Friedrichs integration.")
  //       // need scoped release since we are calling a python method from
  //       // parallelised C++ code here
  //       .def("apply", &lsAdvect<T, D>::apply,
  //            pybind11::call_guard<pybind11::gil_scoped_release>(),
  //            "Perform advection.");
  //   // enums
  //   pybind11::enum_<lsIntegrationSchemeEnum>(module, "lsIntegrationSchemeEnum")
  //       .value("ENGQUIST_OSHER_1ST_ORDER",
  //              lsIntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER)
  //       .value("ENGQUIST_OSHER_2ND_ORDER",
  //              lsIntegrationSchemeEnum::ENGQUIST_OSHER_2ND_ORDER)
  //       .value("LAX_FRIEDRICHS_1ST_ORDER",
  //              lsIntegrationSchemeEnum::LAX_FRIEDRICHS_1ST_ORDER)
  //       .value("LAX_FRIEDRICHS_2ND_ORDER",
  //              lsIntegrationSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER)
  //       .value("LOCAL_LAX_FRIEDRICHS_ANALYTICAL_1ST_ORDER",
  //              lsIntegrationSchemeEnum::LOCAL_LAX_FRIEDRICHS_ANALYTICAL_1ST_ORDER)
  //       .value("LOCAL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER",
  //              lsIntegrationSchemeEnum::LOCAL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER)
  //       .value("LOCAL_LOCAL_LAX_FRIEDRICHS_2ND_ORDER",
  //              lsIntegrationSchemeEnum::LOCAL_LOCAL_LAX_FRIEDRICHS_2ND_ORDER)
  //       .value("LOCAL_LAX_FRIEDRICHS_1ST_ORDER",
  //              lsIntegrationSchemeEnum::LOCAL_LAX_FRIEDRICHS_1ST_ORDER)
  //       .value("LOCAL_LAX_FRIEDRICHS_2ND_ORDER",
  //              lsIntegrationSchemeEnum::LOCAL_LAX_FRIEDRICHS_2ND_ORDER)
  //       .value("STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER",
  //              lsIntegrationSchemeEnum::STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER);
}
