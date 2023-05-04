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
#include <omp.h>
//#include <utility>
#include <rayRNG.hpp>
#include <rayReflection.hpp>
#include <rayTracingData.hpp>
#include <rayUtil.hpp>

// always use double for python export
typedef double T;
// get dimension from cmake define
constexpr int D = VIENNARAY_PYTHON_DIMENSION;

namespace py = pybind11;
/// The Wrap for the rayAbstractParticle in rayParticle.hpp
template<typename T>
class PYrayAbstractParticle:rayAbstractParticle<T>{
public:
    using rayAbstractParticle<T>::rayAbstractParticle;
    std::unique_ptr<rayAbstractParticle<T>> clone() const override {
        PYBIND11_OVERRIDE_PURE(std::unique_ptr<rayAbstractParticle<T>>,rayAbstractParticle<T>,clone);
    }
    [[nodiscard]] int getRequiredLocalDataSize() const override {
        PYBIND11_OVERRIDE_PURE(int, rayAbstractParticle<T>, getRequiredLocalDataSize,);
    }
    [[nodiscard]] T getSourceDistributionPower() const override{
        PYBIND11_OVERRIDE_PURE(T,rayAbstractParticle<T>,getSourceDistributionPower,);
    }
    [[nodiscard]] std::vector<std::string> getLocalDataLabels() const override{
        PYBIND11_OVERRIDE_PURE(std::vector<std::string>,rayAbstractParticle<T>,getLocalDataLabels,);
    }
     void initNew(rayRNG &Rng) override{
        PYBIND11_OVERRIDE_PURE(void, rayAbstractParticle<T>, Rng);
    }
    std::pair<T, rayTriple<T>>
    surfaceReflection(T rayWeight, const rayTriple<T> &rayDir,
                                const rayTriple<T> &geomNormal,
                                const unsigned int primId, const int materialId,
                                const rayTracingData<T> *globalData,
                                rayRNG &Rng) override{
        using Pair_Replacer = std::pair<T, rayTriple<T>>;
        PYBIND11_OVERRIDE_PURE(Pair_Replacer,rayAbstractParticle<T>,surfaceReflection,rayWeight,rayDir,
                               geomNormal,primId,materialId,globalData,Rng);
    }
    void surfaceCollision(T rayWeight,
                                  const rayTriple<T> &rayDir,
                                  const rayTriple<T> &geomNormal,
                                  const unsigned int primID, const int materialId,
                                  rayTracingData<T> &localData,
                                  const rayTracingData<T> *globalData,
                                  rayRNG &Rng) override{
        PYBIND11_OVERRIDE_PURE(void, rayAbstractParticle<T>,rayWeight,rayDir,
                               geomNormal,primID,materialId,localData,globalData,Rng);
    }

};
template<typename Derived,typename T>
class PyrayParticle : rayParticle<Derived,T> {
public:
//using this class_name, as the PYBIND11_OVERRIDE does not support multiple template arguments
using class_name = rayParticle<Derived,T>;
using rayParticle<Derived,T>::clone;
void initNew(rayRNG &Rng) override {
    PYBIND11_OVERRIDE(void, class_name, initNew,Rng);
}
std::pair<T, rayTriple<T>>
surfaceReflection(T rayWeight, const rayTriple<T> &rayDir,
                  const rayTriple<T> &geomNormal,
                  const unsigned int primId, const int materialId,
                  const rayTracingData<T> *globalData,
                  rayRNG &Rng) override{
using Pair_Replacer = std::pair<T, rayTriple<T>>;
PYBIND11_OVERRIDE_PURE(Pair_Replacer,rayAbstractParticle<T>,surfaceReflection,rayWeight,rayDir,
        geomNormal,primId,materialId,globalData,Rng);
}

void surfaceCollision(T rayWeight,
                     const rayTriple<T> &rayDir,
                     const rayTriple<T> &geomNormal,
                     const unsigned int primID, const int materialId,
                     rayTracingData<T> &localData,
                     const rayTracingData<T> *globalData,
                     rayRNG &Rng) override{
PYBIND11_OVERRIDE_PURE(void, rayAbstractParticle<T>,rayWeight,rayDir,
                       geomNormal,primID,materialId,localData,globalData,Rng);
}

int getRequiredLocalDataSize() const override {
PYBIND11_OVERRIDE(int, class_name , getRequiredLocalDataSize,);
}
[[nodiscard]] T getSourceDistributionPower() const override{
PYBIND11_OVERRIDE(T,class_name ,getSourceDistributionPower,);
}
std::vector<std::string> getLocalDataLabels() const override{
PYBIND11_OVERRIDE(std::vector<std::string>,class_name ,getLocalDataLabels,);
}
protected:
using rayParticle<Derived,T>::rayParticle;
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

  // TODO: Implement bindings
  ///The binding for the rayAbstractParticle class
  //we don't need to wrap in this class the clone function
  //as it will be implemented in the rayParticle class, and at this point
  //it is not useful.
  py::class_<rayAbstractParticle<T>,PYrayAbstractParticle<T>>(module,"rayAbstractParticle")
  .def("getRequiredLocalDataSize",&rayAbstractParticle<T>::getRequiredLocalDataSize)
  .def("getSourceDistributionPower",&rayAbstractParticle<T>::getSourceDistributionPower)
  .def("initNew",&rayAbstractParticle<T>::initNew)
  .def("surfaceReflection",&rayAbstractParticle<T>::surfaceReflection)
  .def("surfaceCollision",&rayAbstractParticle<T>::surfaceCollision)
  .def("getLocalDataLabels",&rayAbstractParticle<T>::getLocalDataLabels);

  ///The binding for the rayParticle Class
    py::class_< rayParticle<rayTestParticle<T>,T> , rayAbstractParticle<T> , PyrayParticle<rayTestParticle<T>,T> >(module, "rayParticle")
    .def("clone",&rayParticle<rayTestParticle<T>,T>::clone)
    .def("initNew",&rayParticle<rayTestParticle<T>,T>::initNew)
    .def("getRequiredLocalDataSize",&rayParticle<rayTestParticle<T>,T>::getRequiredLocalDataSize)
    .def("getLocalDataLabels",&rayParticle<rayTestParticle<T>,T>::getLocalDataLabels)
    .def("getSourceDistributionPower",&rayParticle<rayTestParticle<T>,T>::getSourceDistributionPower)
    .def("surfaceCollision",&rayParticle<rayTestParticle<T>,T>::surfaceCollision)
    .def("surfaceReflection",&rayParticle<rayTestParticle<T>,T>::surfaceReflection)
    .def(py::init<>());

    ///The binding for the rayTestParticle class
    //we also have to bind this one, as when we use the rayParticle.clone(), we will get
    //a copy of this class
    py::class_<rayTestParticle<T>, rayParticle<rayTestParticle<T>,T>>(module, "rayTest")
    .def(py::init<>());
}
