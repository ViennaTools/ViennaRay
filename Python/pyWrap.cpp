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

// all header files which define API functions
#include <lsSmartPointer.hpp>

// always use double for python export
typedef double T;
// get dimension from cmake define
constexpr int D = VIENNARAY_PYTHON_DIMENSION;

PYBIND11_DECLARE_HOLDER_TYPE(TemplateType, lsSmartPointer<TemplateType>);

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
}
