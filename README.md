# ViennaRay
[![Linux Actions Test](https://github.com/ViennaTools/ViennaRay/actions/workflows/linux_test.yml/badge.svg?branch=main)](https://github.com/ViennaTools/ViennaRay/actions/workflows/linux_test.yml)
[![macOS Actions Test](https://github.com/ViennaTools/ViennaRay/actions/workflows/macos_test.yml/badge.svg?branch=main)](https://github.com/ViennaTools/ViennaRay/actions/workflows/macos_test.yml)


ViennaRay is a flux calculation library for topography simulations, based in Intel®'s ray tracing kernel [Embree](https://www.embree.org/). It is designed to provide efficient and high-performance ray tracing, while maintaining a simple and easy to use interface. ViennaRay was developed and optimized for use in conjunction with [ViennaLS](https://github.com/ViennaTools/ViennaLS), which provides the necessary geometry representation. It is however possible to use this as a standalone library, with self-designed geometries.

IMPORTANT NOTE: ViennaRay is under heavy development and improved daily. If you do have suggestions or find bugs, please let us know!

## Support

[Exmaples](https://github.com/ViennaTools/ViennaRay/tree/main/Examples) can be found on Github. Bug reports and suggestions should be filed on GitHub.

## Releases

Releases are tagged on the main branch and available in the [releases section](https://github.com/ViennaTools/ViennaRay/releases).

## Building

### Supported Operating Systems

* Windows (Visual Studio)

* Linux (g++ / clang)

* macOS (XCode)

### System Requirements

* C++17 Compiler with OpenMP support

### Dependencies (installed automatically)

* [Embree](https://github.com/embree/embree)

Since [Embree](https://www.embree.org/) is optimized for CPU's using SSE, AVX, AVX2, and AVX-512 instructions, it requires at least a x86 CPU with support for SSE2 or an Apple M1 CPU.

## Installing 

Since this is a header only project, it does not require any installation. However, we recommend the following procedure in order to set up all dependencies correctly:

```
git clone github.com/ViennaTools/ViennaRay.git
cd ViennaRay
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/your/custom/install/
make # this will install Embree the first time it is called and might take a while
make install
```

This will install the necessary headers and CMake files to the specified path. If DCMAKE_INSTALL_PREFIX is not specified, it will be installed to the standard path for your system, usually /usr/local/ .

## Installing with Embree already installed on the system

If you want to use your own install of Embree, just specify the directory in CMake:

```
git clone github.com/ViennaTools/ViennaRay.git
cd ViennaRay
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/your/custom/install/ -Dembree_DIR=/path/to/embree/install
make install
```

## Integration in CMake projects

In order to use this library in your CMake project, add the following lines to the CMakeLists.txt of your project:

```
set(ViennaRay_DIR "/path/to/your/custom/install/")
find_package(ViennaRay REQUIRED)
add_executable(...)
target_include_directories(${PROJECT_NAME} PUBLIC ${VIENNARAY_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${VIENNARAY_LIBRARIES})
```

### Building examples

The examples can be built using CMake:

```
mkdir build && cd build
cmake .. -DVIENNARAY_BUILD_EXAMPLES=ON
make
```

### Running the Tests

ViennaRay uses CTest to run its tests.
In order to check whether ViennaRay runs without issues on your system, you can run:

```
git clone github.com/ViennaTools/ViennaRay.git
cd ViennaRay
mkdir build && cd build
cmake .. -DVIENNARAY_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=DEBUG
make buildTests # build all tests
make test # run all tests
```

## Contributing

If you want to contribute to ViennaRay, make sure to follow the [LLVM Coding guidelines](https://llvm.org/docs/CodingStandards.html). Before creating a pull request, make sure ALL files have been formatted by clang-format, which can be done using the format-project.sh script in the root directory.

## Authors

Current contributors: Tobias Reiter, Xaver Klemenschits, Alexander Scharinger

Contact us via: viennatools@iue.tuwien.ac.at

ViennaRay was developed under the aegis of the 'Institute for Microelectronics' at the 'TU Wien'.
http://www.iue.tuwien.ac.at/

License
--------------------------
See file LICENSE in the base directory.