<div align="center">

![](https://raw.githubusercontent.com/ViennaTools/ViennaLS/master/assets/logo.png)

<h1>ViennaRay</h1>

[![🧪 Tests](https://github.com/ViennaTools/ViennaRay/actions/workflows/test.yml/badge.svg)](https://github.com/ViennaTools/ViennaRay/actions/workflows/test.yml)

</div>

ViennaRay is a flux calculation library for topography simulations, based in Intel®'s ray tracing kernel [Embree](https://www.embree.org/). It is designed to provide efficient and high-performance ray tracing, while maintaining a simple and easy to use interface. ViennaRay was developed and optimized for use in conjunction with [ViennaLS](https://github.com/ViennaTools/ViennaLS), which provides the necessary geometry representation. It is however possible to use this as a standalone library, with self-designed geometries.

> [!NOTE]
> ViennaRay is under heavy development and improved daily. If you do have suggestions or find bugs, please let us know!

## Support

[Examples](examples/) can be found on Github. Bug reports and suggestions should be filed on GitHub.

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

```bash
git clone https://github.com/ViennaTools/ViennaRay.git
cd ViennaRay

cmake -B build -DCMAKE_INSTALL_PREFIX=/path/to/your/custom/install/
cmake --build build
cmake --install build
```

This will install the necessary headers and CMake files to the specified path. If `-DCMAKE_INSTALL_PREFIX` is not specified, it will be installed to the standard path for your system, usually `/usr/local/`.

## Integration in CMake projects

We recommend using [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake) to consume this library.

* Installation with CPM

  ```cmake
  CPMAddPackage("gh:viennatools/viennaray@3.4.1")
  ```

* With a local installation
    > In case you have ViennaRay installed in a custom directory, make sure to properly specify the [`CMAKE_PREFIX_PATH`](https://cmake.org/cmake/help/latest/envvar/CMAKE_PREFIX_PATH.html#envvar:CMAKE_PREFIX_PATH).

    ```cmake
    list(APPEND CMAKE_PREFIX_PATH "/your/local/installation")

    find_package(ViennaRay)
    target_link_libraries(${PROJECT_NAME} PUBLIC ViennaTools::ViennaRay)
    ```

### Building examples

The examples can be built using CMake:

```bash
cmake -B build -DVIENNARAY_BUILD_EXAMPLES=ON
cmake --build build
```

### Running the Tests

ViennaRay uses CTest to run its tests.
In order to check whether ViennaRay runs without issues on your system, you can run:

```bash
git clone https://github.com/ViennaTools/ViennaRay.git
cd ViennaRay

cmake -B build -DVIENNARAY_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

## Contributing

If you want to contribute to ViennaRay, make sure to follow the [LLVM Coding guidelines](https://llvm.org/docs/CodingStandards.html). Before creating a pull request, make sure ALL files have been formatted by clang-format.

## Authors

Current contributors: Tobias Reiter

Contact us via: viennatools@iue.tuwien.ac.at

ViennaRay was developed under the aegis of the 'Institute for Microelectronics' at the 'TU Wien'.
http://www.iue.tuwien.ac.at/

## License
See [LICENSE](LICENSE) file in the base directory.
