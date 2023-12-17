---
layout: default
title: Installing the Library
nav_order: 3
---

# Installing the Library
{: .fs-9 .fw-700}

---

## Supported Operating Systems

* Windows (Visual Studio)

* Linux (g++ / clang)

* macOS (XCode)

## System Requirements

* C++17 Compiler with OpenMP support

## Dependencies

* [Embree](https://www.embree.org/)

## Installing the Library

Since this is a header only project, it does not require any installation. However, we recommend the following procedure in order to set up all dependencies correctly:

```bash
git clone github.com/ViennaTools/ViennaRay.git
cd ViennaRay
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/your/custom/install/
make buildDependencies # this will install Embree the first time it is called and might take a while
make install
```

This will install the necessary headers and CMake files to the specified path. If `DCMAKE_INSTALL_PREFIX` is not specified, it will be installed to the standard path for your system, usually `/usr/local/` .

## Installing with Embree already installed on the system

{: .warning}
> ViennaRay does currently NOT support Embree > 4.0. If you want to use a local installation of Embree make sure the version is > 3.11 and < 4.0. The pinned version when building with the library is 3.13.0.

If you want to use your own install of Embree, just specify the directory in CMake:

```bash
git clone github.com/ViennaTools/ViennaRay.git
cd ViennaRay
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/your/custom/install/ -Dembree_DIR=/path/to/embree/install
make install
```

{: .note}
> If you have a system-wide installation, for instance, through a package manager like apt or brew, specifying a path is unnecessary. CMake will automatically identify and utilize an existing, suitable system-wide installation of Embree.

## Integration in CMake projects

In order to use this library in your CMake project, add the following lines to the CMakeLists.txt of your project:

```CMake
set(ViennaRay_DIR "/path/to/your/custom/install/")
find_package(ViennaRay REQUIRED)
add_executable(...)
target_include_directories(${PROJECT_NAME} PUBLIC ${VIENNARAY_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${VIENNARAY_LIBRARIES})
```

## Running the Tests

ViennaRay uses CTest to run its tests.
In order to check whether ViennaRay runs without issues on your system, you can run:

```bash
git clone github.com/ViennaTools/ViennaRay.git
cd ViennaRay
mkdir build && cd build
cmake .. -DVIENNARAY_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=DEBUG
make buildTests # build all tests
make test # run all tests
```