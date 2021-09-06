# CMake system description

The project uses CMake for integration and testing purposes.

Configuration phase can be customized by passing additional variables: `cmake -D<var1>=<val1> -D<var2>=<val2> ... -D<varN>=<valN> <sources-dir>`

The following variables are provided for oneDPL configuration:

| Variable                     | Type   | Description                                                                                   | Default value |
|------------------------------|--------|-----------------------------------------------------------------------------------------------|---------------|
| ONEDPL_BACKEND               | STRING | Threading backend; supported values: tbb, dpcpp, dpcpp_only, serial, ...; the default value is defined by compiler: dpcpp for DPC++ and tbb for others | tbb/dpcpp |
| ONEDPL_DEVICE_TYPE           | STRING | Device type, applicable only for DPC++ backends; supported values: GPU, CPU, FPGA_HW, FPGA_EMU | GPU           |
| ONEDPL_DEVICE_BACKEND        | STRING | Device backend type, applicable only for oneDPL DPC++ backends; supported values: opencl, level_zero. | Any(*) |
| ONEDPL_USE_UNNAMED_LAMBDA    | BOOL   | Pass `-fsycl-unnamed-lambda`, `-fno-sycl-unnamed-lambda` compile options or nothing           |               |
| ONEDPL_FPGA_STATIC_REPORT    | BOOL   | Enable the static report generation for the FPGA_HW device type                               | OFF           |
| ONEDPL_USE_AOT_COMPILATION   | BOOL   | Enable the ahead of time compilation via OpenCL™ Offline Compiler (OCLOC)                     | OFF           |
| ONEDPL_ENABLE_SIMD           | BOOL   | Enable SIMD vectorization by passing an OpenMP SIMD flag to the compiler if supported; the flag is passed to user project compilation string if enabled | ON           |
| ONEDPL_AOT_ARCH              | STRING | Architecture options for the ahead of time compilation, supported values can be found [here](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html); the default value `*` means compilation for all available options | *             |

Some useful CMake variables ([here](https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html) you can find a full list of CMake variables for the latest version):

- [`CMAKE_CXX_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html) - C++ compiler used for build, e.g. `CMAKE_CXX_COMPILER=dpcpp`.
- [`CMAKE_BUILD_TYPE`](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html) - build type that affects optimization level and debug options, values: `RelWithDebInfo`, `Debug`, `Release`, ...; e.g. `CMAKE_BUILD_TYPE=RelWithDebInfo`.
- [`CMAKE_CXX_STANDARD`](https://cmake.org/cmake/help/latest/variable/CMAKE_CXX_STANDARD.html) - C++ standard, e.g. `CMAKE_CXX_STANDARD=17`.

## Testing

Steps:

1. Configure project using CMake.
2. Perform build [and run] using build system (e.g. `make`).
3. (optional) Run tests using CTest.

The following targets are available for build system after configuration:

- `<test-name>` - build specific test, e.g. `for_each.pass`;
- `run-<test-name>` - build and run specific test, e.g. `run-for_each.pass`;
- `build-<tests-subdir>` - build all tests from specific subdirectory under `<root>/test`, e.g. `build-std`;
- `run-<tests-subdir>` - build and run all tests from specific subdirectory under `<root>/test`, e.g. `run-std`;
- `build-all` - build all tests;
- `run-all` - build and run all tests.

Sudirectories are added as labels for each test and can be used with `ctest -L <label>`.
For example, `<root>/test/path/to/test.pass.cpp` will have `path` and `to` labels.

## How to use oneDPL from CMake
### Using oneDPL source files

This way allow to integrate oneDPL source code into user project using the [add_subdirectory](https://cmake.org/cmake/help/latest/command/add_subdirectory.html) command. `add_subdirectory(<oneDPL_root_dir> <oneDPL_output_dir>)`, where `<oneDPL_root_dir>` is a relative or absolute path to oneDPL root dir and `<oneDPL_output_dir>` is a relative or absolute path to directory for holding output files of oneDPL, adds oneDPL to user project build. If `<oneDPL_root_dir>` is the relative path, then `<oneDPL_output_dir>` is the optional variable.

For example:

```cmake
project(Foo)
add_executable(foo foo.cpp)

# Add oneDPL to the build.
add_subdirectory(/path/to/oneDPL /path/to/build_oneDPL)
```
When using this way oneDPL is built with the user project simultaneously, so variables affecting oneDPL build can be specified in the user project's CMakeLists.txt file:

```cmake
project(Foo)
add_executable(foo foo.cpp)

# Add oneDPL to the build.
add_subdirectory(/path/to/oneDPL /path/to/build_oneDPL)

# Specify oneDPL backend
target_compile_definitions(foo PRIVATE ONEDPL_BACKEND=tbb)
```
Or passed to cmake call as when building oneDPL separately.

### Using oneDPL package

oneDPLConfig.cmake and oneDPLConfigVersion.cmake are included into oneDPL distribution.

These files allow to integrate oneDPL into user project with the [find_package](https://cmake.org/cmake/help/latest/command/find_package.html) command. Successful invocation of `find_package(oneDPL <options>)` creates imported target `oneDPL` that can be passed to the [target_link_libraries](https://cmake.org/cmake/help/latest/command/target_link_libraries.html) command.

For example:

```cmake
project(Foo)
add_executable(foo foo.cpp)

# Search for oneDPL
find_package(oneDPL REQUIRED)

# Connect oneDPL to foo
target_link_libraries(foo oneDPL)
```

Availability of DPC++ and oneTBB backends is automatically checked during the invocation of `find_package(oneDPL <options>)` or `add_subdirectory(<oneDPL_dir> <output_dir>)`:

- macro `ONEDPL_USE_TBB_BACKEND` is set to `0` if oneTBB is not available;
- macro `ONEDPL_USE_DPCPP_BACKEND` is set to `0` if DPC++ is not available.

Detailed description of these and other macros is available in the [documentation](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-library-guide/top/parallel-stl-overview/macros.html). The macros can be explicitly set from user project.

For example:

```cmake
project(Foo)
add_executable(foo foo.cpp)

# Search for oneDPL
find_package(oneDPL REQUIRED)

# Connect oneDPL to foo
target_link_libraries(foo oneDPL)

# Disable TBB backend in oneDPL
target_compile_definitions(foo PRIVATE ONEDPL_USE_TBB_BACKEND=0)
```

### oneDPLConfig files generation

`cmake/script/generate_config.cmake` is provided to generate oneDPLConfig files for oneDPL package.

How to use:

`cmake [-DOUTPUT_DIR=<output_dir>] -P cmake/script/generate_config.cmake`
