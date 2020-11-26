# CMake system description

The project uses CMake for integration and testing purposes.

Configuration phase can be customized by passing additional variables: `cmake -D<var1>=<val1> -D<var2>=<val2> ... -D<varN>=<valN> <sources-dir>`

The following variables are provided for oneDPL configuration:

| Variable                       | Type   | Description                                                                                   | Default value |
|--------------------------------|--------|-----------------------------------------------------------------------------------------------|---------------|
| ONEDPL_USE_PARALLEL_POLICIES   | BOOL   | Enable parallel policies                                                                      | ON            |
| ONEDPL_BACKEND                 | STRING | Threading backend; supported values: tbb, sycl, sycl_only, ...                                | tbb           |
| ONEDPL_DEVICE_TYPE             | STRING | Device type, applicable only for sycl backends; supported values: GPU, CPU, FPGA_HW, FPGA_EMU | GPU           |
| ONEDPL_USE_UNNAMED_LAMBDA      | BOOL   | Pass `-fsycl-unnamed-lambda` compile option                                                   | OFF           |
| ONEDPL_USE_RANGES_API          | BOOL   | Enable the use of ranges API for algorithms                                                   | OFF           |
| ONEDPL_FPGA_STATIC_REPORT      | BOOL   | Enable the static report generation for the FPGA_HW device type                               | OFF           |
| ONEDPL_USE_OFFLINE_COMPILATION | BOOL   | Enable the offline compilation via OpenCL™ Offline Compiler (OCLOC)                           | OFF           |
| ONEDPL_COMPILE_ARCH            | STRING | Architecture options for the offline compilation, supported values can be found [here](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html); the default value `*` means compilation for all available options | *             |

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
