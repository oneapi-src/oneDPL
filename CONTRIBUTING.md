# How to Contribute

We welcome community contributions to oneAPI DPC++ Library (oneDPL). You can:

- Submit your changes directly with a [pull request](https://github.com/oneapi-src/oneDPL/pulls).
- Log a bug or feedback with an [issue](https://github.com/oneapi-src/oneDPL/issues).

# License

oneDPL is licensed under the terms in [LICENSE](https://github.com/oneapi-src/oneDPL/blob/release_oneDPL/licensing/LICENSE.txt).
By contributing to the project, you agree to the license and copyright terms therein and
release your contribution under these terms.

# Pull Requests

This project follows the
[GitHub flow](https://guides.github.com/introduction/flow/index.html). To submit
your change directly to the repository:

- Make sure your code is in line with our
  [coding conventions](#coding-conventions).
- Test your code locally to identify and resolve simple issues by doing
  [validation testing](#validation-testing).
- Submit a
  [pull request](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) into the
  main branch. You may add a description of your contribution into [CREDITS.txt](https://github.com/oneapi-src/oneDPL/blob/main/CREDITS.txt).

# Coding Conventions

Running clang-format is required, except in the [test folder](https://github.com/oneapi-src/oneDPL/tree/main/test).

# Validation Testing

oneDPL validation tests can be configured through CMake to run on different devices including CPU and GPU by specifying the oneDPL backend to use and
the device type to be used if the DPC++ backend is being tested. The oneTBB and OpenMP backends for oneDPL can be used when the device type is CPU or HOST.

To run oneDPL validation tests on the gpu on your local system using the Intel® oneAPI DPC++ Compiler 

1. Configure the Makefiles by running the following command from the [test folder](https://github.com/oneapi-src/oneDPL/tree/main/test).
```
cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_STANDARD=17 -DONEDPL_BACKEND=dpcpp
-DONEDPL_DEVICE_TYPE=gpu -DCMAKE_BUILD_TYPE=release ..
```

2. Build the tests
```
cmake --build . --target build-onedpl-tests # specify a specific test name (e.g., reduce.pass) to build a single test
```

3. Run the tests
```
ctest --output-on-failure --timeout 1200 -R ^reduce.pass$ # Add -R testname (e.g., -R ^reduce.pass$) to run just one test.
```
