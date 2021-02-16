The standard C++17 algorithms and execution policies
####################################################

Parallel STL is an implementation of the C++ standard library algorithms with support
for execution policies, as specified in ISO/IEC 14882:2017 standard, commonly called C++17.
The implementation also supports the unsequenced execution policy specified in the
final draft for the C++ 20 standard (N4860). For more details see `the standard execution policies
<https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t>`_.

Parallel STL offers efficient support for both parallel and vectorized execution of
algorithms for Intel® processors. For sequential execution, it relies on an available
implementation of the C++ standard library. 

Prerequisites
==============

C++11 is the minimal version of the C++ standard, which oneDPL requires. That means, any use of oneDPL
requires at least a C++11 compiler. Some uses of the library may require a higher version of C++.
To use Parallel STL with the C++ standard policies, you must have the following software installed:

  * C++ compiler with support for OpenMP* 4.0 (or higher) SIMD constructs
  * oneAPI Threading Building Blocks (oneTBB) or Threading Building Blocks (TBB) 2019 and later

To use Parallel STL with the DPC++ execution policies, you must have the following software installed:

  * C++ compiler with support for SYCL 2020

