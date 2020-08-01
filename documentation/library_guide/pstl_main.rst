Parallel STL
#############

Introduction to Parallel STL
=============================

Parallel STL is an implementation of the C++ standard library algorithms with support
for execution policies, as specified in ISO/IEC 14882:2017 standard, commonly called C++17.
The implementation also supports the unsequenced execution policy specified in the
final draft for the C++ 20 standard (N4860).

Parallel STL offers efficient support for both parallel and vectorized execution of
algorithms for Intel® processors. For sequential execution, it relies on an available
implementation of the C++ standard library. The implementation also supports
the unsequenced execution policy specified in the final draft for the next version
of the C++ standard and DPC++ execution policy specified in
`the oneDPL Spec <https://spec.oneapi.com/versions/latest/elements/oneDPL/source/index.html#dpc-execution-policy>`_.

Prerequisites
==============
To use Parallel STL with standard policies, you must have the following software installed:

- C++ compiler with:

  - Support for C++11
  - Support for OpenMP* 4.0 SIMD constructs

- oneAPI Threading Building Blocks (oneTBB), or Intel(R) Threading Building Blocks 2019 and later

Use Parallel STL
=================
Follow these steps to add Parallel STL to your application:

#. Add ``#include <oneapi/dpl/execution>`` to your code. Then add a subset of the following set of lines, depending on the algorithms you intend to use:

   - ``#include <oneapi/dpl/algorithm>``
   - ``#include <oneapi/dpl/numeric>``
   - ``#include <oneapi/dpl/memory>``

#. Pass the policy object to a Parallel STL algorithm, which is defined in the ``oneapi::dpl::execution`` namespace.
#. Compile the code as C++11 (or later) and using compiler options for vectorization.
#. Link with the Intel TBB dynamic library for parallelism.

Example
========

.. code:: cpp

  #include <vector>
  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>

  int main()
  {
      std::vector<int> data( 1000 );
      std::fill(oneapi::dpl::execution::par_unseq, data.begin(), data.end(), 42);
      return 0;
  }
