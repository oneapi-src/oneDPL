Algorithms
####################################################

Parallel STL is an implementation of the C++ standard library algorithms with support for execution
policies, as specified in ISO/IEC 14882:2017 standard, commonly called C++17. The implementation also
supports the unsequenced execution policy and the algorithms shift_left and shift_right, which are specified
in the final draft for the C++ 20 standard (N4860). For more details see the `C++ Reference Standard Execution
Policies <https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t>`_.

Parallel STL offers efficient support for both parallel and vectorized execution of
algorithms for Intel® processors. For sequential execution, it relies on an available
implementation of the C++ standard library. 

Prerequisites
==============

C++11 is the minimal version of the C++ standard, which |onedpl_short| requires. That means, any use of |onedpl_short|
requires at least a C++11 compiler. Some uses of the library may require a higher version of C++.
To use Parallel STL with the C++ standard policies, you must have the following software installed:

* C++ compiler with support for OpenMP* 4.0 (or higher) SIMD constructs
* |onetbb_long| or |tbb_long| 2019 and later

To use Parallel STL with the |dpcpp_short| execution policies, you must have the following software installed:

* C++ compiler with support for SYCL* 2020

Restrictions
============

When used with |dpcpp_short| execution policies, |onedpl_short| algorithms apply the same restrictions as |dpcpp_short|
does (see the |dpcpp_short| specification and the SYCL specification for details), such as:

* Adding buffers to a lambda capture list is not allowed for lambdas passed to an algorithm.
* Passing data types, which are not trivially constructible, is only allowed in USM,
  but not in buffers or host-allocated containers.

Known Limitations
=================

For ``transform_exclusive_scan``, ``transform_inclusive_scan`` algorithms result of
unary operation should be convertible to the type of the initial value if one is provided,
otherwise to the type of values in the processed data sequence
(``std::iterator_traits<IteratorType>::value_type``).

Build Your Code with |onedpl_short|
===================================

Use these steps to build your code with |onedpl_short|:

#. To build with the |dpcpp_cpp|, see the `Get Started with the Intel® oneAPI DPC++/C++ Compiler
   <https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-dpcpp-compiler/top.html>`_
   for details.
#. Set the environment for |onedpl_short| and |onetbb_short|.
#. To avoid naming device policy objects explicitly, add the ``-fsycl-unnamed-lambda`` option.

Below is an example of a command line used to compile code that contains
|onedpl_short| parallel algorithms on Linux* (depending on the code, parameters within [] could be unnecessary):

.. code::

  dpcpp [-fsycl-unnamed-lambda] test.cpp [-ltbb] -o test