|onedpl_long| Introduction
#######################################

Parallel API can be used with the `C++ Standard Execution
Policies <https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t>`_
to enable parallelism on the host.

The |onedpl_long| (|onedpl_short|) is implemented in accordance with the `oneDPL
Specification <https://spec.oneapi.io/versions/latest/elements/oneDPL/source/index.html>`_.

To support heterogeneity, |onedpl_short| works with the DPC++ API. More information can be found in the
`oneAPI Specification <https://spec.oneapi.io/versions/latest/elements/sycl/source/index.html>`_.

Before You Begin
================

Visit the |onedpl_short| `Release Notes
<https://www.intel.com/content/www/us/en/developer/articles/release-notes/intel-oneapi-dpcpp-library-release-notes.html>`_
page for:

* Where to Find the Release
* Overview
* New Features
* Fixed Issues
* Deprecation Notice
* Known Issues and Limitations
* Previous Release Notes 

Install the `Intel® oneAPI Base Toolkit (Base Kit) <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html#gs.xaontv>`_
to use |onedpl_short|.

All |onedpl_short| header files are in the ``oneapi/dpl`` directory. To use the |onedpl_short| API,
include the corresponding header in your source code with the ``#include <oneapi/dpl/…>`` directive.
|onedpl_short| introduces the namespace ``oneapi::dpl`` for most its classes and functions.

To use tested C++ standard APIs, you need to include the corresponding C++ standard header files
and use the ``std`` namespace.

System Requirements
===================

Prerequisites
*************

C++17 is the minimal supported version of the C++ standard.
That means, any use of |onedpl_short| may require a C++17 compiler.
While some APIs of the library may accidentally work with earlier versions of the C++ standard, it is no more guaranteed.
 
To call Parallel API with the C++ standard policies, you need to install the following software:

* A C++ compiler with support for OpenMP* 4.0 (or higher) SIMD constructs
* Depending on what parallel backend you want to use install either:

  * |onetbb_long| or |tbb_long| 2019 and later
  * A C++ compiler with support for OpenMP 4.5 (or higher)

For more information about parallel backends, see :doc:`Execution Policies <parallel_api/execution_policies>`

To use Parallel API with the device execution policies, you need to install the following software:

* A C++ compiler with support for SYCL 2020

Difference with Standard C++ Parallel Algorithms
************************************************

* oneDPL execution policies only result in parallel execution if random access iterators are provided,
  the execution will remain serial for other iterator types.
* For the following algorithms, par_unseq and unseq policies do not result in vectorized execution:
  ``includes``, ``inplace_merge``, ``merge``, ``set_difference``, ``set_intersection``,
  ``set_symmetric_difference``, ``set_union``, ``stable_partition``, ``unique``.
* The following algorithms require additional O(n) memory space for parallel execution:
  ``copy_if``, ``inplace_merge``, ``partial_sort``, ``partial_sort_copy``, ``partition_copy``,
  ``remove``, ``remove_if``, ``rotate``, ``sort``, ``stable_sort``, ``unique``, ``unique_copy``.


Restrictions
************

When called with |dpcpp_short| execution policies, |onedpl_short| algorithms apply the same restrictions as
|dpcpp_short| does (see the |dpcpp_short| specification and the SYCL specification for details), such as:

* Adding buffers to a lambda capture list is not allowed for lambdas passed to an algorithm.
* Passing data types, which are not trivially copyable, is only allowed via USM,
  but not via buffers or host-allocated containers.
* The definition of lambda functions used with parallel algorithms should not depend on preprocessor macros
  that makes it different for the host and the device. Otherwise, the behavior is undefined.
* When used within SYCL kernels or transferred to/from a device, a container class can only hold objects
  whose type meets SYCL requirements for use in kernels and for data transfer, respectively.
* Calling the API that throws exception is not allowed within callable objects passed to an algorithm.

Known Limitations
*****************

* For ``transform_exclusive_scan``, ``transform_inclusive_scan`` algorithms the result of the unary operation should be
  convertible to the type of the initial value if one is provided, otherwise it is convertible to the type of values
  in the processed data sequence: ``std::iterator_traits<IteratorType>::value_type``.
* ``exclusive_scan`` and ``transform_exclusive_scan`` algorithms may provide wrong results with
  vector execution policies when building a program with GCC 10 and using ``-O0`` option.
* Compiling ``reduce`` and ``transform_reduce`` algorithms with the Intel DPC++ Compiler, versions 2021 and older,
  may result in a runtime error. To fix this issue, use an Intel DPC++ Compiler version 2022 or newer.
* The use of |onedpl_short| together with the GNU C++ standard library (libstdc++) version 9 or 10 may lead to
  compilation errors (caused by oneTBB API changes).
  Using libstdc++ version 9 requires TBB version 2020 for the header file. This may result in compilation errors when
  using C++17 or C++20 and TBB is not found in the environment, even if its use in |onedpl_short| is switched off.
  To overcome these issues, include |onedpl_short| header files before the standard C++ header files,
  or disable parallel algorithms support in the standard library. 
  For more information, please see `Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`_.
* The ``using namespace oneapi;`` directive in a |onedpl_short| program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, explicitly use
  ``oneapi::dpl`` namespace, or create a namespace alias. 
* ``std::array::at`` member function cannot be used in kernels because it may throw an exception;
  use ``std::array::operator[]`` instead.
* Due to specifics of Microsoft* Visual C++, some standard floating-point math functions
  (including ``std::ldexp``, ``std::frexp``, ``std::sqrt(std::complex<float>)``) require device support
  for double precision. 
* The initial value type for ``exclusive_scan``, ``inclusive_scan``, ``exclusive_scan_by_segment``,
  ``inclusive_scan_by_segment``, ``transform_exclusive_scan``, ``transform_inclusive_scan`` should satisfy
  the ``DefaultConstructible`` requirements. Additionally, a default-constructed instance of that type should be
  the identity element for the provided scan binary operation. 
* The initial value type for ``exclusive_scan``, ``inclusive_scan``, ``exclusive_scan_by_segment``,
  ``inclusive_scan_by_segment``, ``reduce``, ``reduce_by_segment``, ``transform_reduce``, ``transform_exclusive_scan``,
  ``transform_inclusive_scan`` should satisfy the ``MoveAssignable`` and the ``CopyConstructible`` requirements.
* For ``max_element``, ``min_element``, ``minmax_element``, ``partial_sort``, ``partial_sort_copy``, ``sort``, ``stable_sort``
  the dereferenced value type of the provided iterators should satisfy the ``DefaultConstructible`` requirements.
* For ``remove``, ``remove_if``, ``unique`` the dereferenced value type of the provided
  iterators should be ``MoveConstructible``.
        

Build Your Code with |onedpl_short|
===================================

Follow the steps below to build your code with |onedpl_short|:

#. To build with the |dpcpp_cpp|, see the `Get Started with the Intel® oneAPI DPC++/C++ Compiler
   <https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/get-started-guide/current/overview.html>`_
   for details.
#. Set the environment variables for |onedpl_short| and |onetbb_short|.
#. To avoid naming device policy objects explicitly, add the ``-fsycl-unnamed-lambda`` option.

Below is an example of a command line used to compile code that contains
|onedpl_short| parallel algorithms on Linux* (depending on the code, parameters within [] could be unnecessary):

.. code:: cpp

  dpcpp [-fsycl-unnamed-lambda] test.cpp [-ltbb|-fopenmp] -o test

.. _`Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`: https://www.intel.com/content/www/us/en/developer/articles/release-notes/intel-oneapi-threading-building-blocks-release-notes.html
