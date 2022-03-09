|onedpl_long| Overview
#######################################

Parallel API can be used with the `C++ Standard Execution
Policies <https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t>`_
to enable parallelism on the host.

The |onedpl_long| is implemented in accordance with the `oneDPL
Specification <https://spec.oneapi.com/versions/latest/elements/oneDPL/source/index.html>`_.

To support heterogeneity, |onedpl_short| works with the Data Parallel C++ (DPC++) API. More information can be found in the
`DPC++ Specification <https://spec.oneapi.com/versions/latest/elements/dpcpp/source/index.html#dpc>`_.

Before You Begin
================

Visit the |onedpl_long| `Release Notes
<https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-dpcpp-library-release-notes.html>`_
page for:

* Where to Find the Release
* Overview
* New Features
* Fixed Issues
* Known Issues and Limitations

Install the `Intel® oneAPI Base Toolkit (Base Kit) <https://software.intel.com/en-us/oneapi/base-kit>`_
to use |onedpl_short|.

All |onedpl_short| header files are in the ``oneapi/dpl`` directory. To use the |onedpl_short| API,
include the corresponding header in your source code with the ``#include <oneapi/dpl/…>`` directive.
|onedpl_short| introduces the namespace ``oneapi::dpl`` for most its classes and functions.

To use tested C++ standard APIs, you need to include the corresponding C++ standard header files
and use the ``std`` namespace.

Prerequisites
=============

Since |onedpl_short| 2021.6, C++17 is the minimal supported version of the C++ standard.
That means, any use of |onedpl_short| may require a C++17 compiler.
While some APIs of the library may accidentally work with earlier versions of the C++ standard, it is no more guaranteed.
 
To call Parallel API with the C++ standard policies, you need to install the following software:

* A C++ compiler with support for OpenMP* 4.0 (or higher) SIMD constructs
* Depending on what parallel backend you want to use install either:

  * |onetbb_long| or |tbb_long| 2019 and later
  * A C++ compiler with support for OpenMP 4.5 (or higher)

For more information about parallel backends, see :doc:`Execution Policies <parallel_api/execution_policies>`

To use Parallel API with the |dpcpp_short| execution policies, you need to install the following software:

* A C++ compiler with support for SYCL* 2020

Restrictions
============

When called with |dpcpp_short| execution policies, |onedpl_short| algorithms apply the same restrictions as |dpcpp_short|
does (see the |dpcpp_short| specification and the SYCL specification for details), such as:

* Adding buffers to a lambda capture list is not allowed for lambdas passed to an algorithm.
* Passing data types, which are not trivially copyable, is only allowed via USM,
  but not via buffers or host-allocated containers.
* The definition of lambda functions used with parallel algorithms should not depend on preprocessor macros
  that makes it different for the host and the device. Otherwise, the behavior is undefined.
* When used within DPC++ kernels or transferred to/from a device, a container class can only hold objects
  whose type meets DPC++ requirements for use in kernels and for data transfer, respectively.
* Calling the API that throws exception is not allowed within callable objects passed to an algorithm.

Known Limitations
=================

* For ``transform_exclusive_scan``, ``transform_inclusive_scan`` algorithms, the result of the unary operation should be
  convertible to the type of the initial value if one is provided, otherwise it is convertible to the type of values
  in the processed data sequence: ``std::iterator_traits<IteratorType>::value_type``.
* ``exclusive_scan`` and ``transform_exclusive_scan`` algorithms may provide wrong results with
  vector execution policies when building a program with GCC 10 and using ``-O0`` option.
* The use of oneDPL together with the GNU C++ standard library (libstdc++) version 9 or 10 may lead to
  compilation errors (caused by oneTBB API changes). 
  To overcome these issues, include oneDPL header files before the standard C++ header files,
  or disable parallel algorithms support in the standard library. 
  For more information, please see `Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`_.
* When using oneDPL with serial or OpenMP backend and libstdc++ version 9, and C++17 or C++20, a compilation error occurs
  if oneTBB is not present in the environment. It happens because libstdc++ version 9 does not check
  oneTBB availability. To get rid of the error, disable support for Parallel STL algorithms by setting the macro
  ``PSTL_USE_PARALLEL_POLICIES`` to zero before including the first standard header file in each translation unit,
  or include oneDPL headers before the rest of the headers.
* The ``using namespace oneapi;`` directive in a oneDPL program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, explicitly use
  ``oneapi::dpl`` namespace, or create a namespace alias. 
* ``std::array::at`` member function cannot be used in kernels because it may throw an exception;
  use ``std::array::operator[]`` instead.
* Due to specifics of Microsoft* Visual C++, some standard floating-point math functions
  (including ``std::ldexp``, ``std::frexp``, ``std::sqrt(std::complex<float>)``) require device support
  for double precision. 

Build Your Code with |onedpl_short|
===================================

Follow the steps below to build your code with |onedpl_short|:

#. To build with the |dpcpp_cpp|, see the `Get Started with the Intel® oneAPI DPC++/C++ Compiler
   <https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-dpcpp-compiler/top.html>`_
   for details.
#. Set the environment variables for |onedpl_short| and |onetbb_short|.
#. To avoid naming device policy objects explicitly, add the ``-fsycl-unnamed-lambda`` option.

Below is an example of a command line used to compile code that contains
|onedpl_short| parallel algorithms on Linux* (depending on the code, parameters within [] could be unnecessary):

.. code:: cpp

  dpcpp [-fsycl-unnamed-lambda] test.cpp [-ltbb|-fopenmp] -o test

.. _`Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`: https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-threading-building-blocks-release-notes.html
