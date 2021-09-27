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

C++11 is the minimal version of the C++ standard that |onedpl_short| requires. That means, any use of |onedpl_short|
requires at least a C++11 compiler. Some APIs of the library may require a higher version of C++.
To call Parallel API with the C++ standard policies, you need to install the following software:

* A C++ compiler with support for OpenMP* 4.0 (or higher) SIMD constructs
* |onetbb_long| or |tbb_long| 2019 and later

To use Parallel API with the |dpcpp_short| execution policies, you need to install the following software:

* A C++ compiler with support for SYCL* 2020

Restrictions
============

When called with |dpcpp_short| execution policies, |onedpl_short| algorithms apply the same restrictions as |dpcpp_short|
does (see the |dpcpp_short| specification and the SYCL specification for details), such as:

* Adding buffers to a lambda capture list is not allowed for lambdas passed to an algorithm.
* Passing data types, which are not trivially copyable, is only allowed via USM,
  but not via buffers or host-allocated containers.

Known Limitations
=================

For ``transform_exclusive_scan``, ``transform_inclusive_scan`` algorithms, the result of the unary operation should be
convertible to the type of the initial value if (one is provided), otherwise it is convertible to the type of values
in the processed data sequence: (``std::iterator_traits<IteratorType>::value_type``).

Build Your Code with |onedpl_short|
===================================

Follow the steps below to build your code with |onedpl_short|:

#. To build with the |dpcpp_cpp|, see the `Get Started with the Intel® oneAPI DPC++/C++ Compiler
   <https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-dpcpp-compiler/top.html>`_
   for details.
#. Set the environments for |onedpl_short| and |onetbb_short|.
#. To avoid naming device policy objects explicitly, add the ``-fsycl-unnamed-lambda`` option.

Below is an example of a command line used to compile code that contains
|onedpl_short| parallel algorithms on Linux* (depending on the code, parameters within [] could be unnecessary):

.. code:: cpp

  dpcpp [-fsycl-unnamed-lambda] test.cpp [-ltbb] -o test

