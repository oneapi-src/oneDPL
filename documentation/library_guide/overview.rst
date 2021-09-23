oneDPL overview
###############

Parallel API can be used with `C++ Standard Execution
Policies <https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t>`_
to enable parallelism on the host.

|onedpl_long| is implemented in accrodiance with `oneDPL
Specification <https://spec.oneapi.com/versions/latest/elements/oneDPL/source/index.html>`_.

To support heterogeneity |onedpl_short| iteroperates with DPC++ API. More information can be found in
`DPC++ Specification <https://spec.oneapi.com/versions/latest/elements/dpcpp/source/index.html#dpc>`_.

Before You Begin
================

Visit the |onedpl_long| `Release Notes
<https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-dpcpp-library-release-notes.html>`_
page for:

* Where to Find the Release
* Overview
* New in this Release
* Known Issues

Install the `Intel® oneAPI Base Toolkit (Base Kit) <https://software.intel.com/en-us/oneapi/base-kit>`_
to use |onedpl_short|.

All |onedpl_short| header files are in the ``oneapi/dpl`` directory. To use |onedpl_short| API,
include the correspondingheader in your source code with ``#include <oneapi/dpl/…>`` directive.
|onedpl_short| introduces the namespace ``oneapi::dpl`` for most its classes and functions.

To use tested C++ standard APIs, you need to include the corresponding C++ standard header files
and use the ``std`` namespace.

Prerequisites
=============

C++11 is the minimal version of the C++ standard, which |onedpl_short| requires. That means, any use of |onedpl_short|
requires at least a C++11 compiler. Some API of the library may require a higher version of C++.
To call Parallel API with the C++ standard policies, you shall have the following software installed:

* C++ compiler with support for OpenMP* 4.0 (or higher) SIMD constructs
* |onetbb_long| or |tbb_long| 2019 and later

To use Parallel API with the |dpcpp_short| execution policies, you must have the following software installed:

* C++ compiler with support for SYCL* 2020

Restrictions
============

When called with |dpcpp_short| execution policies, |onedpl_short| algorithms apply the same restrictions as |dpcpp_short|
does (see the |dpcpp_short| specification and the SYCL specification for details), such as:

* Adding buffers to a lambda capture list is not allowed for lambdas passed to an algorithm.
* Passing data types, which are not trivially copiable, is only allowed via USM,
  but not via buffers or host-allocated containers.

Known Limitations
=================

For ``transform_exclusive_scan``, ``transform_inclusive_scan`` algorithms result of
unary operation should be convertible to the type of the initial value if one is provided,
otherwise to the type of values in the processed data sequence
(``std::iterator_traits<IteratorType>::value_type``).

Build Your Code with |onedpl_short|
===================================

Follow the steps below to build your code with |onedpl_short|:

#. To build with the |dpcpp_cpp|, see the `Get Started with the Intel® oneAPI DPC++/C++ Compiler
   <https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-dpcpp-compiler/top.html>`_
   for details.
#. Set the environment for |onedpl_short| and |onetbb_short|.
#. To avoid naming device policy objects explicitly, add the ``-fsycl-unnamed-lambda`` option.

Below is an example of a command line used to compile code that contains
|onedpl_short| parallel algorithms on Linux* (depending on the code, parameters within [] could be unnecessary):

.. code::

  dpcpp [-fsycl-unnamed-lambda] test.cpp [-ltbb] -o test

