Intel® oneAPI DPC++ Library (oneDPL) Release Notes
###################################################

Overview
=========

The Intel® oneAPI DPC++ Library (oneDPL) accompanies the Intel® oneAPI DPC++/C++ Compiler
and provides high-productivity APIs aimed to minimize programming efforts of C++ developers
creating efficient heterogeneous applications.

New in 2021.1 Gold
===================

Key Features
-------------
- This version implements `the oneDPL Specification`_ v1.0, including parallel algorithms,
  DPC++ execution policies, special iterators, and other utilities.
- oneDPL algorithms can work with data in DPC++ buffers as well as in unified shared memory (USM).
- For several algorithms, experimental API that accepts ranges (similar to C++20) is additionally provided.
- A subset of the standard C++ libraries for Microsoft* Visual C++, GCC, and Clang is supported
  in DPC++ kernels, including ``<array>``, ``<complex>``, ``<functional>``, ``<tuple>``,
  ``<type_traits>``, ``<utility>`` and other standard library API.
  For the detailed list, please refer to `the oneDPL User Guide`_.
- Standard C++ random number generators and distributions for use in DPC++ kernels.


Known Issues and Limitations
-----------------------------
- The use of oneDPL together with the GNU C++ standard library (libstdc++) version 9 or 10 may lead to
  compilation errors (caused by oneTBB API changes).
  To overcome these issues, include oneDPL header files before the standard C++ header files,
  or disable parallel algorithms support in the standard library. For more information, please see `Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`_.
- The ``using namespace oneapi;`` directive in a oneDPL program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, explicitly use
  ``oneapi::dpl`` namespace, or create a namespace alias.
- The ``partial_sort_copy``, ``sort`` and ``stable_sort`` algorithms are prone to ``CL_BUILD_PROGRAM_FAILURE``
  when using Radix sort [#fnote1]_ in debug mode on oneAPI CPU devices.
- The implementation does not yet provide ``namespace oneapi::std`` as defined in `the oneDPL Specification`_.
- The use of the range-based API requires C++17 and the C++ standard libraries coming with GCC 8.1 (or higher)
  or Clang 7 (or higher).
- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- When used within DPC++ kernels or transferred to/from a device, ``std::array`` can only hold objects
  whose type meets DPC++ requirements for use in kernels and for data transfer, respectively.
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception;
  use ``std::array::operator[]`` instead.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- Due to specifics of Microsoft* Visual C++, some standard floating-point math functions
  (including ``std::ldexp``, ``std::frexp``, ``std::sqrt(std::complex<float>)``) require device support
  for double precision.

.. [#fnote1] The sorting algorithms in oneDPL use Radix sort for arithmetic data types compared with
   ``std::less`` or ``std::greater``, otherwise Merge sort.
.. _`the oneDPL Specification`: https://spec.oneapi.com/versions/latest/elements/oneDPL/source/index.html
.. _`the oneDPL User Guide`: https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-library-guide/top.html
.. _`Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`: https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-threading-building-blocks-release-notes.html
