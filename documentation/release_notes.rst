Intel® oneAPI DPC++ Library (oneDPL) Release Notes
###################################################

Overview
=========

The Intel® oneAPI DPC++ Library (oneDPL) accompanies the Intel® oneAPI DPC++/C++ Compiler
and provides high-productivity APIs aimed to minimize programming efforts of C++ developers
creating efficient heterogeneous applications.

New in 2021.3
=============

New Features
------------
-  Added the range-based versions of the following algorithms: ``all_of``, ``any_of``, ``count``,
   ``count_if``, ``equal``, ``move``, ``remove``, ``remove_if``, ``replace``, ``replace_if``.
-  Added the following utility ranges (views): ``generate``, ``fill``, ``rotate``.

Changes to Existing Features
-----------------------------
-  Improved performance of ``discard_block_engine`` (including ``ranlux24``, ``ranlux48``,
   ``ranlux24_vec``, ``ranlux48_vec`` predefined engines) and ``normal_distribution``.
- Added two constructors to ``transform_iterator``: the default constructor and a constructor from an iterator without a transformation.
  ``transform_iterator`` constructed these ways uses transformation functor of type passed in template arguments.
- ``transform_iterator`` can now work on top of forward iterators.

Fixed Issues
------------
-  Fixed execution of ``swap_ranges`` algorithm with ``unseq``, ``par`` execution policies.
-  Fixed an issue causing memory corruption and double freeing in scan-based algorithms compiled with
   -O0 and -g options and run on CPU devices.
-  Fixed incorrect behavior in the ``exclusive_scan`` algorithm that occurred when the input and output iterator ranges overlapped.
-  Fixed error propagation for async runtime exceptions by consistently calling ``sycl::event::wait_and_throw`` internally.
-  Fixed the warning: ``local variable will be copied despite being returned by name [-Wreturn-std-move]``.

Known Issues and Limitations
-----------------------------
- No new issues in this release. 

Existing Issues
^^^^^^^^^^^^^^^^
- ``exclusive_scan`` and ``transform_exclusive_scan`` algorithms may provide wrong results with vector execution policies
  when building a program with GCC 10 and using -O0 option.
- Some algorithms may hang when a program is built with -O0 option, executed on GPU devices and large number of elements is to be processed.
- The use of oneDPL together with the GNU C++ standard library (libstdc++) version 9 or 10 may lead to
  compilation errors (caused by oneTBB API changes).
  To overcome these issues, include oneDPL header files before the standard C++ header files,
  or disable parallel algorithms support in the standard library.
  For more information, please see `Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`_.
- The ``using namespace oneapi;`` directive in a oneDPL program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, explicitly use
  ``oneapi::dpl`` namespace, or create a namespace alias.
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

New in 2021.2
=============

New Features
------------
-  Added support of parallel, vector and DPC++ execution policies for the following algorithms: ``shift_left``, ``shift_right``.
-  Added the range-based versions of the following algorithms: ``sort``, ``stable_sort``, ``merge``.
-  Added non-blocking versions of the following algorithms: ``copy``, ``fill``, ``for_each``, ``reduce``, ``sort``, ``transform``, ``transform_reduce``.
   These algorithms are declared in ``oneapi::dpl::experimental`` namespace with suffix _async and implemented only for DPC++ policies.
   In order to make these algorithms available the ``<oneapi/dpl/async>`` header should be included. Use of the non-blocking API requires C++11.
-  Utility function ``wait_for_all`` enables waiting for completion of an arbitrary number of events.
-  Added the ``ONEDPL_USE_PREDEFINED_POLICIES`` macro, which enables predefined policy objects and
   ``make_device_policy``, ``make_fpga_policy`` functions without arguments. It is turned on by default.

Changes to Existing Features
-----------------------------
- Improved performance of the following algorithms: ``count``, ``count_if``, ``is_partitioned``,
  ``lexicographical_compare``, ``max_element``, ``min_element``,  ``minmax_element``, ``reduce``, ``transform_reduce``,
  and ``sort``, ``stable_sort`` when using Radix sort [#fnote1]_.
- Improved performance of the linear_congruential_engine RNG engine (including ``minstd_rand``, ``minstd_rand0``,
  ``minstd_rand_vec``, ``minstd_rand0_vec`` predefined engines).

Fixed Issues
------------
- Fixed runtime errors occurring with ``find_end``, ``search``, ``search_n`` algorithms when a program is built with -O0 option and executed on CPU devices.
- Fixed the majority of unused parameter warnings.

Known Issues and Limitations
-----------------------------
- ``exclusive_scan`` and ``transform_exclusive_scan`` algorithms may provide wrong results with vector execution policies
  when building a program with GCC 10 and using -O0 option.
- Some algorithms may hang when a program is built with -O0 option, executed on GPU devices and large number of elements is to be processed.
- The use of oneDPL together with the GNU C++ standard library (libstdc++) version 9 or 10 may lead to
  compilation errors (caused by oneTBB API changes).
  To overcome these issues, include oneDPL header files before the standard C++ header files,
  or disable parallel algorithms support in the standard library.
  For more information, please see `Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`_.
- The ``using namespace oneapi;`` directive in a oneDPL program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, explicitly use
  ``oneapi::dpl`` namespace, or create a namespace alias.
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
  or disable parallel algorithms support in the standard library.
  For more information, please see `Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`_.
- The ``using namespace oneapi;`` directive in a oneDPL program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, explicitly use
  ``oneapi::dpl`` namespace, or create a namespace alias.
- The ``partial_sort_copy``, ``sort`` and ``stable_sort`` algorithms are prone to ``CL_BUILD_PROGRAM_FAILURE``
  when a program uses Radix sort [#fnote1]_, is built with -O0 option and executed on CPU devices.
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
