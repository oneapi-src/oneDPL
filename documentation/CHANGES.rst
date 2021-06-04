Intel® oneAPI DPC++ Library (oneDPL) CHANGES
##########################################################

Overview
========

The list of the most significant changes made over time in oneDPL.

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

New in 2021.2
=============

New Features
------------
-  Added support of parallel, vector and DPC++ execution policies for the following algorithms: ``shift_left``, ``shift_right``.
-  Added the Range-based versions of the following algorithms: ``sort``, ``stable_sort``, ``merge``.
-  Added non-blocking versions of the following algorithms: ``copy``, ``fill``, ``for_each``, ``reduce``, ``sort``, ``transform``, ``transform_reduce``. These algorithms are declared in ``oneapi::dpl::experimental`` namespace with suffix _async and implemented only for DPC++ policies. In order to make these algorithms available the ``<oneapi/dpl/async>`` header should be included. Use of the non-blocking API requires C++11.
-  Utility function ``wait_for_all`` enables waiting for completion of an arbitrary number of events.
-  Added the ``ONEDPL_USE_PREDEFINED_POLICIES`` macro, which enables predefined policy objects and ``make_device_policy``, ``make_fpga_policy`` functions without arguments. It is turned on by default.

Changes to Existing Features
-----------------------------
- Improved performance of the following algorithms: ``count``, ``count_if``, ``is_partitioned``, ``lexicographical_compare``, ``max_element``, ``min_element``, ``minmax_element``, ``reduce``, ``transform_reduce``, and ``sort``, ``stable_sort`` when using Radix sort [#fnote1]_.
- Improved performance of the linear_congruential_engine RNG engine (including ``minstd_rand``, ``minstd_rand0``, ``minstd_rand_vec``, ``minstd_rand0_vec`` predefined engines).

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
  or disable parallel algorithms support in the standard library. For more information, please see `Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`_.
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
====================

New Features
------------
-  Added ``ONEDPL_VERSION_MAJOR``, ``ONEDPL_VERSION_MINOR`` and ``ONEDPL_VERSION_PATCH`` macros which provide major version, minor version and patch of the library.

Changes to Existing Features
-----------------------------
- Previously deprecated interfaces were removed.

Fixed Issues
-------------
- Fixed compilation errors of oneDPL parallel algorithms when using "-sycl-std=2020" compiler switch.
- Fixed the segmentation fault issue on CPU devices in the ``exclusive_scan`` and ``transform_exclusive_scan`` algorithms.
- Fixed the failures of the ``partial_sort_copy``, ``sort`` and ``stable_sort`` algorithms when using Radix sort with 32-bit ``float`` data type.
- Fixed compilation issues that occurred using libstdc++9 or newer.
- Got rid of unused variables. 
- Fixed the issue of the ``is_sorted`` algorithm with use the C++ Standard Execution Policies ``par`` and  ``par_unseq``.

Known Issues and Limitations
----------------------------
- The use of oneDPL together with the GNU C++ standard library (libstdc++) version 9 or 10 may lead to
  compilation errors (caused by oneTBB API changes). To overcome these issues, include oneDPL header files before the standard C++ header files,
  or disable parallel algorithms support in the standard library. For more information, please see `Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`_.
- The ``using namespace oneapi;`` directive in a oneDPL program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, use fully qualified
  names or namespace aliases.
- The ``partial_sort_copy``, ``sort`` and ``stable_sort`` algorithms are prone to ``CL_BUILD_PROGRAM_FAILURE``
  when a program uses Radix sort [#fnote1]_, is built with -O0 option and executed on CPU devices.
- Some algorithms with a DPC++ policy may fail on CPU or on FPGA emulator.
- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- When used within DPC++ kernels or transferred to/from a device, ``std::array`` can only hold objects whose type meets DPC++ requirements for use in kernels
  and for data transfer, respectively.
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception;
  use ``std::array::operator[]`` instead.
- ``std::array`` member function swap cannot be used in DPC++ kernels on Windows platform.
- ``std::swap`` for ``std::array`` cannot work in DPC++ kernels on Windows platform.
- Not all functions in <cmath> are supported currently, please refer to `DPC++ library guide <https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-library-guide/top.html>`_ guide for detail list.
- Due to specifics of Microsoft Visual C++ implementation, some standard math functions for float
  (including: ``std::ldexp``, ``std::frexp``, ``std::sqrt(std::complex<float>)``) require device support
  for double precision.
- The use of the range-based API requires C++17 and the C++ standard libraries coming with GCC 8.1 (or higher) or Clang 7 (or higher).

New in 2021.1-beta10
====================

New Features
------------
- All oneDPL functionality, including the parallel algorithm functions, is accessible via the ``oneapi::dpl`` namespace.

Changes to Existing Features
-----------------------------
- The following methods of the permutation_iterator have been renamed: ``get_source_iterator()`` is renamed to ``base()``, ``get_map_iterator()`` is renamed to ``map()``.
- Improved performance of the following algorithms: ``copy_if``, ``count``, ``count_if``, ``exclusive_scan``, ``inclusive_scan``, ``is_partitioned``, ``lexicographical_compare``, ``max_element``, ``min_element``, ``minmax_element``, ``partition``, ``partition_copy``, ``reduce``, ``remove``, ``remove_copy``, ``remove_copy_if``, ``remove_if``, ``set_difference``, ``set_intersection``, ``set_symmetric_difference``, ``set_union``, ``stable_partition``, ``transform_exclusive_scan``, ``transform_inclusive_scan``, ``transform_reduce``, ``unique``, ``unique_copy``.
- Improved performance of the ``nth_element`` algorithm when input contains large number of duplicates.

Fixed Issues
-------------
- Fixed the failures of the ``sort``, ``stable_sort`` algorithms when using Radix sort on CPU devices.

Known Issues and Limitations
----------------------------
- The use of oneDPL together with the GNU C++ standard library (libstdc++) version 9 or 10 may lead to compilation errors (caused by oneTBB API changes).
  To overcome these, switch off the use of TBB for parallel execution policies in the standard library.
- The use of the -sycl-std=2020 option may lead to compilation errors for oneDPL parallel algorithms.
- The ``using namespace oneapi;`` directive in a oneDPL program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, use fully qualified
  names or namespace aliases.
- The ``partial_sort_copy``, ``sort`` and ``stable_sort`` algorithms are prone to ``CL_BUILD_PROGRAM_FAILURE``
  when a program uses Radix sort [#fnote1]_, is built with -O0 option and executed on CPU devices.
- The ``partial_sort_copy``, ``sort`` and ``stable_sort`` algorithms may produce incorrect result
  when using Radix sort with 32-bit ``float`` data type.
- Some algorithms with a DPC++ policy may fail on CPU or on FPGA emulator.
- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- When used within DPC++ kernels or transferred to/from a device, ``std::array`` can only hold objects whose type meets DPC++ requirements for use in kernels
  and for data transfer, respectively.
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception;
  use ``std::array::operator[]`` instead.
- ``std::array`` member function swap cannot be used in DPC++ kernels on Windows platform.
- ``std::swap`` for ``std::array`` cannot work in DPC++ kernels on Windows platform.
- Not all functions in <cmath> are supported currently, please refer to `DPC++ library guide <https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-library-guide/top.html>`_ for detail list.
- Due to specifics of Microsoft Visual C++ implementation, some standard math functions for float
  (including: ``std::ldexp``, ``std::frexp``, ``std::sqrt(std::complex<float>)``) require device support
  for double precision.


New in 2021.1-beta09
====================

New Features
------------
- Added the Range-based versions of following algorithms: ``for_each``, ``copy``, ``transform``,
  ``find``, ``find_if``, ``find_if_not``, ``find_end``, ``find_first_of``, ``search``, ``is_sorted``,
  ``is_sorted_until``, ``reduce``, ``transform_reduce``, ``min_element``, ``max_element``, ``minmax_element``,
  ``exclusive_scan``, ``inclusive_scan``, ``transform_exclusive_scan``, ``transform_inclusive_scan``.
  These algorithms are declared in ``oneapi::dpl::experimental::ranges`` namespace and implemented only for DPC++ policies.
  In order to make these algorithm available the ``<oneapi/dpl/ranges>`` header should be included.
  Use of the range-based API requires C++17 and the C++ standard libraries coming with GCC 8.1 (or higher) or Clang 7 (or higher).

Changes to Existing Features
-----------------------------
- Changed the order of template parameters for ``transform_iterator``, so the source iterator type is provided first (e.g., ``transform_iterator<Iterator, UnaryFunctor>``).
- Improved performance of the following algorithms: ``copy_if``, ``exclusive_scan``, ``inclusive_scan``, ``partition_copy``, ``remove_copy``, ``remove_copy_if``, ``transform_exclusive_scan``, ``transform_inclusive_scan`` using DPC++ policies.
- Improved performance of the ``sort`` and ``stable_sort`` algorithms when using Radix sort.
- Tested Standard C++ APIs are added to namespace ``oneapi::std`` and ``oneapi::dpl``. In order to use Tested Standard C++ APIs via ``oneapi::std`` or ``oneapi::dpl``, corresponding headers in ``<oneapi/dpl/...>`` must be included (e.g., ``#include <oneapi/dpl/utility>``).

Fixed Issues
-------------
- Fixed an error when local memory usage is out of limit.
- Eliminated warnings about ``std::result_of`` deprecation compiling with C++17 on Windows platform.

Known Issues and Limitations
----------------------------
- The conversion from ``zip_iterator::value_type`` to ``std::tuple`` may produce incorrect result.
- The ``using namespace oneapi;`` directive in a oneDPL program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, use fully qualified
  names or namespace aliases.
- On the DPC++ CPU device, RNG sequences produced by ``discard_block_engine`` may deviate
  from those generated by other implementations of the engine.
- The ``sort``, ``stable_sort``, ``partial_sort``, ``partial_sort_copy`` algorithms
  may work incorrectly on CPU device.
- The ``partial_sort_copy``, ``sort`` and ``stable_sort`` algorithms are prone to ``CL_BUILD_PROGRAM_FAILURE``
  when a program uses Radix sort [#fnote1]_, is built with -O0 option and executed on CPU devices.
- The ``partial_sort_copy``, ``sort`` and ``stable_sort`` algorithms may produce incorrect result
  when using Radix sort with 32-bit ``float`` data type.
- Some algorithms with a DPC++ policy may fail on CPU or on FPGA emulator.
- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- When used within DPC++ kernels or transferred to/from a device, ``std::array`` can only hold objects whose type meets DPC++ requirements for use in kernels
  and for data transfer, respectively.
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception;
  use ``std::array::operator[]`` instead.
- ``std::array`` member function swap cannot be used in DPC++ kernels on Windows platform.
- ``std::swap`` for ``std::array`` cannot work in DPC++ kernels on Windows platform.
- Not all functions in <cmath> are supported currently, please refer to `DPC++ library guide <https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-library-guide/top.html>`_ for detail list.
- Due to specifics of Microsoft Visual C++ implementation, some standard math functions for float
  (including: ``std::ldexp``, ``std::frexp``, ``std::sqrt(std::complex<float>)``) require device support
  for double precision.

New in 2021.1-beta08
====================

New Features
------------
- Added random number generation functionality in ``<oneapi/dpl/random>``:

  - ``linear_congruential_engine``, ``subtract_with_carry_engine``, ``discard_block_engine``;
  - predefined engine instantiations, including ``minstd_rand`` and ``ranlux48``;
  - ``uniform_real_distribution``, ``uniform_int_distribution``, ``normal_distribution``.

- Added implicit conversion of a DPC++ policy to ``sycl::queue``.
- Added the ``ONEDPL_STANDARD_POLICIES_ONLY`` macro (defaults to 0) that makes
  the DPC++ policies unavailable, avoiding dependency on the DPC++ compiler
  and limiting oneDPL algorithms to only use the standard C++ policies
  (``seq``, ``par``, ``unseq``, ``par_unseq``) for the host CPUs.
  It replaces the former ``_PSTL_BACKEND_SYCL`` macro with the opposite meaning.
- Added ``permutation_iterator`` and ``discard_iterator`` in ``<oneapi/dpl/iterator>``.

Changes to Existing Features
-----------------------------
- Improved performance of the ``sort`` and ``stable_sort`` algorithms
  with ``device_policy`` for non-arithmetic data types.
- The ``dpstd`` include folder was renamed. Include ``<oneapi/dpl/...>`` headers
  instead of ``<dpstd/...>``.
- The main namespace of the library changed to ``oneapi::dpl``. The ``dpstd``
  namespace is deprecated, and will be removed in one of the next releases.

- The following API elements of oneDPL were changed or removed:

  - the ``default_policy`` object was renamed to ``dpcpp_default``;
  - the ``fpga_policy`` object was renamed to ``dpcpp_fpga``;
  - the ``fpga_device_policy`` class was renamed to ``fpga_policy``;
  - the ``_PSTL_FPGA_DEVICE`` macro was renamed to ``ONEDPL_FPGA_DEVICE``;
  - the ``_PSTL_FPGA_EMU`` macro was renamed to ``ONEDPL_FPGA_EMULATOR``;
  - the ``_PSTL_COMPILE_KERNEL`` macro was removed;
  - the ``_PSTL_BACKEND_SYCL`` macro was removed.

  The ``default_policy``, ``fpga_device_policy`` names are deprecated,
  and will be removed in one of the next releases. Other previous names
  are no more valid.

Fixed Issues
-------------
- Fixed scan-based algorithms to not rely on independent forward progress for workgroups.

Known Issues and Limitations
----------------------------
- On the DPC++ CPU device, RNG sequences produced by ``discard_block_engine`` may deviate
  from those generated by other implementations of the engine.
- If ``<oneapi/dpl/random>`` is included into code before other oneDPL header files, compilation can fail.
  In order to avoid failures, include ``<oneapi/dpl/random>`` after any other oneDPL header file.
- The following algorithms may be significantly slower with ``device_policy``
  than in previous Beta releases: ``copy_if``, ``exclusive_scan``, ``inclusive_scan``, ``partition``,
  ``partition_copy``, ``remove``, ``remove_copy``, ``remove_if``, ``set_difference``,
  ``set_intersection``, ``set_symmetric_difference``, ``set_union``, ``stable_partition``,
  ``transform_exclusive_scan``, ``transform_inclusive_scan``, ``unique``, ``unique_copy``.
- ``sort``, ``stable_sort``, ``partial_sort``, ``partial_sort_copy`` algorithms
  may work incorrectly on CPU device and on GPU with DPC++ L0 backend.
- Some algorithms with a DPC++ policy may fail on CPU or on FPGA emulator.
- ``std::tuple`` cannot be used with SYCL buffers to transfer data between host and device.
- When used within DPC++ kernels or transferred to/from a device, ``std::tuple, std::pair``,
  and ``std::array`` can only hold objects whose type meets DPC++ requirements for use in kernels
  and for data transfer, respectively.
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception;
  use ``std::array::operator[]`` instead.
- ``std::array`` member function swap cannot be used in DPC++ kernels on Windows platform.
- ``std::swap`` for ``std::array`` cannot work in DPC++ kernels on Windows platform.
- Not all functions in <cmath> are supported currently, please refer to `DPC++ library guide <https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-library-guide/top.html>`_ for detail list.
- Due to specifics of Microsoft Visual C++ implementation, some standard math functions for float
  (including: ``std::ldexp``, ``std::frexp``, ``std::sqrt(std::complex<float>)``) require device support
  for double precision.

New in 2021.1-beta07
====================

New Features
------------
- The Microsoft Visual C++ implementation of ``std::complex`` is supported in device code.

Changes to Existing Features
----------------------------
- ``dpstd/iterators.h`` is deprecated and replaced with ``dpstd/iterator``.
- Improved performance of the ``any_of``, ``adjacent_find``, ``all_of``, ``equal``, ``find``, ``find_end``, ``find_first_of``, ``find_if``, ``find_if_not``, ``includes``, ``is_heap``, ``is_heap_until``, ``is_sorted``, ``is_sorted_until``, ``mismatch``, ``none_of``, ``search``,`` search_n`` algorithms using DPC++ policies.

Fixed Issues
-------------
- Fixed error with usage of ``dpstd::zip_iterator`` on Windows platform.
- Fixed ``exclusive_scan`` compilation errors with GCC* 9 and Clang* 9 in C++17 mode.
- Eliminated warnings about deprecated sub-group interfaces.

Known Issues and Limitations
----------------------------
- ``sort``, ``stable_sort``, ``partial_sort``, ``partial_sort_copy`` algorithms may work incorrectly in debug mode.
- Some algorithms with a DPC++ policy may fail on CPU or on FPGA emulator.
- ``std::tuple`` cannot be used with SYCL buffers to transfer data between host and device.
- When used within DPC++ kernels or transferred to/from a device, ``std::tuple, std::pair`` and ``std::array`` can only hold objects whose type meets DPC++ requirements for use in kernels and for data transfer, respectively.
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception; use ``std::array::operator[]`` instead.
- ``std::array`` member function swap cannot be used in DPC++ kernels on Windows platform.
- ``std::swap`` for ``std::array`` cannot work in DPC++ kernels on Windows platform.
- Not all functions in <cmath> are supported currently, please refer to `DPC++ library guide <https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-library-guide/top.html>`_ for detail list.
- Due to specifics of Microsoft Visual C++ implementation, some standard math functions for float (including: ``std::ldexp``, ``std::frexp``, ``std::sqrt(std::complex<float>)``) require device support for double precision.
- There is a known issue on Windows platform with trying to use clGetPlatformInfo and ClGetDeviceInfo when using a graphics driver older than 27.20.100.8280.
  If you run into this issue, please upgrade to the latest driver of at least version 27.20.100.8280 from the `Download Center <https://downloadcenter.intel.com/product/80939/Graphics>`_.

New in 2021.1-beta06
====================

New Features
-----------------------------
- Added ``fpga_device_policy`` class, ``make_fpga_policy`` function and ``fpga_policy`` object. It may help to achieve better performance on FPGA hardware.
- Added support for <cmath> on Windows platform.
- Added vectorized search algorithms ``binary_search``, ``lower_bound`` and ``upper_bound``.

Changes to Existing Features
-----------------------------
- Host side (synchronous) exceptions are no more handled, and instead pass through algorithms to the calling function.
- For better performance sorting algorithms are specialized to use Radix sort with arithmetic data types and ``std::less``, ``std::greater`` comparators.
- Improved performance of algorithms when used together with Intel(R) DPC++ Compatibility Tool iterator and pointer types.
- Improved performance of the ``merge`` algorithm with a DPC++ ``device_policy``.

Fixed Issues
-------------
- Fixed errors with usage of ``std::tuple`` in user-provided functors when ``dpstd::zip_iterator`` is passed to Parallel STL algorithms. 

Known Issues and Limitations
----------------------------
- ``sort``, ``stable_sort``, ``partial_sort``, ``partial_sort_copy`` algorithms may work incorrectly in debug mode.
- Using DPC++ policy some algorithms might fail on CPU.
- ``std::tuple`` cannot be used with SYCL buffers to transfer data between host and device.
- When used within DPC++ kernels or transferred to/from a device, ``std::tuple, std::pair`` and ``std::array`` can only hold objects whose type meets DPC++ requirements for use in kernels and for data transfer, respectively.
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception; use ``std::array::operator[]`` instead.
- ``std::array`` member function swap cannot be used in DPC++ kernels on Windows platform.
- ``std::swap`` for ``std::array`` cannot work in DPC++ kernels on Windows platform.
- Not all functions in <cmath> are supported currently, please refer to `DPC++ library guide <https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-library-guide/top.html>`_ for detail list.
- ``std::complex`` division may fail in kernel code on some CPU platform.

New in 2021.1-beta05
====================

Changes to Existing Features
-----------------------------
- Improved USM pointers support.

Note: Non-USM pointers are not supported by the DPC++ execution policies anymore.
- A performance optimization for partial_sort, partial_sort_copy algorithms using standard C++ policies.

Fixed Issues
-------------
- Fix for non-trivial user’s type using the ``remove_if``, ``unique``, ``rotate``, ``partial_sort_copy``, ``set_intersetion``, ``set_union``, ``set_difference``, ``set_symmetric_difference`` algorithms with standard C++ policies.

Known Issues and Limitations
----------------------------
- Some algorithms might fail on CPU when using DPC++ policy.
- ``std::tuple`` cannot be used with SYCL buffers to transfer data between host and device.
- When used within DPC++ kernels or transferred to/from a device, ``std::tuple, std::pair`` and ``std::array`` can only hold objects whose type meets DPC++ requirements for use in kernels and for data transfer, respectively.
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception; use ``std::array::operator[]`` instead.
- ``std::array`` member function swap cannot be used in DPC++ kernels on Windows platform.
- ``std::swap`` for ``std::array`` cannot work in DPC++ kernels on Windows platform.
- Not all functions in <cmath> are supported currently, please refer to `DPC++ library guide <https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-library-guide/top.html>`_ for detail list.
- ``std::complex`` division may fail in kernel code on some CPU platform.

New in 2021.1-beta04
====================

New Features
-------------
- Added 64-bit atomics support.
- Added the following to Tested standard C++ APIs:

  - ``<complex>`` and most functions in ``<cmath>`` (GNU* libstdc++);
  - ``<ratio>`` (GNU libstdc++, LLVM* libc++, MSVC*);
  - ``std::numeric_limits`` (GNU libstdc++, MSVC).


Changes to Existing Features
-----------------------------
- The following DPC++ execution policies were renamed:

  - From ``dpstd::execution::sycl_policy`` to ``dpstd::execution::device_policy``.
  - From ``dpstd::execution::make_sycl_policy`` to ``dpstd::execution::make_device_policy``.
  - From ``dpstd::execution::sycl`` object to ``dpstd::execution::default_policy``.

``dpstd::execution::sycl_policy, dpstd::execution::make_sycl_policy, dpstd::execution::sycl`` were deprecated.

- The following algorithms in Extension API were renamed:

  - From ``reduce_by_key`` to ``reduce_by_segment``.
  - From ``inclusive_scan_by_key`` to ``inclusive_scan_by_segment``.
  - From ``exclusive_scan_by_key`` to ``exclusive_scan_by_segment``.


Known Issues and Limitations
----------------------------
- Using DPC++ policy some algorithms might fail on CPU.
- ``std::tuple`` cannot be used with SYCL buffers to transfer data between host and device.
- When used within DPC++ kernels or transferred to/from a device, ``std::tuple, std::pair`` and ``std::array`` can only hold objects whose type meets DPC++ requirements for use in kernels and for data transfer, respectively.
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception; use ``std::array::operator[]`` instead.
- ``std::array`` member function swap cannot be used in DPC++ kernels on Windows platform.
- ``std::swap`` for ``std::array`` cannot work in DPC++ kernels on Windows platform.
- Not all functions in <cmath> are supported currently, please refer to `DPC++ library guide <https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-library-guide/top.html>`_ for detail list.
- ``std::complex`` division may fail in kernel code on some CPU platform.

New in 2021.1-beta03
====================

New Features
-------------
- Added support for Data Parallel C++ (DPC++) to Parallel STL algorithms. For a complete list of Parallel STL algorithms see the ISO/IEC 14882:2017 standard (C++17).
- Added ``dpstd::begin``, ``dpstd::end`` helper functions to pass the ``cl::sycl::buffer`` into Parallel STL algorithms.
- Added initial support for Unified Shared Memory in Parallel STL algorithms.
- More than 80 C++ standard APIs from ``<algorithm>``, ``<array>``, ``<tuple>``, ``<utility>``, ``<functional>``, ``<type_traits>``, ``<initializer_list>`` were tested for use in DPC++ kernels. For more information, see the library guide.
- Added ``counting_iterator``, ``zip_iterator``, ``transform_iterator``, ``reduce_by_key``, ``inclusive_scan_by_key``, and ``exclusive_scan_by_key`` to the extension API.
- Added functional utility classes that include identity, minimum, maximum to the extension API.

Changes to Existing Features
----------------------------
- Construction of a DPC++ execution policy from the ``cl::sycl::ordered`` queue.

Fixed Issues
------------
- Errors no longer appear when the ``<dpstd/execution>`` header is included after other the oneDPL headers.
- Algorithms now work with zip iterators if standard C++ execution policies are used.

Known Issues and Limitations
----------------------------
- Algorithms ``adjacent_find``, ``find``, ``find_end``, ``find_first_of``, ``find_if``, ``find_if_not``, ``is_sorted``, ``is_sorted_until``, ``mismatch``, ``search``, and ``search_n`` do not use iterators with the size of difference_type more than 32 bits.
- ``std::tuple`` cannot be used with SYCL* buffers to transfer data between the host and device.
- When used within DPC++ kernels or transferred to or from a device, ``std::tuple``, ``std::pair``, and ``std::array`` can only hold objects whose type meets DPC++ requirements for use in kernels and for data transfer, respectively.
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception; use ``std::array::operator[]`` instead.
- A ``std::array`` member function swap and ``std::swap`` for ``std::array`` cannot be used in DPC++ kernels on Windows* platforms.


`*` Other names and brands may be claimed as the property of others.

.. [#fnote1] The sorting algorithms in oneDPL use Radix sort for arithmetic data types compared with
   ``std::less`` or ``std::greater``, otherwise Merge sort.
.. _`the oneDPL Specification`: https://spec.oneapi.com/versions/latest/elements/oneDPL/source/index.html
.. _`Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`: https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-threading-building-blocks-release-notes.html
