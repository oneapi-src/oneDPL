Intel® oneAPI DPC++ Library (oneDPL) Release Notes (Beta)
##########################################################

Overview
========

Learn more about the new features and known issues for this library.

New in 2021.1-beta10
====================

New Features
-----------------------------
- All oneDPL functionality, including the parallel algorithm functions, is accessible via the ``oneapi::dpl`` namespace.

Changes to Existing Features
-----------------------------
- The following methods of the permutation_iterator have been renamed: ``get_source_iterator()`` is renamed to ``base()``, ``get_map_iterator()`` is renamed to ``map()``.
- Improved performance of the following algorithms: ``copy_if``, ``count``, ``count_if``, ``exclusive_scan``, ``inclusive_scan``, ``is_partitioned``, ``lexicographical_compare``, ``max_element``, ``min_element``, ``minmax_element``, ``partition``, ``partition_copy``, ``reduce``, ``remove``, ``remove_copy``, ``remove_copy_if``, ``remove_if``, ``set_difference``, ``set_intersection``, ``set_symmetric_difference``, ``set_union``, ``stable_partition``, ``transform_exclusive_scan``, ``transform_inclusive_scan``, ``transform_reduce``, ``unique``, ``unique_copy``.
- Improved performance of the ``nth_element`` algorithm when input contains large number of duplicates.

Fixed Issues
-------------
- Fixed the failures of the ``sort``, ``stable_sort`` algorithms when using Radix sort [#fnote1]_ on oneAPI
  CPU devices.

Known Issues
-------------
- The use of oneDPL together with the GNU C++ standard library (libstdc++) version 9 or 10 may lead to compilation errors (caused by oneTBB API changes).
  To overcome these, switch off the use of TBB for parallel execution policies in the standard library.
- The use of the -sycl-std=2020 option may lead to compilation errors for oneDPL parallel algorithms.
- The ``using namespace oneapi;`` directive in a oneDPL program code may result in compilation errors
  with some compilers including GCC 7 and earlier. Instead of this directive, use fully qualified
  names or namespace aliases.
- The ``partial_sort_copy``, ``sort`` and ``stable_sort`` algorithms are prone to ``CL_BUILD_PROGRAM_FAILURE``
  when using Radix sort in debug mode on oneAPI CPU devices .
- The ``partial_sort_copy``, ``sort`` and ``stable_sort`` algorithms may produce incorrect result
  when using Radix sort with 32-bit ``float`` data type.
- Some algorithms with a DPC++ policy may fail on CPU or on FPGA emulator.
- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- When used within DPC++ kernels or transferred to/from a device, ``std::array`` can only hold objects whose type meets DPC++ requirements for use in kernels
  and for data transfer, respectively.
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception;
  use ``std::array::operator[]`` instead.
- ``std::array`` member function swap cannot be used in DPC++ kernels on Windows* platform.
- ``std::swap`` for ``std::array`` cannot work in DPC++ kernels on Windows* platform.
- Not all functions in <cmath> are supported currently, please refer to DPC++ library guide for detail list.
- Due to specifics of Microsoft* Visual C++ implementation, some standard math functions for float
  (including: ``std::ldexp``, ``std::frexp``, ``std::sqrt(std::complex<float>)``) require device support
  for double precision.

.. [#fnote1] The sorting algorithms in oneDPL use Radix sort for arithmetic data types compared with ``std::less`` or ``std::greater``, otherwise Merge sort.
