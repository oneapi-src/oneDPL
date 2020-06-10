Intel® oneAPI DPC++ Library (oneDPL) Release Notes (Beta)
##########################################################

Overview
========

Learn more about the new features and known issues for this library.

New in 2021.1-beta07
====================

New Features
-----------------------------
- The Microsoft* Visual C++ implementation of ``std::complex`` is supported in device code.

Changes to Existing Features
-----------------------------
- ``dpstd/iterators.h`` is deprecated. Use ``dpstd/iterator`` instead.
- Improved performance of the ``any_of``, ``adjacent_find``, ``all_of``, ``equal``, ``find``, ``find_end``, ``find_first_of``, ``find_if``, ``find_if_not``, ``includes``, ``is_heap``, ``is_heap_until``, ``is_sorted``, ``is_sorted_until``, ``mismatch``, ``none_of``, ``search``,`` search_n`` algorithms using DPC++ policies.

Fixed Issues
-------------
- Fixed error with usage of ``dpstd::zip_iterator`` on Windows.
- Fixed ``exclusive_scan`` compilation errors with GCC 9 and Clang 9 in C++17 mode.
- Eliminated warnings about deprecated sub-group interfaces.

Known Issues
-------------
- ``sort``, ``stable_sort``, ``partial_sort``, ``partial_sort_copy`` algorithms may work incorrectly in debug mode.
- Some algorithms with a DPC++ policy may fail on CPU or on FPGA emulator.
- ``std::tuple`` cannot be used with SYCL buffers to transfer data between host and device.
- When used within DPC++ kernels or transferred to/from a device, ``std::tuple, std::pair`` and ``std::array`` can only hold objects whose type meets DPC++ requirements for use in kernels and for data transfer, respectively.
- ``std::array::at`` member function cannot be used in kernels because it may throw an exception; use ``std::array::operator[]`` instead.
- ``std::array`` member function swap cannot be used in DPC++ kernels on Windows* platform.
- ``std::swap`` for ``std::array`` cannot work in DPC++ kernels on Windows platform.
- Not all functions in <cmath> are supported currently, please refer to DPC++ library guide for detail list.
- Due to specifics of Microsoft* Visual C++ implementation, some standard math functions for float (including: ``std::ldexp``, ``std::frexp``, ``std::sqrt(std::complex<float>)``) require device support for double precision.
