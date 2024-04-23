Intel® oneAPI DPC++ Library (oneDPL) Release Notes
###################################################

Overview
=========

The Intel® oneAPI DPC++ Library (oneDPL) accompanies the Intel® oneAPI DPC++/C++ Compiler
and provides high-productivity APIs aimed to minimize programming efforts of C++ developers
creating efficient heterogeneous applications.

New in 2022.5.0
===============

New Features
------------
- Added new ``histogram`` algorithms for generating a histogram from an input sequence into
  an output sequence representing either equally spaced or user-defined bins.
  These algorithms are currently only available for device execution policies.
- Supported zip_iterator for ``transform`` algorithm.

Fixed Issues
------------
- Fixed handling of ``permutation_iterator`` as input to oneDPL algorithms for a variety of
  source iterator and permutation types which caused issues.
- Fixed ``zip_iterator`` to be `sycl device copyable`_ for trivially copyable source iterator types.
- Added a workaround for reduction algorithm failures with 64-bit data types. Define
  the ``ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION`` macro to ``1`` before including oneDPL header files.

Known Issues and Limitations
----------------------------
New in This Release
^^^^^^^^^^^^^^^^^^^
- Crashes or incorrect results may occur when using ``oneapi::dpl::reverse_iterator`` or
  ``std::reverse_iterator`` as input to oneDPL algorithms with device execution policies.

Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- When compiled with ``-fsycl-pstl-offload`` option of Intel oneAPI DPC++/C++ compiler and with
  ``libstdc++`` version 8 or ``libc++``, ``oneapi::dpl::execution::par_unseq`` offloads
  standard parallel algorithms to the SYCL device similarly to ``std::execution::par_unseq``
  in accordance with the ``-fsycl-pstl-offload`` option value.
- When using the dpl modulefile to initialize the user's environment and compiling with ``-fsycl-pstl-offload``
  option of Intel® oneAPI DPC++/C++ compiler, a linking issue or program crash may be encountered due to the directory
  containing libpstloffload.so not being included in the search path. Use the env/vars.sh to configure the working
  environment to avoid the issue.
- Compilation issues may be encountered when passing zip iterators to ``exclusive_scan_by_segment`` on Windows.
- Incorrect results may be produced by ``set_intersection`` with a DPC++ execution policy,
  where elements are copied from the second input range rather than the first input range. 
- For ``transform_exclusive_scan`` and ``exclusive_scan`` to run in-place (that is, with the same data
  used for both input and destination) and with an execution policy of ``unseq`` or ``par_unseq``, 
  it is required that the provided input and destination iterators are equality comparable.
  Furthermore, the equality comparison of the input and destination iterator must evaluate to true.
  If these conditions are not met, the result of these algorithm calls is undefined.
- ``sort``, ``stable_sort``, ``sort_by_key``, ``partial_sort_copy`` algorithms may work incorrectly or cause
  a segmentation fault when used a DPC++ execution policy for CPU device, and built
  on Linux with Intel® oneAPI DPC++/C++ Compiler and -O0 -g compiler options.
  To avoid the issue, pass ``-fsycl-device-code-split=per_kernel`` option to the compiler.
- Incorrect results may be produced by ``exclusive_scan``, ``inclusive_scan``, ``transform_exclusive_scan``,
  ``transform_inclusive_scan``, ``exclusive_scan_by_segment``, ``inclusive_scan_by_segment``, ``reduce_by_segment``
  with ``unseq`` or ``par_unseq`` policy when compiled by Intel® oneAPI DPC++/C++ Compiler
  with ``-fiopenmp``, ``-fiopenmp-simd``, ``-qopenmp``, ``-qopenmp-simd`` options on Linux.
  To avoid the issue, pass ``-fopenmp`` or ``-fopenmp-simd`` option instead.
- Incorrect results may be produced by ``reduce``, ``reduce_by_segment``, and ``transform_reduce``
  with 64-bit data types when compiled by Intel® oneAPI DPC++/C++ Compiler versions 2021.3 and newer
  and executed on GPU devices.
  For a workaround, define the ``ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION`` macro to ``1`` before
  including oneDPL header files.
- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.
- STL algorithm functions (such as ``std::for_each``) used in DPC++ kernels do not compile with the debug version of
  the Microsoft* Visual C++ standard library.

New in 2022.4.0
===============

New Features
------------
- Added experimental ``radix_sort`` and ``radix_sort_by_key`` algorithms residing in
  the ``oneapi::dpl::experimental::kt::esimd`` namespace. These algorithms are first
  in the family of _kernel templates_ that allow configuring a variety of parameters
  including the number of elements to process by a work item, and the size of a workgroup.
  The algorithms only work with Intel® Data Center GPU Max Series.
- Added new ``transform_if`` algorithm for applying a transform function conditionally
  based on a predicate, with overloads provided for one and two input sequences
  that use correspondingly unary and binary operations and predicates.
- Optimizations used with Intel® oneAPI DPC++/C++ Compiler are expanded to the open source oneAPI DPC++ compiler.

Known Issues and Limitations
----------------------------
New in This Release
^^^^^^^^^^^^^^^^^^^
- ``esimd::radix_sort`` and ``esimd::radix_sort_by_key`` kernel templates fail to compile when a program
  is built with -g, -O0, -O1 compiler options.
- ``esimd::radix_sort_by_key`` kernel template produces wrong results with the following combinations
  of ``kernel_param`` and types of keys and values:
    - ``sizeof(key_type) + sizeof(val_type) == 12``, ``kernel_param::workgroup_size == 64``, and ``kernel_param::data_per_workitem == 96``
    - ``sizeof(key_type) + sizeof(val_type) == 16``, ``kernel_param::workgroup_size == 64``, and ``kernel_param::data_per_workitem == 64``

New in 2022.3.0
===============

New Features
------------
- Added an experimental feature to dynamically select an execution context, e.g., a SYCL queue.
  The feature provides selection functions such as ``select``, ``submit`` and ``submit_and_wait``,
  and several selection policies: ``fixed_resource_policy``, ``round_robin_policy``,
  ``dynamic_load_policy``, and ``auto_tune_policy``.
- ``unseq`` and ``par_unseq`` policies now enable vectorization also for Intel oneAPI DPC++/C++ Compiler.
- Added support for passing zip iterators as segment value data in ``reduce_by_segment``, ``exclusive_scan_by_segment``,
  and ``inclusive_scan_by_segment``.
- Improved performance of the ``merge``, ``sort``, ``stable_sort``, ``sort_by_key``,
  ``reduce``, ``min_element``, ``max_element``, ``minmax_element``, ``is_partitioned``, and
  ``lexicographical_compare`` algorithms with DPC++ execution policies.

Fixed Issues
------------
- Fixed the ``reduce_async`` function to not ignore the provided binary operation.

Known Issues and Limitations
----------------------------
New in This Release
^^^^^^^^^^^^^^^^^^^
- When compiled with ``-fsycl-pstl-offload`` option of Intel oneAPI DPC++/C++ compiler and with
  ``libstdc++`` version 8 or ``libc++``, ``oneapi::dpl::execution::par_unseq`` offloads
  standard parallel algorithms to the SYCL device similarly to ``std::execution::par_unseq``
  in accordance with the ``-fsycl-pstl-offload`` option value.
- When using the dpl modulefile to initialize the user's environment and compiling with ``-fsycl-pstl-offload``
  option of Intel® oneAPI DPC++/C++ compiler, a linking issue or program crash may be encountered due to the directory
  containing libpstloffload.so not being included in the search path. Use the env/vars.sh to configure the working
  environment to avoid the issue.
- Compilation issues may be encountered when passing zip iterators to ``exclusive_scan_by_segment`` on Windows.
- Incorrect results may be produced by ``set_intersection`` with a DPC++ execution policy,
  where elements are copied from the second input range rather than the first input range. 
- For ``transform_exclusive_scan`` and ``exclusive_scan`` to run in-place (that is, with the same data
  used for both input and destination) and with an execution policy of ``unseq`` or ``par_unseq``, 
  it is required that the provided input and destination iterators are equality comparable.
  Furthermore, the equality comparison of the input and destination iterator must evaluate to true.
  If these conditions are not met, the result of these algorithm calls is undefined.
- ``sort``, ``stable_sort``, ``sort_by_key``, ``partial_sort_copy`` algorithms may work incorrectly or cause
  a segmentation fault when used a DPC++ execution policy for CPU device, and built
  on Linux with Intel® oneAPI DPC++/C++ Compiler and -O0 -g compiler options.
  To avoid the issue, pass ``-fsycl-device-code-split=per_kernel`` option to the compiler.
- Incorrect results may be produced by ``exclusive_scan``, ``inclusive_scan``, ``transform_exclusive_scan``,
  ``transform_inclusive_scan``, ``exclusive_scan_by_segment``, ``inclusive_scan_by_segment``, ``reduce_by_segment``
  with ``unseq`` or ``par_unseq`` policy when compiled by Intel® oneAPI DPC++/C++ Compiler
  with ``-fiopenmp``, ``-fiopenmp-simd``, ``-qopenmp``, ``-qopenmp-simd`` options on Linux.
  To avoid the issue, pass ``-fopenmp`` or ``-fopenmp-simd`` option instead.
- Incorrect results may be produced by ``reduce``, ``reduce_by_segment``, and ``transform_reduce``
  with 64-bit data types when compiled by Intel® oneAPI DPC++/C++ Compiler versions 2021.3 and newer
  and executed on GPU devices.

Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.
- STL algorithm functions (such as ``std::for_each``) used in DPC++ kernels do not compile with the debug version of
  the Microsoft* Visual C++ standard library.

New in 2022.2.0
===============

New Features
------------
- Added ``sort_by_key`` algorithm for key-value sorting.
- Improved performance of the ``reduce``, ``min_element``, ``max_element``, ``minmax_element``,
  ``is_partitioned``, and ``lexicographical_compare`` algorithms with DPC++ execution policies.
- Improved performance of the ``reduce_by_segment``, ``inclusive_scan_by_segment``, and
  ``exclusive_scan_by_segment`` algorithms for binary operators with known identities
  when using DPC++ execution policies.
- Added ``value_type`` to all views in ``oneapi::dpl::experimental::ranges``. 
- Extended ``oneapi::dpl::experimental::ranges::sort`` to support projections applied to the range elements prior to comparison.

Fixed Issues
------------
- The minimally required CMake version is raised to 3.11 on Linux and 3.20 on Windows.
- Added new CMake package ``oneDPLIntelLLVMConfig.cmake`` to resolve issues using CMake 3.20+ on Windows for icx and icx-cl.
- Fixed an error in the ``sort`` and ``stable_sort`` algorithms when performing a descending sort
  on signed numeric types with negative values.
- Fixed an error in ``reduce_by_segment`` algorithm when a non-commutative predicate is used.
- Fixed an error in ``sort`` and ``stable_sort`` algorithms for integral types wider than 4 bytes.
- Fixed an error for some compilers where OpenMP or SYCL backend was selected by CMake scripts without full compiler support.

Known Issues and Limitations
----------------------------
New in This Release
^^^^^^^^^^^^^^^^^^^
- Incorrect results may be produced with in-place scans using ``unseq`` and ``par_unseq`` policies on
  CPUs with the Intel® C++ Compiler 2021.8.

Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.
- STL algorithm functions (such as ``std::for_each``) used in DPC++ kernels do not compile with the debug version of
  the Microsoft* Visual C++ standard library.

New in 2022.1.1
===============

New Features
------------
- Improved ``sort`` algorithm performance for the arithmetic data types with ``std::less`` or ``std::greater`` comparison operator and DPC++ policy.

Fixes Issues
------------
- Fixed an error that caused segmentation faults in ``transform_reduce``, ``minmax_element``, and related algorithms when ran on CPU devices. 
- Fixed a compilation error in ``transform_reduce``, ``minmax_element``, and related algorithms on FPGAs.
- Fixed ``permutation_iterator`` to support C-style array as a permutation map.
- Fixed a radix-sort issue with 64-bit signed integer types.

New in 2022.1.0
===============

New Features
------------
- Added ``generate``, ``generate_n``, ``transform`` algorithms to `Tested Standard C++ API`_.
- Improved performance of the ``inclusive_scan``, ``exclusive_scan``, ``reduce`` and
  ``max_element`` algorithms with DPC++ execution policies.

Fixed Issues
------------
- Added a workaround for the ``TBB headers not found`` issue occurring with libstdc++ version 9 when
  oneTBB headers are not present in the environment. The workaround requires inclusion of
  the oneDPL headers before the libstdc++ headers.
- When possible, oneDPL CMake scripts now enforce C++17 as the minimally required language version.
- Fixed an error in the ``exclusive_scan`` algorithm when the output iterator is equal to the
  input iterator (in-place scan).

Known Issues and Limitations
----------------------------
Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.
- STL algorithm functions (such as ``std::for_each``) used in DPC++ kernels do not compile with the debug version of
  the Microsoft* Visual C++ standard library.


New in 2022.0.0
===============

New Features
------------
- Added the functionality from ``<complex>`` and more APIs from ``<cmath>`` and ``<limits>``
  standard headers to `Tested Standard C++ API`_.
- Improved performance of ``sort`` and ``stable_sort``  algorithms on GPU devices when using Radix sort [#fnote1]_.

Fixed Issues
------------
- Fixed permutation_iterator to work with C++ lambda functions for index permutation.
- Fixed an error in ``oneapi::dpl::experimental::ranges::guard_view`` and ``oneapi::dpl::experimental::ranges::zip_view``
  when using ``operator[]`` with an index exceeding the limits of a 32 bit integer type.
- Fixed errors when data size is 0 in ``upper_bound``, ``lower_bound`` and ``binary_search`` algorithms.

Changes affecting backward compatibility
----------------------------------------
- Removed support of C++11 and C++14.
- Changed the size and the layout of the ``discard_block_engine`` class template.
  
  For further details, please refer to `2022.0 Changes`_.

Known Issues and Limitations
----------------------------
Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.
- STL algorithm functions (such as ``std::for_each``) used in DPC++ kernels do not compile with the debug version of
  the Microsoft* Visual C++ standard library.

New in 2021.7.1
===============

New Features
------------
- Added possibility to construct a zip_iterator out of a std::tuple of iterators.
- Added 9 more serial-based versions of algorithms: ``is_heap``, ``is_heap_until``, ``make_heap``, ``push_heap``,
  ``pop_heap``, ``is_sorted``, ``is_sorted_until``, ``partial_sort``, ``partial_sort_copy``.
  Please refer to `Tested Standard C++ API`_.
  
Fixed Issues
------------
- Added namespace alias ``dpl = oneapi::dpl`` into all public headers.
- Fixed error in ``reduce_by_segment`` algorithm.
- Fixed wrong results error in algorithms call with permutation iterator.
  
Known Issues and Limitations
----------------------------
Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.
- STL algorithm functions (such as ``std::for_each``) used in DPC++ kernels do not compile with the debug version of
  the Microsoft* Visual C++ standard library.
  
New in 2021.7.0
===============

Deprecation Notice
------------------
- Deprecated support of C++11 for Parallel API with host execution policies (``seq``, ``unseq``, ``par``, ``par_unseq``).
  C++17 is the minimal required version going forward.

Fixed Issues
------------
- Fixed a kernel name definition error in range-based algorithms and ``reduce_by_segment`` used with
  a device_policy object that has no explicit kernel name.

Known Issues and Limitations
----------------------------
New in This Release
^^^^^^^^^^^^^^^^^^^
- STL algorithm functions (such as ``std::for_each``) used in DPC++ kernels do not compile with the debug version of
  the Microsoft* Visual C++ standard library.

New in 2021.6.1
===============

Fixed Issues
------------
- Fixed compilation errors with C++20.
- Fixed ``CL_OUT_OF_RESOURCES`` issue for Radix sort algorithm executed on CPU devices.
- Fixed crashes in ``exclusive_scan_by_segment``, ``inclusive_scan_by_segment``, ``reduce_by_segment`` algorithms applied to
  device-allocated USM.

Known Issues and Limitations
----------------------------
- No new issues in this release. 

Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.

New in 2021.6
=============

New Features
------------
- Added a new implementation for ``par`` and ``par_unseq`` execution policies based on OpenMP* 4.5 pragmas.
  It can be enabled with the ``ONEDPL_USE_OPENMP_BACKEND`` macro.
  For more details, see `Macros`_ page in oneDPL Guide.
- Added the range-based version of the ``reduce_by_segment`` algorithm and improved performance of
  the iterator-based ``reduce_by_segment`` APIs. 
  Please note that the use of the ``reduce_by_segment`` algorithm requires C++17.
- Added the following algorithms (serial versions) to `Tested Standard C++ API`_: ``for_each_n``, ``copy``,
  ``copy_backward``, ``copy_if``, ``copy_n``, ``is_permutation``, ``fill``, ``fill_n``, ``move``, ``move_backward``.

Changes affecting backward compatibility
----------------------------------------
- Fixed ``param_type`` API of random number distributions to satisfy C++ standard requirements.
  The new definitions of ``param_type`` are not compatible with incorrect definitions in previous library versions.
  Recompilation is recommended for all codes that might use ``param_type``.

Fixed Issues
------------
- Fixed hangs and errors when oneDPL is used together with oneAPI Math Kernel Library (oneMKL) in
  Data Parallel C++ (DPC++) programs.
- Fixed possible data races in the following algorithms used with DPC++ execution
  policies: ``sort``, ``stable_sort``, ``partial_sort``, ``nth_element``.

Known Issues and Limitations
----------------------------
- No new issues in this release.

Existing Issues
^^^^^^^^^^^^^^^
See oneDPL Guide for other `restrictions and known limitations`_.

- ``std::tuple``, ``std::pair`` cannot be used with SYCL buffers to transfer data between host and device.
- ``std::array`` cannot be swapped in DPC++ kernels with ``std::swap`` function or ``swap`` member function
  in the Microsoft* Visual C++ standard library.
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.

New in 2021.5
=============

New Features
------------
- Added new random number distributions: ``exponential_distribution``, ``bernoulli_distribution``,
  ``geometric_distribution``, ``lognormal_distribution``, ``weibull_distribution``, ``cachy_distribution``,
  ``extreme_value_distribution``.
- Added the following algorithms (serial versions) to `Tested Standard C++ API`_: ``all_of``, ``any_of``, 
  ``none_of``, ``count``, ``count_if``, ``for_each``, ``find``, ``find_if``, ``find_if_not``.
- Improved performance of ``search`` and ``find_end`` algorithms on GPU devices.

Fixed Issues
------------
- Fixed SYCL* 2020 features deprecation warnings.
- Fixed some corner cases of ``normal_distribution`` functionality.
- Fixed a floating point exception occurring on CPU devices when a program uses a lot of oneDPL algorithms and DPC++ kernels.
- Fixed possible hanging and data races of the following algorithms used with DPC++ execution policies: ``count``, ``count_if``, ``is_partitioned``, ``lexicographical_compare``, ``max_element``, ``min_element``, ``minmax_element``,    ``reduce``, ``transform_reduce``.

Known Issues and Limitations
----------------------------

New in This Release
^^^^^^^^^^^^^^^^^^^
- The definition of lambda functions used with parallel algorithms should not depend on preprocessor macros
  that makes it different for the host and the device. Otherwise, the behavior is undefined.

Existing Issues
^^^^^^^^^^^^^^^
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
- The ``oneapi::dpl::experimental::ranges::reverse`` algorithm is not available with ``-fno-sycl-unnamed-lambda`` option.

New in 2021.4
=============

New Features
------------
-  Added the range-based versions of the following algorithms: ``any_of``, ``adjacent_find``,
   ``copy_if``, ``none_of``, ``remove_copy_if``, ``remove_copy``, ``replace_copy``, 
   ``replace_copy_if``, ``reverse``, ``reverse_copy``, ``rotate_copy``, ``swap_ranges``,
   ``unique``, ``unique_copy``.
-  Added new asynchronous algorithms: ``inclusive_scan_async``, ``exclusive_scan_async``,
   ``transform_inclusive_scan_async``, ``transform_exclusive_scan_async``.
-  Added structured binding support for ``zip_iterator::value_type``.

Fixed Issues
------------
-  Fixed an issue with asynchronous algorithms returning ``future<ptr>`` with unified shared memory (USM).

Known Issues and Limitations
----------------------------

New in This Release
^^^^^^^^^^^^^^^^^^^
-  With Intel® oneAPI DPC++/C++ Compiler, ``unseq`` and ``par_unseq`` execution policies do not use OpenMP SIMD pragmas
   due to compilation issues with the ``-fopenm-simd`` option, possibly resulting in suboptimal performance.
-  The ``oneapi::dpl::experimental::ranges::reverse`` algorithm does not compile with ``-fno-sycl-unnamed-lambda`` option.

Existing Issues
^^^^^^^^^^^^^^^
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
-  Added experimental asynchronous algorithms: ``copy_async``, ``fill_async``, ``for_each_async``, ``reduce_async``, ``sort_async``, ``transform_async``, ``transform_reduce_async``.
   These algorithms are declared in ``oneapi::dpl::experimental`` namespace and implemented only for DPC++ policies.
   In order to make these algorithms available the ``<oneapi/dpl/async>`` header should be included. Use of the asynchronous API requires C++11.
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
  For the detailed list, please refer to `oneDPL Guide`_.
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
.. _`oneDPL Guide`: https://oneapi-src.github.io/oneDPL/index.html
.. _`Intel® oneAPI Threading Building Blocks (oneTBB) Release Notes`: https://www.intel.com/content/www/us/en/developer/articles/release-notes/intel-oneapi-threading-building-blocks-release-notes.html
.. _`restrictions and known limitations`: https://oneapi-src.github.io/oneDPL/introduction.html#restrictions.
.. _`Tested Standard C++ API`: https://oneapi-src.github.io/oneDPL/api_for_sycl_kernels/tested_standard_cpp_api.html#tested-standard-c-api-reference
.. _`Macros`: https://oneapi-src.github.io/oneDPL/macros.html
.. _`2022.0 Changes`: https://oneapi-src.github.io/oneDPL/oneDPL_2022.0_changes.html
.. _`sycl device copyable`: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec::device.copyable
