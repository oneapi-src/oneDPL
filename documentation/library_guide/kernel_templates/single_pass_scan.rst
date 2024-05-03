Single-Pass Scan
##########

-----------------------------------------------------------
``inclusive_scan`` Function Template
-----------------------------------------------------------

The ``inclusive_scan`` function computes the inclusive prefix sum using a given binary operation.
The function implements a single-pass algorithm, where each input element is read exactly once from
global memory and each output element is written to exactly once in global memory.

A synopsis of the ``inclusive_scan`` function is provided below:

.. code:: cpp

   // defined in <oneapi/dpl/experimental/kernel_templates>

   namespace oneapi::dpl::experimental::kt::gpu {

   template <typename InIterator, typename OutIterator, typename BinaryOp>
   sycl::event
   inclusive_scan (sycl::queue q, InIterator in_begin, InIterator in_end, OutIterator out_begin,
                   BinaryOp binary_op, KernelParam param); // (1)

   template <typename InRng, typename OutRng, typename BinaryOp>
   sycl::event
   inclusive_scan (sycl::queue q, InRng in_rng, OutRng out_rng, BinaryOp binary_op,
                   KernelParam param) // (2)

   }


Parameters
----------

+-----------------------------------------------+---------------------------------------------------------------------+
| Name                                          | Description                                                         |
+===============================================+=====================================================================+
|  ``q``                                        | SYCL* queue to submit the kernels to.                               |
+-----------------------------------------------+---------------------------------------------------------------------+
|                                               | The sequences of elements to apply the algorithm to.                |
|  - ``in_begin``, ``in_end``, ``out_begin`` (1)| Supported sequence types:                                           |
|  - ``in_rng``, ``out_rng`` (2)                |                                                                     |
|                                               | - ``sycl::buffer`` (2),                                             |
|                                               | - :ref:`oneapi::dpl::experimental::ranges::views::all               |
|                                               |   <viewable-ranges>` (2),                                           |
|                                               | - :ref:`oneapi::dpl::experimental::ranges::views::subrange          |
|                                               |   <viewable-ranges>` (2),                                           |
|                                               | - :ref:`USM pointers <use-usm>` (1),                                |
|                                               | - :ref:`oneapi::dpl::begin and oneapi::dpl::end                     |
|                                               |   <use-buffer-wrappers>` (1).                                       |
|                                               |                                                                     |
+-----------------------------------------------+---------------------------------------------------------------------+
|  ``param``                                    | A :doc:`kernel_param <../kernel_configuration>` object.             |
|                                               |                                                                     |
+-----------------------------------------------+---------------------------------------------------------------------+


**Type Requirements**:

- The element type of sequence(s) to scan must be a C++ integral or floating-point type
  other than ``bool`` with a width of up to 64 bits.

.. note::

   Current limitations:

   - The function will internally block until the issued kernels have completed execution
   - The SYCL device associated with the provided queue must support 64-bit atomic operations

Return Value
------------

A ``sycl::event`` object representing the status of the algorithm execution.

--------------
Usage Examples
--------------


``inclusive_scan`` Example
----------------------

.. code:: cpp

   // possible build and run commands:
   //    icpx -fsycl inclusive_scan.cpp -o inclusive_scan -I /path/to/oneDPL/include && ./inclusive_scan

   #include <cstdint>
   #include <iostream>
   #include <sycl/sycl.hpp>

   #include <oneapi/dpl/experimental/kernel_templates>

   namespace kt = oneapi::dpl::experimental::kt;

   int main()
   {
      std::size_t n = 6;
      sycl::queue q{sycl::gpu_selector_v};
      std::uint32_t* arr = sycl::malloc_shared<std::uint32_t>(n, q);
      std::uint32_t* out = sycl::malloc_shared<std::uint32_t>(n, q);

      // initialize
      arr[0] = 1, arr[1] = 2, arr[2] = 1, arr[3] = 3, arr[4] = 1, arr[5] = 2;

      // scan
      auto e = kt::gpu::inclusive_scan(q, arr, arr + n, out, std::plus<std::uint32_t>{}, kt::kernel_param<256, 8>{});
      e.wait();

      // print
      for(std::size_t i = 0; i < n; ++i)
         std::cout << out[i] << ' ';
      std::cout << '\n';

      sycl::free(arr, q);
      sycl::free(out, q);
      return 0;
   }

**Output:**

.. code:: none

   1 3 4 7 8 10

.. _memory-requirements:

-------------------
Memory Requirements
-------------------

The algorithms use global and local device memory (see `SYCL 2020 Specification
<https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_sycl_device_memory_model>`_)
for intermediate data storage. For the algorithms to operate correctly, there must be enough memory
on the device; otherwise, the behavior is undefined. The amount of memory that is required
depends on input data and configuration parameters, as described below.

Global Memory Requirements
--------------------------

Global memory is used for copying the input sequence and storing internal data such as status flags.
The used amount depends on many parameters; below is an upper bound approximation:

2 * V * N \ :sub:`flags` + F * N \ :sub:`flags`

where V is the number of bytes needed to store the input value type and F is the number of bytes needed to store the flags.
Currently, F is hard-coded to 4 as the flag type is 32-bits.

The value of N\ :sub:`flags` represents the number of work-groups and depends on ``param.data_per_workitem`` and ``param.workgroup_size``.
It can be approxmiated by dividing the number of input elements N by the product of ``param.data_per_workitem`` and ``param.workgroup_size``
and adding 33 for padding.

.. note::

   If the number of input elements can be efficiently processed by a single work-group,
   the kernel template is executed by a single work-group and does not use any global memory.


.. _local-memory:

Local Memory Requirements
-------------------------

Local memory is used for storing elements of the input that are to be scanned by a single work group.
The used amount is denoted as N\ :sub:`elems_per_workgroup`, which equals to ``sizeof(key_type) * param.data_per_workitem * param.workgroup_size``.

Some amount of local memory is also used by the calls to SYCL's group reduction and group scan. The amount of memory used particularly
for these calls is implementation dependant.

-----------------------------------------
Recommended Settings for Best Performance
-----------------------------------------

The general advice is to choose kernel parameters based on performance measurements and profiling information.
The initial configuration may be selected according to these high-level guidelines:


- When the number of elements to scan (N) is small (~8K or less),
  generally it is more efficient to process the algorithm by a single work-group.
  In this case, the choice of kernel parameters is irrelevant, as we internally dispatch to a single
  work-group implementation.

- Generally, utilizing all available
  compute cores is key for better performance. Allow creating enough work chunks to feed all
  X\ :sup:`e`-cores [#fnote1]_ on a GPU: ``param.data_per_workitem * param.workgroup_size ≈ N / xe_core_count``.


.. warning::

   Avoid setting too large ``param.data_per_workitem`` and ``param.workgroup_size`` values.
   Make sure that :ref:`Memory requirements <memory-requirements>` are satisfied.

.. [#fnote1] The X\ :sup:`e`-core term is described in the `oneAPI GPU Optimization Guide
   <https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-0/intel-xe-gpu-architecture.html#XE-CORE>`_.
   Check the number of cores in the device specification, such as `Intel® Data Center GPU Max specification
   <https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series/products.html>`_.
