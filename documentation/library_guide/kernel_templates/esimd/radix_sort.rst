Radix Sort
##########

---------------------------------------------------
radix_sort and radix_sort_by_key Function Templates
---------------------------------------------------

The ``radix_sort`` and ``radix_sort_by_key`` functions sort data using the radix sort algorithm.
The sorting is stable, ensuring the preservation of the relative order of elements with equal keys.
The functions implement a Onesweep* [#fnote1]_ algorithm variant. Both in-place and out-of-place
overloads are provided. For out-of-place overloads, the input data order is preserved.

A synopsis of the ``radix_sort`` and ``radix_sort_by_key`` functions is provided below:

.. code:: cpp

   // defined in <oneapi/dpl/experimental/kernel_templates>

   namespace oneapi::dpl::experimental::kt::gpu::esimd {

   // Sort a single sequence

   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename Range>
   sycl::event
   radix_sort (sycl::queue q, Range&& r, KernelParam param); // (1)

   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename Iterator>
   sycl::event
   radix_sort (sycl::queue q, Iterator first, Iterator last,
               KernelParam param); // (2)

   // Sort a single sequence out-of-place

   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename Range1, typename Range2>
   sycl::event
   radix_sort (sycl::queue q, Range1&& r, Range2&& r_out,
               KernelParam param) // (3)

   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename Iterator1,
             typename Iterator2>
   sycl::event
   radix_sort (sycl::queue q, Iterator1 first, Iterator1 last,
               Iterator2 first_out, KernelParam param) // (4)

   // Sort a sequence of keys and apply the same order to a sequence of values

   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename KeysRng, typename ValuesRng>
   sycl::event
   radix_sort_by_key (sycl::queue q, KeysRng&& keys,
                      ValuesRng&& values, KernelParam param); // (5)

   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename Iterator1, typename Iterator2>
   sycl::event
   radix_sort_by_key (sycl::queue q, Iterator1 keys_first, Iterator1 keys_last,
                      Iterator2 values_first, KernelParam param); // (6)

   // Sort a sequence of keys and values out-of-place

   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename KeysRng1, typename ValsRng1,
             typename KeysRng2, typename ValsRng2>
   sycl::event
   radix_sort_by_key (sycl::queue q, KeysRng1&& keys, ValsRng1&& values,
                      KeysRng2&& keys_out, ValsRng2&& vals_out,
                      KernelParam param) // (7)

   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename KeysIterator1,
             typename ValsIterator1, typename KeysIterator2,
             typename ValsIterator2>
   sycl::event
   radix_sort_by_key (sycl::queue q, KeysIterator1 keys_first,
                      KeysIterator1 keys_last, ValsIterator1 vals_first,
                      KeysIterator2 keys_out_first, ValsIterator2 vals_out_first,
                      KernelParam param) // (8)

   }


Template Parameters
--------------------

+-----------------------------+---------------------------------------------------------------------------------------+
| Name                        | Description                                                                           |
+=============================+=======================================================================================+
| ``bool IsAscending``        | The sort order. Ascending: ``true``; Descending: ``false``.                           |
+-----------------------------+---------------------------------------------------------------------------------------+
| ``std::uint8_t RadixBits``  | The number of bits to sort for each radix sort algorithm pass.                        |
+-----------------------------+---------------------------------------------------------------------------------------+


Parameters
----------

+-----------------------------------------------+---------------------------------------------------------------------+
| Name                                          | Description                                                         |
+===============================================+=====================================================================+
| ``q``                                         |  The SYCL* queue where kernels are submitted.                       |
+-----------------------------------------------+---------------------------------------------------------------------+
|                                               |                                                                     |
|                                               | The sequences to apply the algorithm to.                            |
| - ``r`` (1),                                  | Supported sequence types:                                           |
| - ``first``, ``last`` (2),                    |                                                                     |
| - ``r``, ``r_out`` (3),                       | - ``sycl::buffer`` (1,3,5,7),                                       |
| - ``first``, ``last``, ``first_out`` (4),     | - :ref:`oneapi::dpl::experimental::ranges::views::all               |
| - ``keys``, ``values`` (5),                   |   <viewable-ranges>` (1,3,5,7),                                     |
| - ``keys_first``, ``keys_last``,              | - :ref:`oneapi::dpl::experimental::ranges::views::subrange          |
|   ``values_first`` (6),                       |   <viewable-ranges>` (1,3,5,7),                                     |
| - ``keys``, ``values``,                       | - :ref:`USM pointers <use-usm>` (2,4,6,8),                          |
|   ``keys_out``, ``values_out`` (7),           | - :ref:`oneapi::dpl::begin and oneapi::dpl::end                     |
| - ``keys_first``, ``keys_last``,              |   <use-buffer-wrappers>` (2,4,6,8).                                 |
|   ``vals_first``, ``keys_out_first``,         |                                                                     |
|   ``values_out_first`` (8)                    |                                                                     |
|                                               |                                                                     |
+-----------------------------------------------+---------------------------------------------------------------------+
| ``param``                                     | A :doc:`kernel_param <../kernel_configuration>` object.             |
|                                               | Its ``data_per_workitem`` must be a positive multiple of 32.        |
|                                               |                                                                     |
|                                               |                                                                     |
+-----------------------------------------------+---------------------------------------------------------------------+


**Type Requirements**:

- The element type of sequence(s) to sort must be a C++ integral or floating-point type
  other than ``bool`` with a width of up to 64 bits.

.. note::

   Current limitations:

   - Number of elements to sort must not exceed `2^30`.
   - ``RadixBits`` can only be `8`.
   - ``param.workgroup_size`` can only be `64`.

Return Value
------------

A ``sycl::event`` object representing the status of the algorithm execution.

--------------
Usage Examples
--------------


radix_sort In-Place Example
---------------------------

.. code:: cpp

   // possible build and run commands:
   //    icpx -fsycl radix_sort.cpp -o radix_sort -I /path/to/oneDPL/include && ./radix_sort

   #include <cstdint>
   #include <iostream>
   #include <sycl/sycl.hpp>

   #include <oneapi/dpl/experimental/kernel_templates>

   namespace kt = oneapi::dpl::experimental::kt;

   int main()
   {
      std::size_t n = 6;
      sycl::queue q{sycl::gpu_selector_v};
      std::uint32_t* keys = sycl::malloc_shared<std::uint32_t>(n, q);

      // initialize
      keys[0] = 3, keys[1] = 2, keys[2] = 1, keys[3] = 5, keys[4] = 3, keys[5] = 3;

      // sort
      auto e = kt::gpu::esimd::radix_sort<false, 8>(q, keys, keys + n, kt::kernel_param<416, 64>{}); // (2)
      e.wait();

      // print
      for(std::size_t i = 0; i < n; ++i)
         std::cout << keys[i] << ' ';
      std::cout << '\n';

      sycl::free(keys, q);
      return 0;
   }

**Output:**

.. code:: none

   5 3 3 3 2 1


radix_sort_by_key In-Place Example
----------------------------------

.. code:: cpp

   // possible build and run commands:
   //    icpx -fsycl radix_sort_by_key.cpp -o radix_sort_by_key -I /path/to/oneDPL/include && ./radix_sort_by_key

   #include <cstdint>
   #include <iostream>
   #include <sycl/sycl.hpp>

   #include <oneapi/dpl/experimental/kernel_templates>

   namespace kt = oneapi::dpl::experimental::kt;

   int main()
   {
      std::size_t n = 6;
      sycl::queue q{sycl::gpu_selector_v};
      sycl::buffer<std::uint32_t> keys{sycl::range<1>(n)};
      sycl::buffer<char> values{sycl::range<1>(n)};

      // initialize
      {
         sycl::host_accessor k_acc{keys, sycl::write_only};
         k_acc[0] = 3, k_acc[1] = 2, k_acc[2] = 1, k_acc[3] = 5, k_acc[4] = 3, k_acc[5] = 3;

         sycl::host_accessor v_acc{values, sycl::write_only};
         v_acc[0] = 'r', v_acc[1] = 'o', v_acc[2] = 's', v_acc[3] = 'd', v_acc[4] = 't', v_acc[5] = 'e';
      }

      // sort
      auto e = kt::gpu::esimd::radix_sort_by_key<true, 8>(q, keys, values, kt::kernel_param<96, 64>{}); // (3)
      e.wait();

      // print
      {
         sycl::host_accessor k_acc{keys, sycl::read_only};
         for(std::size_t i = 0; i < n; ++i)
               std::cout << k_acc[i] << ' ';
         std::cout << '\n';

         sycl::host_accessor v_acc{values, sycl::read_only};
         for(std::size_t i = 0; i < n; ++i)
               std::cout << v_acc[i] << ' ';
         std::cout << '\n';
      }

      return 0;
   }

**Output:**

.. code:: none

   1 2 3 3 3 5
   s o r t e d

radix_sort Out-of-Place Example
-------------------------------

.. code:: cpp

   // possible build and run commands:
   //    icpx -fsycl radix_sort.cpp -o radix_sort -I /path/to/oneDPL/include && ./radix_sort

   #include <cstdint>
   #include <iostream>
   #include <sycl/sycl.hpp>

   #include <oneapi/dpl/experimental/kernel_templates>

   namespace kt = oneapi::dpl::experimental::kt;

   int main()
   {
      std::size_t n = 6;
      sycl::queue q{sycl::gpu_selector_v};
      std::uint32_t* keys = sycl::malloc_shared<std::uint32_t>(n, q);
      std::uint32_t* keys_out = sycl::malloc_shared<std::uint32_t>(n, q);

      // initialize
      keys[0] = 3, keys[1] = 2, keys[2] = 1, keys[3] = 5, keys[4] = 3, keys[5] = 3;

      // sort
      auto e = kt::gpu::esimd::radix_sort<false, 8>(q, keys, keys + n, keys_out, kt::kernel_param<416, 64>{}); // (4)
      e.wait();

      // print
      for(std::size_t i = 0; i < n; ++i)
         std::cout << keys[i] << ' ';
      std::cout << '\n';
      for(std::size_t i = 0; i < n; ++i)
         std::cout << keys_out[i] << ' ';
      std::cout << '\n';

      sycl::free(keys, q);
      sycl::free(keys_out, q);
      return 0;
   }

**Output:**

.. code:: none

   3 2 1 5 3 3
   5 3 3 3 2 1

radix_sort_by_key Out-of-Place Example
--------------------------------------

.. code:: cpp

   // possible build and run commands:
   //    icpx -fsycl radix_sort_by_key.cpp -o radix_sort_by_key -I /path/to/oneDPL/include && ./radix_sort_by_key

   #include <cstdint>
   #include <iostream>
   #include <sycl/sycl.hpp>

   #include <oneapi/dpl/experimental/kernel_templates>

   namespace kt = oneapi::dpl::experimental::kt;

   int main()
   {
      std::size_t n = 6;
      sycl::queue q{sycl::gpu_selector_v};
      sycl::buffer<std::uint32_t> keys{sycl::range<1>(n)};
      sycl::buffer<std::uint32_t> keys_out{sycl::range<1>(n)};
      sycl::buffer<char> values{sycl::range<1>(n)};
      sycl::buffer<char> values_out{sycl::range<1>(n)};


      // initialize
      {
         sycl::host_accessor k_acc{keys, sycl::write_only};
         k_acc[0] = 3, k_acc[1] = 2, k_acc[2] = 1, k_acc[3] = 5, k_acc[4] = 3, k_acc[5] = 3;

         sycl::host_accessor v_acc{values, sycl::write_only};
         v_acc[0] = 'r', v_acc[1] = 'o', v_acc[2] = 's', v_acc[3] = 'd', v_acc[4] = 't', v_acc[5] = 'e';
      }

      // sort
      auto e = kt::gpu::esimd::radix_sort_by_key<true, 8>(q, keys, values, keys_out, values_out,
                                                     kt::kernel_param<96, 64>{}); // (7)
      e.wait();

      // print
      {
         sycl::host_accessor k_acc{keys, sycl::read_only};
         for(std::size_t i = 0; i < n; ++i)
               std::cout << k_acc[i] << ' ';
         std::cout << '\n';

         sycl::host_accessor v_acc{values, sycl::read_only};
         for(std::size_t i = 0; i < n; ++i)
               std::cout << v_acc[i] << ' ';
         std::cout << "\n\n";
         
         sycl::host_accessor k_out_acc{keys_out, sycl::read_only};
         for(std::size_t i = 0; i < n; ++i)
               std::cout << k_out_acc[i] << ' ';
         std::cout << '\n';

         sycl::host_accessor v_out_acc{values_out, sycl::read_only};
         for(std::size_t i = 0; i < n; ++i)
               std::cout << v_out_acc[i] << ' ';
         std::cout << '\n';
      }

      return 0;
   }

**Output:**

.. code:: none

   3 2 1 5 3 3
   r o s d t e

   1 2 3 3 3 5
   s o r t e d


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

Global memory is used for copying the input sequence(s) and storing internal data such as radix value counters.
The used amount depends on many parameters; below is an upper bound approximation:

:``radix_sort``: N\ :sub:`keys` + C * N\ :sub:`keys`

:``radix_sort_by_key``: N\ :sub:`keys` + N\ :sub:`values` + C * N\ :sub:`keys`

where the sequence with keys takes N\ :sub:`keys` space, the sequence with values takes N\ :sub:`values` space,
and the additional space is C * N\ :sub:`keys`.

The value of `C` depends on ``param.data_per_workitem``, ``param.workgroup_size``, and ``RadixBits``.
For ``param.data_per_workitem`` set to `32`, ``param.workgroup_size`` to `64`, and ``RadixBits`` to `8`,
`C` approximately equals to `1`.
Incrementing ``RadixBits`` increases `C` up to twice, while doubling either
``param.data_per_workitem`` or ``param.workgroup_size`` leads to a halving of `C`.

.. note::

   If the number of elements to sort does not exceed ``param.data_per_workitem * param.workgroup_size``,
   ``radix_sort`` is executed by a single work-group and does not use any global memory.
   For ``radix_sort_by_key`` there is no single work-group implementation yet.

..
   The estimation above is not very precise and it seems it is not necessary for the global memory.
   The C coefficient base is actually 0.53 instead of 1.
   An increment of RadixBits multiplies C by the factor of ~1.5 on average.

   Additionally, C exceeds 1 for radix_sort_by_key,
   when N is small and the global histogram takes more space than the sequences.
   This space is small, single WG implementation will be added, therefore this is neglected.

.. _local-memory:

Local Memory Requirements
-------------------------

Local memory is used for reordering keys or key-value pairs within a work-group,
and for storing internal data such as radix value counters.
The used amount depends on many parameters; below is an upper bound approximation:

:``radix_sort``: N\ :sub:`keys_per_workgroup` + C

:``radix_sort_by_key``: N\ :sub:`keys_per_workgroup` + N\ :sub:`values_per_workgroup` + C

where N\ :sub:`keys_per_workgroup` and N\ :sub:`values_per_workgroup` are the amounts of memory
to store keys and values, respectively.  `C` is some additional space for storing internal data.

N\ :sub:`keys_per_workgroup` equals to ``sizeof(key_type) * param.data_per_workitem * param.workgroup_size``,
N\ :sub:`values_per_workgroup` equals to ``sizeof(value_type) * param.data_per_workitem * param.workgroup_size``,
`C` does not exceed `4KB`.

..
   C as 4KB stands on these points:
   1) Extra space is needed to store a histogram to distribute keys. It's size is 4 * (2^RadixBits).
   The estimation is correct for RadixBits 9 (2KB) and smaller. Support of larger RadixBits is not expected.
   1) N_keys + N_values is rounded up at 2KB border (temporarily as a workaround for a GPU driver bug).

..
   The estimation assumes that reordering keys/pairs takes more space than ranking keys.
   The ranking takes approximatelly "2 * workgroup_size * (2^RadixBits)" bytes.
   It suprpasses Intel Data Center GPU Max SLM capacity in only marginal cases,
   e.g., when RadixBits is 10 and workgroup_size is 64, or when RadixBits is 9 and workgroup_size is 128.
   It is ignored as an unrealistic case.

-----------------------------------------
Recommended Settings for Best Performance
-----------------------------------------

The general advice is to choose kernel parameters based on performance measurements and profiling information.
The initial configuration may be selected according to these high-level guidelines:

..
   TODO: add this part when param.workgroup_size supports more than one value:
   Increasing ``param.data_per_workitem`` should usually be preferred to increasing ``param.workgroup_size``,
   to avoid extra synchronization overhead within a work-group.

- When the number of elements to sort (N) is small (~16K or less) and the algorithm is ``radix_sort``,
  generally sorting is done more efficiently by a single work-group.
  Increase the ``param`` values to make ``N <= param.data_per_workitem * param.workgroup_size``.

- When the number of elements to sort ``N`` is between 16K and 1M, utilizing all available
  compute cores is key for better performance. Allow creating enough work chunks to feed all
  X\ :sup:`e`-cores [#fnote2]_ on a GPU: ``param.data_per_workitem * param.workgroup_size ≈ N / xe_core_count``.

- When the number of elements to sort is large (more than ~1M), maximizing the number of elements
  processed by a work-group, which equals to ``param.data_per_workitem * param.workgroup_size``,
  reduces synchronization overheads between work-groups and usually benefits the overall performance.

.. warning::

   Avoid setting too large ``param.data_per_workitem`` and ``param.workgroup_size`` values.
   Make sure that :ref:`Memory requirements <memory-requirements>` are satisfied.

.. note::

   ``param.data_per_workitem`` is the only available parameter to tune the performance,
   since ``param.workgroup_size`` currently supports only one value (`64`).


.. [#fnote1] Andy Adinets and Duane Merrill (2022). Onesweep: A Faster Least Significant Digit Radix Sort for GPUs. Retrieved from https://arxiv.org/abs/2206.01784.
.. [#fnote2] The X\ :sup:`e`-core term is described in the `oneAPI GPU Optimization Guide
   <https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-0/intel-xe-gpu-architecture.html#XE-CORE>`_.
   Check the number of cores in the device specification, such as `Intel® Data Center GPU Max specification
   <https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series/products.html>`_.
