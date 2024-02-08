Radix Sort
##########

-----------------------------------------------------------
``radix_sort`` and ``radix_sort_by_key`` Function Templates
-----------------------------------------------------------

The ``radix_sort`` and ``radix_sort_by_key`` functions sort data using the radix sort algorithm.
The sorting is stable, ensuring the preservation of the relative order of elements with equal keys.
The functions implement Onesweep* [#fnote1]_ algorithm variant.

A synopsis of the ``radix_sort`` and ``radix_sort_by_key`` functions is provided below:

.. code:: cpp

   // defined in <oneapi/dpl/experimental/kernel_templates>

   namespace oneapi::dpl::experimental::kt::esimd {

   // Sort a single sequence

   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename Range>
   sycl::event
   radix_sort (sycl::queue q, Range&& r, KernelParam param); // (1)

   template <bool IsAscending = true,  std::uint8_t RadixBits = 8,
             typename KernelParam, typename Iterator>
   sycl::event
   radix_sort (sycl::queue q, Iterator first, Iterator last,
               KernelParam param); // (2)


   // Sort a sequence of keys and apply the same order to a sequence of values

   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename KeysRng, typename ValuesRng>
   sycl::event
   radix_sort_by_key (sycl::queue q, KeysRng&& keys,
                      ValuesRng&& values, KernelParam param); // (3)

   template <bool IsAscending = true, std::uint8_t RadixBits = 8,
             typename KernelParam, typename Iterator1, typename Iterator2>
   sycl::event
   radix_sort_by_key (sycl::queue q, Iterator1 keys_first, Iterator1 keys_last,
                      Iterator2 values_first, KernelParam param); // (4)

   }


.. _template-parameters:

Template Parameters
--------------------

+-----------------------------+---------------------------------------------------------------------------------------+
| Name                        | Description                                                                           |
+=============================+=======================================================================================+
| ``bool IsAscending``        | The sort order. Ascending: ``true``; Descending: ``false``.                           |
+-----------------------------+---------------------------------------------------------------------------------------+
| ``std::uint8_t RadixBits``  | The number of bits to sort per a radix sort algorithm pass.                           |
+-----------------------------+---------------------------------------------------------------------------------------+


.. _parameters:

Parameters
----------

+-----------------------------------------------+---------------------------------------------------------------------+
| Name                                          | Description                                                         |
+===============================================+=====================================================================+
|  ``q``                                        | SYCL* queue to submit the kernels to.                               |
+-----------------------------------------------+---------------------------------------------------------------------+
|                                               | The sequences of elements to apply the algorithm to.                |
|  - ``r`` (1)                                  | Supported sequence types:                                           |
|  - ``first``, ``last`` (2)                    |                                                                     |
|  - ``keys``, ``values`` (3)                   | - ``sycl::buffer`` (1,3),                                           |
|  - ``keys_first``, ``keys_last``,             | - ``oneapi::dpl::experimental::ranges::views::all`` (1,3),          |
|    ``values_first`` (4)                       | - ``oneapi::dpl::experimental::ranges::views::subrange`` (1,3),     |
|                                               | - USM pointers (2,4),                                               |
|                                               | - ``oneapi::dpl::begin`` and ``oneapi::dpl::end`` (2,4).            |
|                                               |                                                                     |
+-----------------------------------------------+---------------------------------------------------------------------+
|  ``param``                                    | A :doc:`kernel_param <../kernel_configuration>` object.             |
|                                               | Its ``data_per_workitem`` must be a positive multiple of 32.        |
|                                               |                                                                     |
|                                               |                                                                     |
+-----------------------------------------------+---------------------------------------------------------------------+


**Type Requirements**:

- The element type of sequence(s) to sort must be any
  C++ integral and floating-point type with a width of up to 64 bits, except for ``bool``.

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


``radix_sort`` Example
----------------------

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
      auto e = kt::esimd::radix_sort<false, 8>(q, keys, keys + n, kt::kernel_param<416, 64>{}); // (2)
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


``radix_sort_by_key`` Example
-----------------------------

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
      auto e = kt::esimd::radix_sort_by_key<true, 8>(q, keys, values, kt::kernel_param<96, 64>{}); // (3)
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


-------------------
Memory Requirements
-------------------

The algorithms use global and local device memory (see `SYCL 2020 Specification
<https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_sycl_device_memory_model>`_)
for intermediate data storage. For the algorithms to operate correctly, there must be enough memory
on the device; otherwise, the behavior is undefined. How much memory is needed depends on input data
and configuration parameters, as described below.

Global Memory Requirements
--------------------------

The algorithms require memory for copying the input sequence(s) and some additional space to distribute elements.
The used amount depends on many parameters; below is an upper bound approximation:

- ``radix_sort``:

  N\ :sub:`1` + max (16KB, N\ :sub:`1`)

- ``radix_sort_by_key``:

  N\ :sub:`1` + N\ :sub:`2` + max (16KB, N\ :sub:`1`)

where the sequence with keys takes N\ :sub:`1` space and the sequence with values takes N\ :sub:`2` space.

..
   This is a rough upper bound approximation. High precision seems to be not necessary for global memory.
   It works for RadixBits <= 8, the data_per_workitem >= 32 and workgroup_size >= 64.
   Reevaluate it, once bigger RadixBits, or smaller data_per_workitem and workgroup_size are supported.

.. note::

   For ``N <= param.data_per_workitem * param.workgroup_size``, where ``N`` is a number of elements to sort,
   ``radix_sort`` is executed by a single work-group and does not use any global memory.

.. _local-memory:

Local Memory Requirements
-------------------------

The algorithms require local memory to rank keys, reorder keys, or key-value pairs.
The used amount depends on many parameters; below is an upper bound approximation:

- ``radix_sort``:

  max (36KB, sizeof(``key_type``) * ``param.data_per_workitem`` * ``param.workgroup_size`` + 2KB)

- ``radix_sort_by_key``:

  max (36KB, (sizeof(``key_type``) + sizeof(``value_type``)) * ``param.data_per_workitem`` * ``param.workgroup_size`` + 2KB)

where ``key_type``, ``value_type`` are the types of the input keys, values respectively.

..
   This is an upper bound approximation, which is close to the real value.
   High precision is essential as SLM usage has high impact on performance.
   It works for RadixBits = 8, the data_per_workitem >= 32 and workgroup_size >= 64.
   Reevaluate it, once bigger RadixBits, or smaller data_per_workitem and workgroup_size are supported.

-----------------------------------------
Recommended Settings for Best Performance
-----------------------------------------

The general advice is to choose kernel parameters based on performance measurements and profiling information.
The initial configuration may be selected according to these high-level guidelines:

- When the number of elements to sort is small (~16K or less) and the algorithm is ``radix_sort``,
  then the elements can be processed by a single-work-group sort, which generally outperforms multiple-work-group sort.
  Increase the ``param`` values, so ``N <= param.data_per_workitem * param.workgroup_size``,
  where ``N`` is the number of elements to sort.

.. note::

   ``radix_sort_by_key`` does not have a single-work-group implementation yet.

- When the number of elements to sort ``N`` is between 16K and 1M, utilizing all available
  compute cores is key for better performance. Allow creating enough work chunks to feed all
  Xe-cores on a GPU: ``param.data_per_workitem * param.workgroup_size ≈ N / device_xe_core_count``.

  ..
     TODO: add this part when param.workgroup_size supports more than one value:
     A larger ``param.workgroup_size`` in ``param.data_per_workitem * param.workgroup_size``
     combination is preferred to reduce the number of work-groups and the synchronization overhead.

- When the number of elements to sort is large (more than ~1M), then the work-groups preempt each other.
  Increase the occupancy to hide the latency with ``param.data_per_workitem * param.workgroup_size ≈< N / (device_xe_core_count * desired_occupancy)``.
  The occupancy depends on the local memory usage, which is determined by
  ``key_type``, ``value_type``, ``RadixBits``, ``param.data_per_workitem`` and ``param.workgroup_size`` parameters.
  Refer to :ref:`Local Memory Requirements <local-memory>` section for the calculation.

.. note::

   ``param.data_per_workitem`` is the only available parameter to tune the performance,
   since ``param.workgroup_size`` currently supports only one value (`64`).


.. [#fnote1] Andy Adinets and Duane Merrill (2022). Onesweep: A Faster Least Significant Digit Radix Sort for GPUs. Retrieved from https://arxiv.org/abs/2206.01784.
