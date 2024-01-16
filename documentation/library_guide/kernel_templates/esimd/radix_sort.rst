Radix Sort
##########

.. code:: cpp

   // Defined in header <oneapi/dpl/experimental/kernel_templates>

   namespace esimd {

   template <bool IsAscending = true, std::uint8_t RadixBits = 8, typename KernelParam, typename Range>
   sycl::event
   radix_sort(sycl::queue q, Range&& rng, KernelParam param); // (1)

   template <bool IsAscending = true,  std::uint8_t RadixBits = 8, typename KernelParam, typename Iter>
   sycl::event
   radix_sort(sycl::queue q, Iter first, Iter last, KernelParam param); // (2)

   template <bool IsAscending = true, std::uint8_t RadixBits = 8, typename KernelParam, typename KeysRng, typename ValsRng>
   sycl::event
   radix_sort_by_key(sycl::queue q, KeysRng&& keys_rng, ValsRng&& vals_rng, KernelParam param); // (3)

   template <bool IsAscending = true, std::uint8_t RadixBits = 8, typename KernelParam, typename KeysIter, typename ValsIter>
   sycl::event
   radix_sort_by_key(sycl::queue q, KeysIter keys_first, KeysIter keys_last, ValsIter vals_first, KernelParam param); // (4)

   } // namespace esimd

The functions sort data using the radix sort algorithm. For a small number of elements to sort, they invoke a single-work-group implementation; otherwise, they use a multiple-work-group implementation based on the Onesweep* [#fnote1]_ algorithm variant.

Template Parameters
--------------------

+-----------------------------+----------------------------------------------------------+
| Name                        | Description                                              |
+=============================+==========================================================+
| ``bool IsAscending``        | Sort order. Ascending: ``true``; Descending: ``false``.  |
+-----------------------------+----------------------------------------------------------+
| ``std::uint8_t RadixBits``  | Number of bits to sort per a radix sort algorithm pass.  |
|                             | Only ``8`` is currently supported.                       |
+-----------------------------+----------------------------------------------------------+


Parameters
----------

+------------------------------------------------------+------------------------------------------------------------------+
| Name                                                 | Description                                                      |
+======================================================+==================================================================+
|  ``q``                                               | SYCL queue to submit the kernel template to.                     |
+------------------------------------------------------+------------------------------------------------------------------+
|                                                      | The sequences(s) of elements to apply the algorithm to.          |
|  - ``rng`` (1)                                       | They can be provided as:                                         |
|  - ``first``, ``last`` (2)                           |                                                                  |
|  - ``keys_rng``, ``vals_rng`` (3)                    | - ``sycl::buffer`` (1,3),                                        |
|  - ``keys_first``, ``keys_last``, ``vals_first`` (4) | - ``oneapi::dpl::experimental::ranges::views::all`` (1,3)        |
|                                                      | - ``oneapi::dpl::experimental::ranges::views::subrange`` (1,3)   |
|                                                      | - USM pointer (2,4)                                              |
|                                                      | - ``oneapi::dpl::begin`` and ``oneapi::dpl::end`` (2,4)          |
+------------------------------------------------------+------------------------------------------------------------------+
|  ``param``                                           | Kernel configuration structure. ``data_per_workitem``,           |
|                                                      | can be any value among ``32``, ``64``, ``96``,..., ``k * 32``;   |
|                                                      | ``workgroup_size`` can be only ``64``.                           |
+------------------------------------------------------+------------------------------------------------------------------+


Return Value
------------

``sycl::event`` object representing a status of the algorithm execution.


Memory Usage
------------

.. _local-memory:

Local Memory
~~~~~~~~~~~~

The local memory is allocated as shown in the pseudo-code blocks below:

- ``radix_sort`` (1,2):

  .. code:: python

     ranks = 2 * (2 ^ radix_bits) * workgroup_size + (2 * workgroup_size) + 4 * (2 ^ radix_bits)
     reorder = sizeof(key_type) * data_per_workitem * workgroup_size + 4 * (2 ^ radix_bits)
     allocated_bytes = round_up_to_nearest_multiple(max(ranks, reorder), 2048)


- ``radix_sort_by_key`` (3,4):

  .. code:: python

     ranks = 2 * (2 ^ radix_bits) * workgroup_size + (2 * workgroup_size) + 4 * (2 ^ radix_bits)
     reorder = (sizeof(key_type) + sizeof(value_type)) * data_per_workitem * workgroup_size + 4 * (2 ^ radix_bits)
     allocated_bytes = round_up_to_nearest_multiple(max(ranks, reorder), 2048)

The device must have enough local memory to execute the selected configuration.


Global Memory
~~~~~~~~~~~~~

The global (USM device) memory is allocated as shown in the pseudo-code blocks below:

- ``radix_sort`` (1,2):

  .. code:: python

     histogram_bytes = (2 ^ radix_bits) * ceiling_division(sizeof(key_type) * 8, radix_bits)
     tmp_buffer_bytes = N * sizeof(key_type)
     allocated_bytes = tmp_buffer_bytes + histogram_bytes

- ``radix_sort_by_key`` (3,4):

  .. code:: python

     histogram_bytes = (2 ^ radix_bits) * ceiling_division(sizeof(key_type) * 8, radix_bits)
     tmp_buffer_bytes = N * (sizeof(key_type) + sizeof(value_type))
     allocated_bytes = tmp_buffer_bytes + histogram_bytes


Examples
--------

.. code:: cpp

   // example1.cpp
   // icpx -fsycl example1.cpp -o example1 -I /path/to/oneDPL/include && ./example1

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

-----

.. code:: cpp

   // example2.cpp
   // icpx -fsycl example2.cpp -o example2 -I /path/to/oneDPL/include && ./example2

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


Recommended Settings for Best Performance
-----------------------------------------

The general advice is to set your configuration according to the performance measurements and profiling information. The initial configuration may be selected according to these points:

- When the number of elements to sort is small (~16K or less) and the algorithm is ``radix_sort``, then the elements can be processed by a single work-group. Increase the param values, so ``N <= param.data_per_workitem * param.workgroup_size``.

- When the number of elements to sort is medium (between ~16K and ~1M), then all the work-groups can execute simultaneously. Make sure the device is saturated: ``param.data_per_workitem * param.workgroup_size ≈ N / device_xe_core_count``. A larger ``param.workgroup_size`` in ``param.data_per_workitem * param.workgroup_size`` combination is preferred to reduce the number of work-groups and the synchronization overhead.

- When the number of elements to sort is large (more than ~1M), then the work-groups preempt each other. Increase the occupancy to hide the latency with ``param.data_per_workitem * param.workgroup_size ≈< N / (device_xe_core_count * desired_occupancy)``. The occupancy depends on the local memory usage, which is determined by ``key_type``, ``value_type``, ``radix_bits``, ``param.data_per_workitem`` and ``param.workgroup_size`` parameters. Refer to :ref:`Local Memory <local-memory>` section for the calculation.


.. _limitations:

Limitations
-----------

- Algorithms can only process C++ integral and floating-point types with a width of up to 64 bits (except for ``bool``).
- Number of elements to sort must not exceed `2^30`.
- ``radix_bits`` can only be `8`.
- ``param.data_per_workitem`` has discreteness of `32`.
- ``param.workgroup_size`` can only be `64`.
- Local memory is always used to rank keys, reorder keys, or key-value pairs, which limits possible values of ``param.data_per_workitem`` and ``param.workgroup_size``
- ``radix_sort_by_key`` does not have single-work-group implementation yet.


.. _possible-api-extensions:

Possible API Extensions
-----------------------

- Allow passing externally allocated memory.
- Allow passing dependent events.
- Allow passing a range of bits to sort.
- Allow out-of-place sorting (for example, through a double-buffer or an output sequence).
- Allow configuration of kernels other than the most time-consuming kernel (for example, of a kernel computing histograms).
- Allow range transformations (for example, range pipes or transform iterators).


.. _system-requirements:

System Requirements
-------------------

- Hardware: Intel® Data Center GPU Max Series.
- Compiler: Intel® oneAPI DPC++/C++ 2023.2 and newer.
- OS: RHEL 9.2, SLES 15 SP5, Ubuntu 22.04. Other distributions and their versions listed in `<https://dgpu-docs.intel.com/driver/installation.html>`_ should be supported accordingly however they have not been tested.


Known Issues
------------

- Use of -g, -O0, -O1 compiler options may lead to compilation issues.
- Combinations of ``param.data_per_workitem`` and ``param.work_group_size`` with large values may lead to device-code compilation errors due to allocation of local memory amounts beyond the device capabilities. Refer to "Local memory usage" paragraph for the details regarding allocation.
- ``radix_sort_by_key`` produces wrong results with the following combinations of ``kt::kernel_param`` and types of keys and values:

  - ``sizeof(key_type) + sizeof(value_type) = 12``, ``param.workgroup_size = 64`` and ``param.data_per_workitem = 96``
  - ``sizeof(key_type) + sizeof(value_type) = 16``, ``param.workgroup_size = 64`` and ``param.data_per_workitem = 64``

.. note::

   The following may be changed in the future:

   - The API may be expanded (see :ref:`Possible API Extensions <possible-api-extensions>`). As a result, it may become incompatible with the previous versions.
   - :ref:`Limitations <limitations>` may be relaxed.
   - List of supported hardware, compilers and operative systems shown on :ref:`System Requirements <system-requirements>` may be expanded.


.. [#fnote1] Andy Adinets and Duane Merrill (2022). Onesweep: A Faster Least Significant Digit Radix Sort for GPUs. Retrieved from https://arxiv.org/abs/2206.01784.
