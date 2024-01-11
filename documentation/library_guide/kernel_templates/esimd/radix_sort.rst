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

The functions sort data using the radix sort algorithm. For a small number of elements to sort, they invoke a single-work-group implementation; otherwise, they use a multiple-work-group implementation based on the onesweep algorithm variant.

**Template Parameters**

+-----------------------------+----------------------------------------------------------+
| Name                        | Description                                              |
+=============================+==========================================================+
| ``bool IsAscending``        | Sort order. Ascending: ``true``; Descending: ``false``.  |
+-----------------------------+----------------------------------------------------------+
| ``std::uint8_t RadixBits``  | Number of bits to sort per a radix sort algorithm pass.  |
|                             | Only ``8`` is currently supported.                       |
+-----------------------------+----------------------------------------------------------+


**Parameters**

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


**Return Value**

``sycl::event`` object representing the status of the algorithm execution.

**Local memory usage**

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

**Invocation examples**

.. code:: cpp

   #include <oneapi/dpl/experimental/kernel_templates>
   namespace kt = oneapi::dpl::experimental::kt;
   ...
   std::size_t n = 10'000'000;
   sycl::queue q{sycl::gpu_selector_v};
   std::uint32_t* keys = sycl::malloc_device<std::uint32_t>(q, n);
   ...
   auto e = kt::esimd::radix_sort<false, 8>(q, keys, keys + n, kt::kernel_param<416, 32>{}); // (2)
   e.wait(); // keys are now sorted in descending order


.. code:: cpp

   #include <oneapi/dpl/experimental/kernel_templates>
   namespace kt = oneapi::dpl::experimental::kt;
   ...
   std::size_t n = 500'000;
   sycl::queue q{sycl::gpu_selector_v};
   sycl::buffer<std::uint32_t> keys{sycl::range<1>(n)};
   sycl::buffer<float> values(sycl::range<1>(n));
   ...
   auto e = kt::esimd::radix_sort_by_key<true, 8>(q, keys, values, kt::kernel_param<96, 64>{}); // (3)
   e.wait(); // key-value pairs are now sorted in ascedning order


**Recommended settings for the best performance**

General advice is to set the configuration according to the performance measurements and profiling information.

But the initial configuration may be selected according to these points:

a. The number of elements to sort is small (~16K or less) and the algorithm is ``radix_sort``. The elements can be processed by a single work-group.

   - Increase ``param`` values, so ``N <= param.data_per_workitem * param.workgroup_size``.

b. The number of elements to sort is medium (between ~16K and ~1M). All the work-groups can execute simultaneously.

   - Make sure the device is saturated: ``param.data_per_workitem * param.workgroup_size ≈ N / device_xe_core_count``. Prefer larger ``param.workgroup_size`` in ``param.data_per_workitem * param.workgroup_size`` combination to reduce the number of work-groups and thus synchronization overhead.

c. The number of elements to sort is large (more than ~1M). The work-groups preempt each other.

   - Increase the occupancy to hide the latency: ``param.data_per_workitem * param.workgroup_size ≈< N / (device_xe_core_count * desired_occupancy)``. The occupancy depends on the local memory usage which is determined by ``key_type``, ``value_type``, ``radix_bits``, ``param.data_per_workitem`` and ``param.workgroup_size`` parameters. Refer to "Local memory usage" chapter for the calculation.


**Limitations (may be relaxed in the future)**

- Algorithms can process only C++ integral and floating-point types with the width up to 64-bits (except for ``bool``).
- Number of elements to sort must not exceed `2^30`.
- ``radix_bits`` can only be `8`.
- ``param.data_per_workitem`` has discreteness of `32`.
- ``param.workgroup_size`` can only be `64`.
- Local memory is always used to rank keys, reorder keys or key-value pairs which limits possible values of ``param.data_per_workitem`` and ``param.workgroup_size``.
- ``radix_sort_by_key`` does not have single-work-group implementation yet.


**Possible API extensions (may be implemented in the future)**

- Allow passing externally allocated memory.
- Allow passing dependent events.
- Allow passing a range of bits to sort.
- Allow out-of-place sorting, e.g. with a double-buffer or an output sequence(s)
- Allow configuration of kernels other than the most time-consuming kernel (e.g. of a kernel computing histograms).
- Allow range transformations (e.g. range pipes or transform iterators).


**System requirements (coverage my be extended in the future)**

- Hardware: Intel® Data Center GPU Max Series.
- Compiler: Intel® oneAPI DPC++/C++ 2023.2 and newer.
- OS: RHEL 9.2, SLES 15 SP5, Ubuntu 22.04. Other distributions and their versions listed in `<https://dgpu-docs.intel.com/driver/installation.html>` should be supported accordingly.


**Known Issues**

- Use of -g, -O0, -O1 compiler options may lead to compilation issues.
- Combinations of ``param.data_per_workitem`` and ``param.work_group_size`` with large values may lead to device-code compilation errors due to allocation of local memory amounts beyond the device capabilities. Refer to "Local memory usage" paragraph for the details regarding allocation.
- ``radix_sort_by_key`` produces wrong results with the following combinations of ``kt::kernel_param`` and types of keys and values:

  - ``sizeof(key_type) + sizeof(value_type) = 12``, ``param.workgroup_size = 64`` and ``param.data_per_workitem = 96``
  - ``sizeof(key_type) + sizeof(value_type) = 16``, ``param.workgroup_size = 64`` and ``param.data_per_workitem = 64``
