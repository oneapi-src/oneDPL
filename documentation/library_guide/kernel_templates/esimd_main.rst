ESIMD-Based Kernel Templates
############################

All API on this section reside in ``oneapi::dpl::experimental::kt`` namespace and
are available through inclusion of ``oneapi/dpl/experimental/kernel_templates`` header file
This namespace is omitted in the rest of the page, while the nested namespaces are specified.

The ESIMD kernel templates are based on `Explicit SIMD SYCL (ESIMD) <https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2024-0/explicit-simd-sycl-extension.html>`_.
This technology only supports Intel GPU devices.

.. note::

   The ``kernel_param::data_per_workitem`` parameter has a special meaning in ESIMD-based kernel templates.
   Usually, each work-item processes ``data_per_workitem`` sequentially.
   However, work-items in ESIMD-based kernel templates perform vectorization,
   so the sequential work is ``data_per_workitem / vector_length``, where ``vector_length`` is an implementation-defined vectorization factor.

List of kernel templates:

* :doc:`radix_sort and radix_sort_by_key <esimd/radix_sort>`

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :glob:
   :hidden:

   esimd/radix_sort
