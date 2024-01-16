ESIMD-Based Kernel Templates
############################

The ESIMD kernel templates are based on `Explicit SIMD SYCL (ESIMD) <https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2024-0/explicit-simd-sycl-extension.html>`_.
This technology only supports Intel GPU devices.

.. note::
   The ``kernel_param::data_per_workitem`` parameter has a special meaning in ESIMD-based kernel templates.
   Usually, each work-item processes ``data_per_workitem`` sequentially.
   However, work-items in ESIMD-based kernel templates perform vectorization,
   so the sequential work is ``data_per_workitem / vector_lenght``, where ``vector_lenght`` is an implementation-defined vectorization factor.

List of Kernel Templates:

* :doc:`Radix Sort <esimd/radix_sort>`

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :glob:
   :hidden:

   esimd/radix_sort
