ESIMD-based Kernel Templates
############################

ESIMD Kernel Templates are based on `Explicit SIMD SYCL (ESIMD) <https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2024-0/explicit-simd-sycl-extension.html>`_.
This technology supports only Intel GPU devices.

.. note::
   ``kernel_param::data_per_workitem`` parameter has a special meaning in ESIMD-based Kernel Templates.
   Usually, each work-item processes ``data_per_workitem`` sequentially.
   However, work-items in ESIMD-based Kernel Templates perform vectorization,
   so the sequential work is in fact ``data_per_workitem / vector_lenght``, where ``vector_lenght`` is an implementation-defined vectorization factor.

------------

List of Kernel Templates:

* :doc:`Radix Sort Kernel Templates <esimd/radix_sort>`.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :glob:

   esimd/radix_sort
