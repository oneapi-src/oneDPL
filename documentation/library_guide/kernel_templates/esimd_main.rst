ESIMD-Based Kernel Templates
############################

The ESIMD kernel templates are based on `Explicit SIMD SYCL extension
<https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2024-0/explicit-simd-sycl-extension.html>`_
of Intel® oneAPI DPC++/C++ Compiler.
This technology only supports Intel GPU devices.

These templates are available in the ``oneapi::dpl::experimental::kt::esimd`` namespace. The following are implemented:

* :doc:`radix_sort and radix_sort_by_key <esimd/radix_sort>`

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :glob:
   :hidden:

   esimd/radix_sort
