Kernel Templates
################

Kernel Templates is an experimental API that includes algorithms which can be configured based on a variety of parameters, such as the number of elements to be processed by a work-item and the size of a work-group.
Use it when you can specialize an algorithm for a specific workload (e.g., number of elements and their type) or a device to achieve better performance.

.. toctree::
   :maxdepth: 2

   kernel_templates/kernel_configuration: Generic structure for configuring a Kernel Template.
   kernel_templates/esimd_main: Kernel Templates based on Intel "Explicit SIMD" SYCL extension. They work with Intel GPUs.
