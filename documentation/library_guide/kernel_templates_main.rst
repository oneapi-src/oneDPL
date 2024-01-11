Kernel Templates API
####################

Kernel Templates is an experimental API that includes algorithms which can be configured based on a variety of parameters, such as the number of elements to be processed by a work-item and the size of a work-group.
Use it when you can specialize an algorithm for a specific workload (e.g., number of elements and their type) or a device to achieve better performance.

* :doc:`ESIMD-based Kernel Templates <kernel_templates/esimd_main>`. Kernel Templates based on Intel "Explicit SIMD" SYCL extension.
* :doc:`Kernel Configuration <kernel_templates/kernel_configuration>`. Generic structure for configuring a Kernel Template.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :glob:
   :hidden:

   kernel_templates/esimd_main
   kernel_templates/kernel_configuration
