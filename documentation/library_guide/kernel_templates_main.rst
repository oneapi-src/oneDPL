Kernel Templates API
####################

Kernel templates is an experimental API that includes algorithms that can be configured based on various parameters, such as the number of elements to be processed by a work-item and the size of a work-group.
Use it when you can specialize an algorithm for a specific workload (for example, the number of elements and their type) or a device to achieve better performance.

* :doc:`ESIMD-based kernel templates <kernel_templates/esimd_main>`. Kernel templates based on Intel "Explicit SIMD" SYCL extension.
* :doc:`Kernel Configuration <kernel_templates/kernel_configuration>`. Generic structure for configuring a kernel template.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :glob:
   :hidden:

   kernel_templates/esimd_main
   kernel_templates/kernel_configuration
