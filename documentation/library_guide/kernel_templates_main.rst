Kernel Templates
################

Kernel Templates are experimental API which allow fine-grained performance tuning.
Use them when you need better performance and
you can sacrifice performance-portability as a side-effect of precise tuning for a specific device, number of elements, data type, etc.

Generic structure for configuring a Kernel Template:
* :doc:`Kernel Configuration <kernel_templates/kernel_configuration>`.

Sets of Kernel Templates with a specific backend:
* :doc:`ESIMD-based Kernel Templates <kernel_templates/esimd_main>`.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :glob:
   :hidden:

   kernel_templates/kernel_configuration
   kernel_templates/esimd_main
