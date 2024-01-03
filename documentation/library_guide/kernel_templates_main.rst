Kernel Templates
################

|onedpl_long| (|onedpl_short|) Kernel Templates are experimental API which allow fine-grained performance tuning.
Use them when you need better performance (e.g. compared to standard API like ``oneapi::dpl::sort``) and
you can sacrifice performance-portability as a side-effect of precise tuning for a specific device, number of elements, data type, etc.

All public API for SYCL kernel templates reside in ``namespace oneapi::dpl::experimental::kt``.
This, the main, namespace is omitted in the rest of this document, while the nested namespaces are specified.

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
