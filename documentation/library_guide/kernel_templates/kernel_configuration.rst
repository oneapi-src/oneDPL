Kernel Configuration
####################

.. code:: cpp

   // Defined in header <oneapi/dpl/experimental/kernel_templates>

   template <std::uint16_t DataPerWorkItem,
             std::uint16_t WorkGroupSize,
             typename KernelName = oneapi::dpl::execution::DefaultKernelName>
   struct kernel_param;

Each Kernel Template is supplied by a `kernel_param` object, generic structure for configuring a kernel.
The supported and the most performant values depend on the Kernel Template itself, device capabilities, the data type and the data size.

Member constants
----------------

+------------------------------------------------------+---------------------+------------------------------------------------------+
| Name                                                 | Value               | Description                                          |
+======================================================+=====================+======================================================+
| ``static constexpr std::uint16_t data_per_workitem`` | ``DataPerWorkItem`` | Number of iterations to be processed by a work-item. |
+------------------------------------------------------+---------------------+------------------------------------------------------+
| ``static constexpr std::uint16_t workgroup_size``    | ``WorkGroupSize``   | Number of work-items within a work-group.            |
+------------------------------------------------------+---------------------+------------------------------------------------------+


Member types
------------

+-----------------+----------------+----------------------------------------------------------------------------------------+
| Type            | Definition     | Description                                                                            |
+=================+================+========================================================================================+
|                 |                | Name of the kernel.                                                                    |
|                 |                | It can be augmented for a ``kernel_param`` object passed in a multi-kernel template    |
|                 |                | to guarantee uniqueness of the individual kernels.                                     |
|                 |                | For ``DefaultKernelName``, unique implementation-defined name is generated even across |
| ``kernel_name`` | ``KernelName`` | multiple kernel templates if ``fsycl-unnamed-lambda`` is enabled.                      |
+-----------------+----------------+----------------------------------------------------------------------------------------+
