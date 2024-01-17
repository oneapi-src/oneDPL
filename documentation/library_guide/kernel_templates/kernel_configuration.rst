Kernel Configuration
####################

.. code:: cpp

   // Defined in header <oneapi/dpl/experimental/kernel_templates>

   template <std::uint16_t DataPerWorkItem,
             std::uint16_t WorkGroupSize,
             typename KernelName = oneapi::dpl::execution::DefaultKernelName>
   struct kernel_param;

Each kernel template is supplied by a ``kernel_param`` object, a generic structure for configuring a kernel.
The supported and the most performant values depend on the kernel template itself, device capabilities, the data type, and the data size.

Member Constants
----------------

+------------------------------------------------------+---------------------+------------------------------------------------------+
| Name                                                 | Value               | Description                                          |
+======================================================+=====================+======================================================+
| ``static constexpr std::uint16_t data_per_workitem`` | ``DataPerWorkItem`` | Number of iterations to be processed by a work-item. |
+------------------------------------------------------+---------------------+------------------------------------------------------+
| ``static constexpr std::uint16_t workgroup_size``    | ``WorkGroupSize``   | Number of work-items within a work-group.            |
+------------------------------------------------------+---------------------+------------------------------------------------------+


Member Types
------------

+-----------------+----------------+-----------------------------------------------------------------------------------------+
| Type            | Definition     | Description                                                                             |
+=================+================+=========================================================================================+
| ``kernel_name`` | ``KernelName`` | An optional parameter that is used to set a kernel name.                                |
|                 |                | The behavior is different whether the parameter is provided or not:                     |
|                 |                |                                                                                         |
|                 |                | a. Provided: It is passed as is or augmented to guarantee                               |
|                 |                |    unique kernel names in the kernel template if it has multiple kernels.               |
|                 |                | b. Not provided: The default value (``DefaultKernelName``) results in                   |
|                 |                |    the implementation-defined generation of unique kernels names across the whole       |
|                 |                |    program.                                                                             |
|                 |                |                                                                                         |
|                 |                | .. note::                                                                               |
|                 |                |                                                                                         |
|                 |                |    Automatic kernel name generation (the **not provided** option) assumes that the used |
|                 |                |    compiler and runtime comply with SYCL* 2020 and support optional kernel names.       |
|                 |                |    For example, Intel® oneAPI DPC++/C++ Compiler supports it by default.                |
+-----------------+----------------+-----------------------------------------------------------------------------------------+
