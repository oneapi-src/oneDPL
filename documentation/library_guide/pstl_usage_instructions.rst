Parallel STL Usage Instructions
################################
Follow these steps to use Parallel STL.

Use the DPC++ Policy
=====================
The DPC++ execution policy specifies where and how a Parallel STL algorithm runs. It inherits a standard C++ execution policy, encapsulates a SYCL* device or queue, and allows you to set an optional kernel name. DPC++ execution policies can be used with all standard C++ algorithms that support execution policies according to the ISO/IEC 14882:2017 standard.

Use the policy:

#. Add ``#include <dpstd/execution>`` to your code.
#. Create a policy object by providing a standard policy type (currently, only the ``parallel_unsequenced_policy`` is supported), a class type for a unique kernel name as a template argument (this is optional, see the **Note** below), and one of the following constructor arguments:

    a. a SYCL queue;
    #. a SYCL device;
    #. a SYCL device selector;
    #. an existing policy object with a different kernel name.

#. Pass the created policy object to a Parallel STL algorithm.

``dpstd::execution::default_policy`` object is a predefined object of the ``device_policy`` class created with default kernel name and default queue. Use it to create customized policy objects, or pass directly when invoking an algorithm.

:Note: Providing a kernel name for a policy is optional if the host code used to invoke the kernel is compiled with the Intel® oneAPI DPC++ Compiler. Otherwise you can instead add the ``-fsycl-unnamed-lambda`` option to the compilation command. This compilation option is required if you use the ``dpstd::execution::default_policy`` policy object in the code.

DPC++ Policy Usage Examples
============================
Code examples below assume ``using namespace dpstd::execution;`` and ``using namespace cl::sycl;`` directives when refer to policy classes and functions:

.. code:: cpp

  auto policy_a = device_policy<parallel_unsequenced_policy, class PolicyA> {queue{}};
  std::for_each(policy_a, …);
  
.. code:: cpp

  auto policy_b = device_policy<parallel_unsequenced_policy, class PolicyB> {device{gpu_selector{}}};
  std::for_each(policy_b, …);

.. code:: cpp

  auto policy_c = device_policy<parallel_unsequenced_policy, class PolicyС> {default_selector{}};
  std::for_each(policy_c, …);

.. code:: cpp

  auto policy_d = make_device_policy<class PolicyD>(default_policy);
  std::for_each(policy_d, …);

.. code:: cpp

  auto policy_e = make_device_policy<class PolicyE>(queue{});
  std::for_each(policy_e, …);

.. code:: cpp

  auto policy_f = make_device_policy<class PolicyF>(queue{property::queue::in_order()});
  std::for_each(policy_f, …);

Use the FPGA policy
====================
The ``fpga_device_policy`` class is a DPC++ policy tailored to achieve better performance of parallel algorithms on FPGA hardware devices.

Use the policy when you're going to run the application on FPGA hardware device or FPGA emulation device:

#. Define the ``_PSTL_FPGA_DEVICE`` macro to run on FPGA devices and additionally ``_PSTL_FPGA_EMU`` to run on FPGA emulation device.
#. Add ``#include <dpstd/execution>`` to your code.
#. Create a policy object by providing a class type for a unique kernel name and an unroll factor (see the **Note** below) as template arguments (both optional), and one of the following constructor arguments:

    a. A SYCL queue constructed for `the FPGA selector <https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/IntelFPGA/FPGASelector.md>`_ (the behavior is undefined with any other queue).
    #. An existing FPGA policy object with a different kernel name and/or unroll factor.

#. Pass the created policy object to a Parallel STL algorithm.

The default constructor of ``fpga_device_policy`` creates an object with a SYCL queue constructed for ``fpga_selector``, or for ``fpga_emulator_selector`` if ``_PSTL_FPGA_EMU`` is defined.

``dpstd::execution::fpga_policy`` is a predefined object of the ``fpga_device_policy`` class created with default kernel name and default unroll factor. Use it to create customized policy objects, or pass directly when invoking an algorithm.

:Note: Specifying unroll factor for a policy enables loop unrolling in the implementation of algorithms. Default value is 1. To find out how to choose a better value, you can refer to `unroll Pragma <https://software.intel.com/en-us/oneapi-fpga-optimization-guide-unroll-pragma>`_ and `Loops Analysis <https://software.intel.com/en-us/oneapi-fpga-optimization-guide-loops-analysis>`_ chapters of the `Intel(R) oneAPI DPC++ FPGA Optimization Guide <https://software.intel.com/en-us/oneapi-fpga-optimization-guide>`_.

FPGA Policy Usage Examples
===========================
The code below assumes ``using namespace dpstd::execution;`` for policies and ``using namespace cl::sycl;`` for queues and device selectors:

.. code:: cpp

  auto fpga_policy_a = fpga_device_policy<class FPGAPolicyA>{};
  auto fpga_policy_b = make_fpga_policy(queue{intel::fpga_selector{}});
  constexpr auto unroll_factor = 8;
  auto fpga_policy_c = make_fpga_policy<class FPGAPolicyC, unroll_factor>(fpga_policy);

Include Parallel STL Header Files
==================================
To include Parallel STL header files, add a subset of the following set of lines. These lines are dependent on the algorithms you intend to use:

- ``#include <dpstd/algorithm>``
- ``#include <dpstd/numeric>``
- ``#include <dpstd/memory>``

Use dpstd::begin, dpstd::end Functions
=======================================

The ``dpstd::begin`` and ``dpstd::end`` are special helper functions that allow you to pass SYCL buffers to Parallel STL algorithms. These functions accept a SYCL buffer and return an object of an unspecified type that satisfies the following requirements:

- Is ``CopyConstructible``, ``CopyAssignable``, and comparable with operators == and !=.
- The following expressions are valid: ``a + n``, ``a - n``, and ``a - b``, where ``a`` and ``b`` are objects of the type, and ``n`` is an integer value.
- Has a ``get_buffer`` method with no arguments. The method returns the SYCL buffer passed to ``dpstd::begin`` and ``dpstd::end`` functions.

To use the functions, add ``#include <dpstd/iterator>`` to your code.

Example:

.. code:: cpp

  #include <CL/sycl.hpp>
  #include <dpstd/execution>
  #include <dpstd/algorithm>
  #include <dpstd/iterator>
  int main(){
    cl::sycl::queue q;
    cl::sycl::buffer<int> buf { 1000 };
    auto buf_begin = dpstd::begin(buf);
    auto buf_end   = dpstd::end(buf);
    auto policy = dpstd::execution::make_device_policy<class fill>( q );
    std::fill(policy, buf_begin, buf_end, 42);
    return 0;
  }

:Note: Parallel STL algorithms can be called with ordinary (host-side) iterators, as seen in the code example below. In this case, a temporary SYCL buffer is created and the data is copied to this buffer. After processing of the temporary buffer on a device is complete, the data is copied back to the host. Working with SYCL buffers is recommended to reduce data copying between the host and device.

Example:

.. code:: cpp

  #include <vector>
  #include <dpstd/execution>
  #include <dpstd/algorithm>
  int main(){
    std::vector<int> v( 1000000 );
    std::fill(dpstd::execution::default_policy, v.begin(), v.end(), 42);
    // each element of vec equals to 42
    return 0;
  }

Use Parallel STL with Unified Shared Memory (USM)
==================================================
The following examples demonstrate two ways to use the Parallel STL algorithms with USM:

- USM pointers
- USM allocators

If you have a USM-allocated buffer, pass the pointers to the start and past the end of the buffer to a parallel algorithm. Make sure that the execution policy and the buffer were created for the same queue or context.

If the same buffer is processed by several algorithms, either use an ordered queue or explicitly wait for completion of each algorithm before passing the buffer to the next one. Also wait for completion before accessing the data at the host.

.. code:: cpp

  #include <CL/sycl.hpp>
  #include <dpstd/execution>
  #include <dpstd/algorithm>
  int main(){
    cl::sycl::queue q;
    const int n = 1000;
    int* d_head = static_cast<int*>(cl::sycl::malloc_device(n * sizeof(int),
        q.get_device(), q.get_context()));

    std::fill(dpstd::execution::make_device_policy(q), d_head, d_head + n, 42);
    q.wait();
    cl::sycl::free(d_head, q.get_context());
    return 0;
  }

Alternatively, use ``std::vector`` with a USM allocator:

.. code:: cpp

  #include <CL/sycl.hpp>
  #include <dpstd/execution>
  #include <dpstd/algorithm>
  int main(){
    cl::sycl::queue q;
    const int n = 1000;
    cl::sycl::usm_allocator<int, cl::sycl::usm::alloc::shared> alloc(q.get_context(), q.get_device());
    std::vector<int, decltype(alloc)> vec(n, alloc);

    std::fill(dpstd::execution::make_device_policy(q), vec.begin(), vec.end(), 42);
    q.wait();

    return 0;
  }

Error handling with DPC++ execution policies
=============================================
The DPC++ error handling model supports two types of errors. In case of *synchronous* errors DPC++ host runtime libraries throw exceptions, while *asynchronous* errors may only be processed in a user-supplied error handler associated with a DPC++ queue.

For Parallel STL algorithms executed with DPC++ policies, handling all errors, synchronous or asynchronous, is a responsibility of the caller.
Specifically,

* no exceptions are thrown explicitly by algorithms;
* exceptions thrown by runtime libraries at the host CPU, including DPC++ synchronous exceptions, are passed through to the caller;
* DPC++ asynchronous errors are not handled.

In order to process DPC++ asynchronous errors, the queue associated with a DPC++ policy must be created with an error handler object.
The predefined policy objects (``default_policy`` etc.) have no error handlers; do not use those if you need to process asynchronous errors.

Additional Macros
==================

================================= ==============================
Macro                             Description
================================= ==============================
``_PSTL_BACKEND_SYCL``            This macro enables the use of the DPC++ policy. (This is enabled by default when compiling with the Intel® oneAPI DPC++ Compiler, otherwise it is disabled.)
--------------------------------- ------------------------------
``_PSTL_FPGA_DEVICE``             Use this macro to build your code containing Parallel STL algorithms for FPGA devices. (Disabled by default.)
--------------------------------- ------------------------------
``_PSTL_FPGA_EMU``                Use this macro to build your code containing Parallel STL algorithms for FPGA emulation device. (Disabled by default.)
--------------------------------- ------------------------------
``_PSTL_COMPILE_KERNEL``          Use this macro to get rid of the ``CL_OUT_OF_RESOURCES`` exception that may occur during some invocations of Parallel STL algorithms on CPU and FPGA devices. The macro may increase the execution time of the algorithms. (Enabled by default.)
================================= ==============================

:Note: Define both ``_PSTL_FPGA_DEVICE`` and ``_PSTL_FPGA_EMU`` macros in the same application to run on FPGA emulation device. To run on FPGA hardware device only ``_PSTL_FPGA_DEVICE`` should be defined.

Build Your Code with Parallel STL for DPC++
============================================
Use these steps to build your code with Parallel STL for DPC++.

#. To build with the Intel® oneAPI DPC++ Compiler, see the Get Started with the Intel® oneAPI DPC++ Compiler for details.
#. Set the environment for oneAPI Data Parallel C++ Library and oneAPI Threading Building Blocks.
#. To avoid naming device policy objects explicitly, add the ``–fsycl-unnamed-lambda`` option.

Below is an example of a command line used to compile code that contains Parallel STL algorithms on Linux (depending on the code, parameters within [] could be unnecessary):

.. code::

  dpcpp [–fsycl-unnamed-lambda] test.cpp [-ltbb] -o test
