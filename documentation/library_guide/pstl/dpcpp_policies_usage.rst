Use the DPC++ Execution Policies
#################################

The DPC++ execution policy specifies where and how a Parallel STL algorithm runs.
It encapsulates a SYCL* device or queue, and
allows you to set an optional kernel name. DPC++ execution policies can be used with all
standard C++ algorithms that support execution policies.

To use the policy create a policy object by providing a class type for a unique kernel name as a template argument (this is optional, see the **Note** below), and one of the following constructor arguments:

  #. a SYCL queue;
  #. a SYCL device;
  #. a SYCL device selector;
  #. an existing policy object with a different kernel name.

``oneapi::dpl::execution::dpcpp_default`` object is a predefined object of
the ``device_policy`` class created with default kernel name and default queue.
Use it to create customized policy objects, or pass directly when invoking an algorithm.

:Note: Providing a kernel name for a policy is optional if the host code used to invoke the kernel is compiled with the Intel® oneAPI DPC++ Compiler. Otherwise you can instead add the ``-fsycl-unnamed-lambda`` option to the compilation command. This compilation option is required if you use the ``oneapi::dpl::execution::dpcpp_default`` policy object for more than one algorithm in the code.

The ``make_device_policy`` function templates simplify ``device_policy`` creation.

Usage Examples
===============
Code examples below assume ``using namespace oneapi::dpl::execution;``
and ``using namespace cl::sycl;`` directive when refer to policy classes and functions:

.. code:: cpp

  auto policy_a = device_policy<class PolicyA> {};
  std::for_each(policy_a, …);
  
.. code:: cpp

  auto policy_b = device_policy<class PolicyB> {device{gpu_selector{}}};
  std::for_each(policy_b, …);

.. code:: cpp

  auto policy_c = device_policy<class PolicyС> {cpu_selector{}};
  std::for_each(policy_c, …);

.. code:: cpp

  auto policy_d = make_device_policy<class PolicyD>(dpcpp_default);
  std::for_each(policy_d, …);

.. code:: cpp

  auto policy_e = make_device_policy(queue{property::queue::in_order()});
  std::for_each(policy_e, …);

Use the FPGA policy
====================
The ``fpga_policy`` class is a DPC++ policy tailored to achieve
better performance of parallel algorithms on FPGA hardware devices.

Use the policy when you're going to run the application on FPGA hardware device or FPGA emulation device:

#. Define the ``ONEDPL_FPGA_DEVICE`` macro to run on FPGA devices and additionally ``ONEDPL_FPGA_EMULATOR`` to run on FPGA emulation device.
#. Add ``#include <oneapi/dpl/execution>`` to your code.
#. Create a policy object by providing an unroll factor (see the **Note** below) and a class type for a unique kernel name as template arguments (both optional), and one of the following constructor arguments:

    a. A SYCL queue constructed for `the FPGA selector <https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/IntelFPGA/FPGASelector.md>`_ (the behavior is undefined with any other queue).
    #. An existing FPGA policy object with a different kernel name and/or unroll factor.

#. Pass the created policy object to a Parallel STL algorithm.

The default constructor of ``fpga_policy`` creates an object with a
SYCL queue constructed for ``fpga_selector``, or for ``fpga_emulator_selector``
if ``ONEDPL_FPGA_EMULATOR`` is defined.

``oneapi::dpl::execution::dpcpp_fpga`` is a predefined object of
the ``fpga_policy`` class created with a default unroll factor and a default kernel name.
Use it to create customized policy objects, or pass directly when invoking an algorithm.

:Note: Specifying unroll factor for a policy enables loop unrolling in the implementation of algorithms. Default value is 1. To find out how to choose a better value, you can refer to `unroll Pragma <https://software.intel.com/en-us/oneapi-fpga-optimization-guide-unroll-pragma>`_ and `Loops Analysis <https://software.intel.com/en-us/oneapi-fpga-optimization-guide-loops-analysis>`_ chapters of the `Intel(R) oneAPI DPC++ FPGA Optimization Guide <https://software.intel.com/en-us/oneapi-fpga-optimization-guide>`_.

The ``make_fpga_policy`` function templates simplify ``fpga_policy`` creation.

FPGA Policy Usage Examples
===========================
The code below assumes ``using namespace oneapi::dpl::execution;`` for policies and
``using namespace cl::sycl;`` for queues and device selectors:

.. code:: cpp

  constexpr auto unroll_factor = 8;
  auto fpga_policy_a = fpga_policy<unroll_factor, class FPGAPolicyA>{};
  auto fpga_policy_b = make_fpga_policy(queue{intel::fpga_selector{}});
  auto fpga_policy_c = make_fpga_policy<unroll_factor, class FPGAPolicyC>();

Use oneapi::dpl::begin, oneapi::dpl::end Functions
===================================================

The ``oneapi::dpl::begin`` and ``oneapi::dpl::end`` are special helper functions that allow you to pass SYCL buffers to Parallel STL algorithms. These functions accept a SYCL buffer and return an object of an unspecified type that satisfies the following requirements:

- Is ``CopyConstructible``, ``CopyAssignable``, and comparable with operators == and !=.
- The following expressions are valid: ``a + n``, ``a - n``, and ``a - b``, where ``a`` and ``b`` are objects of the type, and ``n`` is an integer value.
- Has a ``get_buffer`` method with no arguments. The method returns the SYCL buffer passed to ``oneapi::dpl::begin`` and ``oneapi::dpl::end`` functions.

To use the functions, add ``#include <oneapi/dpl/iterator>`` to your code.

Example:

.. code:: cpp

  #include <CL/sycl.hpp>
  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <oneapi/dpl/iterator>
  int main(){
    cl::sycl::buffer<int> buf { 1000 };
    auto buf_begin = oneapi::dpl::begin(buf);
    auto buf_end   = oneapi::dpl::end(buf);
    std::fill(oneapi::dpl::execution::dpcpp_default, buf_begin, buf_end, 42);
    return 0;
  }

:Note: Parallel STL algorithms can be called with ordinary (host-side) iterators, as seen in the code example below. In this case, a temporary SYCL buffer is created and the data is copied to this buffer. After processing of the temporary buffer on a device is complete, the data is copied back to the host. Working with SYCL buffers is recommended to reduce data copying between the host and device.

Example:

.. code:: cpp

  #include <vector>
  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  int main(){
    std::vector<int> v( 1000 );
    std::fill(oneapi::dpl::execution::dpcpp_default, v.begin(), v.end(), 42);
    // each element of vec equals to 42
    return 0;
  }

Use Parallel STL with Unified Shared Memory (USM)
==================================================
The following examples demonstrate two ways to use the Parallel STL algorithms with USM:

- USM pointers
- USM allocators

If you have a USM-allocated buffer, pass the pointers to the start and past the end
of the buffer to a parallel algorithm. Make sure that the execution policy and
the buffer were created for the same queue.

If the same buffer is processed by several algorithms, either use an ordered queue
or explicitly wait for completion of each algorithm before passing the buffer
to the next one. Also wait for completion before accessing the data at the host.

.. code:: cpp

  #include <CL/sycl.hpp>
  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  int main(){
    cl::sycl::queue q;
    const int n = 1000;
    int* d_head = cl::sycl::malloc_device<int>(n q);

    std::fill(oneapi::dpl::execution::make_device_policy(q), d_head, d_head + n, 42);
    q.wait();

    cl::sycl::free(d_head, q);
    return 0;
  }

Alternatively, use ``std::vector`` with a USM allocator:

.. code:: cpp

  #include <CL/sycl.hpp>
  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  int main(){
    cl::sycl::queue q;
    const int n = 1000;
    cl::sycl::usm_allocator<int, cl::sycl::usm::alloc::shared> alloc(q);
    std::vector<int, decltype(alloc)> vec(n, alloc);

    std::fill(oneapi::dpl::execution::make_device_policy(q), vec.begin(), vec.end(), 42);
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
The predefined policy objects (``dpcpp_default`` etc.) have no error handlers; do not use those if you need to process asynchronous errors.

Build Your Code with Parallel STL for DPC++
============================================
Use these steps to build your code with Parallel STL for DPC++.

#. To build with the Intel® oneAPI DPC++ Compiler, see the Get Started with the Intel® oneAPI DPC++ Compiler for details.
#. Set the environment for oneAPI Data Parallel C++ Library and oneAPI Threading Building Blocks.
#. To avoid naming device policy objects explicitly, add the ``–fsycl-unnamed-lambda`` option.

Below is an example of a command line used to compile code that contains Parallel STL algorithms on Linux (depending on the code, parameters within [] could be unnecessary):

.. code::

  dpcpp [–fsycl-unnamed-lambda] test.cpp [-ltbb] -o test
