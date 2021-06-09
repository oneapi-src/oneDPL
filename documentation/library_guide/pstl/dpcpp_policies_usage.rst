Parallel STL Usage
##################

Follow these steps to add Parallel STL to your application:

#. Add ``#include <oneapi/dpl/execution>`` to your code.
   Then include one or more of the following header files, depending on the algorithms you
   intend to use:

   #. ``#include <oneapi/dpl/algorithm>``
   #. ``#include <oneapi/dpl/numeric>``
   #. ``#include <oneapi/dpl/memory>``

   For better coexistence with the C++ standard library,
   include |onedpl_long| header files before the standard C++ header files.

#. Pass a |onedpl_short| execution policy object, defined in the ``oneapi::dpl::execution``
   namespace, to a parallel algorithm.
#. Use the C++ Standard Execution Policies:

   #. Compile the code with options that enable OpenMP* vectorization pragmas.
   #. Link with the |onetbb_long| or |tbb_long| dynamic library for parallelism.

#. Use the |dpcpp_long| Execution Policies:

   #. Compile the code with options that enable support for SYCL* 2020.

Use the C++ Standard Execution Policies
=======================================

Example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <vector>

  int main()
  {
      std::vector<int> data( 1000 );
      std::fill(oneapi::dpl::execution::par_unseq, data.begin(), data.end(), 42);
      return 0;
  }

Use the |dpcpp_short| Execution Policies
========================================

The |dpcpp_short| execution policy specifies where a parallel algorithm runs.
It encapsulates a SYCL* device or queue, and
allows you to set an optional kernel name. |dpcpp_short| execution policies can be used with all
standard C++ algorithms that support execution policies.

To use the policy, create a policy object by providing a class type for a unique kernel name
as a template argument, and one of the following constructor arguments:

* A SYCL queue
* A SYCL device
* A SYCL device selector
* An existing policy object with a different kernel name

Providing a kernel name for a policy is optional if the used compiler supports implicit
names for SYCL kernel functions. The |dpcpp_cpp| supports it by default;
for other compilers it may need to be enabled with compilation options such as
``-fsycl-unnamed-lambda``. Refer to your compiler documentation for more information.

The ``oneapi::dpl::execution::dpcpp_default`` object is a predefined object of
the ``device_policy`` class. It is created with a default kernel name and a default queue.
Use it to create customized policy objects, or pass directly when invoking an algorithm.

If ``dpcpp_default`` is passed directly to more than one algorithm, you must enable implicit
kernel names (see above) for compilation.

The ``make_device_policy`` function templates simplify ``device_policy`` creation.

Usage Examples
==============

The code examples below assume you are ``using namespace oneapi::dpl::execution;``
and ``using namespace sycl;`` directives when referring to policy classes and functions:

.. code::

  auto policy_a = device_policy<class PolicyA> {};
  std::for_each(policy_a, …);

.. code::

  auto policy_b = device_policy<class PolicyB> {device{gpu_selector{}}};
  std::for_each(policy_b, …);

.. code::

  auto policy_c = device_policy<class PolicyС> {cpu_selector{}};
  std::for_each(policy_c, …);

.. code::

  auto policy_d = make_device_policy<class PolicyD>(dpcpp_default);
  std::for_each(policy_d, …);

.. code::

  auto policy_e = make_device_policy(queue{property::queue::in_order()});
  std::for_each(policy_e, …);

Use the FPGA Policy
===================

The ``fpga_policy`` class is a |dpcpp_short| policy tailored to achieve
better performance of parallel algorithms on FPGA hardware devices.

Use the policy when you run the application on a FPGA hardware device or FPGA emulation device:

#. Define the ``ONEDPL_FPGA_DEVICE`` macro to run on FPGA devices and the ``ONEDPL_FPGA_EMULATOR``
   to run on FPGA emulation devices.
#. Add ``#include <oneapi/dpl/execution>`` to your code.
#. Create a policy object by providing an unroll factor (see the **Note** below) and
   a class type for a unique kernel name as template arguments (both optional), and one of the
   following constructor arguments:

   #. A SYCL queue constructed for the
      `FPGA Selector <https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/IntelFPGA/FPGASelector.md>`_
      (the behavior is undefined with any other queue).
   #. An existing FPGA policy object with a different kernel name and/or unroll factor.

#. Pass the created policy object to a parallel algorithm.

The default constructor of ``fpga_policy`` creates an object with a
SYCL queue constructed for ``fpga_selector``, or for ``fpga_emulator_selector``
if the ``ONEDPL_FPGA_EMULATOR`` is defined.

``oneapi::dpl::execution::dpcpp_fpga`` is a predefined object of
the ``fpga_policy`` class created with a default unroll factor and a default kernel name.
Use it to create customized policy objects, or pass directly when invoking an algorithm.

.. Note::

   Specifying unroll factor for a policy enables loop unrolling in the implementation of
   algorithms. Default value is 1.
   To find out how to choose a better value, you can refer to the `unroll Pragma <https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/fpga-optimization-flags-attributes-pragmas-and-extensions/loop-directives/unroll-pragma.html>`_
   and `Loops Analysis <https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top/analyze-your-design/analyze-the-fpga-early-image/review-the-report-html-file/loops-analysis.html>`_ chapters of
   the `Intel® oneAPI DPC++ FPGA Optimization Guide
   <https://software.intel.com/content/www/us/en/develop/documentation/oneapi-fpga-optimization-guide/top.html>`_.

The ``make_fpga_policy`` function templates simplify ``fpga_policy`` creation.

FPGA Policy Usage Examples
==========================

The code below assumes ``using namespace oneapi::dpl::execution;`` for policies and
``using namespace sycl;`` for queues and device selectors:

.. code:: cpp

  constexpr auto unroll_factor = 8;
  auto fpga_policy_a = fpga_policy<unroll_factor, class FPGAPolicyA>{};
  auto fpga_policy_b = make_fpga_policy(queue{intel::fpga_selector{}});
  auto fpga_policy_c = make_fpga_policy<unroll_factor, class FPGAPolicyC>();

Pass Data to Algorithms
=======================

You can use one of the following ways to pass data to an algorithm executed with a |dpcpp_short| policy:

* ``oneapi:dpl::begin`` and ``oneapi::dpl::end`` functions
* Unified shared memory (USM) pointers and ``std::vector`` with USM allocators
* Iterators of host-side ``std::vector``

Use oneapi::dpl::begin and oneapi::dpl::end Functions
-----------------------------------------------------

``oneapi::dpl::begin`` and ``oneapi::dpl::end`` are special helper functions that
allow you to pass SYCL buffers to parallel algorithms. These functions accept
a SYCL buffer and return an object of an unspecified type that provides the following
API:

* it satisfies ``CopyConstructible``, ``CopyAssignable`` C++ named requirements and comparable with ``operator==`` and ``operator!=``
* the following expressions are valid: ``a + n``, ``a - n``, and ``a - b``, where ``a`` and ``b``
  are objects of the type, and ``n`` is an integer value. Effect for those operations is the same as for the type
  that satisfies ``LegacyRandomAccessIterator`` C++ named requirement
* it provides the ``get_buffer`` method that returns the buffer passed to the ``begin`` and ``end`` functions

``begin``, ``end`` can optionally take SYCL 2020 deduction tags and ``sycl::no_init`` as arguments
to explicitly mention, which access mode should be applied to the buffer accessor when submitting
DPC++ kernel to a device. For example:

.. code:: cpp

  auto first1 = begin(buf, sycl::read_only);
  auto first2 = begin(buf, sycl::write_only, sycl::no_init);
  auto first3 = begin(buf, sycl::no_init);

It allows you to control the access mode for the particular buffer passing to a parallel algorithm.

To use the functions, add ``#include <oneapi/dpl/iterator>`` to your code.

Example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <oneapi/dpl/iterator>
  #include <CL/sycl.hpp>
  int main(){
    sycl::buffer<int> buf { 1000 };
    auto buf_begin = oneapi::dpl::begin(buf);
    auto buf_end   = oneapi::dpl::end(buf);
    std::fill(oneapi::dpl::execution::dpcpp_default, buf_begin, buf_end, 42);
    return 0;
  }

Use Unified Shared Memory
-------------------------

The following examples demonstrate two ways to use the parallel algorithms with USM:

* USM pointers
* USM allocators

If you have a USM-allocated buffer, pass the pointers to the start and past the end
of the buffer to a parallel algorithm. Make sure that the execution policy and
the buffer were created for the same queue.

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <CL/sycl.hpp>
  int main(){
    sycl::queue q;
    const int n = 1000;
    int* d_head = sycl::malloc_device<int>(n, q);

    std::fill(oneapi::dpl::execution::make_device_policy(q), d_head, d_head + n, 42);

    sycl::free(d_head, q);
    return 0;
  }

Alternatively, use ``std::vector`` with a USM allocator:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <CL/sycl.hpp>
  int main(){
    const int n = 1000;
    auto policy = oneapi::dpl::execution::dpcpp_default;
    sycl::usm_allocator<int, sycl::usm::alloc::shared> alloc(policy.queue());
    std::vector<int, decltype(alloc)> vec(n, alloc);

    std::fill(policy, vec.begin(), vec.end(), 42);

    return 0;
  }

Use Host-Side ``std::vector``
-----------------------------

|onedpl_short| parallel algorithms can be called with ordinary (host-side) iterators, as seen in the
example below.
In this case, a temporary SYCL buffer is created and the data is copied to this buffer.
After processing of the temporary buffer on a device is complete, the data is copied back
to the host. Working with SYCL buffers is recommended to reduce data copying between the host and device.

Example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <vector>
  int main(){
    std::vector<int> v( 1000 );
    std::fill(oneapi::dpl::execution::dpcpp_default, v.begin(), v.end(), 42);
    // each element of vec equals to 42
    return 0;
  }

Error Handling with |dpcpp_short| Execution Policies
====================================================

The |dpcpp_short| error handling model supports two types of errors. In cases of synchronous errors
|dpcpp_short| host runtime libraries throw exceptions, while asynchronous errors may only
be processed in a user-supplied error handler associated with a |dpcpp_short| queue.

For algorithms executed with |dpcpp_short| policies, handling all errors, synchronous or asynchronous, is a
responsibility of the caller. Specifically:

* No exceptions are thrown explicitly by algorithms.
* Exceptions thrown by runtime libraries at the host CPU, including |dpcpp_short| synchronous exceptions,
  are passed through to the caller.
* |dpcpp_short| asynchronous errors are not handled.

In order to process |dpcpp_short| asynchronous errors, the queue associated with a |dpcpp_short| policy must be
created with an error handler object. The predefined policy objects (``dpcpp_default`` etc.) have
no error handlers; do not use those if you need to process asynchronous errors.

Restrictions
============

When used with |dpcpp_short| execution policies, |onedpl_short| algorithms apply the same restrictions as |dpcpp_short|
does (see the |dpcpp_short| specification and the SYCL specification for details), such as:

* Adding buffers to a lambda capture list is not allowed for lambdas passed to an algorithm.
* Passing data types, which are not trivially constructible, is only allowed in USM,
  but not in buffers or host-allocated containers.

Known Limitations
=================

For ``transform_exclusive_scan``, ``transform_inclusive_scan`` algorithms result of
unary operation should be convertible to the type of the initial value if one is provided,
otherwise to the type of values in the processed data sequence
(``std::iterator_traits<IteratorType>::value_type``).

Build Your Code with |onedpl_short|
===================================

Use these steps to build your code with |onedpl_short|:

#. To build with the |dpcpp_cpp|, see the `Get Started with the Intel® oneAPI DPC++/C++ Compiler
   <https://software.intel.com/content/www/us/en/develop/documentation/get-started-with-dpcpp-compiler/top.html>`_
   for details.
#. Set the environment for |onedpl_short| and |onetbb_short|.
#. To avoid naming device policy objects explicitly, add the ``–fsycl-unnamed-lambda`` option.

Below is an example of a command line used to compile code that contains
|onedpl_short| parallel algorithms on Linux* (depending on the code, parameters within [] could be unnecessary):

.. code::

  dpcpp [–fsycl-unnamed-lambda] test.cpp [-ltbb] -o test
