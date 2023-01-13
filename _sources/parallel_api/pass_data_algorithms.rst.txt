Pass Data to Algorithms
#######################

You can use one of the following ways to pass data to an algorithm executed with a device policy:

* ``oneapi:dpl::begin`` and ``oneapi::dpl::end`` functions
* Unified shared memory (USM) pointers and ``std::vector`` with USM allocators
* Iterators of host-side ``std::vector``

Use oneapi::dpl::begin and oneapi::dpl::end Functions
-----------------------------------------------------

``oneapi::dpl::begin`` and ``oneapi::dpl::end`` are special helper functions that
allow you to pass SYCL buffers to parallel algorithms. These functions accept
a SYCL buffer and return an object of an unspecified type that provides the following API:

* It satisfies ``CopyConstructible`` and ``CopyAssignable`` C++ named requirements and comparable with
  ``operator==`` and ``operator!=``.
* It gives the following valid expressions: ``a + n``, ``a - n``, and ``a - b``, where ``a`` and ``b``
  are objects of the type, and ``n`` is an integer value. The effect of those operations is the same as for the type
  that satisfies the ``LegacyRandomAccessIterator``, a C++ named requirement.
* It provides the ``get_buffer`` method, which returns the buffer passed to the ``begin`` and ``end`` functions.

The ``begin`` and ``end`` functions can take SYCL 2020 deduction tags and ``sycl::no_init`` as arguments
to explicitly mention which access mode should be applied to the buffer accessor when submitting a
SYCL kernel to a device. For example:

.. code:: cpp

  auto first1 = begin(buf, sycl::read_only);
  auto first2 = begin(buf, sycl::write_only, sycl::no_init);
  auto first3 = begin(buf, sycl::no_init);

The example above allows you to control the access mode for the particular buffer passing to a parallel algorithm.

To use the functions, add ``#include <oneapi/dpl/iterator>`` to your code. For example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <oneapi/dpl/iterator>
  #include <sycl/sycl.hpp>
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
the buffer were created for the same queue. For example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <sycl/sycl.hpp>
  int main(){
    sycl::queue q;
    const int n = 1000;
    int* d_head = sycl::malloc_shared<int>(n, q);

    std::fill(oneapi::dpl::execution::make_device_policy(q), d_head, d_head + n, 42);

    sycl::free(d_head, q);
    return 0;
  }

Alternatively, use ``std::vector`` with a USM allocator. For example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <sycl/sycl.hpp>
  int main(){
    const int n = 1000;
    auto policy = oneapi::dpl::execution::dpcpp_default;
    sycl::usm_allocator<int, sycl::usm::alloc::shared> alloc(policy.queue());
    std::vector<int, decltype(alloc)> vec(n, alloc);

    std::fill(policy, vec.begin(), vec.end(), 42);

    return 0;
  }

When using device USM, such as allocated by ``malloc_device``, manually copy data to this memory
before calling oneDPL algorithms, and copy it back once the algorithms have finished execution.

Use Host-Side std::vector
-----------------------------

|onedpl_long| parallel algorithms can be called with ordinary (host-side) iterators, as seen in the
example below.
In this case, a temporary SYCL buffer is created, and the data is copied to this buffer.
After processing on a device is complete, the modified data is copied from the temporary buffer back
to the host container.
For example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <vector>
  int main(){
    std::vector<int> vec( 1000 );
    std::fill(oneapi::dpl::execution::dpcpp_default, vec.begin(), vec.end(), 42);
    // each element of vec equals to 42
    return 0;
  }

Working with SYCL buffers is recommended to reduce data copying between the host and device.