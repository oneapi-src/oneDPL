Pass Data to Algorithms
#######################

For an algorithm to access data, it is important that the used execution policy matches the data storage type.
The following table shows which execution policies can be used with various data storage types.

================================================ ========================== =============
Data Storage                                     Device Policies            Host Policies
================================================ ========================== =============
`SYCL buffer`_                                   Yes                        No
Device-allocated `unified shared memory`_ (USM)  Yes                        No
Shared and host-allocated USM                    Yes                        Yes
``std::vector`` with ``sycl::usm_allocator``     Yes                        Yes
``std::vector`` with a host allocator            See :ref:`use-std-vector`  Yes
Other host-allocated data                        No                         Yes
================================================ ========================== =============

When using the standard-aligned (or *host*) execution policies, |onedpl_short| supports data being passed
to its algorithms as specified in the C++ standard (C++17 for algorithms working with iterators,
C++20 for parallel range algorithms), with :ref:`known restrictions and limitations <library-restrictions>`.

According to the standard, the calling code must prevent data races when using algorithms
with parallel execution policies.

.. note::
   Implementations of ``std::vector<bool>`` are not required to avoid data races for concurrent modifications
   of vector elements. Some implementations may optimize multiple ``bool`` elements into a bitfield, making it unsafe
   for multithreading. For this reason, it is recommended to avoid ``std::vector<bool>`` for anything but a read-only
   input with the standard-aligned execution policies.

The following subsections describe proper ways to pass data to an algorithm invoked with a device execution policy.

.. _use-buffer-wrappers:

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
to explicitly control which access mode should be applied to a particular buffer when submitting
a SYCL kernel to a device:

.. code:: cpp

  sycl::buffer<int> buf{/*...*/};
  auto first_ro = oneapi::dpl::begin(buf, sycl::read_only);
  auto first_wo = oneapi::dpl::begin(buf, sycl::write_only, sycl::no_init);
  auto first_ni = oneapi::dpl::begin(buf, sycl::no_init);

To use the functions, add ``#include <oneapi/dpl/iterator>`` to your code. For example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <oneapi/dpl/iterator>
  #include <random>
  #include <sycl/sycl.hpp>

  int main(){
    std::vector<int> vec(1000);
    std::generate(vec.begin(), vec.end(), std::minstd_rand{});

    //create a buffer from host memory
    sycl::buffer<int> buf{ vec.data(), vec.size() };
    auto buf_begin = oneapi::dpl::begin(buf);
    auto buf_end   = oneapi::dpl::end(buf);

    oneapi::dpl::sort(oneapi::dpl::execution::dpcpp_default, buf_begin, buf_end);
    return 0;
  }

.. _use-usm:

Use Unified Shared Memory
-------------------------

If you have USM-allocated memory, pass the pointers to the start and past the end
of the sequence to a parallel algorithm. Make sure that the execution policy and
the USM-allocated memory were created for the same queue. For example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <random>
  #include <sycl/sycl.hpp>

  int main(){
    sycl::queue q;
    const int n = 1000;
    int* d_head = sycl::malloc_shared<int>(n, q);
    std::generate(d_head, d_head + n, std::minstd_rand{});

    oneapi::dpl::sort(oneapi::dpl::execution::make_device_policy(q), d_head, d_head + n);

    sycl::free(d_head, q);
    return 0;
  }

.. note::
   Use of non-USM pointers is not supported for algorithms with device execution policies.

When using device USM, such as allocated by ``malloc_device``, you are responsible for data
transfers to and from the device to ensure that input data is device accessible during oneDPL
algorithm execution and that the result is available to the subsequent operations.

.. _use-std-vector:

Use std::vector
---------------

You can use iterators to host-allocated ``std::vector`` data, as shown in the following example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <random>
  #include <vector>

  int main(){
    std::vector<int> vec( 1000 );
    std::generate(vec.begin(), vec.end(), std::minstd_rand{});

    oneapi::dpl::sort(oneapi::dpl::execution::dpcpp_default, vec.begin(), vec.end());
    return 0;
  }

When using iterators to host-allocated data, a temporary SYCL buffer is created, and the data
is copied to this buffer. After processing on a device is complete, the modified data is copied
from the temporary buffer back to the host container. While convenient, using host-allocated
data can lead to unintended copying between host and device. We recommend working with SYCL buffers
or USM memory to reduce data copying between the host and device.

.. note::
   For specialized memory algorithms that begin or end the lifetime of data objects, that is,
   ``uninitialized_*`` and ``destroy*`` families of functions, the data to initialize or destroy
   should be accessible on the device without extra copying. Therefore for these algorithms
   host-allocated data storage may not be used with device execution policies.

You can also use ``std::vector`` with a ``sycl::usm_allocator``, as shown in the following example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <random>
  #include <vector>
  #include <sycl/sycl.hpp>

  int main(){
    const int n = 1000;
    auto policy = oneapi::dpl::execution::dpcpp_default;
    sycl::usm_allocator<int, sycl::usm::alloc::shared> alloc(policy.queue());
    std::vector<int, decltype(alloc)> vec(n, alloc);
    std::generate(vec.begin(), vec.end(), std::minstd_rand{});

    // Recommended to use USM pointers:
    oneapi::dpl::sort(policy, vec.data(), vec.data() + vec.size());
  /*
    // Iterators for USM allocators might require extra copying - not a recommended method
    oneapi::dpl::sort(policy, vec.begin(), vec.end());
  */
    return 0;
  }

Make sure that the execution policy and the USM-allocated memory were created for the same queue.

For ``std::vector`` with a USM allocator we recommend to use ``std::vector::data()`` in
combination with ``std::vector::size()`` as shown in the example above, rather than iterators to
``std::vector``. That is because for some implementations of the C++ Standard Library it might not
be possible for |onedpl_short| to detect that iterators are pointing to USM-allocated data. In that
case the data will be treated as if it were host-allocated, with an extra copy made to a SYCL buffer.
Retrieving USM pointers from ``std::vector`` as shown guarantees no unintended copying.

Use Range Views
---------------

For :doc:`parallel range algorithms <parallel_range_algorithms>` with device execution policies,
place the data in USM or a USM-allocated ``std::vector``, and pass it to an algorithm
via a device-copyable range or view object such as ``std::ranges::subrange`` or ``std::span``.

.. note::
   Use of ``std::ranges::views::all`` is not supported for algorithms with device execution policies.

These data ranges as well as supported range adaptors and factories may be combined into
data transformation pipelines that also can be used with parallel range algorithms. For example:

.. code:: cpp

  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/algorithm>
  #include <random>
  #include <vector>
  #include <span>
  #include <ranges>
  #include <functional>
  #include <sycl/sycl.hpp>

  int main(){
    const int n = 1000;
    auto policy = oneapi::dpl::execution::dpcpp_default;
    sycl::queue q = policy.queue();

    int* d_head = sycl::malloc_host<int>(n, q);
    std::generate(d_head, d_head + n, std::minstd_rand{});

    sycl::usm_allocator<int, sycl::usm::alloc::shared> alloc(q);
    std::vector<int, decltype(alloc)> vec(n, alloc);

    oneapi::dpl::ranges::copy(policy,
        std::ranges::subrange(d_head, d_head + n) | std::views::transform(std::negate{}),
        std::span(vec));

    oneapi::dpl::ranges::sort(policy, std::span(vec));

    sycl::free(d_head, q);
    return 0;
  }

.. _`SYCL buffer`: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:buffers
.. _`unified shared memory`: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:usm
