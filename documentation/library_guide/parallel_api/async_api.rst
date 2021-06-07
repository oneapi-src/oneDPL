Asynchronous API algorithms
###########################

The functions defined in the STL ``<algorithm>`` or ``<numeric>`` headers are traditionally blocking. |onedpl_short| extends the
functionality of C++17 parallel algorithms by providing asynchronous algorithms with non-blocking behavior.
This experimental feature enables you to express a concurrent control flow by building dependency chains and interleaving algorithm calls
and interoperability with |dpcpp_short| and SYCL* kernels. 

The current implementation for async algorithms is limited to |dpcpp_short| Execution Policies.
All the functionality described below is available in the ``oneapi::dpl::experimental`` namespace.

The following async algorithms are currently supported:

* ``copy_async``
* ``fill_async``
* ``for_each_async``
* ``reduce_async``
* ``transform_async``
* ``transform_reduce_async``
* ``sort_async``

All the interfaces listed above are a subset of C++17 STL algorithms,
where the suffix ``_async`` is added to the corresponding name (for example: ``reduce``, ``sort``, etc.).
The behavior and signatures are overlapping with the C++17 STL algorithm with the following changes:

* Do not block the execution.
* Take an arbitrary number of events (including 0) as last arguments to allow expressing input dependencies.
* Return future-like object that allows ``wait`` for completion and ``get`` the result.

The type of the future-like object returned from asynchronous algorithm is unspecified. The following member functions are present:

* ``get()`` returns the result.
* ``wait()`` waits for the result to become available.

If the returned object is the result of an algorithm with device policy, it can be converted into a ``sycl::event``.
Lifetime of any resources the algorithm allocates (for example: temporary storage) is bound to the lifetime of the returned object.

Utility functions:

* ``wait_for_all(…)`` waits for an arbitrary number of objects that are convertible into ``sycl::event`` to become ready.


Example of Async API Usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

    #include <oneapi/dpl/execution>
    #include <oneapi/dpl/async>
    #include <CL/sycl.hpp>
    
    int main() {
        using namespace oneapi;
        {
            /* Build and compute a simple dependency chain: Fill buffer -> Transform -> Reduce */
            sycl::buffer<int> a{10};
 
            auto fut1 = dpl::experimental::fill_async(dpl::execution::dpcpp_default, 
                                                      dpl::begin(a),dpl::end(a),7);
            
            auto fut2 = dpl::experimental::transform_async(dpl::execution::dpcpp_default,
                                                           dpl::begin(a),dpl::end(a),dpl::begin(a),
                                                           [&](const int& x){return x + 1; },fut1);
            auto ret_val = dpl::experimental::reduce_async(dpl::execution::dpcpp_default,
                                                           dpl::begin(a),dpl::end(a),fut1,fut2).get();
        }
        return 0;
    }
